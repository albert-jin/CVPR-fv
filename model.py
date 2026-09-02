"""CVPR-FV LightningModule.

Two heads share one LoRA-adapted PLM:

* ``fv``  — decomposition head. Given a prompted input ``X_fv^{(k)}``
             encoded from (evidence, claim, hypothesis-state ``s_k``),
             returns a Yes/No confidence ``q_k``.
* ``cvp`` — verifiability head. Given ``X_cvp = CVP-PROMPT ⊕ c``,
             returns ``v = p(Verifiable | c)`` — a scalar in [0, 1].

Training loss (main.tex Eq. after "joint optimization objective"):

    L = L_fv + λ_cvp · L_cvp

Both losses are cross-entropy over the two-answer candidate space,
computed via the T-Few length-normalised scoring:

    NLL_i = Σ_t 1[y_t≠pad] · (-log p(y_t)) / |{t : y_t≠pad}|

Aggregation (Section 3.2):

    q_true = 1 - NLL_Yes_true / (NLL_Yes_true + NLL_No_true)        (softmax over choices)
    q_false = same, but on the ``false`` prefix variant
    q_uncertain = same, but on the ``uncertain`` prefix variant

    ℓ_Sup = q_true·(1-q_false)·(1-q_uncertain)
    ℓ_Ref = q_false·(1-q_true)·(1-q_uncertain)
    ℓ_NEI = q_uncertain·(1-q_true)·(1-q_false)
    d_y = (ℓ_y + ε) / Σ(ℓ + ε)

    r_NEI(v) = (1-v) + γ·v
    r_SUP(v) = r_REF(v) = (1-γ)·v/2

    a_y = d_y · r_y(v)^λ
    ŷ = argmax_y a_y

The normalized ``a_y`` values are decision scores, not a Bayesian posterior.
"""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score, classification_report
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, Adafactor
from torch.optim.lr_scheduler import LambdaLR

import configs
from configs import resolve_backbone_path, BACKBONE_KIND
from utils import modify_with_lora, count_trainable_params


class CVPRFV(LightningModule):
    """Verifiability-aware fact-verification model."""

    def __init__(self, backbone: str = None, config_module=None):
        super().__init__()
        self.cfg = config_module or configs
        backbone = (backbone or self.cfg.backbone).lower()
        self.backbone_name = backbone
        self.kind = BACKBONE_KIND[backbone]
        path = resolve_backbone_path(backbone)

        print(f'load PLM ({backbone}, kind={self.kind}) from {path} ...')
        loader = AutoModelForSeq2SeqLM if self.kind == 'seq2seq' else AutoModelForCausalLM
        target_dtype = torch.bfloat16 if self.cfg.compute_precision == 'bf16' else torch.float32
        self.transformer = loader.from_pretrained(
            path,
            local_files_only=os.path.isdir(path),
            low_cpu_mem_usage=True,
            torch_dtype=target_dtype,
        )
        # Freeze backbone and inject LoRA adapters.
        modify_with_lora(self.transformer, kind=self.kind)
        # Newly-created adapter parameters are fp32 by default — cast the
        # whole module (adapters included) to the backbone dtype so forward
        # activations and weights stay consistent.
        self.transformer.to(target_dtype)

        self.trainable_param_names: set = set()
        self._val_predictions: List[Dict] = []
        self._val_cvp_v: Dict[int, float] = {}
        self._best_metric = -1.0
        self._patience_ctr = 0

        print('trainable params: {:,}'.format(count_trainable_params(self.transformer)))

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        param_groups = []
        for name, p in self.transformer.named_parameters():
            if re.fullmatch(configs.trainable_param_names_re, name):
                assert p.requires_grad, name
                param_groups.append(p)
                self.trainable_param_names.add(name)
            else:
                p.requires_grad = False
        optimizer = Adafactor(
            [{'params': param_groups}],
            lr=configs.lr, weight_decay=configs.weight_decay,
            scale_parameter=configs.scale_parameter,
            relative_step=False, warmup_init=False,
        )
        total_steps = max(1, configs.num_steps)
        num_warmup = int(total_steps * configs.warmup_ratio)

        def lr_lambda(step: int):
            if step < num_warmup:
                return float(step) / float(max(1, num_warmup))
            return max(0.0, float(total_steps - step) / float(max(1, total_steps - num_warmup)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    # ------------------------------------------------------------------
    # Backbone forward — handles both seq2seq and causal LM.
    # ------------------------------------------------------------------

    def _forward_choices(self, input_ids: torch.Tensor, choices_ids: torch.Tensor):
        """Return (choices_scores, token_ce) where ``choices_scores`` is
        the length-normalised NLL for each candidate."""
        pad_id = self.cfg.PAD_TOKEN_ID
        bsz, n_choices, choice_len = choices_ids.shape
        attention_mask = (input_ids != pad_id).long()

        if self.kind == 'seq2seq':
            encoder_hidden = self.transformer.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
            )[0]
            enc_hidden_rep = encoder_hidden.unsqueeze(1).repeat(1, n_choices, 1, 1).flatten(0, 1)
            attn_rep = attention_mask.unsqueeze(1).repeat(1, n_choices, 1).flatten(0, 1).float()
            flat_choices = choices_ids.flatten(0, 1)
            decoder_input_ids = torch.cat([torch.zeros_like(flat_choices[:, :1]), flat_choices[:, :-1]], dim=1)
            decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.float)
            out = self.transformer(
                attention_mask=attn_rep,
                encoder_outputs=[enc_hidden_rep],
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            logits = out.logits
            lm_target = flat_choices.clone()
            lm_target[flat_choices == pad_id] = -100
            token_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), lm_target.reshape(-1),
                reduction='none',
            ).reshape(bsz, n_choices, choice_len)
        else:
            # Causal LM: concatenate prompt + candidate answer, compute
            # per-candidate NLL only over the answer tokens.
            prompt_len = input_ids.size(1)
            token_ce = input_ids.new_zeros(bsz, n_choices, choice_len, dtype=torch.float)
            for c_idx in range(n_choices):
                cand = choices_ids[:, c_idx, :]                      # (B, L_c)
                concat = torch.cat([input_ids, cand], dim=1)         # (B, L_p+L_c)
                attn = torch.cat([attention_mask,
                                  (cand != pad_id).long()], dim=1)
                out = self.transformer(input_ids=concat, attention_mask=attn)
                logits = out.logits.float()                           # (B, L_p+L_c, V)
                # Shift for next-token prediction: predict token t from position t-1.
                # We only score positions ``prompt_len .. prompt_len+choice_len-1``.
                shift_logits = logits[:, prompt_len - 1: prompt_len + choice_len - 1, :]
                lm_target = cand.clone()
                lm_target[cand == pad_id] = -100
                ce = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    lm_target.reshape(-1), reduction='none',
                ).reshape(bsz, choice_len)
                token_ce[:, c_idx, :] = ce

        choice_lens = (choices_ids != pad_id).sum(dim=-1).clamp(min=1).float()  # (B, C)
        choices_scores = token_ce.sum(dim=-1) / choice_lens
        return choices_scores, token_ce

    # ------------------------------------------------------------------
    # Loss (shared by FV and CVP heads because both are two-way choice)
    # ------------------------------------------------------------------

    @staticmethod
    def _triple_loss(choices_scores: torch.Tensor, token_ce: torch.Tensor,
                     choices_ids: torch.Tensor, answer_label: torch.Tensor,
                     pad_id: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        bsz, n_choices, choice_len = choices_ids.shape

        # L_LM: cross-entropy over the correct target sequence.
        gold_ce = token_ce.gather(1, answer_label.view(bsz, 1, 1)
                                             .expand(bsz, 1, choice_len)).squeeze(1)
        gold_mask = (choices_ids.gather(1, answer_label.view(bsz, 1, 1)
                                                     .expand(bsz, 1, choice_len))
                                .squeeze(1) != pad_id).float()
        lm_loss = (gold_ce * gold_mask).sum() / gold_mask.sum().clamp(min=1)

        # L_UK: unlikelihood on incorrect targets.
        pad_mask = (choices_ids != pad_id).float()
        one_minus_p = (1.0 - torch.exp(-token_ce)).clamp(min=1e-6)
        cand_log = torch.log(one_minus_p) * pad_mask
        cand_log_masked = cand_log.clone()
        cand_log_masked.scatter_(1, answer_label.view(bsz, 1, 1)
                                                .expand(bsz, 1, choice_len), 0.0)
        pad_mask_masked = pad_mask.clone()
        pad_mask_masked.scatter_(1, answer_label.view(bsz, 1, 1)
                                                .expand(bsz, 1, choice_len), 0.0)
        unlikely_loss = -cand_log_masked.sum() / pad_mask_masked.sum().clamp(min=1)

        # L_LN: length-normalised softmax cross-entropy over the candidates.
        ln_loss = F.cross_entropy(-choices_scores, answer_label)

        loss = lm_loss + unlikely_loss + ln_loss
        return loss, {
            'lm_loss': lm_loss.detach().float(),
            'unlikely_loss': unlikely_loss.detach().float(),
            'ln_loss': ln_loss.detach().float(),
        }

    # ------------------------------------------------------------------
    # Training step: joint FV + CVP objective
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        choices_ids = batch['choices_ids']
        answer_label = batch['answer_label']
        task_flag = batch['task_flag']

        choices_scores, token_ce = self._forward_choices(input_ids, choices_ids)
        bsz = input_ids.size(0)

        # Split rows by task and compute the two losses on their respective masks.
        fv_mask = task_flag == 0
        cvp_mask = task_flag == 1

        total_loss = choices_scores.sum() * 0.0  # keeps device / dtype
        log_dict = {}

        if fv_mask.any():
            fv_loss, fv_log = self._triple_loss(
                choices_scores[fv_mask], token_ce[fv_mask],
                choices_ids[fv_mask], answer_label[fv_mask],
                pad_id=self.cfg.PAD_TOKEN_ID,
            )
            total_loss = total_loss + fv_loss
            for k, v in fv_log.items():
                log_dict[f'train/fv_{k}'] = v

        if cvp_mask.any() and self.cfg.use_cvp:
            cvp_loss, cvp_log = self._triple_loss(
                choices_scores[cvp_mask], token_ce[cvp_mask],
                choices_ids[cvp_mask], answer_label[cvp_mask],
                pad_id=self.cfg.PAD_TOKEN_ID,
            )
            total_loss = total_loss + configs.lam_cvp * cvp_loss
            for k, v in cvp_log.items():
                log_dict[f'train/cvp_{k}'] = v
            log_dict['train/cvp_loss'] = cvp_loss.detach().float()

        log_dict['train/total_loss'] = total_loss.detach().float()
        self.log_dict(log_dict, on_step=True, prog_bar=False, batch_size=bsz)
        return total_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        choices_ids = batch['choices_ids']
        meta = batch['meta']
        v_label = batch['v_label_idx']
        task_flag = batch['task_flag']

        with torch.no_grad():
            choices_scores, _ = self._forward_choices(input_ids, choices_ids)
        # q_yes = softmax over (−NLL_yes, −NLL_no).
        # We store both scores so the aggregation step can use them.
        yes_nll = choices_scores[:, 0]
        no_nll = choices_scores[:, 1]
        # Convert length-normalised NLL to Yes-probability.
        yes_prob = torch.softmax(torch.stack([-yes_nll, -no_nll], dim=-1), dim=-1)[:, 0]

        for i in range(input_ids.size(0)):
            self._val_predictions.append({
                'instance_idx': meta[i]['instance_idx'],
                'instance_id': meta[i]['instance_id'],
                'int_prefix_idx': meta[i]['int_prefix_idx'],
                'template_idx': meta[i]['template_idx'],
                'yes_prob': float(yes_prob[i].item()),
                'gold_v_label': int(v_label[i].item()),
                'task_flag': int(task_flag[i].item()),
                'claim': meta[i]['claim'],
            })
        return None

    # ------------------------------------------------------------------
    # CVP inference — batched over the claims we saw in validation.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def cvp_predict(self, claims: List[str], batch_size: int = 16) -> Dict[str, float]:
        """Compute ``v = p(Verifiable | claim)`` for each unique claim.

        Batches through the same LoRA-adapted backbone using the CVP prompt.
        """
        if not claims or not self.cfg.use_cvp:
            return {c: 0.5 for c in claims}
        from data_reader import build_cvp_inference_batch
        out: Dict[str, float] = {}
        for start in range(0, len(claims), batch_size):
            chunk = claims[start:start + batch_size]
            batch = build_cvp_inference_batch(chunk)
            input_ids = batch['input_ids'].to(self.device)
            choices_ids = batch['choices_ids'].to(self.device)
            choices_scores, _ = self._forward_choices(input_ids, choices_ids)
            v_prob = torch.softmax(-choices_scores, dim=-1)[:, 0]
            for i, claim in enumerate(chunk):
                out[claim] = float(v_prob[i].item())
        return out

    # ------------------------------------------------------------------
    # Heuristic score aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(q_true: float, q_false: float, q_uncertain: float,
                   v: float, lam: float, gamma: float) -> Tuple[int, np.ndarray]:
        """Return (predicted_v_label_idx, normalized_decision_scores).

        The output indexing is ``VLabel2Idx`` (SUPPORT=0, REFUTE=1, NEI=2).
        Normalization is convenient for logging only; these values are not
        interpreted as a Bayesian posterior.
        """
        eps = 1e-12
        # Decomposition-based compatibility scores (main text Eqs 6-8).
        l_sup = q_true * (1 - q_false) * (1 - q_uncertain)
        l_ref = q_false * (1 - q_true) * (1 - q_uncertain)
        l_nei = q_uncertain * (1 - q_true) * (1 - q_false)
        raw_scores = np.array([l_sup, l_ref, l_nei])
        det_scores = (raw_scores + eps) / (raw_scores + eps).sum()

        # Verifiability compatibility scores (main text Eqs 9-10).
        r_nei = (1 - v) + gamma * v
        r_sr = (1 - gamma) * v / 2.0
        compatibility = np.array([r_sr, r_sr, r_nei])          # SUPPORT, REFUTE, NEI

        # Log-linear heuristic fusion (main text Eq 11).
        log_a = np.log(det_scores + eps) + lam * np.log(compatibility + eps)
        decision_scores = np.exp(log_a - log_a.max())
        decision_scores = decision_scores / max(decision_scores.sum(), eps)
        return int(decision_scores.argmax()), decision_scores

    # ------------------------------------------------------------------
    # Validation epoch end — aggregate and report per-class F1 + macro-F1
    # ------------------------------------------------------------------

    def validation_epoch_end(self, outputs):
        # Group by instance (FV rows only).
        buckets: Dict[int, List[Dict]] = defaultdict(list)
        for row in self._val_predictions:
            if row['task_flag'] == 0:
                buckets[row['instance_idx']].append(row)

        if not buckets:
            self._val_predictions = []
            return {}

        # Compute CVP scores for every unique claim seen at eval time.
        unique_claims: List[str] = []
        claim_of_instance: Dict[int, str] = {}
        seen = set()
        for instance_idx, rows in buckets.items():
            claim = rows[0]['claim']
            claim_of_instance[instance_idx] = claim
            if claim not in seen:
                seen.add(claim)
                unique_claims.append(claim)
        v_map = self.cvp_predict(unique_claims) if self.cfg.use_cvp else {c: 0.5 for c in unique_claims}

        y_true, y_pred, y_v = [], [], []
        export_rows: List[Dict] = []
        for instance_idx, rows in buckets.items():
            # Pick one row per prefix (first one — templates are randomised).
            q = {}
            for r in rows:
                q.setdefault(r['int_prefix_idx'], r['yes_prob'])
            q_true = q.get(0, 0.5)
            q_uncertain = q.get(1, 0.5)
            q_false = q.get(2, 0.5)
            claim = claim_of_instance[instance_idx]
            v = v_map.get(claim, 0.5)
            pred, decision_scores = self._aggregate(
                q_true, q_false, q_uncertain, v,
                lam=configs.lam_prior, gamma=configs.nei_floor_gamma,
            )
            y_true.append(rows[0]['gold_v_label'])
            y_pred.append(pred)
            y_v.append(v)

            binary_answers = [
                0 if q_true >= 0.5 else 1,
                0 if q_uncertain >= 0.5 else 1,
                0 if q_false >= 0.5 else 1,
            ]
            valid_answer_rows = {(0, 1, 1), (1, 0, 1), (1, 1, 0)}
            export_rows.append({
                'instance_idx': int(instance_idx),
                'instance_id': rows[0]['instance_id'],
                'claim': claim,
                'gold_label': configs.Idx2VLabel[int(rows[0]['gold_v_label'])],
                'q_true': float(q_true),
                'q_false': float(q_false),
                'q_uncertain': float(q_uncertain),
                'v': float(v),
                'binary_answers_true_uncertain_false': binary_answers,
                'lookup_conflict': tuple(binary_answers) not in valid_answer_rows,
                'prediction': configs.Idx2VLabel[int(pred)],
                'decision_scores_support_refute_nei': decision_scores.tolist(),
            })

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        # Per-class F1 (SUPPORT, REFUTE, NEI in VLabel2Idx order).
        per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
        for cls_idx, cls_name in enumerate(['SUPPORT', 'REFUTE', 'NEI']):
            self.log(f'val/f1_{cls_name}', float(per_class[cls_idx]), prog_bar=False)
        self.log('val/macro_f1', float(macro_f1), prog_bar=True)
        print(f'\n[val] macro_f1={macro_f1:.4f} | '
              f'SUPPORT={per_class[0]:.4f} REFUTE={per_class[1]:.4f} NEI={per_class[2]:.4f} | '
              f'mean v={np.mean(y_v):.3f}')

        # Overwrite on every validation pass so the final/best-model evaluation
        # leaves one self-contained per-instance file for downstream analyses.
        prediction_dir = os.path.join(configs.exp_root, configs.exp_name)
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_path = os.path.join(prediction_dir, 'predictions.jsonl')
        with open(prediction_path, 'w', encoding='utf-8') as stream:
            for row in export_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'[val] wrote {len(export_rows)} predictions to {prediction_path}')

        # Early stopping.
        if macro_f1 > self._best_metric:
            self._best_metric = macro_f1
            self._patience_ctr = 0
            self.save_model(tag='best')
        else:
            self._patience_ctr += 1
            if self._patience_ctr >= self.cfg.patience:
                print(f'[early-stop] no improvement for {self._patience_ctr} evals.')
                self.trainer.should_stop = True

        self._val_predictions = []
        return {'macro_f1': macro_f1}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def load_adapter_weights(self, path: str):
        if not path or not os.path.exists(path):
            return
        state = torch.load(path, map_location='cpu')
        load_result = self.transformer.load_state_dict(state, strict=False)
        assert len(load_result.unexpected_keys) == 0, (
            f'Unexpected keys while loading adapter weights: {load_result.unexpected_keys}'
        )
        print(f'[CVPR-FV] loaded adapter weights from {path}')

    def save_model(self, tag: str = 'step'):
        if not self.cfg.save_model:
            return
        exp_dir = os.path.join(self.cfg.exp_root, self.cfg.exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        if tag == 'best':
            path = os.path.join(exp_dir, 'best.pt')
        elif tag == 'finish':
            path = os.path.join(exp_dir, 'finish.pt')
        else:
            path = os.path.join(exp_dir, f'global_step{self.global_step}.pt')
        states = {k: v.detach().cpu()
                  for k, v in self.transformer.state_dict().items()
                  if k in self.trainable_param_names}
        torch.save(states, path)
        print(f'[CVPR-FV] saved adapter weights to {path}')

    def on_train_end(self):
        self.save_model(tag='finish')


if __name__ == '__main__':
    m = CVPRFV()
    m.configure_optimizers()
    print('CVPR-FV instantiated.')
