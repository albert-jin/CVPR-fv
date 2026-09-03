"""Det2Ver LightningModule — cross-task rumor-detection to fact-verification
framework built on top of T0-3B.

The module implements:

* Backbone: ``AutoModelForSeq2SeqLM`` (T0-3B) with a LoRA / (IA)^3 adapter
  swapped in by ``utils.modify_with_lora``. Only the adapter parameters
  are optimized (Section III-B).

* Losses (Section III-B, Equations 1--4):

    L_LM  — cross-entropy on the correct Yes/No target
    L_UK  — unlikelihood on the incorrect target
    L_LN  — length-normalised softmax cross-entropy over the candidates
    L_tot = L_LM + L_UK + L_LN

  RD examples share the same loss triple (the input is the raw claim
  and the target is Yes / No).

* Label Words Synchronization Engine (Section III-C, Table I,
  Equations 5--8): after gathering the length-normalised cross-entropies
  for the three internal prefixes of a single FV instance the module
  first tries a lookup-table match; on failure it falls back to the
  probability-ranking rule that picks the class minimising
  sum_{d in D_s} (-Logit_{d|s}).

At validation time the model aggregates every variant that shares the
same ``instance_idx`` and returns one three-way prediction per FV
instance so that Macro-F1 can be computed.
"""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score
from transformers import AutoModelForSeq2SeqLM, Adafactor
from torch.optim.lr_scheduler import LambdaLR

import configs
from configs import (
    pretrained_model_path, _load_local_only,
    trainable_param_names_re, lr, scale_parameter, weight_decay,
    num_steps, warmup_ratio,
    IntPres, MapTab, DLabel2Idx, VLabel2Idx, Idx2VLabel, n_ways,
)
from utils import modify_with_lora, count_trainable_params


# Static row of Table I encoded as (verification-label-id, internal-prefix-id) -> answer-id.
# ``VLabel2Idx`` fixes the row order (SUPPORT=0 / REFUTE=1 / NEI=2). Column
# order matches ``IntPres`` = [true, uncertain, false]. The engine relies on
# this row order matching the gold-label indexing.
_MAP_ROWS = torch.tensor([
    [DLabel2Idx[MapTab[Idx2VLabel[i]][j]] for j in range(len(IntPres))]
    for i in range(n_ways)
], dtype=torch.long)  # shape (3, 3)


class Det2Ver(LightningModule):
    """Cross-task Det2Ver framework."""

    def __init__(self, config_module=None):
        super().__init__()
        self.cfg = config_module or configs
        print('load LLM...')
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            local_files_only=_load_local_only,
            low_cpu_mem_usage=True,
        )
        print('load Success!')
        # Freeze the backbone, wrap with (IA)^3 / LoRA adapters.
        modify_with_lora(self.transformer)
        # Bookkeeping.
        self.trainable_param_names: set = set()
        self._val_predictions: List[Dict] = []
        self._best_metric = -1.0
        self._patience_ctr = 0
        self._last_saved_step = -1

        print('trainable params: {:,}'.format(count_trainable_params(self.transformer)))

    # ---------------------------------------------------------------
    # Optimizer / scheduler
    # ---------------------------------------------------------------

    def configure_optimizers(self):
        param_groups = []
        for name, p in self.transformer.named_parameters():
            if re.fullmatch(trainable_param_names_re, name):
                assert p.requires_grad, name
                param_groups.append(p)
                self.trainable_param_names.add(name)
            else:
                p.requires_grad = False
        optimizer = Adafactor(
            [{'params': param_groups}],
            lr=lr,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=False,
            warmup_init=False,
        )
        total_steps = max(1, num_steps)
        num_warmup = int(total_steps * warmup_ratio)

        def lr_lambda(current_step: int):
            if current_step < num_warmup:
                return float(current_step) / float(max(1, num_warmup))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - num_warmup)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

    # ---------------------------------------------------------------
    # Forward helpers
    # ---------------------------------------------------------------

    def _encode(self, input_ids: torch.Tensor):
        pad_id = self.cfg.PAD_TOKEN_ID
        attention_mask = (input_ids != pad_id).float()
        encoder_hidden = self.transformer.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
        )[0]
        return attention_mask, encoder_hidden

    def _forward_choices(self, input_ids: torch.Tensor, choices_ids: torch.Tensor):
        """Run the seq2seq model on every candidate answer.

        Returns
        -------
        choices_scores : (B, C)  length-normalised NLL for each candidate
        lm_target      : (B, C, L_c)  targets with -100 on pad tokens
        lm_target_neg  : (B, C, L_c)  same target but the unlikelihood
                                     branch may still refer to negative
                                     rows via its own indexing.
        raw_logits     : (B, C, L_c, V)
        """
        pad_id = self.cfg.PAD_TOKEN_ID
        bsz, n_choices, choice_len = choices_ids.shape
        attention_mask, encoder_hidden = self._encode(input_ids)

        # Repeat encoder outputs for every candidate.
        enc_hidden_rep = encoder_hidden.unsqueeze(1).repeat(1, n_choices, 1, 1).flatten(0, 1)
        attn_rep = attention_mask.unsqueeze(1).repeat(1, n_choices, 1).flatten(0, 1)

        flat_choices = choices_ids.flatten(0, 1)  # (B*C, L_c)
        # Right-shift for the T5 decoder input.
        decoder_input_ids = torch.cat([torch.zeros_like(flat_choices[:, :1]), flat_choices[:, :-1]], dim=1)
        decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.float)

        model_output = self.transformer(
            attention_mask=attn_rep,
            encoder_outputs=[enc_hidden_rep],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        logits = model_output.logits  # (B*C, L_c, V)

        # Build targets with -100 on pad positions so cross_entropy ignores pads.
        lm_target = flat_choices.clone()
        lm_target[flat_choices == pad_id] = -100

        token_ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), lm_target.reshape(-1), reduction='none',
        ).reshape(bsz, n_choices, choice_len)

        choice_lens = (choices_ids != pad_id).sum(dim=-1).clamp(min=1).float()  # (B, C)
        choices_scores = token_ce.sum(dim=-1) / choice_lens  # length-normalised NLL

        return choices_scores, token_ce, logits.reshape(bsz, n_choices, choice_len, -1), lm_target.reshape(bsz, n_choices, choice_len)

    # ---------------------------------------------------------------
    # Training step: L_LM + L_UK + L_LN
    # ---------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        choices_ids = batch['choices_ids']  # (B, 2, L_c)
        d_label = batch['d_label_idx']       # (B,) — index into choices dim

        choices_scores, token_ce, logits, lm_target = self._forward_choices(input_ids, choices_ids)
        bsz, n_choices, choice_len = choices_ids.shape

        # ---- L_LM (correct-sequence cross-entropy) --------------------
        gold_ce = token_ce.gather(1, d_label.view(bsz, 1, 1).expand(bsz, 1, choice_len)).squeeze(1)
        gold_mask = (choices_ids.gather(1, d_label.view(bsz, 1, 1).expand(bsz, 1, choice_len)).squeeze(1)
                     != self.cfg.PAD_TOKEN_ID).float()
        lm_loss = (gold_ce * gold_mask).sum() / gold_mask.sum().clamp(min=1)

        # ---- L_UK (unlikelihood on incorrect targets) -----------------
        # p(y_hat) for each incorrect token.
        # Instead of computing p from logits & lm_target directly, we can
        # use the fact that token_ce = -log p(target); therefore
        # log(1 - exp(-token_ce)) is the desired term.
        pad_mask = (choices_ids != self.cfg.PAD_TOKEN_ID).float()  # (B, C, L_c)
        one_minus_p = (1.0 - torch.exp(-token_ce)).clamp(min=1e-6)
        cand_log = torch.log(one_minus_p) * pad_mask  # only real tokens contribute

        # zero-out the correct row
        cand_log_masked = cand_log.clone()
        cand_log_masked.scatter_(
            1, d_label.view(bsz, 1, 1).expand(bsz, 1, choice_len), 0.0,
        )
        pad_mask_masked = pad_mask.clone()
        pad_mask_masked.scatter_(
            1, d_label.view(bsz, 1, 1).expand(bsz, 1, choice_len), 0.0,
        )
        unlikely_loss = -cand_log_masked.sum() / pad_mask_masked.sum().clamp(min=1)

        # ---- L_LN (length-normalised softmax cross-entropy) -----------
        # ``choices_scores`` is *negative log likelihood* averaged over tokens.
        # δ(X, Y) = -choices_scores  → softmax over δ, target is correct choice.
        delta = -choices_scores  # (B, C)
        ln_loss = F.cross_entropy(delta, d_label)

        loss = lm_loss + unlikely_loss + ln_loss
        self.log('train/lm_loss', lm_loss.detach(), on_step=True, prog_bar=False, batch_size=bsz)
        self.log('train/unlikely_loss', unlikely_loss.detach(), on_step=True, prog_bar=False, batch_size=bsz)
        self.log('train/ln_loss', ln_loss.detach(), on_step=True, prog_bar=False, batch_size=bsz)
        self.log('train/loss', loss.detach(), on_step=True, prog_bar=True, batch_size=bsz)
        return loss

    # ---------------------------------------------------------------
    # Validation step
    # ---------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        choices_ids = batch['choices_ids']
        d_label = batch['d_label_idx']
        v_label = batch['v_label_idx']
        meta = batch['meta']

        with torch.no_grad():
            choices_scores, _, _, _ = self._forward_choices(input_ids, choices_ids)

        # log-normalised likelihood ranking: predicted binary answer = argmin NLL.
        pred_binary = choices_scores.argmin(dim=-1)

        for i in range(input_ids.size(0)):
            self._val_predictions.append({
                'instance_idx': meta[i]['instance_idx'],
                'instance_id': meta[i]['instance_id'],
                'int_prefix_idx': meta[i]['int_prefix_idx'],
                'template_idx': meta[i]['template_idx'],
                'binary_pred': int(pred_binary[i].item()),
                'gold_binary': int(d_label[i].item()),
                'gold_v_label': int(v_label[i].item()),
                'choices_scores': choices_scores[i].detach().cpu().tolist(),
                'task_flag': int(batch['task_flag'][i].item()),
                'claim': meta[i].get('claim', ''),
            })
        return None

    # ---------------------------------------------------------------
    # Label Words Synchronization Engine
    # ---------------------------------------------------------------

    def _sync_labels(self, per_instance: List[Dict]) -> int:
        """Map three binary predictions from the three internal prefixes to
        a single ternary FV label.

        ``per_instance`` is a list of dicts (one per internal prefix) with:

        * ``int_prefix_idx``    : 0 (true) / 1 (uncertain) / 2 (false)
        * ``binary_pred``       : 0 (Yes) / 1 (No)
        * ``choices_scores``    : [nll_yes, nll_no] length-normalised
        """
        # Build the row-vector of predictions in the (true, uncertain, false) order.
        preds = [None] * len(IntPres)
        scores = [None] * len(IntPres)
        for row in per_instance:
            preds[row['int_prefix_idx']] = row['binary_pred']
            scores[row['int_prefix_idx']] = row['choices_scores']
        # Missing prefixes shouldn't happen for FV eval, but be defensive:
        # default to the majority verification class (NEI) so the index
        # remains a valid ternary label.
        if any(p is None for p in preds):
            return VLabel2Idx['NEI']

        preds_t = torch.tensor(preds, dtype=torch.long)              # (3,)
        # Lookup table (Table I).
        map_rows = _MAP_ROWS.to(preds_t.device)                      # (3, 3)
        matches = (map_rows == preds_t.unsqueeze(0)).all(dim=1)      # (3,)
        if matches.any():
            return int(torch.nonzero(matches, as_tuple=False)[0].item())

        # Fallback: sequence probability ranking (Equation 8).
        scores_t = torch.tensor(scores)                              # (3, 2)
        # -Logit_{d|s} = -scores[int_prefix, d_target]. Higher = more confidence.
        # For every candidate s we look up the target answers Ds via MapTab.
        totals = []
        for s in range(n_ways):
            total = 0.0
            for prefix_idx in range(len(IntPres)):
                target_answer_idx = int(map_rows[s, prefix_idx].item())
                total += (-scores_t[prefix_idx, target_answer_idx].item())
            totals.append(total)
        return int(torch.tensor(totals).argmax().item())

    # ---------------------------------------------------------------
    # Aggregate ternary predictions and compute Macro-F1
    # ---------------------------------------------------------------

    def validation_epoch_end(self, outputs):
        # Group by instance for FV rows only.
        buckets: Dict[int, List[Dict]] = defaultdict(list)
        for row in self._val_predictions:
            if row['task_flag'] == 0:  # FV
                buckets[row['instance_idx']].append(row)

        y_true, y_pred = [], []
        export_rows: List[Dict] = []
        for instance_idx, rows in buckets.items():
            gold = rows[0]['gold_v_label']
            pred = self._sync_labels(rows)
            y_true.append(gold)
            y_pred.append(pred)

            ordered = {row['int_prefix_idx']: row for row in rows}
            binary_answers = [ordered[i]['binary_pred'] for i in range(len(IntPres))]
            answer_tensor = torch.tensor(binary_answers, dtype=torch.long)
            lookup_conflict = not bool((_MAP_ROWS == answer_tensor.unsqueeze(0)).all(dim=1).any())

            yes_probs = []
            for i in range(len(IntPres)):
                scores = torch.tensor(ordered[i]['choices_scores'], dtype=torch.float32)
                yes_probs.append(float(torch.softmax(-scores, dim=0)[0].item()))
            export_rows.append({
                'instance_idx': int(instance_idx),
                'instance_id': rows[0]['instance_id'],
                'claim': rows[0].get('claim', ''),
                'gold_label': Idx2VLabel[int(gold)],
                'q_true': yes_probs[0],
                'q_uncertain': yes_probs[1],
                'q_false': yes_probs[2],
                'binary_answers_true_uncertain_false': binary_answers,
                'lookup_conflict': lookup_conflict,
                'prediction': Idx2VLabel[int(pred)],
            })

        if y_true:
            macro_f1 = f1_score(y_true, y_pred, average='macro')
        else:
            macro_f1 = 0.0
        print(f'\n[val] Macro-F1 = {macro_f1:.4f}   ({len(y_true)} instances)')
        self.log('val/macro_f1', macro_f1, prog_bar=True)

        prediction_dir = os.path.join(configs.exp_root, configs.exp_name)
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_path = os.path.join(prediction_dir, 'predictions.jsonl')
        with open(prediction_path, 'w', encoding='utf-8') as stream:
            for row in export_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'[val] wrote {len(export_rows)} predictions to {prediction_path}')

        # Early stopping bookkeeping.
        if macro_f1 > self._best_metric:
            self._best_metric = macro_f1
            self._patience_ctr = 0
            self.save_model(tag='best')
        else:
            self._patience_ctr += 1
            if self._patience_ctr >= self.cfg.patience:
                print(f'[early-stop] no improvement for {self._patience_ctr} evaluations.')
                self.trainer.should_stop = True

        self._val_predictions = []
        return {'macro_f1': macro_f1}

    # ---------------------------------------------------------------
    # Checkpointing helpers
    # ---------------------------------------------------------------

    def load_adapter_weights(self, path: str):
        """Load only the adapter (LoRA / IA3) parameters saved via
        ``save_model``."""
        if not path or not os.path.exists(path):
            return
        state = torch.load(path, map_location='cpu')
        load_result = self.transformer.load_state_dict(state, strict=False)
        assert len(load_result.unexpected_keys) == 0, (
            f'Unexpected keys while loading adapter weights: {load_result.unexpected_keys}'
        )
        print(f'[Det2Ver] adapter weights loaded from {path}')

    def save_model(self, tag: str = 'step'):
        if not self.cfg.save_model:
            return
        os.makedirs(self.cfg.exp_root, exist_ok=True)
        exp_dir = os.path.join(self.cfg.exp_root, self.cfg.exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        if tag == 'best':
            path = os.path.join(exp_dir, 'best.pt')
        elif tag == 'finish':
            path = os.path.join(exp_dir, 'finish.pt')
        else:
            path = os.path.join(exp_dir, f'global_step{self.global_step}.pt')
        trainable_states = {
            k: v.detach().cpu()
            for k, v in self.transformer.state_dict().items()
            if k in self.trainable_param_names
        }
        torch.save(trainable_states, path)
        self._last_saved_step = int(self.global_step)
        print(f'[Det2Ver] saved adapter weights to {path}')

    def on_train_end(self):
        self.save_model(tag='finish')


if __name__ == '__main__':
    # Instantiation smoke test (does not require CUDA — only for structure).
    m = Det2Ver()
    m.configure_optimizers()
    print('Det2Ver initialised.')
