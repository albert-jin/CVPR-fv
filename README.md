# CVPR-FV

**Claim Verifiability Prediction-based Rumor detection for Fact
Verification.** A verifiability-aware fact-verification framework that
extends Det2Ver with an auxiliary *Claim Verifiability Prediction* (CVP)
task and a *CVP-guided score aggregator* over the
decomposition-based intermediate judgments.

## Motivation

<p align="center">
  <img src="figures/introfig.png" width="68%" alt="Motivation: intrinsically unverifiable claims cause inconsistent intermediate judgments in decomposition-based fact verification."/>
</p>

Decomposition-based fact verification (Det2Ver-style) turns a ternary
verification decision into three binary yes/no judgments over
`true / uncertain / false` hypothesis states, then synchronises them
back into `{SUPPORT, REFUTE, NEI}` via a hard lookup table. This works
when the claim is genuinely falsifiable, but it becomes brittle for
claims that are **intrinsically unverifiable**, e.g. normative,
vague, or unfalsifiable statements, where the three binary judgments
frequently contradict each other and force the label synchroniser into
spurious flips.

## Framework

<p align="center">
  <img src="figures/CVPRfvmodelnewest.png" width="98%" alt="CVPR-FV framework: CVP auxiliary task plus CVP-guided heuristic score aggregation."/>
</p>

CVPR-FV addresses this by adding two pieces on top of Det2Ver:

* **Claim Verifiability Prediction (CVP)**, an auxiliary head that
  estimates whether a claim is intrinsically verifiable
  (`v = p(Verifiable | c) ∈ [0, 1]`), trained with weak supervision
  from rumor-detection corpora using pseudo-verifiability labels.
* **CVP-guided score aggregation**, the three per-decomposition
  Yes-probabilities `q_true, q_false, q_uncertain` are turned into
  decomposition scores, then combined with verifiability compatibility
  scores in log-linear form:

  ```
  ℓ_Sup = q_true·(1-q_false)·(1-q_uncertain)
  ℓ_Ref = q_false·(1-q_true)·(1-q_uncertain)
  ℓ_NEI = q_uncertain·(1-q_true)·(1-q_false)
  d_y = (ℓ_y + ε) / Σ(ℓ + ε)
  r_NEI(v) = (1-v) + γ·v
  r_SUP(v) = r_REF(v) = (1-γ)·v / 2
  a_y = d_y · r_y(v)^λ
  ```

  `d_y`, `r_y(v)`, and `a_y` are decision scores. The fusion is a
  documented heuristic, not a Bayesian posterior or conditional-probability
  derivation.

Both heads share one LoRA-adapted backbone (T0-3B, Qwen2.5-3B, or
Llama-3.1-8B).

---

## What's new relative to Det2Ver

| Component | Det2Ver | CVPR-FV |
|-----------|---------|---------|
| Auxiliary task | Rumor detection (binary real/fake) | **Claim Verifiability Prediction** (`Verifiable / Unverifiable`) with pseudo-labels |
| Label synchronization | Hard lookup table + probability-ranking fallback | **Soft decomposition scores** + verifiability-guided score fusion `d_y r_y(v)^λ` |
| Backbones | T0-3B only | **T0-3B, Qwen2.5-3B, Llama-3.1-8B** (single CLI flag) |
| Reporting | Macro-F1 only | Macro-F1 **+ per-class F1** (`SUPPORT / REFUTE / NEI`) and mean±std over seeds |

---

## Directory layout

```
CVPR_FV/
├── configs.py               # global config (prompts, CVP cues, aggregator λ / γ, backbones)
├── utils.py                 # LoRA / (IA)^3 wrapper, seeding, helpers
├── cvp_pseudo_labeler.py    # heuristic u(c) score + weak-supervision label rule
├── data_reader.py           # FV + CVP datasets, consolidation prompting, DataModule
├── model.py                 # CVPR-FV LightningModule (joint loss + score aggregation)
├── train.py                 # CLI entry point
├── reproduce_conflict_rate.py # checkpoint-to-table conflict experiment driver
├── analyze_conflict_rate.py # audited per-instance conflict/F1 analysis
├── tests/                   # dependency-free conflict-analysis tests
├── patches/                 # Det2Ver per-instance export compatibility patch
├── requirements.txt
├── run_fs.sh                # few-shot experiments
├── run_zs.sh                # zero-shot experiments
├── run_ablations.sh         # ablation studies
├── data/
│   ├── fever_train.jsonl    FEVER / VitaminC / SciFACT
│   ├── scifact_*.jsonl
│   ├── vc_*.jsonl
│   ├── few_shot/            auto-created K-shot caches
│   └── rumor/
│       ├── liar_*.jsonl                LIAR   (8146 / 1036)
│       ├── fnn_*.jsonl                 FakeNewsNet (20877 / 2319)
│       ├── covid_*.jsonl               COVID-19 Fake News (6420 / 2140)
│       ├── prepare_rumor_data.py       raw → JSONL converter
│       ├── cvp_cache/                  auto-created CVP few-shot caches
│       └── few_shot/                   auto-created RD few-shot caches
└── output/                  auto-created experiment root
```

## Environment

```bash
conda create -n cvprfv python=3.10 -y
conda activate cvprfv
pip install -r requirements.txt
```

Backbone weights are resolved in this order:

| Env var | Backbone |
|---------|----------|
| `CVPRFV_T0_PATH`    | T0-3B local snapshot |
| `CVPRFV_QWEN_PATH`  | Qwen2.5-3B-Instruct local snapshot |
| `CVPRFV_LLAMA_PATH` | Llama-3.1-8B-Instruct local snapshot |

If none is set, `transformers` falls back to the HuggingFace hub
identifiers listed in `configs.BACKBONES`.

## Data

All six JSONL files are already committed. Every row of a fact
verification file looks like

```json
{"id": 42, "claim": "...", "gold_evidence_text": "...", "label": "SUPPORT"}
```

Every row of a rumor detection file:

```json
{"id": "liar_324.json", "claim": "...", "label": "REAL"}
```

Re-run the raw → JSONL conversion at any time with
`data/rumor/prepare_rumor_data.py`.

## Training

### Few-shot

```bash
python train.py \
    --dataset fever --shot_num 4 --seed 0 \
    --backbone t0-3b \
    --use_cvp true --cvp_total_per_dataset 200 \
    --lam_prior 0.5 --nei_floor_gamma 0.1 --lam_cvp 1.0 \
    --lr 1e-5 --num_epochs 10 --patience 5 \
    --train_batch_size 8 --eval_batch_size 8 \
    --exp_name fever_K4_seed0
```

Full sweep, `bash run_fs.sh` (5 seeds × 3 backbones × 4 K-shot × 3 datasets).

### Zero-shot

```bash
python train.py --dataset scifact --few_shot false --zero_shot true \
    --backbone llama-3.1-8b --exp_name scifact_zs_llama_seed0
bash run_zs.sh
```

### Ablations

`bash run_ablations.sh` covers the no-CVP baseline, the λ sweep, the γ
sweep, and the three single-source CVP configurations. To pin the
backbone:

```bash
BACKBONE=qwen2.5-3b bash run_ablations.sh
```

**Effect of removing the CVP module (`--use_cvp false`).**

<p align="center">
  <img src="figures/ablaNoCVP.png" width="82%" alt="Ablation: performance drop after removing the CVP module."/>
</p>

**CVP-guided score aggregation vs. Det2Ver synchronization.**

<p align="center">
  <img src="figures/ablaHardSynC2.png" width="82%" alt="Ablation: CVP-guided score aggregation versus Det2Ver lookup-then-fallback synchronization."/>
</p>

**Hyperparameter sensitivity (λ, γ).**

<p align="center">
  <img src="figures/ablaHyper1.png" width="82%" alt="Hyperparameter sensitivity for λ (score-fusion strength)."/>
</p>

<p align="center">
  <img src="figures/nei_floor_lambda_heatmap.png" width="70%" alt="Joint λ × γ heatmap on the NEI class."/>
</p>

## Pseudo-verifiability labelling

The verifiability scoring rule lives in `cvp_pseudo_labeler.py`. You
can inspect or dump labels manually:

```bash
python cvp_pseudo_labeler.py \
    --input  data/rumor/liar_train.jsonl \
    --output data/rumor/liar_cvp.jsonl \
    --tau 2
```

The default LLM flag (cue **f**) is an offline regex proxy, fully
deterministic and dependency-free. To use an external API-based judge,
wrap it in a Python function and pass it to
`label_rd_corpus(..., llm_flag_fn=my_llm_flag)`.

## Outputs

Every `--exp_name` writes to `output/<exp_name>/`:

* `best.pt`, adapter weights at the highest validation Macro-F1.
* `finish.pt`, adapter weights at the end of training.
* `predictions.jsonl`, one row per validation instance with the three
  Yes-probabilities, conflict indicator, CVP score (CVPR-FV), and final label.
* `log/version_0/events.out.*`, TensorBoard scalars
  (`train/*_loss`, `val/macro_f1`, `val/f1_SUPPORT`, `val/f1_REFUTE`,
  `val/f1_NEI`).

## Reproducing the spurious-conflict analysis

The experiment has two explicit stages: both models export one row per FEVER
validation instance, and the analysis joins those exports and recomputes the
lookup-table mismatch. The analysis does **not** trust a precomputed conflict
flag. It derives the `(true, uncertain, false)` Yes/No triple from
`q_true`, `q_uncertain`, and `q_false` at threshold 0.5 and checks it against
the three valid Det2Ver lookup rows.

### From trained checkpoints (complete reproduction)

Provide the two trained adapter checkpoints used for the T0-3B, seed-0,
`K=4` FEVER experiment:

```bash
git clone https://github.com/albert-jin/Det2Ver.git ../Det2Ver
git -C ../Det2Ver apply ../CVPR-fv/patches/det2ver_conflict_export.patch
```

Then run:

```bash
python reproduce_conflict_rate.py \
  --det2ver-checkpoint ../Det2Ver/output/fever_K4_seed0/best.pt \
  --cvpr-checkpoint output/fever_K4_seed0/best.pt
```

The driver performs evaluation-only passes over all 9,985 FEVER validation
instances and writes three artifacts:

* `RESULTS/conflict_rate_reproduced.md`, the human-readable table with integer
  conflict counts, rates, and both models' NEI-F1;
* `RESULTS/conflict_rate_summary.json`, the same results in machine-readable
  form plus SHA-256 fingerprints of both prediction inputs;
* `RESULTS/conflict_rate_instances.jsonl`, the joined per-instance audit trail.

Model checkpoints are not included in this source tree. To make the published
experiment independently executable, release the two checkpoints (or their
already-generated `predictions.jsonl` files) and record their download URL and
hashes in `RESULTS/conflict_rate.md`.

### From existing prediction exports

If the evaluation passes have already been run, re-analyze their exports with:

```bash
python reproduce_conflict_rate.py \
  --det2ver-pred ../Det2Ver/output/fever_K4_seed0/predictions.jsonl \
  --cvpr-pred output/fever_K4_seed0/predictions.jsonl
```

Current model exports carry the original dataset `instance_id`, local
`instance_idx`, claim, gold/final labels, all three Yes-probabilities, and the
redundant binary-answer/conflict fields. The analyzer joins on `instance_id`,
checks IDs, claims, gold labels, probability ranges, exported binary answers,
and exported conflict flags, and fails loudly on any mismatch. Older exports
without `instance_id` remain supported through a clearly signalled
`instance_idx` fallback.

Run the lightweight checks with:

```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

* **OOM**, drop `--precision` to `bf16`, reduce `--train_batch_size`,
  and increase `--grad_accum_factor`.
* **Slow eval**, lower `--eval_batch_size` or shorten `--max_seq_len`
  to 200.
* **CVP class imbalance**, inspect the counts printed by
  `cvp_pseudo_labeler.py`; if `Unverifiable` rows are < 50, lower τ or
  extend the cue lexicon in `configs.CVP_CUES`.
* **Cache collision**, delete `data/few_shot/<dataset>/*.jsonl` and
  `data/rumor/cvp_cache/<rd>/*.jsonl` to force resampling.

## Citation

```bibtex
@article{jin2026cvprfv,
  title  = {Verifiability-Aware Fact Verification: Leveraging Weak Supervised Task Transfer from Rumor Detection for Reliable Claim Assessment},
  journal = {Pattern Recognition Letter},
  year   = {2026},
  note   = {under review}
}
```

## Acknowledgement

CVPR-FV builds on top of Det2Ver, the T-Few
parameter-efficient fine-tuning stack, and the ProToCo consistency
framework.
