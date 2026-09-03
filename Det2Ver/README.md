# Vendored Det2Ver baseline

This directory contains the Det2Ver implementation used for the matched-budget
baseline and the spurious-conflict export in the CVPR-FV revision. It is kept
inside the CVPR-FV release so the paper's baseline and analysis scripts do not
depend on an untracked sibling checkout.

The code implements the original two-stage synchronization rule:

1. match a `(true, uncertain, false)` Yes/No triple against the three valid
   lookup rows;
2. if no row matches, rank the three verification labels by the summed
   sequence scores of their required Yes/No answers.

## Shared data

To avoid duplicating the large JSONL files, this vendored baseline reads the
parent repository's `data/` directory by default:

```text
CVPR_FV/
├── data/
└── Det2Ver/
    ├── train.py
    ├── data_reader.py
    └── model.py
```

Set `DET2VER_DATA_DIR` to use a different prepared data directory. The expected
files are the same FEVER, VitaminC, SciFACT, LIAR, FakeNewsNet, and COVID-19
Fake News JSONL files documented in the parent README.

## Matched-budget run

From the parent repository root, the exact Table 1 endpoint sweep is:

```bash
bash run_matched_budget.sh
```

The equivalent single run is:

```bash
python Det2Ver/train.py \
  --dataset fever --shot_num 4 --seed 0 \
  --few_shot true --zero_shot false \
  --use_rumor_detection true --rd_total_per_dataset 200 \
  --exp_name det2ver_fever_K4_rd200_seed0
```

`--rd_total_per_dataset 200` means 200 rumor instances from each source corpus,
split evenly between the two rumor classes (600 auxiliary instances total).

## Conflict-analysis export

The validation output includes the source instance ID, gold/final labels,
the three Yes-probabilities, lookup conflict, and final prediction. Use the
parent workflow to regenerate the reported stratified analysis:

```bash
bash run_conflict_rate.sh
```

## Environment

The parent `requirements.txt` is sufficient. T0-3B is resolved from
`DET2VER_T0_PATH`, then from the Hugging Face model identifier.
