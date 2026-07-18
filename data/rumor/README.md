# Rumor Detection datasets for Det2Ver

Det2Ver uses **three** rumor-detection corpora as the cross-task supervisory
signal (Section IV-B of the paper):

| dataset id | paper name | source | rows shipped |
|------------|------------|--------|-------------:|
| `liar`     | LIAR       | Wang, ACL 2017 | 8,146 train / 1,036 val |
| `fnn`      | FakeNewsNet| Shu et al., Big Data 2020 | 20,877 train / 2,319 val |
| `covid`    | COVID-19 Fake News | Patwa et al., CONSTRAINT 2021 | 6,420 train / 2,140 val |

All three JSONLs are already generated (see `liar_train.jsonl`,
`fnn_train.jsonl`, `covid_train.jsonl` next to this README). You can
delete the auto-created `few_shot/` cache to resample.

## Expected file layout

```
data/rumor/
├── liar_train.jsonl
├── liar_validation.jsonl        (optional, kept for parity)
├── fnn_train.jsonl
├── fnn_validation.jsonl         (optional)
├── covid_train.jsonl
└── covid_validation.jsonl       (optional)
```

Every line is a JSON object with **exactly** two required fields:

```json
{"id": "unique-instance-id", "claim": "the rumor / news text", "label": "REAL"}
```

* `label` must be one of `{"REAL", "FAKE"}` (case-insensitive; the loader
  also accepts LIAR's original 6-way labels and normalises them).
* Extra fields such as `subject`, `speaker`, `source_url`, … are ignored.

## Recommended download commands (run manually)

### 1. LIAR

```bash
cd data/rumor
curl -L -o liar_raw.zip https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
unzip liar_raw.zip -d liar_raw
python prepare_rumor_data.py --dataset liar --raw_dir liar_raw --out_dir .
```

`liar_raw/{train,valid,test}.tsv` are TSV files. The preparation script
maps the 6-way labels as follows (Section IV-B strategy):

* REAL ← `true`, `mostly-true`
* FAKE ← `false`, `pants-fire`, `barely-true`
* dropped ← `half-true` (ambiguous)

### 2. FakeNewsNet (FNN)

The full dataset requires the Twitter API. For Det2Ver only the
headline / claim text is needed. The simplest self-contained mirror is:

```bash
cd data/rumor
git clone --depth 1 https://github.com/KaiDMML/FakeNewsNet.git fnn_raw
python prepare_rumor_data.py --dataset fnn --raw_dir fnn_raw --out_dir .
```

`fnn_raw/dataset/politifact_{real,fake}.csv` and
`fnn_raw/dataset/gossipcop_{real,fake}.csv` are read; the `title` column
becomes the `claim` field.

### 3. COVID-19 Fake News

```bash
cd data/rumor
git clone --depth 1 https://github.com/diptamath/covid_fake_news.git covid_raw
python prepare_rumor_data.py --dataset covid --raw_dir covid_raw --out_dir .
```

The preparation script reads
`covid_raw/data/Constraint_Train.csv` and normalises `tweet -> claim`
and `label -> {REAL, FAKE}`.

## After downloading

Every subsequent Det2Ver few-shot experiment reuses the cache
`data/rumor/few_shot/<dataset>/<N>-per-class.jsonl` (see
`configs.rd_cache_file_path`). Delete a cache file to force resampling.

## Do I have to download all three?

No. Det2Ver gracefully degrades:

* If a file is missing, `RumorDataReader` prints a warning and treats
  that dataset as empty.
* Only having **one** RD dataset is a valid ablation (`Det2Ver(LIAR)`,
  `Det2Ver(COVID)`, `Det2Ver(FNN)` in Figure 3).
* Not downloading any RD dataset reduces the framework to `Det2Ver(0)`
  (Table V), still faithful to the paper.

If you cannot download the real corpora at all, run

```bash
python prepare_rumor_data.py --dataset dummy --out_dir .
```

to synthesise a tiny placeholder train file that lets you smoke-test
the training pipeline.
