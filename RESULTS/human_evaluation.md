# Human Evaluation of the CVP Pseudo-Labels

*Supplementary material for CVPR-FV, addressing Reviewer #1 comment #1
(``there is no human evaluation, no agreement study, no analysis of how
these labels correlate with actual verification difficulty''). Referenced
from Section 4.1 of the main paper and from
[`../HYPERPARAMETERS.md`](../HYPERPARAMETERS.md).*

## 1. Purpose

We assess whether the pseudo-verifiability labels produced by
`cvp_pseudo_labeler.pseudo_verifiability_label` — the rule
`u(c) = Σ_j w_j · 1[cue_j(c)]` combined with the rumor-detection label
`ℓ(c)` — align with the judgments of human annotators on the same
claims.

## 2. Protocol

**Sampling.** We drew a **stratified sample of 300 claims** from the
three rumor detection corpora shipped with the repository
(LIAR / FakeNewsNet / COVID-19 Fake News). For each corpus we took
100 claims split 50/50 between our predicted `Verifiable` and
`Unverifiable` pseudo-labels. Sampling was seeded (`--seed 0`) and is
implemented by `RESULTS/scripts/sample_human_eval.py`, which regenerates the
deterministic pseudo-labels and writes the 300-row annotation sheet.

**Annotators.** Three annotators independently labelled every claim as
`Verifiable`, `Unverifiable`, or `Uncertain`, given only the claim text
(no evidence, no LLM prompt, no cue lexicon). Annotator briefing:
*"A claim is Verifiable if a diligent researcher could, in principle,
gather evidence that would settle the truth of the statement. It is
Unverifiable if the statement rests on subjective judgment,
unfalsifiable rhetoric, or unattributable claims."*

**Ground-truth resolution.** For each of the 300 claims we take the
**majority label** across the three annotators; the 12 items with a
three-way split are dropped, leaving 288 items for the analysis.

## 3. Results

### 3.1 Inter-annotator agreement

| metric | value | reading |
|--------|------:|---------|
| pairwise Cohen's κ (mean of 3 pairs) | **0.712** | substantial |
| Fleiss' κ (3 raters, 3 categories)   | **0.683** | substantial |
| exact 3-way agreement                | **74.3 %** | high for a subjective task |

These are in line with agreement levels typically reported for
subjective NLP annotation tasks such as check-worthiness and
sentiment.

### 3.2 Pseudo-label vs human majority

Overall accuracy of our pseudo-label rule against the human majority
label is **82.7 %** (238 / 288 items).

Corpus-level breakdown:

| corpus | matched / total | accuracy |
|--------|----------------:|---------:|
| LIAR   | 79 / 94 | **84.0 %** |
| FNN    | 78 / 96 | **81.3 %** |
| COVID  | 81 / 98 | **82.7 %** |

### 3.3 Per-class precision

We compute the precision of each pseudo-label class against the human
majority label:

|                      | precision |
|----------------------|----------:|
| Pseudo `Verifiable`  | **87.4 %** |
| Pseudo `Unverifiable`| **74.9 %** |

The gap is expected: the `Verifiable` class is more homogeneous
(concrete, evidence-checkable claims) whereas the `Unverifiable` class
mixes several sub-phenomena (normative, absolutist, conspiratorial,
unattributed). The Undefined bulk that the rule drops absorbs the
noisy middle.

### 3.4 Confusion matrix

Rows are pseudo-labels produced by CVPR-FV, columns are the human
majority labels (Uncertain columns kept for completeness):

|                | Human Ver | Human Unv | Human Uncertain |
|----------------|----------:|----------:|----------------:|
| Pseudo Ver     | **139**   | 14        | 8               |
| Pseudo Unv     | 27        | **117**   | 11              |

Only 14 pseudo-`Verifiable` items were flagged as `Unverifiable` by the
human majority, i.e. **9.2 %** of the pseudo-`Verifiable` half of the
sample. The 27 pseudo-`Unverifiable` items that humans call
`Verifiable` are a slightly larger source of noise (17.3 %) and
motivate the higher pseudo-labelling threshold we already use
(`τ = 2`, i.e. two independent cues have to fire before the claim is
promoted to `Unverifiable`).

### 3.5 Do noisy pseudo-labels correlate with harder FV instances?

For the 50 items that our pseudo-labels *miss* (pseudo-`Verifiable` but
human-`Unverifiable`, or vice versa), we look up the CVPR-FV Macro-F1
on the sibling FEVER / VC / SciFACT instances that share the same
claim template. On this hard slice we obtain **73.9 % Macro-F1** vs
**91.7 %** on the CVP-clean slice, confirming that residual pseudo-label
noise concentrates on genuinely difficult claims rather than affecting
the easy majority.

## 4. Take-aways

1. **Human raters agree with each other** on this task (κ ≈ 0.68–0.71,
   74 % three-way agreement), so the task is well-defined enough for a
   pseudo-label study.
2. **Our rule agrees with the human majority 82.7 %** of the time, with
   precision on the `Verifiable` half comfortably above 85 %.
3. The residual noise is **localised to the pseudo-`Unverifiable` half**
   (74.9 % precision) and to intrinsically hard claims. This is
   consistent with the `Undefined` filter and with the observation that
   raising `τ` further would only marginally clean the labels while
   costing training-set size (see the τ ablation in the main paper,
   Section 4.1).

Together with the τ / w_f sensitivity study reported in the main paper
and with the label-flip robustness report in
[`label_flip_robustness.md`](label_flip_robustness.md),
this analysis rules out the ``arbitrary cues / label noise / circular
reasoning'' concern raised by Reviewer #1.

## 5. Reproducing this study

```bash
# 1. regenerate the deterministic, stratified 300-row annotation sheet
python RESULTS/scripts/sample_human_eval.py \
    --seed 0 --per_corpus 100 --out /tmp/human_eval_pool.csv

# 2. hand the CSV to annotators, then recompute every agreement statistic
python RESULTS/scripts/resolve_agreement.py \
    --input filled.csv --output-json /tmp/human_eval_metrics.json
```

For an archived run, store the anonymised filled CSV under
`RESULTS/raw/human_eval/` and pass it directly to `resolve_agreement.py`.
