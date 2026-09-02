# Supplementary Empirical Results

Extra analyses that support the CVPR-FV paper but do not fit in the
main text. Each file is self-contained and points back to the paper
section it answers.

| file | topic | reviewer anchor |
|------|-------|----------------|
| [`human_evaluation.md`](human_evaluation.md) | 300-claim human study of pseudo-verifiability labels | R1 #1 |
| [`label_flip_robustness.md`](label_flip_robustness.md) | label-flip robustness of CVPR-FV | R1 #1 |
| [`case_study.md`](case_study.md) | full case study with intermediate confidences, evidence passages, per-instance analysis | R1 #6 |
| [`conflict_rate.md`](conflict_rate.md) | spurious-conflict rate stratified by CVP score v | R2 #5 |
| [`matched_budget.md`](matched_budget.md) | Det2Ver(200) vs CVPR-FV(200) matched-budget comparison | R2 #6 |
| [`std_wilcoxon.md`](std_wilcoxon.md) | mean ± std over 5 seeds + Wilcoxon signed-rank p-values | R2 #10 |

The conflict-rate experiment is executable from the repository root with
`python reproduce_conflict_rate.py ...`. It produces a count-bearing Markdown
table, a JSON summary with input hashes, and a per-instance JSONL audit trail;
see [`conflict_rate.md`](conflict_rate.md#reproduction) for both supported input
modes.
