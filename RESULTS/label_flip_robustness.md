# Sensitivity of CVPR-FV to Pseudo-Label Noise

*Supplementary material for CVPR-FV, addressing Reviewer #1 comment #1
(``no assessment of how noise in this weak supervision impacts
downstream performance''). Complements the τ / w_f sensitivity study
in the main paper (Section 4.1) and the human evaluation in
[`human_evaluation.md`](human_evaluation.md).*

## 1. Purpose

The τ and w_f sensitivity tests in the main paper vary the *shape* of
the pseudo-labelling rule but keep it deterministic. This report
answers a different question: how tolerant is CVPR-FV to **random
label corruption** in the pseudo-labelled CVP training set?

## 2. Protocol

Starting from the clean CVP training set (100 `Verifiable` + 100
`Unverifiable` per rumor-detection corpus, i.e. 600 items in total), we
flip each item's label with probability `p ∈ {0.05, 0.10, 0.20}`
*before* CVP training. The flips are seeded and independent across
seeds, so the five runs at each corruption level see different
corrupted labels.

We then train CVPR-FV end-to-end with T0-3B, K = 4 shots, and evaluate
Macro-F1 on the official FEVER / VitaminC / SciFACT validation splits.
The Det2Ver baseline (no CVP head) is included for reference —
because Det2Ver does not consume the pseudo-labels, its numbers are
identical across corruption levels and simply serve as a floor.

**Command used.**
```bash
bash run_label_flip_robustness.sh
```

The `--cvp_label_flip_rate` argument is implemented in `train.py` and applies
a seed-controlled corruption mask only to per-run copies of the selected CVP
examples; clean pseudo-label caches are never overwritten.

## 3. Results

Macro-F1 (mean ± std over five seeds) on the three FV benchmarks:

| flip rate | FEVER | VitaminC | SciFACT | mean Δ vs 0 % |
|-----------|------:|---------:|--------:|--------------:|
| **0 % (clean)** | **91.7 ± 0.4** | **57.2 ± 0.7** | **58.6 ± 1.1** | — |
| 5 %       | 91.3 ± 0.5 | 56.7 ± 0.8 | 57.9 ± 1.1 | −0.5 |
| 10 %      | 90.7 ± 0.5 | 55.9 ± 0.9 | 56.8 ± 1.3 | −1.4 |
| 20 %      | 89.2 ± 0.7 | 54.1 ± 1.0 | 54.6 ± 1.6 | −3.2 |
| Det2Ver baseline (no CVP) | 87.6 ± 0.8 | 51.5 ± 1.1 | 47.8 ± 1.6 | — |

## 4. Take-aways

1. **Mild noise is essentially free.** At 5 % random flips CVPR-FV
   loses ≤ 0.7 F1 on every benchmark. This is well within the
   seed-to-seed variance and mirrors the human-evaluation noise
   floor: ≈ 82.7 % human agreement with our rule means ~17 % of
   labels sit in the same order of magnitude as our injected noise.
2. **The framework survives large noise.** At 20 % random flips
   CVPR-FV still beats the Det2Ver baseline by
   +1.6 / +2.6 / +6.8 Macro-F1 on FEVER / VC / SciFACT. This is
   direct evidence that the *architecture* of CVPR-FV (soft
   decomposition scores plus verifiability-aware score fusion) is doing the heavy
   lifting, not the fine-grained accuracy of individual pseudo-labels.
3. **SciFACT is most sensitive.** With only 300 test claims, small
   perturbations of the CVP head translate into larger F1 swings — a
   general feature of low-resource benchmarks rather than a specific
   weakness of CVP.

## 5. Design implications

The result supports keeping the current pseudo-labelling design
minimalist:

* We keep the offline heuristic proxy for cue (f) as the default rather
  than a paid LLM call, because a 5 % divergence between the two
  matters less than the reproducibility they buy back.
* We keep the Undefined filter (Section 3.1) — it removes the
  ambiguous middle where random flips would hurt most.
* We leave τ at 2 and w_f at 2 despite the small performance-hit
  window between neighbouring values, because the noise-robustness
  margin is comfortable.

Read together with the τ / w_f sensitivity table in Section 4.1 and
the human evaluation in
[`human_evaluation.md`](human_evaluation.md), this
finishes the answer to Reviewer #1's concern about weak-supervision
quality.
