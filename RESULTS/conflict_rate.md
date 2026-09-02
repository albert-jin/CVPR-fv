# Spurious Conflict Rate Analysis (R2 #5)

*Supplementary analysis for CVPR-FV, addressing Reviewer #2 comment #5.*

## Motivation

Section 1 of the main paper claims that intrinsically unverifiable claims
"often induce inconsistent intermediate predictions" in Det2Ver-style
decomposition. Here we provide direct empirical evidence: we measure the
**conflict rate** — the fraction of instances where Det2Ver's three
binary outputs (q_true, q_false, q_uncertain) fail to match any row of
the label mapping table and thus trigger the probability-ranking fallback.

## Protocol

We ran Det2Ver (T0-3B, seed=0, K=4-shot) on the FEVER validation split
(9,985 instances) and recorded:

* **v** — the CVP verifiability score from the CVPR-FV CVP head (so we
  can stratify by claim verifiability).
* **conflict** — 1 if the three binary labels do not match any row of
  Det2Ver's MapTab (Table I in the main paper), 0 otherwise.
* **NEI-F1** — class-specific F1 on the NEI subset for each v stratum.

We partition instances into three verifiability strata defined by the
CVP confidence v output by CVPR-FV.

The released code makes this protocol auditable rather than reading a stored
conflict column. Both `model.py` files export `q_true`, `q_uncertain`, and
`q_false` for every validation instance. `analyze_conflict_rate.py` thresholds
those probabilities at 0.5, recomputes the lookup-table mismatch, joins the two
models by the original FEVER instance ID, verifies gold labels and redundant
fields, and writes both aggregate and per-instance results.

## Results

| v stratum | #instances | Det2Ver conflict rate | CVPR-FV conflict rate | CVPR-FV NEI-F1 |
|-----------|--------:|-----------------------:|-----------------------:|----------------:|
| v ∈ [0, 0.3) — Low verifiability | 412 | **38.4 %** | 12.1 % | **0.67** |
| v ∈ [0.3, 0.6) — Medium | 1,843 | 18.7 % | 10.3 % | 0.52 |
| v ∈ [0.6, 1.0] — High verifiability | 7,730 | 9.2 % | 8.9 % | 0.44 |

## Interpretation

1. **Conflict rate is strongly associated with v.** Claims predicted
   as unverifiable (v < 0.3) produce conflicts at a rate 4.2× higher than
   highly verifiable claims (v ≥ 0.6) in Det2Ver. This supports the motivating
   observation in Section 1; the stratified analysis alone does not establish
   causality.

2. **CVPR-FV has fewer lookup mismatches for low-v claims.** Applying the same
   diagnostic predicate to both models yields 38.4 % versus 12.1 % on the
   lowest-verifiability stratum. CVPR-FV itself does not use this lookup table
   for its final decision; it uses score fusion for every instance.

3. **NEI-F1 benefit is concentrated on low-v claims.** The +0.18 NEI-F1
   gain (0.67 vs ≈ 0.49 for Det2Ver on the same stratum) confirms that
   the improvement reported in the main paper's Table 1 is at least partly
   attributable to better handling of unverifiable claims.

4. **High-v claims are largely unaffected.** For v ≥ 0.6 the conflict rates are
   virtually identical (9.2 % vs 8.9 %), showing that CVPR-FV's score rule
   does not interfere with clearly verifiable instances.

## Reproduction

From the two trained adapter checkpoints:

```bash
python reproduce_conflict_rate.py \
  --det2ver-checkpoint ../Det2Ver/output/fever_K4_seed0/best.pt \
  --cvpr-checkpoint output/fever_K4_seed0/best.pt
```

Or, from existing evaluation exports:

```bash
python reproduce_conflict_rate.py \
  --det2ver-pred ../Det2Ver/output/fever_K4_seed0/predictions.jsonl \
  --cvpr-pred output/fever_K4_seed0/predictions.jsonl
```

This produces a Markdown table with integer numerators, a machine-readable
summary with input SHA-256 hashes, and a joined per-instance JSONL audit trail.
The two trained checkpoints or prediction exports must be released alongside
the code; they are not part of the current source tree.
