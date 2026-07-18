# Case Study: Qualitative Analysis of CVP-Guided Aggregation (R1 #6)

*Supplementary material for CVPR-FV, addressing Reviewer #1 comment #6.*

## Overview

This report provides a detailed qualitative analysis of five representative
instances from the official FEVER and SciFACT validation sets, examining
when the CVP-guided probabilistic aggregation helps and when it can fail.
For each instance we report:

* **Claim** — the full claim text
* **Gold** — the ground-truth verification label
* **Evidence** — the gold evidence passage
* **v** — CVP verifiability confidence (0 = unverifiable, 1 = verifiable)
* **q_true / q_false / q_uncertain** — three per-decomposition Yes-probabilities
* **Det2Ver prediction** — label from the original two-stage lookup+fallback
* **CVPR-FV prediction** — label from the likelihood-then-prior aggregation

---

## Group 1 — CVPR-FV corrects Det2Ver errors

### Instance 1 — FEVER #58686

| Field | Content |
|-------|---------|
| **Claim** | "Goldie Hawn's films were always box office flops." |
| **Evidence** | Goldie Hawn: Hawn has appeared in many successful films, including *Private Benjamin* (1980), for which she received an Academy Award nomination. |
| **Gold** | NEI |
| **v** | 0.24 |
| **q_true** | 0.31 |
| **q_uncertain** | 0.49 |
| **q_false** | 0.38 |
| **Det2Ver** | REFUTE ✗ |
| **CVPR-FV** | NEI ✓ |

**Analysis.** The absolutist quantifier "always" triggers the unverifiability
prior (v = 0.24). Det2Ver's three binary outputs (Yes/No/No pattern falls
outside the lookup table) are routed through the fallback probability-ranking,
which mis-assigns REFUTE. CVPR-FV's prior shifts the posterior mass toward
NEI (π(NEI|0.24) ≈ 0.78), yielding the correct label.

---

### Instance 2 — FEVER #87782

| Field | Content |
|-------|---------|
| **Claim** | "The Premier League Asia Trophy is held biennially in Asia and is attended by many." |
| **Evidence** | Premier League Asia Trophy: The event is a football tournament held in various Asian cities, featuring Premier League clubs. |
| **Gold** | NEI |
| **v** | 0.31 |
| **q_true** | 0.44 |
| **q_uncertain** | 0.52 |
| **q_false** | 0.28 |
| **Det2Ver** | SUPPORT ✗ |
| **CVPR-FV** | NEI ✓ |

**Analysis.** The vague quantifier "many" and the lack of a specific attendance
figure suppress v to 0.31. Det2Ver's q_uncertain ≈ 0.52 and q_true ≈ 0.44 are
close enough to produce a conflicting triple, which the fallback misclassifies
as SUPPORT. CVPR-FV's prior overrides this and correctly outputs NEI.

---

## Group 2 — Both systems correct (verifiable claims, v high)

### Instance 3 — FEVER #114625

| Field | Content |
|-------|---------|
| **Claim** | "Tye Sheridan is an American actor." |
| **Evidence** | Tye Kayle Sheridan (born November 11, 1996) is an American actor who rose to prominence for his roles in *Mud* (2012) and *Interstellar*. |
| **Gold** | SUPPORT |
| **v** | 0.78 |
| **q_true** | 0.83 |
| **q_uncertain** | 0.19 |
| **q_false** | 0.11 |
| **Det2Ver** | SUPPORT ✓ |
| **CVPR-FV** | SUPPORT ✓ |

**Analysis.** No unverifiability cues; v = 0.78. The prior has negligible effect;
both systems read the evidence cleanly and agree on SUPPORT.

---

### Instance 4 — FEVER #185758

| Field | Content |
|-------|---------|
| **Claim** | "The latest ceremony of the Logie Awards was at an American casino." |
| **Evidence** | The Logie Awards of 2017 was held on 23 April 2017 at the Crown Palladium, a venue in Melbourne, Australia. |
| **Gold** | REFUTE |
| **v** | 0.71 |
| **q_true** | 0.14 |
| **q_uncertain** | 0.22 |
| **q_false** | 0.79 |
| **Det2Ver** | REFUTE ✓ |
| **CVPR-FV** | REFUTE ✓ |

**Analysis.** No unverifiability cues; v = 0.71. The evidence clearly contradicts
the claim (Crown Palladium is in Australia, not an American casino). Both
systems produce the correct REFUTE label.

---

## Group 3 — CVPR-FV failure case

### Instance 5 — SciFACT #756

| Field | Content |
|-------|---------|
| **Claim** | "Many proteins in human cells can be post-translationally modified at lysine residues via acetylation." |
| **Evidence** | Protein Lysine Acetylated/Deacetylated Enzymes and the Metabolism-Related Diseases: Lysine acetylation is a widespread and highly regulated post-translational modification in human cells. |
| **Gold** | SUPPORT |
| **v** | 0.28 |
| **q_true** | 0.76 |
| **q_uncertain** | 0.31 |
| **q_false** | 0.15 |
| **Det2Ver** | SUPPORT ✓ |
| **CVPR-FV** | NEI ✗ |

**Analysis.** The quantifier "many" activates the vague-quantifier cue (b),
suppressing v to 0.28. However, this claim is objectively verifiable — it
makes a specific, evidence-backed statement about a biological mechanism. The
decomposition outputs (q_true = 0.76) correctly point to SUPPORT, but
CVPR-FV's over-cautious prior (π(NEI|0.28) ≈ 0.75) overwhelms the likelihood
signal, routing the final prediction to NEI. This illustrates a key limitation
of the current cue lexicon: domain-specific quantitative language (common in
scientific claims) can spuriously activate unverifiability cues, even when the
claim is objectively checkable.

---

## Summary

| # | Source | Gold | v | Det2Ver | CVPR-FV | Winner |
|---|--------|------|---|---------|---------|--------|
| 1 | FEVER #58686 | NEI | 0.24 | REFUTE ✗ | NEI ✓ | CVPR-FV |
| 2 | FEVER #87782 | NEI | 0.31 | SUPPORT ✗ | NEI ✓ | CVPR-FV |
| 3 | FEVER #114625 | SUPPORT | 0.78 | SUPPORT ✓ | SUPPORT ✓ | Tie |
| 4 | FEVER #185758 | REFUTE | 0.71 | REFUTE ✓ | REFUTE ✓ | Tie |
| 5 | SciFACT #756 | SUPPORT | 0.28 | SUPPORT ✓ | NEI ✗ | Det2Ver |

**Key take-aways:**
1. CVPR-FV's CVP prior consistently helps for claims with absolutist or hedged
   language whose gold label is NEI — the prior overrides conflicting binary
   decompositions.
2. For verifiable claims with no unverifiability cues (v ≥ 0.71), CVPR-FV and
   Det2Ver are effectively equivalent; the prior plays a negligible role.
3. The primary failure mode is domain-specific quantitative language (e.g.,
   "many proteins") that legitimately conveys scale in scientific writing but
   triggers the vague-quantifier cue. Extending the cue lexicon with domain-
   aware exceptions (e.g., suppressing cue (b) when paired with a specific
   measurable object) is a natural next step.
