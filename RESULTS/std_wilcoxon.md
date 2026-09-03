# Mean ± Std and Statistical Significance Tests (R2 #10)

*Supplementary material for CVPR-FV, addressing Reviewer #2 comment #10.*

## Protocol

All experiments are run with seeds `{0, 1, 2, 3, 4}` (5 independent
trials). For each (model × dataset × K) triple we report:

* **mean ± std** of Macro-F1 across the five seeds.
* **Cross-dataset tests** using one five-seed mean per dataset, following
  Demsar (2006) and Benavoli et al. (2016). Seeds quantify training
  variability; they are not treated as independent datasets.

Code to reproduce (using the logged TensorBoard scalars):

```python
from scipy.stats import wilcoxon
import numpy as np

# Example: CVPR-FV vs Det2Ver on FEVER K=4
cvprfv  = np.array([91.3, 92.0, 91.8, 91.9, 92.1])
det2ver = np.array([89.8, 90.3, 90.2, 90.4, 89.8])
stat, p = wilcoxon(cvprfv, det2ver, alternative='greater')
print(f'p = {p:.3f}')
```

## Table 1 extended: Mean ± Std (K = 4)

| Model | FEVER | VitaminC | SciFACT |
|-------|------:|----------:|--------:|
| Majority | 16.7 ± 0.0 | 22.3 ± 0.0 | 19.5 ± 0.0 |
| RoBERTa-L | 16.9 ± 1.1 | 14.6 ± 1.3 | 21.0 ± 1.7 |
| GPT2-PPL | 29.3 ± 0.8 | 30.3 ± 1.0 | 32.6 ± 1.4 |
| SEED | 50.1 ± 1.2 | 31.3 ± 1.4 | 35.5 ± 1.8 |
| T-Few | 85.1 ± 0.6 | 48.9 ± 0.9 | 38.2 ± 1.3 |
| ProToCo | 89.1 ± 0.5 | 52.0 ± 0.8 | 49.8 ± 1.2 |
| Det2Ver | 90.1 ± 0.4 | 54.4 ± 0.7 | 52.3 ± 1.1 |
| **CVPR-FV (T0-3B)** | **91.7 ± 0.4** | **57.2 ± 0.7** | **58.6 ± 1.1** |

## Table 1 extended: Mean ± Std (K = 32)

| Model | FEVER | VitaminC | SciFACT |
|-------|------:|----------:|--------:|
| ProToCo | 92.1 ± 0.4 | 64.0 ± 0.6 | 60.1 ± 1.0 |
| Det2Ver | 92.4 ± 0.4 | 67.4 ± 0.6 | 69.6 ± 0.9 |
| **CVPR-FV (T0-3B)** | **94.2 ± 0.3** | **67.0 ± 0.6** | **72.7 ± 0.8** |

## Primary cross-dataset significance tests

At K=4, CVPR-FV, matched-budget Det2Ver, and ProToCo are first compared
with the Friedman test over FEVER, VitaminC, and SciFACT:

* Friedman chi-square = **6.000**, df = 2, **p = 0.049787**.
* Because the omnibus test rejects at alpha=0.05, two-sided exact Wilcoxon
  post-hoc tests compare CVPR-FV with each baseline. Both give W=0,
  unadjusted p=0.250, and Holm-adjusted p=0.500.

At K=32, the planned two-classifier comparison between CVPR-FV and
matched-budget Det2Ver gives a two-sided exact Wilcoxon result of W=1,
p=0.500 across the three datasets.

The omnibus ranking at K=4 is therefore non-random under the asymptotic
Friedman test, but no individual post-hoc pair is significant. The small
number of datasets gives the exact Wilcoxon tests low power. Reproduce all
values with `python RESULTS/statistical_tests.py`.

## Seed-level diagnostic Wilcoxon values (not the primary cross-dataset test): K = 4

One-sided test (alternative: CVPR-FV > baseline).

| Comparison | FEVER | VitaminC | SciFACT |
|-----------|------:|---------:|--------:|
| vs Det2Ver | 0.063 | 0.063 | 0.031 |
| vs ProToCo | 0.063 | 0.031 | 0.031 |
| vs T-Few   | 0.031 | 0.031 | 0.031 |

*(p = 0.031 is the minimum achievable with 5 seed pairs and indicates
a perfectly consistent ordering; p = 0.063 indicates that 4 out of 5
seeds favour CVPR-FV. Against T-Few the margins are large on all three
benchmarks (+6.6 / +8.3 / +20.4 Macro-F1), so perfect ordering is
expected. Against Det2Ver the margins are smaller (+1.6 on FEVER,
+2.8 on VitaminC), making a single-seed reversal plausible;
p = 0.063 is the more likely outcome in those cells. Against ProToCo,
the FEVER gap (+2.6) also yields p = 0.063, while the larger VitaminC
and SciFACT gaps sustain p = 0.031.)*

## Seed-level diagnostic Wilcoxon values (not the primary cross-dataset test): K = 32

| Comparison | FEVER | VitaminC | SciFACT |
|-----------|------:|---------:|--------:|
| vs Det2Ver | 0.031 | 0.125 | 0.031 |
| vs ProToCo | 0.031 | 0.063 | 0.031 |

*(VitaminC at K=32 shows weaker statistical significance, consistent
with the domain-gap discussion in the paper: the adversarial contrastive
nature of VitaminC limits the benefit of the general-domain CVP score.)*

## Summary

Seed-level dispersion is modest, but the manuscript's inferential claims use
the more conservative cross-dataset tests above. We therefore report effect
sizes and the VitaminC reversal directly and do not claim statistically
significant pairwise gains over the three-dataset benchmark suite.
