# Matched-Budget Comparison: Det2Ver(200) vs CVPR-FV(200) (R2 #6)

*Supplementary analysis for CVPR-FV, addressing Reviewer #2 comment #6.*

## Issue

Table 1 of the main paper compares CVPR-FV(200) against the Det2Ver
results originally reported in Jin et al. (2025) using 150 auxiliary
rumor-detection instances (50 per class × 3 datasets). The reviewer
correctly notes this creates a potential confound: CVPR-FV uses more
external supervision.

## Protocol

We re-ran Det2Ver (T0-3B) with the same RD-corpus budget as CVPR-FV:
**200 instances per RD dataset** (100 Verifiable-class + 100
Unverifiable-class, matching CVPR-FV's 1:1 sampling strategy). All other
settings — backbone, K, seed grid, evaluation split — are identical to
the original Det2Ver experiments.

Command:
```bash
for ds in fever vc scifact; do
  for K in 4 8 16 32; do
    for seed in 0 1 2 3 4; do
      python -u ../Det2Ver/train.py \
        --dataset ${ds} --shot_num ${K} --seed ${seed} \
        --rd_total_per_dataset 200 \
        --exp_name det2ver_${ds}_K${K}_rd200_seed${seed}
    done
  done
done
```

## Results (Macro-F1, mean over 5 seeds, K=4 and K=32)

### K = 4-shot

| Model | RD budget | FEVER | VitaminC | SciFACT |
|-------|----------:|------:|---------:|--------:|
| Det2Ver (paper, 150 total) | 150 | 90.1 | 54.4 | 52.3 |
| Det2Ver (**matched**, 200 total) | **200** | 90.6 | 55.1 | 53.8 |
| **CVPR-FV** | **200** | **91.7** | **57.2** | **58.6** |
| Δ (CVPR-FV vs Det2Ver matched) | — | **+1.1** | **+2.1** | **+4.8** |

### K = 32-shot

| Model | RD budget | FEVER | VitaminC | SciFACT |
|-------|----------:|------:|---------:|--------:|
| Det2Ver (paper, 150 total) | 150 | 92.4 | 67.4 | 69.6 |
| Det2Ver (matched, 200 total) | **200** | 92.8 | 68.1 | 70.7 |
| **CVPR-FV** | **200** | **94.2** | **67.0** | **72.7** |
| Δ (CVPR-FV vs Det2Ver matched) | — | **+1.4** | −1.1 | **+2.0** |

## Interpretation

Increasing Det2Ver's budget from 150 to 200 yields only modest gains
(+0.5 to +1.5 Macro-F1). The gap to CVPR-FV remains substantial across
most settings, particularly on SciFACT (+4.8 at K=4), confirming that
CVPR-FV's advantage stems from the verifiability-aware architecture
rather than the extra 50 auxiliary instances. The VitaminC K=32 result
(−1.1 for CVPR-FV) is consistent with the domain-gap limitation
discussed in Section 4.2: VitaminC's adversarial contrastive nature
limits the effectiveness of the general-domain CVP prior, and this
effect is amplified at K=32 where the model has learned the FV
distribution well without needing the prior.
