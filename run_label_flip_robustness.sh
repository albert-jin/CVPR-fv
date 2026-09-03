#!/usr/bin/env bash
# Reproduce the pseudo-label corruption experiment in
# RESULTS/label_flip_robustness.md.

set -euo pipefail

OUTPUT_PATH=${OUTPUT_PATH:-output}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASETS=(fever vc scifact)
SEEDS=(0 1 2 3 4)
FLIP_RATES=(0 0.05 0.10 0.20)

for rate in "${FLIP_RATES[@]}"; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            exp_name="${ds}_K4_t0-3b_flip${rate}_seed${seed}"
            echo "=== ${exp_name} ==="
            "${PYTHON_BIN}" -u train.py \
                --dataset "${ds}" --shot_num 4 --seed "${seed}" \
                --backbone t0-3b --use_cvp true \
                --cvp_total_per_dataset 200 \
                --cvp_label_flip_rate "${rate}" \
                --aggregation_mode score_fusion \
                --lam_prior 0.5 --nei_floor_gamma 0.1 --lam_cvp 1.0 \
                --lr 1e-5 --num_epochs 10 --patience 5 \
                --train_batch_size 8 --eval_batch_size 8 \
                --exp_root "${OUTPUT_PATH}" --exp_name "${exp_name}"
        done
    done
done
