#!/bin/bash
# ----------------------------------------------------------------------
# Few-shot reproduction script for CVPR-FV (main.tex Table 1).
#
# Datasets × K-shot × seeds × 3 backbones.
# ----------------------------------------------------------------------

set -e
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUTPUT_PATH=${OUTPUT_PATH:-output}

DATASETS=("fever" "scifact" "vc")
SHOTS=(4 8 16 32)
SEEDS=(0 1 2 3 4)
BACKBONES=("t0-3b" "qwen2.5-3b" "llama-3.1-8b")

for backbone in "${BACKBONES[@]}"; do
    for ds in "${DATASETS[@]}"; do
        for K in "${SHOTS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                EXP="${ds}_K${K}_${backbone}_seed${seed}"
                echo "=== ${EXP} ==="
                python -u train.py \
                    --dataset ${ds} --shot_num ${K} --seed ${seed} \
                    --few_shot true --zero_shot false \
                    --use_cvp true --cvp_total_per_dataset 200 \
                    --backbone ${backbone} \
                    --lam_prior 0.5 --nei_floor_gamma 0.1 --lam_cvp 1.0 \
                    --lr 1e-5 --num_epochs 10 --patience 5 \
                    --train_batch_size 8 --eval_batch_size 8 \
                    --exp_root ${OUTPUT_PATH} --exp_name ${EXP}
            done
        done
    done
done
