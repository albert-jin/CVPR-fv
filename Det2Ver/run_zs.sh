#!/bin/bash
# ----------------------------------------------------------------------
# Zero-shot reproduction script for Det2Ver.
#
# Reproduces Table IV: no FV example, only cross-task RD supervision
# (Det2Ver(20) / Det2Ver(50)).
# ----------------------------------------------------------------------

set -e
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUTPUT_PATH=${OUTPUT_PATH:-output}

DATASETS=("fever" "scifact" "vc")
RD_SIZES=(20 50)
SEEDS=(0 2 3 4)

for ds in "${DATASETS[@]}"; do
    for rd in "${RD_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            EXP="${ds}_zs_rd${rd}_seed${seed}"
            echo "=== ${EXP} ==="
            python -u train.py \
                --dataset ${ds} \
                --seed ${seed} \
                --few_shot false --zero_shot true \
                --use_rumor_detection true \
                --rd_total_per_dataset ${rd} \
                --num_steps 1500 \
                --eval_step_interval 50 \
                --patience 5 \
                --train_batch_size 1 \
                --grad_accum_factor 2 \
                --eval_batch_size 4 \
                --exp_root ${OUTPUT_PATH} \
                --exp_name ${EXP}
        done
    done
done
