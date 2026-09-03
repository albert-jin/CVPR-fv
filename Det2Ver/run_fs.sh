#!/bin/bash
# ----------------------------------------------------------------------
# Few-shot reproduction script for Det2Ver.
#
# Reproduces Table III of the paper: three datasets × four K-shot
# settings × three RD sizes {20, 50, 100} × several random seeds.
#
# Adjust CUDA_VISIBLE_DEVICES / OUTPUT_PATH to your environment.
# ----------------------------------------------------------------------

set -e
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUTPUT_PATH=${OUTPUT_PATH:-output}

DATASETS=("fever" "scifact" "vc")
SHOTS=(4 8 16 32)
RD_SIZES=(20 50 100)
SEEDS=(0 2 3 4)

for ds in "${DATASETS[@]}"; do
    for K in "${SHOTS[@]}"; do
        for rd in "${RD_SIZES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                EXP="${ds}_K${K}_rd${rd}_seed${seed}_fs"
                echo "=== ${EXP} ==="
                python -u train.py \
                    --dataset ${ds} \
                    --shot_num ${K} \
                    --seed ${seed} \
                    --few_shot true --zero_shot false \
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
done
