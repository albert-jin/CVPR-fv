#!/bin/bash
# ----------------------------------------------------------------------
# Ablation script for Det2Ver (Section V-C).
#
# 1) Det2Ver(0)  — no rumor-detection supervision (Table V)
# 2) Det2Ver(LIAR / COVID / FNN) — single-source RD (Figure 3)
#
# Rerun ``prepare_rumor_data.py`` beforehand to make sure the single-
# source RD files exist. For 1) we set --use_rumor_detection false;
# for 2) we override configs.rd_dataset_names via the DET2VER_RD_ONLY
# environment variable read by data_reader.py (see README).
# ----------------------------------------------------------------------

set -e
OUTPUT_PATH=${OUTPUT_PATH:-output}
DATASETS=("fever" "scifact" "vc")
SHOTS=(4 16)
SEEDS=(0 2 3 4)

# -----------------------------
# 1) Det2Ver(0)
# -----------------------------
for ds in "${DATASETS[@]}"; do
    for K in "${SHOTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            EXP="${ds}_K${K}_rd0_seed${seed}"
            python -u train.py \
                --dataset ${ds} --shot_num ${K} --seed ${seed} \
                --few_shot true --use_rumor_detection false \
                --num_steps 1500 --exp_root ${OUTPUT_PATH} --exp_name ${EXP}
        done
    done
done

# -----------------------------
# 2) Det2Ver(LIAR / COVID / FNN) — 60 total examples from ONE source.
# -----------------------------
for src in liar covid fnn; do
    export DET2VER_RD_ONLY=${src}
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            EXP="${ds}_K4_rd60_${src}_seed${seed}"
            python -u train.py \
                --dataset ${ds} --shot_num 4 --seed ${seed} \
                --few_shot true --use_rumor_detection true \
                --rd_total_per_dataset 60 \
                --num_steps 1500 --exp_root ${OUTPUT_PATH} --exp_name ${EXP}
        done
    done
    unset DET2VER_RD_ONLY
done
