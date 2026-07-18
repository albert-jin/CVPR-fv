#!/bin/bash
# ----------------------------------------------------------------------
# Ablation script for CVPR-FV (main.tex Section 4.3).
#
# 1) no-CVP: --use_cvp false           (Figure 3 style)
# 2) λ sweep (prior strength):         (Figure 5)
# 3) γ sweep (NEI floor):              (Figure 6 style, if referenced)
# 4) Single-source CVP: CVPRFV_RD_ONLY (LIAR / FNN / COVID)
# 5) Hard-mapping baseline: --use_cvp false uses aggregation with v=0.5
# ----------------------------------------------------------------------

set -e
OUTPUT_PATH=${OUTPUT_PATH:-output}
DATASETS=("fever" "scifact" "vc")
SEEDS=(0 1 2 3 4)
BACKBONE=${BACKBONE:-t0-3b}
BASE_ARGS="--backbone ${BACKBONE} --lr 1e-5 --num_epochs 10 --patience 5 \
           --train_batch_size 8 --eval_batch_size 8 --exp_root ${OUTPUT_PATH}"

# 1) No-CVP: decomposition only.
for ds in "${DATASETS[@]}"; do
    for K in 4 16; do
        for seed in "${SEEDS[@]}"; do
            EXP="${ds}_K${K}_noCVP_seed${seed}"
            python -u train.py --dataset ${ds} --shot_num ${K} --seed ${seed} \
                --few_shot true --use_cvp false --cvp_total_per_dataset 0 \
                --exp_name ${EXP} ${BASE_ARGS}
        done
    done
done

# 2) λ (prior strength) sweep.
for lam in 0.0 0.25 0.5 0.75 1.0 1.5 2.0; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            EXP="${ds}_K4_lam${lam}_seed${seed}"
            python -u train.py --dataset ${ds} --shot_num 4 --seed ${seed} \
                --few_shot true --use_cvp true --cvp_total_per_dataset 200 \
                --lam_prior ${lam} \
                --exp_name ${EXP} ${BASE_ARGS}
        done
    done
done

# 3) γ (NEI floor) sweep.
for gamma in 0.0 0.05 0.1 0.2 0.3 0.5; do
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            EXP="${ds}_K4_gamma${gamma}_seed${seed}"
            python -u train.py --dataset ${ds} --shot_num 4 --seed ${seed} \
                --few_shot true --use_cvp true --cvp_total_per_dataset 200 \
                --nei_floor_gamma ${gamma} \
                --exp_name ${EXP} ${BASE_ARGS}
        done
    done
done

# 4) Single-source CVP.
for src in liar covid fnn; do
    export CVPRFV_RD_ONLY=${src}
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            EXP="${ds}_K4_${src}only_seed${seed}"
            python -u train.py --dataset ${ds} --shot_num 4 --seed ${seed} \
                --few_shot true --use_cvp true --cvp_total_per_dataset 200 \
                --exp_name ${EXP} ${BASE_ARGS}
        done
    done
    unset CVPRFV_RD_ONLY
done
