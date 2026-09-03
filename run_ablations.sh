#!/usr/bin/env bash
# Reproduce the ablations reported in manuscript Section 4.3.
#
# Select one family with, for example:
#   ABLATION=det2ver_sync bash run_ablations.sh
# Valid values: all, no_cvp, det2ver_sync, lambda, fixed_control,
# gamma_lambda, single_source.

set -euo pipefail

OUTPUT_PATH=${OUTPUT_PATH:-output}
PYTHON_BIN=${PYTHON_BIN:-python}
ABLATION=${ABLATION:-all}
DATASETS=(fever vc scifact)
SEEDS=(0 1 2 3 4)
SHOTS=(4 8 16 32)
LAMBDAS=(0 0.1 0.25 0.5 1 2 4)
GAMMAS=(0 0.05 0.1 0.2 0.3)
GRID_LAMBDAS=(0 0.5 1 2 4)

if [[ -n "${BACKBONE_ONLY:-}" ]]; then
    BACKBONES=("${BACKBONE_ONLY}")
else
    BACKBONES=(t0-3b qwen2.5-3b llama-3.1-8b)
fi

BASE_ARGS=(
    --lr 1e-5 --num_epochs 10 --patience 5
    --train_batch_size 8 --eval_batch_size 8
    --cvp_total_per_dataset 200 --exp_root "${OUTPUT_PATH}"
)

selected() {
    [[ "${ABLATION}" == all || "${ABLATION}" == "$1" ]]
}

run_experiment() {
    local exp_name=$1
    shift
    echo "=== ${exp_name} ==="
    "${PYTHON_BIN}" -u train.py --exp_name "${exp_name}" "${BASE_ARGS[@]}" "$@"
}

# Figure 3: remove claim-specific CVP guidance at K=32 for all backbones.
if selected no_cvp; then
    for backbone in "${BACKBONES[@]}"; do
        for ds in "${DATASETS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_experiment "${ds}_K32_${backbone}_noCVP_seed${seed}" \
                    --dataset "${ds}" --shot_num 32 --seed "${seed}" \
                    --backbone "${backbone}" --use_cvp false \
                    --aggregation_mode score_fusion
            done
        done
    done
fi

# Figure 4: replace score fusion with Det2Ver's exact lookup-then-fallback
# synchronizer while retaining the otherwise identical trained CVPR-FV model.
if selected det2ver_sync; then
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_experiment "${ds}_K16_t0-3b_det2verSync_seed${seed}" \
                --dataset "${ds}" --shot_num 16 --seed "${seed}" \
                --backbone t0-3b --use_cvp true \
                --aggregation_mode det2ver
        done
    done
fi

# Figure 5 solid curves and the compact FEVER table.
if selected lambda; then
    for ds in "${DATASETS[@]}"; do
        for K in "${SHOTS[@]}"; do
            for lam in "${LAMBDAS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    run_experiment "${ds}_K${K}_lambda${lam}_seed${seed}" \
                        --dataset "${ds}" --shot_num "${K}" --seed "${seed}" \
                        --backbone t0-3b --use_cvp true \
                        --aggregation_mode score_fusion --lam_prior "${lam}"
                done
            done
        done
    done
fi

# Figure 5 dashed control: fixed compatibility scores independent of v.
if selected fixed_control; then
    for ds in vc scifact; do
        for K in "${SHOTS[@]}"; do
            for lam in "${LAMBDAS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    run_experiment "${ds}_K${K}_fixedControl_lambda${lam}_seed${seed}" \
                        --dataset "${ds}" --shot_num "${K}" --seed "${seed}" \
                        --backbone t0-3b --use_cvp true \
                        --aggregation_mode fixed_control --lam_prior "${lam}"
                done
            done
        done
    done
fi

# Figure 6: full gamma x lambda grid at K=32.
if selected gamma_lambda; then
    for ds in "${DATASETS[@]}"; do
        for gamma in "${GAMMAS[@]}"; do
            for lam in "${GRID_LAMBDAS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    run_experiment "${ds}_K32_gamma${gamma}_lambda${lam}_seed${seed}" \
                        --dataset "${ds}" --shot_num 32 --seed "${seed}" \
                        --backbone t0-3b --use_cvp true \
                        --aggregation_mode score_fusion \
                        --nei_floor_gamma "${gamma}" --lam_prior "${lam}"
                done
            done
        done
    done
fi

# Auxiliary diagnostic retained from the first revision.
if selected single_source; then
    for src in liar covid fnn; do
        export CVPRFV_RD_ONLY=${src}
        for ds in "${DATASETS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_experiment "${ds}_K4_${src}Only_seed${seed}" \
                    --dataset "${ds}" --shot_num 4 --seed "${seed}" \
                    --backbone t0-3b --use_cvp true \
                    --aggregation_mode score_fusion
            done
        done
        unset CVPRFV_RD_ONLY
    done
fi
