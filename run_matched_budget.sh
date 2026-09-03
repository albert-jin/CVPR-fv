#!/usr/bin/env bash
# Reproduce the endpoint Det2Ver(200/source) rows reported in Table 1 and
# RESULTS/matched_budget.md. A release-compatible Det2Ver implementation is
# vendored in this repository; DET2VER_DIR may override it.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DET2VER_DIR=${DET2VER_DIR:-"${SCRIPT_DIR}/Det2Ver"}
OUTPUT_PATH=${OUTPUT_PATH:-"${DET2VER_DIR}/output"}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASETS=(fever vc scifact)
SHOTS=(4 32)
SEEDS=(0 1 2 3 4)

if [[ ! -f "${DET2VER_DIR}/train.py" ]]; then
    echo "Det2Ver train.py was not found at: ${DET2VER_DIR}" >&2
    echo "Clone the Det2Ver repository there or set DET2VER_DIR." >&2
    exit 2
fi

for ds in "${DATASETS[@]}"; do
    for K in "${SHOTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            exp_name="det2ver_${ds}_K${K}_rd200_seed${seed}"
            echo "=== ${exp_name} ==="
            (
                cd "${DET2VER_DIR}"
                "${PYTHON_BIN}" -u train.py \
                    --dataset "${ds}" --shot_num "${K}" --seed "${seed}" \
                    --few_shot true --zero_shot false \
                    --use_rumor_detection true --rd_total_per_dataset 200 \
                    --lr 1e-5 --num_steps 1500 --patience 5 \
                    --train_batch_size 8 --eval_batch_size 8 \
                    --exp_root "${OUTPUT_PATH}" --exp_name "${exp_name}"
            )
        done
    done
done
