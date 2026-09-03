#!/usr/bin/env bash
set -euo pipefail

# Reproduce the FEVER T0-3B, seed-0, K=4 spurious-conflict table from the
# trained Det2Ver and CVPR-FV adapter checkpoints. Run this script from any
# directory; paths are resolved relative to the CVPR-FV repository root.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DET2VER_CHECKPOINT="${DET2VER_CHECKPOINT:-Det2Ver/output/fever_K4_seed0/best.pt}"
CVPR_CHECKPOINT="${CVPR_CHECKPOINT:-output/fever_K4_seed0/best.pt}"

"$PYTHON_BIN" reproduce_conflict_rate.py \
  --det2ver-checkpoint "$DET2VER_CHECKPOINT" \
  --cvpr-checkpoint "$CVPR_CHECKPOINT" \
  "$@"
