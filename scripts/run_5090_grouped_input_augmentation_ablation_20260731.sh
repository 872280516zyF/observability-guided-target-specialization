#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-all}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/outputs/grouped_input_augmentation_ablation_20260731/logs"
LOG_FILE="${LOG_DIR}/${STAGE}_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

echo "[START] $(date --iso-8601=seconds)"
echo "[STAGE] ${STAGE}"
echo "[PROJECT_ROOT] ${PROJECT_ROOT}"
echo "[LOG] ${LOG_FILE}"

python scripts/run_grouped_input_augmentation_ablation_20260731.py "${STAGE}" \
  --project-root "${PROJECT_ROOT}" \
  --splits-root data/images3/grouped_outer_cv_20260730 \
  --before-dir data/images3/before \
  --after-dir data/images3/after \
  --output-root outputs/grouped_input_augmentation_ablation_20260731 \
  --folds 0,1,2,3,4 \
  --seeds 42,52,62 \
  --epochs 50 \
  --batch-size 32 \
  --num-workers 2 \
  --img-size 224 \
  --bootstrap-iters 5000 \
  2>&1 | tee "${LOG_FILE}"

echo "[DONE] $(date --iso-8601=seconds)"

