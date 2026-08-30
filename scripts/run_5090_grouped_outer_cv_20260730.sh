#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-core}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
SPLITS_ROOT="${SPLITS_ROOT:-${PROJECT_ROOT}/data/images3/grouped_outer_cv_20260730}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/grouped_outer_cv_20260730}"
SEEDS=(42 52 62)
FOLDS=(0 1 2 3 4)

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_ROOT}/logs"
LOG_FILE="${OUTPUT_ROOT}/logs/${STAGE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[START] $(date --iso-8601=seconds)"
echo "[STAGE] ${STAGE}"
echo "[PROJECT_ROOT] ${PROJECT_ROOT}"
echo "[SPLITS_ROOT] ${SPLITS_ROOT}"
echo "[OUTPUT_ROOT] ${OUTPUT_ROOT}"

python -m py_compile \
  scripts/run_plain_cnn_inverse_baseline.py \
  scripts/run_dpi_branch_ablation.py \
  scripts/build_grouped_outer_cv_splits_20260730.py \
  scripts/audit_grouped_image_duplicates_20260730.py \
  scripts/summarize_grouped_observability_20260730.py \
  scripts/calibrate_grouped_outer_run_20260730.py \
  scripts/run_grouped_outer_inverse_suite_20260730.py \
  scripts/aggregate_grouped_outer_inverse_oof_20260730.py \
  scripts/profile_grouped_inverse_models_20260730.py

if [[ ! -f "${SPLITS_ROOT}/split_manifest.json" ]]; then
  python scripts/build_grouped_outer_cv_splits_20260730.py \
    --label-csv data/images3/label_train.csv \
    --label-csv data/images3/label_val.csv \
    --output-dir "${SPLITS_ROOT}" \
    --outer-folds 5 \
    --inner-folds 5 \
    --seed 20260730
fi

case "${STAGE}" in
  sensitivity)
    python scripts/audit_grouped_image_duplicates_20260730.py \
      --splits-root "${SPLITS_ROOT}" \
      --before-dir data/images3/before \
      --after-dir data/images3/after \
      --output-dir "${OUTPUT_ROOT}/duplicate_audit" \
      --near-threshold 2
    python scripts/run_grouped_outer_inverse_suite_20260730.py \
      --splits-root "${SPLITS_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --stage sensitivity \
      --folds "${FOLDS[@]}" \
      --observability-bootstrap 10000 \
      --resume
    ;;
  core)
    python scripts/run_grouped_outer_inverse_suite_20260730.py \
      --splits-root "${SPLITS_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --stage core \
      --folds "${FOLDS[@]}" \
      --seeds "${SEEDS[@]}" \
      --epochs 50 \
      --batch-size 32 \
      --num-workers 2 \
      --selection-metric val_mean_mape \
      --calibration-alpha 0.01 \
      --observability-bootstrap 10000 \
      --resume
    ;;
  controls)
    python scripts/run_grouped_outer_inverse_suite_20260730.py \
      --splits-root "${SPLITS_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --stage controls \
      --folds "${FOLDS[@]}" \
      --seeds "${SEEDS[@]}" \
      --epochs 50 \
      --batch-size 32 \
      --num-workers 2 \
      --selection-metric val_mean_mape \
      --calibration-alpha 0.01 \
      --observability-bootstrap 10000 \
      --resume
    ;;
  traditional)
    python scripts/run_grouped_outer_inverse_suite_20260730.py \
      --splits-root "${SPLITS_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --stage traditional \
      --folds "${FOLDS[@]}" \
      --seeds "${SEEDS[@]}" \
      --num-workers 2 \
      --calibration-alpha 0.01 \
      --observability-bootstrap 10000 \
      --resume
    ;;
  all)
    python scripts/run_grouped_outer_inverse_suite_20260730.py \
      --splits-root "${SPLITS_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --stage all \
      --folds "${FOLDS[@]}" \
      --seeds "${SEEDS[@]}" \
      --epochs 50 \
      --batch-size 32 \
      --num-workers 2 \
      --selection-metric val_mean_mape \
      --calibration-alpha 0.01 \
      --observability-bootstrap 10000 \
      --resume
    ;;
  aggregate)
    python scripts/aggregate_grouped_outer_inverse_oof_20260730.py \
      --splits-root "${SPLITS_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --folds "${FOLDS[@]}" \
      --seeds "${SEEDS[@]}" \
      --bootstrap 10000 \
      --bootstrap-seed 20260730

    python scripts/profile_grouped_inverse_models_20260730.py \
      --output-dir "${OUTPUT_ROOT}/aggregate/complexity" \
      --image-size 224
    ;;
  package)
    ARCHIVE="${PROJECT_ROOT}/grouped_outer_cv_20260730_no_weights.tar.gz"
    tar \
      --exclude='*.pth' \
      --exclude='*.pt' \
      --exclude='*.ckpt' \
      --exclude='__pycache__' \
      -czf "${ARCHIVE}" \
      data/images3/grouped_outer_cv_20260730 \
      outputs/grouped_outer_cv_20260730 \
      scripts/run_plain_cnn_inverse_baseline.py \
      scripts/run_dpi_branch_ablation.py \
      scripts/run_traditional_inverse_baselines.py \
      scripts/build_grouped_outer_cv_splits_20260730.py \
      scripts/audit_grouped_image_duplicates_20260730.py \
      scripts/summarize_grouped_observability_20260730.py \
      scripts/calibrate_grouped_outer_run_20260730.py \
      scripts/run_grouped_outer_inverse_suite_20260730.py \
      scripts/aggregate_grouped_outer_inverse_oof_20260730.py \
      scripts/profile_grouped_inverse_models_20260730.py \
      scripts/run_5090_grouped_outer_cv_20260730.sh
    echo "[ARCHIVE] ${ARCHIVE}"
    ;;
  *)
    echo "Usage: bash scripts/run_5090_grouped_outer_cv_20260730.sh {sensitivity|core|controls|traditional|all|aggregate|package}" >&2
    exit 2
    ;;
esac

echo "[DONE] $(date --iso-8601=seconds)"
echo "[LOG] ${LOG_FILE}"
