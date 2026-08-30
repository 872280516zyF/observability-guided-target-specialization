#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
SPLITS_ROOT="${SPLITS_ROOT:-${PROJECT_ROOT}/data/images3/grouped_outer_cv_20260730}"
CONFIG_ROOT="${CONFIG_ROOT:-${PROJECT_ROOT}/configs/grouped_forward_oof_20260730}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/grouped_forward_oof_20260730}"
SEEDS=(42 52 62)
FOLDS=(0 1 2 3 4)

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${RUN_ROOT}/logs"
LOG_FILE="${RUN_ROOT}/logs/${STAGE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

python -m py_compile \
  scripts/prepare_grouped_forward_experiments_20260730.py \
  scripts/begin_locked_forward_training.py \
  scripts/finalize_locked_forward_training.py \
  scripts/evaluate_locked_forward_evidence.py \
  scripts/aggregate_grouped_forward_oof_20260730.py

if [[ ! -f "${SPLITS_ROOT}/split_manifest.json" ]]; then
  echo "ERROR: grouped outer splits are missing. Run the inverse sensitivity stage first." >&2
  exit 1
fi

prepare_configs() {
  python scripts/prepare_grouped_forward_experiments_20260730.py \
    --base-config configs/forward_model_5090_l1_proxy.yaml \
    --splits-root "${SPLITS_ROOT}" \
    --output-dir "${CONFIG_ROOT}" \
    --run-root "${RUN_ROOT}" \
    --folds 0,1,2,3,4 \
    --seeds 42,52,62
}

case "${STAGE}" in
  prepare)
    prepare_configs
    ;;
  run)
    prepare_configs
    for fold in "${FOLDS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        config="${CONFIG_ROOT}/fold_${fold}/forward_pix2pixhd_grouped_fold${fold}_seed${seed}.yaml"
        seed_root="${RUN_ROOT}/fold_${fold}/seed${seed}"
        checkpoint="${seed_root}/checkpoints/best_forward_model.pth"
        start_manifest="${seed_root}/training_started_manifest.json"
        training_manifest="${seed_root}/checkpoints/grouped_training_manifest.json"
        training_log="${seed_root}/train_console.log"
        training_complete="${seed_root}/TRAINING_COMPLETE.flag"
        evidence_dir="${seed_root}/outer_test_evidence"
        test_csv="${SPLITS_ROOT}/fold_${fold}/label_test.csv"
        mkdir -p "${seed_root}"

        if [[ -f "${evidence_dir}/run_manifest.json" ]]; then
          echo "[SKIP] fold=${fold} seed=${seed}: outer-test evidence exists."
          continue
        fi

        if [[ ! -f "${training_complete}" ]]; then
          if [[ -f "${checkpoint}" || -f "${start_manifest}" ]]; then
            echo "ERROR: incomplete prior forward run at ${seed_root}." >&2
            echo "Do not certify a partial checkpoint. Move that one run directory aside and rerun." >&2
            exit 1
          fi
          python scripts/begin_locked_forward_training.py \
            --config "${config}" \
            --expected-checkpoint "${checkpoint}" \
            --output "${start_manifest}"
          python train_forward_model.py --config "${config}" \
            2>&1 | tee "${training_log}"
          test -f "${checkpoint}" || {
            echo "ERROR: checkpoint was not created: ${checkpoint}" >&2
            exit 1
          }
          touch "${training_complete}"
        fi

        if [[ ! -f "${training_manifest}" ]]; then
          python scripts/finalize_locked_forward_training.py \
            --config "${config}" \
            --checkpoint "${checkpoint}" \
            --start-manifest "${start_manifest}" \
            --training-log "${training_log}" \
            --output "${training_manifest}"
        fi

        python scripts/evaluate_locked_forward_evidence.py \
          --config "${config}" \
          --checkpoint "${checkpoint}" \
          --training-manifest "${training_manifest}" \
          --test-csv "${test_csv}" \
          --output-dir "${evidence_dir}" \
          --batch-size 8 \
          --num-workers 4 \
          --device cuda \
          --counterfactual-step 0.10 \
          --mask-threshold 0.25 \
          --save-predictions \
          2>&1 | tee "${seed_root}/outer_test_evaluation.log"
      done
    done
    ;;
  aggregate)
    python scripts/aggregate_grouped_forward_oof_20260730.py \
      --splits-root "${SPLITS_ROOT}" \
      --run-root "${RUN_ROOT}" \
      --folds "${FOLDS[@]}" \
      --seeds "${SEEDS[@]}"
    ;;
  package)
    ARCHIVE="${PROJECT_ROOT}/grouped_forward_oof_20260730_no_weights.tar.gz"
    tar \
      --exclude='*.pth' \
      --exclude='*.pt' \
      --exclude='*.ckpt' \
      --exclude='__pycache__' \
      -czf "${ARCHIVE}" \
      configs/grouped_forward_oof_20260730 \
      outputs/grouped_forward_oof_20260730 \
      scripts/prepare_grouped_forward_experiments_20260730.py \
      scripts/aggregate_grouped_forward_oof_20260730.py \
      scripts/run_5090_grouped_forward_oof_20260730.sh
    echo "[ARCHIVE] ${ARCHIVE}"
    ;;
  *)
    echo "Usage: bash scripts/run_5090_grouped_forward_oof_20260730.sh {prepare|run|aggregate|package}" >&2
    exit 2
    ;;
esac

echo "[DONE] $(date --iso-8601=seconds)"
echo "[LOG] ${LOG_FILE}"
