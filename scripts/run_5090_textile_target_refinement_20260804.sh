#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/textile_target_refinement_20260804"
PIPELINE="${SCRIPT_DIR}/run_textile_target_refinement_pipeline_20260804.py"
AGGREGATOR="${SCRIPT_DIR}/aggregate_textile_target_refinement_20260804.py"
ARCHIVE="${PROJECT_ROOT}/textile_target_refinement_20260804_no_weights.tar.gz"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {preflight|pilot|select|freeze|confirm|aggregate|package|all} [options]"
  exit 2
fi
STAGE="$1"
shift
cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs"

run_pipeline() {
  "${PYTHON_BIN}" "${PIPELINE}" "$1" --project-root "${PROJECT_ROOT}" "${@:2}"
}

aggregate_results() {
  local count
  count="$(find "${OUTPUT_ROOT}/confirm" -path '*/outer_test/test_summary.json' 2>/dev/null | wc -l)"
  if [[ "${count}" -ne 75 ]]; then
    echo "[ERROR] Expected 75 refinement summaries, found ${count}." >&2
    exit 1
  fi
  "${PYTHON_BIN}" "${AGGREGATOR}" \
    --project-root "${PROJECT_ROOT}" \
    --bootstrap 20000 \
    --bootstrap-seed 20260804
}

package_results() {
  if [[ ! -f "${OUTPUT_ROOT}/aggregate/aggregation_manifest.json" ]]; then
    echo "[ERROR] Aggregate first." >&2
    exit 1
  fi
  tar \
    --exclude='*.pth' \
    --exclude='*.pt' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -czf "${ARCHIVE}" \
    -C "${PROJECT_ROOT}" \
    outputs/textile_target_refinement_20260804 \
    scripts/train_textile_target_refinement_20260804.py \
    scripts/run_textile_target_refinement_pipeline_20260804.py \
    scripts/aggregate_textile_target_refinement_20260804.py \
    scripts/run_5090_textile_target_refinement_20260804.sh \
    TEXTILE_TARGET_REFINEMENT_5090_README_20260804.md
  echo "[ARCHIVE] ${ARCHIVE}"
}

case "${STAGE}" in
  preflight|pilot|select|freeze|confirm)
    run_pipeline "${STAGE}" "$@"
    ;;
  aggregate)
    aggregate_results
    ;;
  package)
    package_results
    ;;
  all)
    run_pipeline preflight
    run_pipeline pilot
    run_pipeline select
    run_pipeline freeze
    run_pipeline confirm
    aggregate_results
    package_results
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac
