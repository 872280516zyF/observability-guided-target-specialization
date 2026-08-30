#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/textile_module_attribution_20260802"
PIPELINE="${SCRIPT_DIR}/run_textile_module_attribution_pipeline_20260802.py"
AGGREGATOR="${SCRIPT_DIR}/aggregate_textile_module_attribution_20260802.py"
ARCHIVE="${PROJECT_ROOT}/textile_module_attribution_20260802_no_weights.tar.gz"

usage() {
  echo "Usage: $0 {preflight|pilot|select|freeze|confirm|aggregate|package|all} [options]"
  echo "Examples:"
  echo "  $0 preflight"
  echo "  $0 pilot --folds 0 2 4"
  echo "  $0 confirm --folds 1 3"
}

if [[ $# -lt 1 ]]; then
  usage
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
  if [[ "${count}" -ne 90 ]]; then
    echo "[ERROR] Expected 90 confirmatory test summaries, found ${count}." >&2
    exit 1
  fi
  "${PYTHON_BIN}" "${AGGREGATOR}" \
    --project-root "${PROJECT_ROOT}" \
    --bootstrap 20000 \
    --bootstrap-seed 20260802
}

package_results() {
  if [[ ! -f "${OUTPUT_ROOT}/aggregate/aggregation_manifest.json" ]]; then
    echo "[ERROR] Aggregate first; aggregation_manifest.json is absent." >&2
    exit 1
  fi
  tar \
    --exclude='*.pth' \
    --exclude='*.pt' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/images3/before' \
    --exclude='data/images3/after' \
    -czf "${ARCHIVE}" \
    -C "${PROJECT_ROOT}" \
    outputs/textile_module_attribution_20260802 \
    scripts/train_textile_module_attribution_20260802.py \
    scripts/run_textile_module_attribution_pipeline_20260802.py \
    scripts/aggregate_textile_module_attribution_20260802.py \
    scripts/run_5090_textile_module_attribution_20260802.sh \
    TEXTILE_MODULE_ATTRIBUTION_5090_README_20260802.md \
    analysis/textile_module_attribution_manuscript_template_20260802.md
  echo "[ARCHIVE] ${ARCHIVE}"
}

case "${STAGE}" in
  preflight|pilot|select|freeze|confirm)
    run_pipeline "${STAGE}" "$@"
    ;;
  aggregate)
    if [[ $# -ne 0 ]]; then
      echo "[ERROR] aggregate does not accept fold options." >&2
      exit 2
    fi
    aggregate_results
    ;;
  package)
    if [[ $# -ne 0 ]]; then
      echo "[ERROR] package does not accept fold options." >&2
      exit 2
    fi
    package_results
    ;;
  all)
    if [[ $# -ne 0 ]]; then
      echo "[ERROR] all does not accept fold options." >&2
      exit 2
    fi
    echo "[WARNING] 'all' is single-terminal and may take a long time."
    run_pipeline preflight
    run_pipeline pilot
    run_pipeline select
    run_pipeline freeze
    run_pipeline confirm
    aggregate_results
    package_results
    ;;
  *)
    usage
    exit 2
    ;;
esac

