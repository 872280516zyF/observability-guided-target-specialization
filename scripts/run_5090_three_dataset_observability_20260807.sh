#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DOMAIN_RUNNER="$PROJECT_ROOT/scripts/run_dataset_specific_refinement_20260807.py"
NORB_PREPARE="$PROJECT_ROOT/scripts/prepare_smallnorb_observability_20260807.py"
NORB_RUNNER="$PROJECT_ROOT/scripts/run_smallnorb_observability_20260807.py"
COMBINER="$PROJECT_ROOT/scripts/aggregate_three_dataset_observability_20260807.py"

cd "$PROJECT_ROOT"

compile_scripts() {
  "$PYTHON_BIN" -m py_compile \
    scripts/train_dataset_specific_refinement_20260807.py \
    scripts/run_dataset_specific_refinement_20260807.py \
    scripts/prepare_smallnorb_observability_20260807.py \
    scripts/train_smallnorb_observability_20260807.py \
    scripts/run_smallnorb_observability_20260807.py \
    scripts/aggregate_three_dataset_observability_20260807.py
}

prepare() {
  echo "[1/2] Download/resume official smallNORB"
  "$PYTHON_BIN" "$NORB_PREPARE" download
  echo "[2/2] Build official-instance splits and training-only observability profiles"
  "$PYTHON_BIN" "$NORB_PREPARE" prepare
}

preflight() {
  compile_scripts
  "$PYTHON_BIN" "$DOMAIN_RUNNER" preflight
  "$PYTHON_BIN" "$NORB_RUNNER" preflight
}

pilot() {
  echo "[PILOT] DED and NIST dataset-specific candidates"
  "$PYTHON_BIN" "$DOMAIN_RUNNER" pilot
  echo "[PILOT] smallNORB spatial specialist candidates"
  "$PYTHON_BIN" "$NORB_RUNNER" pilot
}

freeze() {
  "$PYTHON_BIN" "$DOMAIN_RUNNER" freeze
  "$PYTHON_BIN" "$NORB_RUNNER" freeze
}

confirm() {
  echo "[CONFIRM] DED and NIST frozen grouped outer tests"
  "$PYTHON_BIN" "$DOMAIN_RUNNER" confirm
  echo "[CONFIRM] smallNORB official held-out-object test"
  "$PYTHON_BIN" "$NORB_RUNNER" confirm
}

aggregate() {
  "$PYTHON_BIN" "$DOMAIN_RUNNER" aggregate
  "$PYTHON_BIN" "$NORB_RUNNER" aggregate
  "$PYTHON_BIN" "$COMBINER" aggregate
}

package() {
  "$PYTHON_BIN" "$DOMAIN_RUNNER" package
  "$PYTHON_BIN" "$NORB_RUNNER" package
  "$PYTHON_BIN" "$COMBINER" package
}

stage="${1:-}"
case "$stage" in
  prepare) prepare ;;
  preflight) preflight ;;
  pilot) pilot ;;
  freeze) freeze ;;
  confirm) confirm ;;
  aggregate) aggregate ;;
  package) package ;;
  all)
    prepare
    preflight
    pilot
    freeze
    confirm
    aggregate
    package
    echo "THREE-DATASET PIPELINE COMPLETED SUCCESSFULLY"
    ;;
  *)
    echo "Usage: $0 {prepare|preflight|pilot|freeze|confirm|aggregate|package|all}" >&2
    exit 2
    ;;
esac
