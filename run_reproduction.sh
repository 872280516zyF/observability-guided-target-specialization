#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT
cd "${PROJECT_ROOT}"

echo "[1/4] grouped outer-fold observability"
bash scripts/run_5090_grouped_outer_cv_20260730.sh sensitivity

echo "[2/4] grouped inverse suite"
bash scripts/run_5090_grouped_outer_cv_20260730.sh all
bash scripts/run_5090_grouped_outer_cv_20260730.sh aggregate

echo "[3/4] grouped forward models"
bash scripts/run_5090_grouped_forward_oof_20260730.sh run
bash scripts/run_5090_grouped_forward_oof_20260730.sh aggregate

echo "[4/4] completed"
