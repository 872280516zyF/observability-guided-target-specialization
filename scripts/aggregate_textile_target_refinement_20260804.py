#!/usr/bin/env python3
"""Aggregate iterative exploratory target-refinement OOF predictions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.aggregate_textile_ordered_pairwise_20260803 as prior


MODEL_IDS = [
    "mean_selection_baseline",
    "pobs_selection_baseline",
    "pobs_coral",
    "pobs_isolated_pairwise",
    "selected_advanced_refinement",
]


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def comparison_pairs() -> List[Tuple[str, str]]:
    return [
        ("mean_selection_baseline", "pobs_selection_baseline"),
        ("mean_selection_baseline", "pobs_coral"),
        ("mean_selection_baseline", "pobs_isolated_pairwise"),
        ("mean_selection_baseline", "selected_advanced_refinement"),
        ("pobs_selection_baseline", "pobs_coral"),
        ("pobs_selection_baseline", "pobs_isolated_pairwise"),
        ("pobs_selection_baseline", "selected_advanced_refinement"),
        ("pobs_coral", "selected_advanced_refinement"),
        ("pobs_isolated_pairwise", "selected_advanced_refinement"),
    ]


def lookup(
    frame: pd.DataFrame, baseline: str, candidate: str, metric: str
) -> Dict[str, object]:
    return prior.lookup(frame, baseline, candidate, metric)


def performance_gate(
    summary: pd.DataFrame, comparisons: pd.DataFrame, coverage: pd.DataFrame
) -> Dict[str, object]:
    table = summary.set_index("model_id")
    baseline = table.loc["mean_selection_baseline"]
    candidate = table.loc["selected_advanced_refinement"]
    dpi = lookup(
        comparisons,
        "mean_selection_baseline",
        "selected_advanced_refinement",
        "ape_dpi",
    )
    versus_pobs_selection = lookup(
        comparisons,
        "pobs_selection_baseline",
        "selected_advanced_refinement",
        "ape_dpi",
    )
    overall_delta = float(candidate["mean_ape_mean"] - baseline["mean_ape_mean"])
    nonselected = {
        name: float(candidate["{}_mape_mean".format(name)] - baseline["{}_mape_mean".format(name)])
        for name in prior.PARAM_NAMES
        if name != "dpi"
    }
    checks = {
        "advanced_vs_mean_selection_DPI_CI_high_lt_zero": float(dpi["cluster_bootstrap_ci_high"]) < 0,
        "advanced_vs_pobs_selection_DPI_CI_high_lt_zero": float(versus_pobs_selection["cluster_bootstrap_ci_high"]) < 0,
        "overall_mean_APE_delta_le_0_5pp": overall_delta <= 0.5,
        "each_nonselected_MAPE_delta_le_1pp": max(nonselected.values()) <= 1.0,
        "complete_1240_sample_29_group_5_fold_3_seed_coverage": bool(coverage["complete_oof"].all()),
    }
    passed = bool(all(checks.values()))
    return {
        "passes_exploratory_internal_performance_gate": passed,
        "eligible_for_main_text_model_replacement": False,
        "independent_confirmation_required": True,
        "reason_not_confirmatory": (
            "The same 29 grouped outer-fold test units had been examined in earlier "
            "module experiments before this refinement suite was designed."
        ),
        "checks": checks,
        "advanced_vs_mean_selection_DPI": dpi,
        "advanced_vs_pobs_selection_DPI": versus_pobs_selection,
        "overall_mean_APE_delta_pp": overall_delta,
        "nonselected_MAPE_deltas_pp": nonselected,
        "decision_text": (
            "Freeze this candidate for a genuinely independent future evaluation."
            if passed
            else "Do not advance this refinement; retain the current main model."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--splits-root", default="data/images3/grouped_outer_cv_20260730")
    parser.add_argument("--output-root", default="outputs/textile_target_refinement_20260804")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    args = parser.parse_args()
    args.project_root = Path(args.project_root).resolve()
    for name in ["splits_root", "output_root"]:
        value = Path(getattr(args, name))
        if not value.is_absolute():
            value = args.project_root / value
        setattr(args, name, value.resolve())
    return args


def main() -> None:
    args = parse_args()
    frozen_path = args.output_root / "FROZEN_SELECTION.json"
    if not frozen_path.exists():
        raise FileNotFoundError("Missing frozen protocol: {}".format(frozen_path))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("prior_outer_fold_results_had_been_examined_before_design") is not True:
        raise RuntimeError("Exploratory reuse disclosure is missing")

    prior.MODEL_IDS = list(MODEL_IDS)
    predictions, inventory = prior.load_predictions(args.output_root, args.splits_root)
    coverage = prior.coverage_audit(predictions)
    by_seed, summary = prior.metric_tables(predictions)
    averaged = prior.seed_averaged(predictions)
    original_pairs = prior.comparison_pairs
    prior.comparison_pairs = comparison_pairs
    try:
        comparisons = prior.clustered_comparisons(
            averaged, args.bootstrap, args.bootstrap_seed
        )
    finally:
        prior.comparison_pairs = original_pairs
    gate = performance_gate(summary, comparisons, coverage)

    aggregate = args.output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(aggregate / "run_inventory.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(aggregate / "oof_predictions_all_seeds.csv", index=False, encoding="utf-8-sig")
    averaged.to_csv(aggregate / "oof_predictions_seed_averaged.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(args.output_root / "coverage_audit.csv", index=False, encoding="utf-8-sig")
    by_seed.to_csv(args.output_root / "refinement_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_root / "refinement_oof_summary.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(args.output_root / "paired_group_comparisons.csv", index=False, encoding="utf-8-sig")
    inventory.groupby(["model_id", "variant"], as_index=False).agg(
        min_parameter_count=("parameter_count", "min"),
        max_parameter_count=("parameter_count", "max"),
        runs=("seed", "size"),
    ).to_csv(args.output_root / "parameter_counts.csv", index=False, encoding="utf-8-sig")
    write_json(args.output_root / "exploratory_performance_gate.json", gate)
    write_json(
        args.output_root / "manuscript_values.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "frozen_advanced_winner_by_fold": frozen["pilot"]["winner_by_fold"],
            "model_summary": summary.to_dict(orient="records"),
            "exploratory_performance_gate": gate,
            "allowed_reporting_scope": "Supplementary exploratory analysis only until independent confirmation",
        },
    )
    write_json(
        aggregate / "aggregation_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "bootstrap_unit": "before_id",
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "confirm_runs": int(len(inventory)),
            "model_groups": MODEL_IDS,
            "outer_test_reuse_disclosed": True,
        },
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print("[PASS] target-refinement aggregate")


if __name__ == "__main__":
    main()
