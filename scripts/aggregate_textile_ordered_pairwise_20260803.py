#!/usr/bin/env python3
"""Aggregate frozen grouped OOF ordered/pairwise textile experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PARAM_NAMES = ["frequency", "pulse_width", "speed", "dpi"]
FOLDS = [0, 1, 2, 3, 4]
SEEDS = [42, 52, 62]
MODEL_IDS = [
    "baseline_regression",
    "pairwise_only",
    "ordinal_only",
    "selected_ordered_supervision",
]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def expected_test_samples(splits_root: Path) -> Dict[int, int]:
    return {
        fold: int(len(pd.read_csv(splits_root / "fold_{}".format(fold) / "label_test.csv")))
        for fold in FOLDS
    }


def load_predictions(
    output_root: Path, splits_root: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    expected = expected_test_samples(splits_root)
    prediction_parts = []
    inventory = []
    for fold in FOLDS:
        expected_ids = set(
            pd.read_csv(splits_root / "fold_{}".format(fold) / "label_test.csv")["sample_id"]
            .astype(str)
            .tolist()
        )
        for model_id in MODEL_IDS:
            for seed in SEEDS:
                run_dir = output_root / "confirm" / "fold_{}".format(fold) / model_id / "seed{}".format(seed)
                prediction_path = run_dir / "outer_test" / "test_predictions.csv"
                summary_path = run_dir / "outer_test" / "test_summary.json"
                config_path = run_dir / "config.json"
                if not prediction_path.exists() or not summary_path.exists() or not config_path.exists():
                    raise FileNotFoundError("Incomplete confirm run: {}".format(run_dir))
                frame = pd.read_csv(prediction_path)
                frame["sample_id"] = frame["sample_id"].astype(str)
                frame["before_id"] = frame["before_id"].astype(str)
                if len(frame) != expected[fold] or set(frame["sample_id"]) != expected_ids:
                    raise RuntimeError("Fold {} {} seed {} test coverage mismatch".format(fold, model_id, seed))
                if frame["sample_id"].duplicated().any():
                    raise RuntimeError("Duplicate test sample in {}".format(run_dir))
                config = read_json(config_path)
                summary = read_json(summary_path)
                if summary.get("evaluation_scope") != "frozen_checkpoint_outer_test":
                    raise RuntimeError("Non-frozen test summary in {}".format(run_dir))
                frame["outer_fold"] = fold
                frame["seed"] = seed
                frame["model_id"] = model_id
                frame["variant"] = str(config["variant"])
                frame["selected_target"] = "dpi"
                frame["ape_selected_target"] = frame["ape_dpi"]
                prediction_parts.append(frame)
                inventory.append(
                    {
                        "outer_fold": fold,
                        "seed": seed,
                        "model_id": model_id,
                        "variant": str(config["variant"]),
                        "samples": int(len(frame)),
                        "parameter_count": int(summary["parameter_count"]),
                        "best_epoch": int(summary["best_epoch"]),
                    }
                )
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(inventory)


def coverage_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, seed), group in predictions.groupby(["model_id", "seed"]):
        counts = group["sample_id"].value_counts()
        complete = (
            len(group) == 1240
            and group["sample_id"].nunique() == 1240
            and group["before_id"].nunique() == 29
            and group["outer_fold"].nunique() == 5
            and counts.eq(1).all()
        )
        rows.append(
            {
                "model_id": model_id,
                "seed": int(seed),
                "rows": int(len(group)),
                "unique_samples": int(group["sample_id"].nunique()),
                "unique_initial_image_groups": int(group["before_id"].nunique()),
                "outer_folds": int(group["outer_fold"].nunique()),
                "each_sample_once": bool(counts.eq(1).all()),
                "complete_oof": bool(complete),
            }
        )
    audit = pd.DataFrame(rows)
    if len(audit) != len(MODEL_IDS) * len(SEEDS) or not audit["complete_oof"].all():
        raise RuntimeError("Incomplete OOF coverage\n{}".format(audit.to_string(index=False)))
    return audit


def metric_row(group: pd.DataFrame) -> Dict[str, float]:
    values: Dict[str, float] = {
        "n_samples": int(len(group)),
        "selected_target_mape": float(group["ape_dpi"].mean()),
        "mean_ape": float(group["mean_ape"].mean()),
    }
    for parameter in PARAM_NAMES:
        values["{}_mape".format(parameter)] = float(group["ape_{}".format(parameter)].mean())
    return values


def metric_tables(predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (model_id, seed), group in predictions.groupby(["model_id", "seed"]):
        rows.append({"model_id": model_id, "seed": int(seed), **metric_row(group)})
    by_seed = pd.DataFrame(rows)
    value_columns = [
        "selected_target_mape",
        "mean_ape",
        *["{}_mape".format(parameter) for parameter in PARAM_NAMES],
    ]
    summary_rows = []
    for model_id, group in by_seed.groupby("model_id"):
        row: Dict[str, object] = {
            "model_id": model_id,
            "n_seeds": int(group["seed"].nunique()),
            "n_samples_per_seed": int(group["n_samples"].iloc[0]),
        }
        for column in value_columns:
            row["{}_mean".format(column)] = float(group[column].mean())
            row["{}_sd".format(column)] = float(group[column].std(ddof=1))
        summary_rows.append(row)
    return by_seed, pd.DataFrame(summary_rows)


def seed_averaged(predictions: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mean_ape", "ape_selected_target", *["ape_{}".format(name) for name in PARAM_NAMES]]
    identities = ["model_id", "sample_id", "before_id", "outer_fold", "selected_target"]
    return predictions.groupby(identities, as_index=False)[metrics].mean()


def comparison_pairs() -> List[Tuple[str, str]]:
    return [
        ("baseline_regression", "pairwise_only"),
        ("baseline_regression", "ordinal_only"),
        ("baseline_regression", "selected_ordered_supervision"),
        ("pairwise_only", "selected_ordered_supervision"),
        ("ordinal_only", "selected_ordered_supervision"),
    ]


def clustered_comparisons(
    averaged: pd.DataFrame, bootstrap: int, seed: int
) -> pd.DataFrame:
    rows = []
    metrics = ["mean_ape", "ape_selected_target", *["ape_{}".format(name) for name in PARAM_NAMES]]
    for pair_index, (baseline, candidate) in enumerate(comparison_pairs()):
        base_frame = averaged.loc[averaged["model_id"].eq(baseline)]
        candidate_frame = averaged.loc[averaged["model_id"].eq(candidate)]
        paired = base_frame.merge(
            candidate_frame,
            on=["sample_id", "before_id", "outer_fold", "selected_target"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        if len(paired) != 1240:
            raise RuntimeError("{} vs {} paired {} samples".format(baseline, candidate, len(paired)))
        for metric_index, metric in enumerate(metrics):
            delta = paired["{}_candidate".format(metric)] - paired["{}_baseline".format(metric)]
            group_delta = delta.groupby(paired["before_id"]).mean()
            values = group_delta.to_numpy(float)
            rng = np.random.default_rng(seed + pair_index * 100 + metric_index)
            draws = values[rng.integers(0, len(values), size=(bootstrap, len(values)))].mean(axis=1)
            rows.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "n_samples": 1240,
                    "n_initial_image_groups": int(len(values)),
                    "baseline_sample_mean": float(paired["{}_baseline".format(metric)].mean()),
                    "candidate_sample_mean": float(paired["{}_candidate".format(metric)].mean()),
                    "sample_mean_delta_candidate_minus_baseline": float(delta.mean()),
                    "group_mean_delta_candidate_minus_baseline": float(values.mean()),
                    "cluster_bootstrap_ci_low": float(np.quantile(draws, 0.025)),
                    "cluster_bootstrap_ci_high": float(np.quantile(draws, 0.975)),
                    "improved_groups": int((values < 0).sum()),
                    "worse_groups": int((values > 0).sum()),
                    "tied_groups": int((values == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def lookup(
    frame: pd.DataFrame, baseline: str, candidate: str, metric: str
) -> Dict[str, object]:
    row = frame.loc[
        frame["baseline"].eq(baseline)
        & frame["candidate"].eq(candidate)
        & frame["metric"].eq(metric)
    ]
    if len(row) != 1:
        raise RuntimeError("Missing comparison {} {} {}".format(baseline, candidate, metric))
    return row.iloc[0].to_dict()


def upgrade_decision(
    summary: pd.DataFrame, comparisons: pd.DataFrame, coverage: pd.DataFrame
) -> Dict[str, object]:
    table = summary.set_index("model_id")
    baseline = table.loc["baseline_regression"]
    selected = table.loc["selected_ordered_supervision"]
    dpi = lookup(
        comparisons,
        "baseline_regression",
        "selected_ordered_supervision",
        "ape_dpi",
    )
    overall_delta = float(selected["mean_ape_mean"] - baseline["mean_ape_mean"])
    nonselected = {
        name: float(selected["{}_mape_mean".format(name)] - baseline["{}_mape_mean".format(name)])
        for name in PARAM_NAMES
        if name != "dpi"
    }
    checks = {
        "DPI_group_delta_lt_zero": float(dpi["group_mean_delta_candidate_minus_baseline"]) < 0,
        "DPI_cluster_CI_high_lt_zero": float(dpi["cluster_bootstrap_ci_high"]) < 0,
        "overall_mean_APE_delta_le_0_5pp": overall_delta <= 0.5,
        "each_nonselected_MAPE_delta_le_1pp": max(nonselected.values()) <= 1.0,
        "complete_1240_sample_29_group_5_fold_3_seed_coverage": bool(coverage["complete_oof"].all()),
    }
    return {
        "upgrade_main_model": bool(all(checks.values())),
        "checks": checks,
        "selected_vs_baseline_DPI": dpi,
        "overall_mean_APE_delta_pp": overall_delta,
        "nonselected_MAPE_deltas_pp": nonselected,
        "decision_text": (
            "Replace the main training objective with the frozen ordered-supervision rule."
            if all(checks.values())
            else "Retain the current main model; report ordered/pairwise training as exploratory SI evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--splits-root", default="data/images3/grouped_outer_cv_20260730")
    parser.add_argument("--output-root", default="outputs/textile_ordered_pairwise_20260803")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
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
    frozen = read_json(frozen_path)
    predictions, inventory = load_predictions(args.output_root, args.splits_root)
    coverage = coverage_audit(predictions)
    by_seed, summary = metric_tables(predictions)
    averaged = seed_averaged(predictions)
    comparisons = clustered_comparisons(averaged, args.bootstrap, args.bootstrap_seed)
    decision = upgrade_decision(summary, comparisons, coverage)

    aggregate = args.output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(aggregate / "run_inventory.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(aggregate / "oof_predictions_all_seeds.csv", index=False, encoding="utf-8-sig")
    averaged.to_csv(aggregate / "oof_predictions_seed_averaged.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(args.output_root / "coverage_audit.csv", index=False, encoding="utf-8-sig")
    by_seed.to_csv(args.output_root / "ordered_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_root / "ordered_oof_summary.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(args.output_root / "paired_group_comparisons.csv", index=False, encoding="utf-8-sig")
    inventory.groupby(["model_id", "variant"], as_index=False).agg(
        min_parameter_count=("parameter_count", "min"),
        max_parameter_count=("parameter_count", "max"),
        runs=("seed", "size"),
    ).to_csv(args.output_root / "parameter_counts.csv", index=False, encoding="utf-8-sig")
    write_json(args.output_root / "upgrade_decision.json", decision)
    write_json(
        args.output_root / "manuscript_values.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "frozen_winner_by_fold": frozen["pilot"]["winner_by_fold"],
            "model_summary": summary.to_dict(orient="records"),
            "upgrade_decision": decision,
            "interpretation_scope": (
                "subsequent exploratory ablation under leakage-controlled "
                "retrospective grouped internal validation"
            ),
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
            "expected_samples": 1240,
            "expected_initial_image_groups": 29,
        },
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("[PASS] ordered/pairwise aggregate")


if __name__ == "__main__":
    main()
