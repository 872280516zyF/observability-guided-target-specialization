#!/usr/bin/env python3
"""Aggregate complete grouped outer-fold predictions and clustered statistics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PARAMETERS = ["frequency", "pulse_width", "speed", "dpi"]
EXPECTED_SOURCE_MODEL_IDS = {
    "plain_cnn",
    "shared_head_resnet",
    "selected_texture_expert",
    "selected_nonguided_equal_capacity",
    "texture_expert_frequency",
    "texture_expert_pulse_width",
    "texture_expert_speed",
    "texture_expert_dpi",
    "resnet_rf",
    "resnet_xgboost",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def combination(n: int, k: int) -> int:
    k = min(k, n - k)
    numerator = 1
    denominator = 1
    for value in range(1, k + 1):
        numerator *= n - (k - value)
        denominator *= value
    return numerator // denominator


def binomial_one_sided(successes: int, trials: int) -> float:
    return sum(
        combination(trials, count) * (0.5**trials)
        for count in range(successes, trials + 1)
    )


def canonical_model_id(model_id: str, spec: dict) -> str:
    if model_id == "selected_texture_expert":
        return f"texture_expert_{spec['expert_target']}"
    if model_id == "selected_nonguided_equal_capacity":
        return "nonguided_selected_target"
    return model_id


def load_predictions(
    splits_root: Path,
    output_root: Path,
    folds: list[int],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    run_rows: list[dict] = []
    observability_rows: list[dict] = []
    for fold in folds:
        test_labels = pd.read_csv(
            splits_root / f"fold_{fold}" / "label_test.csv"
        )
        expected_samples = set(test_labels["sample_id"].astype(str))
        selected = read_json(
            output_root
            / f"fold_{fold}"
            / "observability"
            / "grouped"
            / "selected_observability_target.json"
        )
        observability_rows.append(
            {
                "outer_fold": fold,
                "selected_target": selected["selected_target"],
                "selected_score": selected["selected_score"],
                "runner_up": selected["runner_up"],
                "score_margin": selected["score_margin"],
                "rank1_fraction": selected["selected_rank1_fraction"],
                "balanced_groups_per_parameter": selected[
                    "balanced_groups_per_parameter"
                ],
                "leave_one_group_out_estimable": selected.get(
                    "leave_one_group_out_estimable"
                ),
                "leave_one_group_out_selected_fraction": selected.get(
                    "leave_one_group_out_selected_fraction"
                ),
            }
        )
        runs_root = output_root / f"fold_{fold}" / "runs"
        if not runs_root.exists():
            raise FileNotFoundError(runs_root)
        model_dirs = {
            path.name: path
            for path in runs_root.iterdir()
            if path.is_dir() and path.name != "traditional_feature_cache"
        }
        selected_source = "selected_texture_expert"
        nonselected_sources = {
            f"texture_expert_{parameter}"
            for parameter in PARAMETERS
            if parameter != selected["selected_target"]
        }
        expected_for_fold = (
            EXPECTED_SOURCE_MODEL_IDS
            - {f"texture_expert_{selected['selected_target']}"}
        ) | {selected_source} | nonselected_sources
        missing_models = sorted(expected_for_fold - set(model_dirs))
        if missing_models:
            raise FileNotFoundError(
                f"Missing required model directories for fold {fold}: "
                f"{missing_models}"
            )
        for model_id, model_dir in sorted(model_dirs.items()):
            model_id = model_dir.name
            for seed in seeds:
                run_dir = model_dir / f"seed{seed}"
                spec_path = run_dir / "grouped_run_spec.json"
                summary_path = run_dir / "summary.json"
                if not spec_path.exists() or not summary_path.exists():
                    raise FileNotFoundError(
                        f"Incomplete run: fold={fold} model={model_id} seed={seed}"
                    )
                spec = read_json(spec_path)
                summary = read_json(summary_path)
                expected_selection = (
                    "fixed_traditional_baseline_no_test_selection"
                    if model_id in {"resnet_rf", "resnet_xgboost"}
                    else "val_mean_mape"
                )
                if summary.get("selection_metric") != expected_selection:
                    raise RuntimeError(
                        f"Non-uniform checkpoint rule in {run_dir}: "
                        f"{summary.get('selection_metric')}"
                    )
                canonical = canonical_model_id(model_id, spec)
                for calibration, path in [
                    ("raw", run_dir / "predictions" / "test_predictions.csv"),
                    (
                        "uniform_validation_ridge",
                        run_dir / "calibrated" / "test_predictions.csv",
                    ),
                ]:
                    if not path.exists():
                        raise FileNotFoundError(path)
                    frame = pd.read_csv(path)
                    frame["sample_id"] = frame["sample_id"].astype(str)
                    actual_samples = set(frame["sample_id"])
                    if actual_samples != expected_samples:
                        raise RuntimeError(
                            f"Prediction/test-label mismatch in {path}: "
                            f"expected={len(expected_samples)} actual={len(actual_samples)}"
                        )
                    frame["outer_fold"] = fold
                    frame["seed"] = seed
                    frame["source_model_id"] = model_id
                    frame["model_id"] = canonical
                    frame["calibration"] = calibration
                    frame["selected_target"] = selected["selected_target"]
                    frame["expert_target"] = spec.get("expert_target", "")
                    frame["ape_selected_target"] = [
                        row[f"ape_{row['selected_target']}"]
                        for _, row in frame.iterrows()
                    ]
                    prediction_parts.append(frame)
                run_rows.append(
                    {
                        "outer_fold": fold,
                        "seed": seed,
                        "source_model_id": model_id,
                        "model_id": canonical,
                        "selected_target": selected["selected_target"],
                        "expert_target": spec.get("expert_target", ""),
                        "variant": summary.get("variant", ""),
                        "selection_metric": summary.get("selection_metric"),
                        "parameter_count": summary.get("parameter_count"),
                        "train_samples": summary.get("num_train_samples"),
                        "val_samples": summary.get("num_val_samples"),
                        "test_samples": summary.get("num_test_samples"),
                        "run_dir": str(run_dir),
                    }
                )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    runs = pd.DataFrame(run_rows)
    observability = pd.DataFrame(observability_rows)
    return predictions, runs, observability


def add_derived_controls(
    predictions: pd.DataFrame,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    derived: list[pd.DataFrame] = []
    assignments: list[dict] = []
    for (fold, seed, calibration), group in predictions.groupby(
        ["outer_fold", "seed", "calibration"]
    ):
        selected_target = str(group["selected_target"].iloc[0])
        selected_model = f"texture_expert_{selected_target}"
        selected_frame = group.loc[group["model_id"] == selected_model].copy()
        if selected_frame.empty:
            raise RuntimeError(
                f"Missing selected expert {selected_model} for fold={fold}, seed={seed}"
            )
        selected_frame["model_id"] = "selected_texture_expert"
        selected_frame["source_model_id"] = selected_model
        derived.append(selected_frame)

        alternatives = [
            parameter for parameter in PARAMETERS
            if parameter != selected_target
        ]
        random_index = (int(fold) * 1009 + int(seed)) % len(alternatives)
        random_target = alternatives[random_index]
        random_model = f"texture_expert_{random_target}"
        random_frame = group.loc[group["model_id"] == random_model].copy()
        if random_frame.empty:
            raise RuntimeError(
                f"Missing random control {random_model} for fold={fold}, seed={seed}"
            )
        random_frame["model_id"] = "random_nonselected_expert"
        random_frame["source_model_id"] = random_model
        derived.append(random_frame)
        assignments.append(
            {
                "outer_fold": int(fold),
                "seed": int(seed),
                "calibration": calibration,
                "selected_target": selected_target,
                "random_nonselected_target": random_target,
            }
        )
    return (
        pd.concat([predictions, *derived], ignore_index=True),
        pd.DataFrame(assignments),
    )


def validate_oof(
    predictions: pd.DataFrame,
    expected_samples: set[str],
    folds: list[int],
    seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict] = []
    for (model_id, calibration, seed), group in predictions.groupby(
        ["model_id", "calibration", "seed"]
    ):
        counts = group.groupby("sample_id").size()
        coverage = set(counts.index.astype(str))
        exact = coverage == expected_samples and counts.eq(1).all()
        rows.append(
            {
                "model_id": model_id,
                "calibration": calibration,
                "seed": int(seed),
                "n_rows": int(len(group)),
                "n_samples": int(len(coverage)),
                "n_outer_folds": int(group["outer_fold"].nunique()),
                "sample_once_fraction": float(counts.eq(1).mean()),
                "complete_oof": bool(exact),
            }
        )
    audit = pd.DataFrame(rows)
    incomplete = audit.loc[~audit["complete_oof"]]
    if not incomplete.empty:
        raise RuntimeError(
            "Incomplete OOF coverage:\n" + incomplete.to_string(index=False)
        )
    return audit


def metric_row(group: pd.DataFrame) -> dict:
    row = {
        "n_samples": int(group["sample_id"].nunique()),
        "selected_target_mape": float(group["ape_selected_target"].mean()),
        "mean_ape": float(group["mean_ape"].mean()),
    }
    for parameter in PARAMETERS:
        row[f"{parameter}_mape"] = float(group[f"ape_{parameter}"].mean())
    return row


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    fold_rows = []
    for keys, group in predictions.groupby(
        ["model_id", "calibration", "seed"]
    ):
        seed_rows.append(
            {
                "model_id": keys[0],
                "calibration": keys[1],
                "seed": int(keys[2]),
                **metric_row(group),
            }
        )
    for keys, group in predictions.groupby(
        ["model_id", "calibration", "outer_fold", "seed"]
    ):
        fold_rows.append(
            {
                "model_id": keys[0],
                "calibration": keys[1],
                "outer_fold": int(keys[2]),
                "seed": int(keys[3]),
                **metric_row(group),
            }
        )
    by_seed = pd.DataFrame(seed_rows)
    by_fold_seed = pd.DataFrame(fold_rows)
    value_columns = [
        "selected_target_mape",
        "mean_ape",
        *[f"{parameter}_mape" for parameter in PARAMETERS],
    ]
    summary_rows = []
    for (model_id, calibration), group in by_seed.groupby(
        ["model_id", "calibration"]
    ):
        row = {
            "model_id": model_id,
            "calibration": calibration,
            "n_seeds": int(group["seed"].nunique()),
            "n_samples_per_seed": int(group["n_samples"].iloc[0]),
        }
        for column in value_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_sd"] = float(group[column].std(ddof=1))
        summary_rows.append(row)
    return by_seed, by_fold_seed, pd.DataFrame(summary_rows)


def seed_averaged(predictions: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        *[f"ape_{parameter}" for parameter in PARAMETERS],
        "ape_selected_target",
        "mean_ape",
    ]
    identity_columns = [
        "model_id",
        "calibration",
        "sample_id",
        "before_id",
        "outer_fold",
        "selected_target",
    ]
    return (
        predictions.groupby(identity_columns, as_index=False)[metric_columns]
        .mean()
    )


def clustered_comparisons(
    averaged: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    comparisons = [
        ("plain_cnn", "selected_texture_expert"),
        ("resnet_rf", "selected_texture_expert"),
        ("resnet_xgboost", "selected_texture_expert"),
        ("shared_head_resnet", "selected_texture_expert"),
        ("nonguided_selected_target", "selected_texture_expert"),
        ("random_nonselected_expert", "selected_texture_expert"),
    ]
    rng = np.random.default_rng(seed)
    rows = []
    for calibration in ["raw", "uniform_validation_ridge"]:
        for baseline, candidate in comparisons:
            base = averaged.loc[
                (averaged["model_id"] == baseline)
                & (averaged["calibration"] == calibration)
            ]
            cand = averaged.loc[
                (averaged["model_id"] == candidate)
                & (averaged["calibration"] == calibration)
            ]
            paired = base.merge(
                cand,
                on=["sample_id", "before_id", "outer_fold", "selected_target"],
                suffixes=("_baseline", "_candidate"),
            )
            if paired.empty:
                raise RuntimeError(
                    f"No paired predictions for {baseline} vs {candidate}"
                )
            for metric in ["ape_selected_target", "mean_ape"]:
                paired["delta"] = (
                    paired[f"{metric}_candidate"]
                    - paired[f"{metric}_baseline"]
                )
                group_delta = (
                    paired.groupby("before_id", as_index=False)["delta"].mean()
                )
                values = group_delta["delta"].to_numpy(float)
                draws = values[
                    rng.integers(
                        0, len(values), size=(bootstrap, len(values))
                    )
                ].mean(axis=1)
                improved = int((values < 0).sum())
                worse = int((values > 0).sum())
                non_ties = improved + worse
                rows.append(
                    {
                        "calibration": calibration,
                        "baseline": baseline,
                        "candidate": candidate,
                        "metric": metric,
                        "n_samples": int(paired["sample_id"].nunique()),
                        "n_before_groups": int(len(values)),
                        "baseline_sample_mean": float(
                            paired[f"{metric}_baseline"].mean()
                        ),
                        "candidate_sample_mean": float(
                            paired[f"{metric}_candidate"].mean()
                        ),
                        "group_mean_delta_candidate_minus_baseline": float(
                            values.mean()
                        ),
                        "cluster_bootstrap_ci_low": float(
                            np.quantile(draws, 0.025)
                        ),
                        "cluster_bootstrap_ci_high": float(
                            np.quantile(draws, 0.975)
                        ),
                        "improved_groups": improved,
                        "worse_groups": worse,
                        "tied_groups": int((values == 0).sum()),
                        "group_sign_test_one_sided_p": (
                            float(binomial_one_sided(improved, non_ties))
                            if non_ties
                            else 1.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    args = parser.parse_args()

    splits_root = Path(args.splits_root)
    output_root = Path(args.output_root)
    aggregate = output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_csv(
        splits_root / "canonical_labels_excluding_unverified.csv"
    )
    expected_samples = set(canonical["sample_id"].astype(str))
    duplicate_manifest_path = (
        output_root
        / "duplicate_audit"
        / "duplicate_audit_manifest.json"
    )
    if not duplicate_manifest_path.exists():
        raise FileNotFoundError(
            "Missing exact/perceptual duplicate audit: "
            f"{duplicate_manifest_path}"
        )
    duplicate_audit = read_json(duplicate_manifest_path)

    predictions, runs, observability = load_predictions(
        splits_root, output_root, args.folds, args.seeds
    )
    predictions, random_assignments = add_derived_controls(
        predictions, args.seeds
    )
    oof_audit = validate_oof(
        predictions, expected_samples, args.folds, args.seeds
    )
    by_seed, by_fold_seed, summary = metric_tables(predictions)
    averaged = seed_averaged(predictions)
    comparisons = clustered_comparisons(
        averaged, args.bootstrap, args.bootstrap_seed
    )

    outputs = {
        "oof_predictions_all_seeds.csv": predictions,
        "oof_predictions_seed_averaged.csv": averaged,
        "run_inventory.csv": runs,
        "fold_observability_summary.csv": observability,
        "random_nonselected_control_assignment.csv": random_assignments,
        "oof_coverage_audit.csv": oof_audit,
        "oof_metrics_by_seed.csv": by_seed,
        "oof_metrics_by_fold_and_seed.csv": by_fold_seed,
        "oof_model_summary.csv": summary,
        "oof_group_cluster_comparisons.csv": comparisons,
    }
    for filename, frame in outputs.items():
        frame.to_csv(
            aggregate / filename, index=False, encoding="utf-8-sig"
        )
    parameter_counts = (
        runs.groupby("model_id", as_index=False)["parameter_count"]
        .agg(["min", "max"])
        .reset_index()
    )
    parameter_counts.to_csv(
        aggregate / "model_parameter_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation": "before-image-grouped outer CV",
        "outer_folds": args.folds,
        "seeds": args.seeds,
        "expected_samples_per_seed": len(expected_samples),
        "sample_877_excluded": True,
        "checkpoint_metric": (
            "val_mean_mape for every epoch-trained neural model; "
            "not applicable to fixed-fit RF/XGBoost"
        ),
        "calibration": (
            "identical ridge fitted only on each fold's inner-validation "
            "predictions"
        ),
        "oof_complete": bool(oof_audit["complete_oof"].all()),
        "models": sorted(predictions["model_id"].unique()),
        "bootstrap_unit": "before_id",
        "bootstrap_replicates": args.bootstrap,
        "cross_partition_exact_image_duplicates": duplicate_audit[
            "exact_duplicate_pairs"
        ],
        "cross_partition_perceptual_candidates": duplicate_audit[
            "candidate_pairs"
        ],
        "outputs": {
            filename: str(aggregate / filename)
            for filename in outputs
        },
    }
    (aggregate / "GROUPED_OUTER_CV_COMPLETE.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
