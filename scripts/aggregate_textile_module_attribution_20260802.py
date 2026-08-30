#!/usr/bin/env python3
"""Aggregate frozen module-attribution OOF predictions and upgrade rules."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_textile_module_attribution_pipeline_20260802 import (  # noqa: E402
    FOLDS,
    SEEDS,
    confirm_specs,
    read_json,
    selected_target,
    split_dir,
)
from scripts.train_textile_module_attribution_20260802 import (  # noqa: E402
    PARAM_NAMES,
)


ANCHOR_MODEL_IDS = [
    "shared_head_resnet",
    "selected_texture_expert",
    "nonguided_selected_target",
    "random_nonselected_expert",
]
NEW_MODEL_IDS = [
    "specialist_head_plain",
    "residual_adapter_core",
    "selected_literature_winner",
    "current_full_no_attention",
    "capacity_matched_dual_shared",
    "winner_nonselected_placement",
]


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def metric_row(group: pd.DataFrame) -> Dict[str, object]:
    row: Dict[str, object] = {
        "n_samples": int(group["sample_id"].nunique()),
        "selected_target_mape": float(group["ape_selected_target"].mean()),
        "mean_ape": float(group["mean_ape"].mean()),
    }
    for parameter in PARAM_NAMES:
        row["{}_mape".format(parameter)] = float(
            group["ape_{}".format(parameter)].mean()
        )
    return row


def expected_samples(splits_root: Path) -> set:
    values = []
    for fold in FOLDS:
        frame = pd.read_csv(split_dir(splits_root, fold) / "label_test.csv")
        values.extend(frame["sample_id"].astype(str).tolist())
    counts = pd.Series(values).value_counts()
    if len(counts) != 1240 or not counts.eq(1).all():
        raise RuntimeError("Outer-test split union is not the canonical 1,240 samples")
    return set(values)


def load_new_predictions(
    output_root: Path,
    grouped_root: Path,
    splits_root: Path,
    winners: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    inventory_rows = []
    parameter_rows = []
    for fold in FOLDS:
        winner = str(winners[str(fold)])
        selected = selected_target(grouped_root, fold)
        expected_fold = len(
            pd.read_csv(split_dir(splits_root, fold) / "label_test.csv")
        )
        for seed in SEEDS:
            specs = confirm_specs(winner, selected, seed)
            for spec in specs:
                run_dir = (
                    output_root
                    / "confirm"
                    / "fold_{}".format(fold)
                    / spec["model_id"]
                    / "seed{}".format(seed)
                )
                summary_path = run_dir / "outer_test" / "test_summary.json"
                prediction_path = run_dir / "outer_test" / "test_predictions.csv"
                if not summary_path.exists() or not prediction_path.exists():
                    raise FileNotFoundError(
                        "Incomplete confirm run: {}".format(run_dir)
                    )
                summary = read_json(summary_path)
                frame = pd.read_csv(prediction_path)
                if len(frame) != expected_fold:
                    raise RuntimeError(
                        "{} has {} rows, expected {}".format(
                            prediction_path, len(frame), expected_fold
                        )
                    )
                frame["sample_id"] = frame["sample_id"].astype(str)
                frame["before_id"] = frame["before_id"].astype(str)
                if frame["sample_id"].duplicated().any():
                    raise RuntimeError("Duplicate sample in {}".format(prediction_path))
                frame["outer_fold"] = fold
                frame["seed"] = seed
                frame["model_id"] = spec["model_id"]
                frame["source_variant"] = spec["variant"]
                frame["calibration"] = "raw"
                frame["selected_target"] = selected
                frame["expert_target"] = spec["expert_target"]
                frame["ape_selected_target"] = [
                    float(row["ape_{}".format(selected)])
                    for _, row in frame.iterrows()
                ]
                prediction_frames.append(frame)
                inventory_rows.append(
                    {
                        "outer_fold": fold,
                        "seed": seed,
                        "model_id": spec["model_id"],
                        "variant": spec["variant"],
                        "selected_target": selected,
                        "expert_target": spec["expert_target"],
                        "samples": len(frame),
                        "summary": str(summary_path),
                        "predictions": str(prediction_path),
                    }
                )
                parameter_rows.append(
                    {
                        "model_id": spec["model_id"],
                        "variant": spec["variant"],
                        "outer_fold": fold,
                        "seed": seed,
                        "parameter_count": int(summary["parameter_count"]),
                    }
                )
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(inventory_rows),
        pd.DataFrame(parameter_rows),
    )


def load_anchor_predictions(grouped_root: Path) -> pd.DataFrame:
    path = grouped_root / "aggregate" / "oof_predictions_all_seeds.csv"
    if not path.exists():
        raise FileNotFoundError("Missing existing grouped OOF anchor: {}".format(path))
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame["calibration"].eq("raw")
        & frame["model_id"].isin(ANCHOR_MODEL_IDS)
    ].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["before_id"] = frame["before_id"].astype(str)
    frame["source_variant"] = frame.get("source_model_id", frame["model_id"])
    missing = set(ANCHOR_MODEL_IDS) - set(frame["model_id"].unique())
    if missing:
        raise RuntimeError("Missing anchor models: {}".format(sorted(missing)))
    return frame


def coverage_audit(predictions: pd.DataFrame, expected: set) -> pd.DataFrame:
    rows = []
    for (model_id, seed), group in predictions.groupby(["model_id", "seed"]):
        counts = group.groupby("sample_id").size()
        covered = set(counts.index.astype(str))
        complete = covered == expected and counts.eq(1).all()
        rows.append(
            {
                "model_id": model_id,
                "seed": int(seed),
                "rows": int(len(group)),
                "unique_samples": int(len(covered)),
                "outer_folds": int(group["outer_fold"].nunique()),
                "sample_once_fraction": float(counts.eq(1).mean()),
                "complete_oof": bool(complete),
            }
        )
    frame = pd.DataFrame(rows)
    expected_models = set(ANCHOR_MODEL_IDS + NEW_MODEL_IDS)
    missing_models = expected_models - set(frame["model_id"])
    if missing_models or len(frame) != len(expected_models) * len(SEEDS):
        raise RuntimeError(
            "Coverage model inventory mismatch: missing={} rows={}".format(
                sorted(missing_models), len(frame)
            )
        )
    if not frame["complete_oof"].all():
        raise RuntimeError(
            "Incomplete OOF coverage:\n{}".format(
                frame.loc[~frame["complete_oof"]].to_string(index=False)
            )
        )
    return frame


def metric_tables(predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    for (model_id, seed), group in predictions.groupby(["model_id", "seed"]):
        seed_rows.append(
            {"model_id": model_id, "seed": int(seed), **metric_row(group)}
        )
    by_seed = pd.DataFrame(seed_rows)
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
    metrics = [
        *["ape_{}".format(parameter) for parameter in PARAM_NAMES],
        "ape_selected_target",
        "mean_ape",
    ]
    identities = [
        "model_id",
        "sample_id",
        "before_id",
        "outer_fold",
        "selected_target",
    ]
    return predictions.groupby(identities, as_index=False)[metrics].mean()


def binomial_one_sided(successes: int, trials: int) -> float:
    def choose(n: int, k: int) -> int:
        k = min(k, n - k)
        value = 1
        for index in range(1, k + 1):
            value = value * (n - k + index) // index
        return value

    return float(
        sum(choose(trials, k) for k in range(successes, trials + 1))
        / (2.0**trials)
    )


def comparison_pairs() -> List[Tuple[str, str]]:
    return [
        ("shared_head_resnet", "specialist_head_plain"),
        ("shared_head_resnet", "residual_adapter_core"),
        ("shared_head_resnet", "selected_literature_winner"),
        ("residual_adapter_core", "selected_literature_winner"),
        ("selected_texture_expert", "selected_literature_winner"),
        ("winner_nonselected_placement", "selected_literature_winner"),
        ("current_full_no_attention", "selected_texture_expert"),
        ("capacity_matched_dual_shared", "selected_texture_expert"),
        ("nonguided_selected_target", "selected_texture_expert"),
        ("random_nonselected_expert", "selected_texture_expert"),
    ]


def clustered_comparisons(
    averaged: pd.DataFrame, bootstrap: int, seed: int
) -> pd.DataFrame:
    rows = []
    for pair_index, (baseline, candidate) in enumerate(comparison_pairs()):
        base = averaged.loc[averaged["model_id"].eq(baseline)]
        cand = averaged.loc[averaged["model_id"].eq(candidate)]
        paired = base.merge(
            cand,
            on=["sample_id", "before_id", "outer_fold", "selected_target"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        if len(paired) != 1240:
            raise RuntimeError(
                "Comparison {} vs {} has {} paired samples".format(
                    baseline, candidate, len(paired)
                )
            )
        for metric_index, metric in enumerate(["ape_selected_target", "mean_ape"]):
            delta = (
                paired["{}_candidate".format(metric)]
                - paired["{}_baseline".format(metric)]
            )
            group_delta = delta.groupby(paired["before_id"]).mean()
            values = group_delta.to_numpy(float)
            local_rng = np.random.default_rng(seed + 100 * pair_index + metric_index)
            draws = values[
                local_rng.integers(0, len(values), size=(bootstrap, len(values)))
            ].mean(axis=1)
            improved = int((values < 0).sum())
            worse = int((values > 0).sum())
            rows.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "n_samples": 1240,
                    "n_before_groups": int(len(values)),
                    "baseline_sample_mean": float(
                        paired["{}_baseline".format(metric)].mean()
                    ),
                    "candidate_sample_mean": float(
                        paired["{}_candidate".format(metric)].mean()
                    ),
                    "group_mean_delta_candidate_minus_baseline": float(values.mean()),
                    "cluster_bootstrap_ci_low": float(np.quantile(draws, 0.025)),
                    "cluster_bootstrap_ci_high": float(np.quantile(draws, 0.975)),
                    "improved_groups": improved,
                    "worse_groups": worse,
                    "tied_groups": int((values == 0).sum()),
                    "group_sign_test_one_sided_p": binomial_one_sided(
                        improved, improved + worse
                    )
                    if improved + worse
                    else 1.0,
                }
            )
    return pd.DataFrame(rows)


def parameter_table(
    new_parameters: pd.DataFrame, grouped_root: Path
) -> pd.DataFrame:
    rows = []
    for (model_id, variant), group in new_parameters.groupby(
        ["model_id", "variant"]
    ):
        rows.append(
            {
                "model_id": model_id,
                "variant": variant,
                "min_trainable_parameters": int(group["parameter_count"].min()),
                "max_trainable_parameters": int(group["parameter_count"].max()),
                "source": "new frozen module-attribution runs",
            }
        )
    old_path = grouped_root / "aggregate" / "model_parameter_counts.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        mapping = {
            "shared_head_resnet": "shared_head_resnet",
            "selected_texture_expert": "texture_expert_dpi",
            "nonguided_selected_target": "nonguided_selected_target",
        }
        for model_id, source_id in mapping.items():
            source = old.loc[old["model_id"].eq(source_id)]
            if not source.empty:
                rows.append(
                    {
                        "model_id": model_id,
                        "variant": source_id,
                        "min_trainable_parameters": int(float(source.iloc[0]["min"])),
                        "max_trainable_parameters": int(float(source.iloc[0]["max"])),
                        "source": str(old_path),
                    }
                )
    return pd.DataFrame(rows).sort_values(["model_id", "variant"])


def lookup_comparison(
    comparisons: pd.DataFrame, baseline: str, candidate: str, metric: str
) -> Dict[str, object]:
    row = comparisons.loc[
        comparisons["baseline"].eq(baseline)
        & comparisons["candidate"].eq(candidate)
        & comparisons["metric"].eq(metric)
    ]
    if len(row) != 1:
        raise RuntimeError(
            "Expected one comparison row for {} {} {}".format(
                baseline, candidate, metric
            )
        )
    return row.iloc[0].to_dict()


def upgrade_decision(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    winners: Dict[str, str],
) -> Dict[str, object]:
    current = summary.set_index("model_id").loc["selected_texture_expert"]
    candidate = summary.set_index("model_id").loc["selected_literature_winner"]
    versus_current = lookup_comparison(
        comparisons,
        "selected_texture_expert",
        "selected_literature_winner",
        "ape_selected_target",
    )
    versus_placement = lookup_comparison(
        comparisons,
        "winner_nonselected_placement",
        "selected_literature_winner",
        "ape_selected_target",
    )
    overall_delta = float(candidate["mean_ape_mean"] - current["mean_ape_mean"])
    selected = "dpi"
    nonselected_deltas = {
        parameter: float(
            candidate["{}_mape_mean".format(parameter)]
            - current["{}_mape_mean".format(parameter)]
        )
        for parameter in PARAM_NAMES
        if parameter != selected
    }
    checks = {
        "same_literature_variant_selected_in_all_folds": (
            len(set(winners.values())) == 1
        ),
        "candidate_vs_current_group_delta_lt_zero": float(
            versus_current["group_mean_delta_candidate_minus_baseline"]
        )
        < 0,
        "candidate_vs_current_ci_high_lt_zero": float(
            versus_current["cluster_bootstrap_ci_high"]
        )
        < 0,
        "candidate_vs_nonselected_group_delta_lt_zero": float(
            versus_placement["group_mean_delta_candidate_minus_baseline"]
        )
        < 0,
        "candidate_vs_nonselected_ci_high_lt_zero": float(
            versus_placement["cluster_bootstrap_ci_high"]
        )
        < 0,
        "overall_mean_ape_delta_le_0_5pp": overall_delta <= 0.5,
        "all_nonselected_mape_deltas_le_1pp": max(nonselected_deltas.values())
        <= 1.0,
    }
    return {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "upgrade_main_model": bool(all(checks.values())),
        "checks": checks,
        "candidate_vs_current": versus_current,
        "candidate_vs_nonselected_placement": versus_placement,
        "overall_mean_ape_delta_pp": overall_delta,
        "nonselected_mape_deltas_pp": nonselected_deltas,
        "decision_text": (
            "Upgrade the main-text model and expose only the supported module."
            if all(checks.values())
            else "Retain the existing main model; report the new module suite in SI only."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--splits-root", default="data/images3/grouped_outer_cv_20260730"
    )
    parser.add_argument(
        "--grouped-root", default="outputs/grouped_outer_cv_20260730"
    )
    parser.add_argument(
        "--output-root", default="outputs/textile_module_attribution_20260802"
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    args.project_root = project
    for name in ["splits_root", "grouped_root", "output_root"]:
        value = Path(getattr(args, name))
        if not value.is_absolute():
            value = project / value
        setattr(args, name, value.resolve())
    return args


def main() -> None:
    args = parse_args()
    frozen_path = args.output_root / "FROZEN_SELECTION.json"
    if not frozen_path.exists():
        raise FileNotFoundError("Missing frozen protocol: {}".format(frozen_path))
    frozen = read_json(frozen_path)
    winners = {
        str(fold): str(variant)
        for fold, variant in frozen["pilot"]["winner_by_fold"].items()
    }
    if set(winners) != {str(fold) for fold in FOLDS}:
        raise RuntimeError("Frozen fold-specific winner map is incomplete")
    expected = expected_samples(args.splits_root)
    new, inventory, new_parameters = load_new_predictions(
        args.output_root, args.grouped_root, args.splits_root, winners
    )
    anchors = load_anchor_predictions(args.grouped_root)
    common = sorted(set(new.columns) & set(anchors.columns))
    required = {
        "sample_id",
        "before_id",
        "outer_fold",
        "seed",
        "model_id",
        "selected_target",
        "ape_selected_target",
        "mean_ape",
        *["ape_{}".format(parameter) for parameter in PARAM_NAMES],
    }
    missing = required - set(common)
    if missing:
        raise RuntimeError("Combined prediction columns missing {}".format(sorted(missing)))
    combined = pd.concat([anchors[common], new[common]], ignore_index=True)
    combined["sample_id"] = combined["sample_id"].astype(str)
    combined["before_id"] = combined["before_id"].astype(str)

    aggregate = args.output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    coverage = coverage_audit(combined, expected)
    by_seed, summary = metric_tables(combined)
    averaged = seed_averaged(combined)
    comparisons = clustered_comparisons(
        averaged, args.bootstrap, args.bootstrap_seed
    )
    parameters = parameter_table(new_parameters, args.grouped_root)
    decision = upgrade_decision(summary, comparisons, winners)

    inventory.to_csv(
        aggregate / "run_inventory.csv", index=False, encoding="utf-8-sig"
    )
    combined.to_csv(
        aggregate / "oof_predictions_all_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    averaged.to_csv(
        aggregate / "oof_predictions_seed_averaged.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coverage.to_csv(
        args.output_root / "coverage_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_seed.to_csv(
        aggregate / "module_metrics_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output_root / "module_oof_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparisons.to_csv(
        args.output_root / "paired_group_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    parameters.to_csv(
        args.output_root / "parameter_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(args.output_root / "upgrade_decision.json", decision)

    attention = lookup_comparison(
        comparisons,
        "current_full_no_attention",
        "selected_texture_expert",
        "ape_selected_target",
    )
    manuscript_values = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_literature_variants_by_fold": winners,
        "upgrade_decision": decision,
        "current_attention_comparison": attention,
        "model_summary": summary.to_dict(orient="records"),
        "wording_policy": {
            "main_figure": (
                "Expose the selected module only if upgrade_main_model is true; "
                "otherwise show a generic P_obs specialist block."
            ),
            "methods": "Describe every trained module regardless of outcome.",
            "supplement": "Report all pilot and confirmatory variants, including negative results.",
            "validation_label": (
                "subsequent exploratory ablation under the same leakage-controlled "
                "retrospective internal-validation framework"
            ),
        },
    }
    write_json(args.output_root / "manuscript_values.json", manuscript_values)
    write_json(
        aggregate / "aggregation_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "bootstrap_unit": "before_id",
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "expected_samples": 1240,
            "expected_before_groups": 29,
            "new_confirm_runs": int(len(inventory)),
            "new_model_groups": NEW_MODEL_IDS,
            "anchor_model_groups": ANCHOR_MODEL_IDS,
            "selected_literature_variants_by_fold": winners,
        },
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("[PASS] module attribution aggregate")


if __name__ == "__main__":
    main()
