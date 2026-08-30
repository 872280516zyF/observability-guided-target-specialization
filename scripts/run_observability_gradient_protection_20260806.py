#!/usr/bin/env python3
"""Leakage-controlled pilot, freeze and OOF evaluation for gradient protection.

The previously frozen domain-tailored backbone is retained in every fold.  The
new experimental factor is only the multi-task optimization rule.  A single
non-anchor norm ratio is selected per dataset from the five inner-validation
folds, frozen before confirm, and evaluated against the prior shared model,
task-symmetric PCGrad, and same-size alternative anchor-target placements.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import run_domain_tailored_observability_20260806 as previous  # noqa: E402
from scripts import train_observability_gradient_protection_20260806 as trainer  # noqa: E402


TRAINER = PROJECT / "scripts" / "train_observability_gradient_protection_20260806.py"
OUTPUT = PROJECT / "outputs" / "observability_gradient_protection_20260806"
ARCHIVE = PROJECT / "observability_gradient_protection_20260806_no_weights.tar.gz"
PREVIOUS_OUTPUT = PROJECT / "outputs" / "domain_tailored_observability_20260806"
PREVIOUS_FROZEN = PREVIOUS_OUTPUT / "FROZEN_PROTOCOL.json"
DATASETS = previous.DATASETS
FOLDS = list(range(5))
SEEDS = [42, 52, 62]
PILOT_SEED = 42


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute(command: Sequence[str], dry_run: bool) -> None:
    print("[CMD]", " ".join(str(value) for value in command), flush=True)
    if not dry_run:
        subprocess.run(list(command), check=True)


def previous_protocol() -> Dict[str, object]:
    if not PREVIOUS_FROZEN.is_file():
        raise FileNotFoundError(
            "The frozen domain-tailored experiment is required: {}".format(PREVIOUS_FROZEN)
        )
    payload = previous.verify_frozen()
    if payload.get("outer_test_used_in_selection") is not False:
        raise RuntimeError("Previous protocol did not keep outer test out of selection")
    return payload


def fold_selection(protocol: Dict[str, object], dataset: str, fold: int) -> Dict[str, object]:
    return protocol["selection_by_dataset_fold"][dataset][str(fold)]


def ratio_label(value: float) -> str:
    return "r{:03d}".format(int(round(100.0 * float(value))))


def pilot_root(dataset: str, fold: int, role: str) -> Path:
    return OUTPUT / "pilot" / dataset / "fold_{}".format(fold) / role / "seed42"


def confirm_root(dataset: str, fold: int, role: str, seed: int) -> Path:
    return OUTPUT / "confirm" / dataset / "fold_{}".format(fold) / role / "seed{}".format(seed)


def previous_pilot_root(protocol: Dict[str, object], dataset: str, fold: int) -> Path:
    selection = fold_selection(protocol, dataset, fold)
    return previous.pilot_root(
        dataset,
        fold,
        "domain_tailored",
        str(selection["backbone"]),
        "shared_baseline",
    )


def previous_confirm_paths(dataset: str, fold: int) -> List[Path]:
    return [
        PREVIOUS_OUTPUT
        / "confirm"
        / dataset
        / "fold_{}".format(fold)
        / "domain_shared"
        / "seed{}".format(seed)
        / "test_predictions_conditions.csv"
        for seed in SEEDS
    ]


def profile(protocol: Dict[str, object], dataset: str, fold: int) -> Dict[str, object]:
    selection = fold_selection(protocol, dataset, fold)
    return {
        "scores": selection["observability_scores"],
        "high_targets": selection["high_targets"],
    }


def alternative_anchor_sets(
    parameters: Sequence[str], high_targets: Sequence[str]
) -> List[List[str]]:
    size = len(high_targets)
    selected = set(high_targets)
    return [
        list(values)
        for values in itertools.combinations(parameters, size)
        if set(values) != selected
    ]


def train_command(
    args: argparse.Namespace,
    phase: str,
    protocol: Dict[str, object],
    dataset: str,
    fold: int,
    algorithm: str,
    seed: int,
    destination: Path,
    anchor_targets: Sequence[str] = (),
    ratio: float = 1.0,
) -> List[str]:
    spec = DATASETS[dataset]
    selection = fold_selection(protocol, dataset, fold)
    command = [
        sys.executable,
        str(TRAINER),
        "--phase",
        phase,
        "--dataset-id",
        dataset,
        "--backbone",
        str(selection["backbone"]),
        "--algorithm",
        algorithm,
        "--train-csv",
        str(spec.split_path(fold, "train")),
        "--val-csv",
        str(spec.split_path(fold, "val")),
        "--parameters",
        *spec.parameters,
        "--high-targets",
        *selection["high_targets"],
        "--observability-scores",
        *[str(selection["observability_scores"][parameter]) for parameter in spec.parameters],
        "--group-column",
        spec.group_column,
        "--output-dir",
        str(destination),
        "--outer-fold",
        str(fold),
        "--seed",
        str(seed),
        "--epochs",
        str(args.pilot_epochs if phase == "pilot" else args.confirm_epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--set-size",
        str(args.set_size),
        "--image-size",
        str(args.image_size),
        "--freeze-backbone-epochs",
        str(args.freeze_backbone_epochs),
        "--nonanchor-norm-ratio",
        str(ratio),
    ]
    if anchor_targets:
        command.extend(["--anchor-targets", *anchor_targets])
    if phase == "confirm":
        command.extend(["--test-csv", str(spec.split_path(fold, "test"))])
    return command


def stage_preflight(args: argparse.Namespace) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = previous_protocol()
    dependencies = [
        TRAINER,
        Path(__file__).resolve(),
        PROJECT / "scripts" / "train_domain_tailored_observability_20260806.py",
        PROJECT / "scripts" / "run_domain_tailored_observability_20260806.py",
        PREVIOUS_FROZEN,
    ]
    missing = [path for path in dependencies if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    prior_preflight = read_json(PREVIOUS_OUTPUT / "preflight.json")
    if prior_preflight.get("status") != "PASS":
        raise RuntimeError("Previous split preflight did not pass")
    rows = []
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        for fold in FOLDS:
            selection = fold_selection(protocol, dataset, fold)
            high_targets = list(selection["high_targets"])
            alternatives = alternative_anchor_sets(spec.parameters, high_targets)
            if not alternatives:
                raise RuntimeError("No alternative anchor placement for {} fold{}".format(dataset, fold))
            for path in previous_confirm_paths(dataset, fold):
                if not path.is_file():
                    raise FileNotFoundError(path)
            frames = {
                split: pd.read_csv(
                    spec.split_path(fold, split),
                    dtype={"condition_id": str, spec.group_column: str},
                )
                for split in ("train", "val", "test")
            }
            overlap = 0
            for column in (spec.sample_column, "condition_id", spec.group_column):
                for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                    overlap += len(
                        set(frames[left][column].astype(str))
                        & set(frames[right][column].astype(str))
                    )
            if overlap:
                raise RuntimeError("Split overlap in {} fold{}".format(dataset, fold))
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "backbone": selection["backbone"],
                    "high_targets": json.dumps(high_targets),
                    "alternative_anchor_sets": json.dumps(alternatives),
                    "train_conditions": frames["train"]["condition_id"].nunique(),
                    "val_conditions": frames["val"]["condition_id"].nunique(),
                    "test_conditions": frames["test"]["condition_id"].nunique(),
                    "zero_sample_condition_group_overlap": True,
                    "prior_confirm_predictions_present": True,
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT / "preflight_audit.csv", index=False)
    write_json(
        OUTPUT / "preflight.json",
        {
            "status": "PASS",
            "scope": "retrospective cross-domain optimization experiment",
            "dataset_isolation": True,
            "outer_test_used_in_pilot_or_freeze": False,
            "previous_backbone_and_high_target_selection_reused": True,
            "mechanism": (
                "keep high-observability shared gradient unchanged; project conflicting "
                "non-anchor gradients and cap their summed norm"
            ),
            "pilot_ratios": args.anchor_ratios,
            "pilot_receives_test_csv": False,
        },
    )
    print(json.dumps(read_json(OUTPUT / "preflight.json"), indent=2), flush=True)


def stage_pilot(args: argparse.Namespace) -> None:
    protocol = previous_protocol()
    for dataset in args.datasets:
        for fold in args.folds:
            selection = fold_selection(protocol, dataset, fold)
            execute(
                train_command(
                    args,
                    "pilot",
                    protocol,
                    dataset,
                    fold,
                    "symmetric_pcgrad",
                    PILOT_SEED,
                    pilot_root(dataset, fold, "symmetric_pcgrad"),
                ),
                args.dry_run,
            )
            for ratio in args.anchor_ratios:
                role = "pobs_anchor_{}".format(ratio_label(ratio))
                execute(
                    train_command(
                        args,
                        "pilot",
                        protocol,
                        dataset,
                        fold,
                        "pobs_anchor",
                        PILOT_SEED,
                        pilot_root(dataset, fold, role),
                        selection["high_targets"],
                        ratio,
                    ),
                    args.dry_run,
                )


def validation_guardrails(
    dataset: str,
    high_targets: Sequence[str],
    baseline_summary: Dict[str, object],
    candidate_summary: Dict[str, object],
) -> Dict[str, float]:
    parameters = DATASETS[dataset].parameters
    baseline = baseline_summary["validation_group_equal"]
    candidate = candidate_summary["validation_group_equal"]
    low_degradations = [
        100.0
        * (
            float(candidate["nmae_{}".format(parameter)])
            - float(baseline["nmae_{}".format(parameter)])
        )
        for parameter in parameters
        if parameter not in high_targets
    ]
    return {
        "overall_degradation_pp": 100.0
        * (float(candidate["mean_nmae"]) - float(baseline["mean_nmae"])),
        "worst_low_target_degradation_pp": max(low_degradations) if low_degradations else 0.0,
    }


def freeze_inputs(args: argparse.Namespace, protocol: Dict[str, object]) -> List[Path]:
    paths = [
        TRAINER,
        Path(__file__).resolve(),
        PREVIOUS_FROZEN,
        OUTPUT / "preflight.json",
        OUTPUT / "preflight_audit.csv",
        OUTPUT / "pilot_inner_validation.csv",
    ]
    for dataset in args.datasets:
        for fold in FOLDS:
            baseline = previous_pilot_root(protocol, dataset, fold)
            # These prior outer-test predictions are never read by pilot or
            # freeze selection.  Their hashes are recorded only so the shared
            # reference used later by aggregate cannot silently change.
            paths.extend(previous_confirm_paths(dataset, fold))
            paths.extend(
                [
                    baseline / "validation_summary.json",
                    baseline / "validation_predictions_conditions.csv",
                    pilot_root(dataset, fold, "symmetric_pcgrad") / "validation_summary.json",
                    pilot_root(dataset, fold, "symmetric_pcgrad")
                    / "validation_predictions_conditions.csv",
                ]
            )
            for ratio in args.anchor_ratios:
                root = pilot_root(
                    dataset, fold, "pobs_anchor_{}".format(ratio_label(ratio))
                )
                paths.extend(
                    [root / "validation_summary.json", root / "validation_predictions_conditions.csv"]
                )
    return paths


def stage_freeze(args: argparse.Namespace) -> None:
    protocol = previous_protocol()
    rows = []
    selections: Dict[str, Dict[str, object]] = {}
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        for fold in FOLDS:
            selection = fold_selection(protocol, dataset, fold)
            baseline_root = previous_pilot_root(protocol, dataset, fold)
            baseline_summary = read_json(baseline_root / "validation_summary.json")
            for role, ratio in [("symmetric_pcgrad", np.nan)] + [
                ("pobs_anchor_{}".format(ratio_label(value)), float(value))
                for value in args.anchor_ratios
            ]:
                root = pilot_root(dataset, fold, role)
                summary = read_json(root / "validation_summary.json")
                if summary.get("outer_test_was_supplied") is not False:
                    raise RuntimeError("Pilot received outer test: {}".format(root))
                comparison = previous.validation_comparison(
                    args,
                    dataset,
                    fold,
                    baseline_root / "validation_predictions_conditions.csv",
                    root / "validation_predictions_conditions.csv",
                )
                guardrails = validation_guardrails(
                    dataset, selection["high_targets"], baseline_summary, summary
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "role": role,
                        "ratio": ratio,
                        "backbone": selection["backbone"],
                        "high_targets": json.dumps(selection["high_targets"]),
                        "validation_score": float(summary["best_validation_score"]),
                        "validation_high_nmae": float(
                            summary["validation_high_observability_nmae"]
                        ),
                        "high_target_gain_pp": float(comparison["mean_gain_pp"]),
                        "high_target_gain_ci_low": float(comparison["ci_low"]),
                        "high_target_gain_ci_high": float(comparison["ci_high"]),
                        **guardrails,
                    }
                )
        dataset_rows = pd.DataFrame([row for row in rows if row["dataset"] == dataset])
        ratio_rows = dataset_rows[dataset_rows["role"].str.startswith("pobs_anchor_")]
        grouped = (
            ratio_rows.groupby(["role", "ratio"], as_index=False)
            .agg(
                validation_score=("validation_score", "mean"),
                validation_high_nmae=("validation_high_nmae", "mean"),
                high_target_gain_pp=("high_target_gain_pp", "mean"),
                overall_degradation_pp=("overall_degradation_pp", "mean"),
                worst_low_target_degradation_pp=(
                    "worst_low_target_degradation_pp",
                    "max",
                ),
            )
            .sort_values(["validation_score", "ratio"])
        )
        eligible = grouped[
            grouped["overall_degradation_pp"].le(args.validation_overall_guardrail_pp)
            & grouped["worst_low_target_degradation_pp"].le(
                args.validation_parameter_guardrail_pp
            )
        ]
        pool = eligible if not eligible.empty else grouped
        winner = pool.iloc[0].to_dict()
        selections[dataset] = {
            "selected_ratio": float(winner["ratio"]),
            "selected_role": str(winner["role"]),
            "selection_scope": "five inner-validation folds only",
            "selection_rule": (
                "minimum fold-equal mean validation score among guardrail-passing "
                "ratios; if none pass, minimum score with failure recorded"
            ),
            "validation_guardrails_passed": bool(not eligible.empty),
            "candidate_table": grouped.to_dict(orient="records"),
            "fold_configuration": {
                str(fold): {
                    "backbone": fold_selection(protocol, dataset, fold)["backbone"],
                    "high_targets": fold_selection(protocol, dataset, fold)["high_targets"],
                    "observability_scores": fold_selection(protocol, dataset, fold)[
                        "observability_scores"
                    ],
                    "alternative_anchor_sets": alternative_anchor_sets(
                        spec.parameters,
                        fold_selection(protocol, dataset, fold)["high_targets"],
                    ),
                }
                for fold in FOLDS
            },
        }
    pd.DataFrame(rows).to_csv(OUTPUT / "pilot_inner_validation.csv", index=False)
    required = freeze_inputs(args, protocol)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing freeze input {}".format(missing[0]))
    payload = {
        "scope": "retrospective cross-domain gradient-protection evaluation",
        "not_untouched_external_confirmation": True,
        "dataset_isolation": True,
        "outer_test_used_in_selection": False,
        "inherited_high_target_rule": protocol["common_high_target_rule"],
        "mechanism": (
            "observability-anchored asymmetric gradient projection with a "
            "validation-selected non-anchor norm cap"
        ),
        "selection_by_dataset": selections,
        "confirm_seeds": SEEDS,
        "file_sha256": {
            str(path.relative_to(PROJECT)).replace("\\", "/"): sha256(path)
            for path in required
        },
    }
    write_json(OUTPUT / "FROZEN_PROTOCOL.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def verify_frozen() -> Dict[str, object]:
    path = OUTPUT / "FROZEN_PROTOCOL.json"
    if not path.is_file():
        raise FileNotFoundError("Run freeze before confirm")
    payload = read_json(path)
    for relative, expected in payload["file_sha256"].items():
        source = PROJECT / relative
        if not source.is_file() or sha256(source) != expected:
            raise RuntimeError("Frozen input changed: {}".format(source))
    previous_protocol()
    return payload


def stage_confirm(args: argparse.Namespace) -> None:
    frozen = verify_frozen()
    prior_protocol = read_json(PREVIOUS_FROZEN)
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        selected_ratio = float(frozen["selection_by_dataset"][dataset]["selected_ratio"])
        for fold in args.folds:
            selection = fold_selection(prior_protocol, dataset, fold)
            jobs = [
                ("symmetric_pcgrad", "symmetric_pcgrad", [], 1.0),
                (
                    "pobs_anchor",
                    "pobs_anchor",
                    selection["high_targets"],
                    selected_ratio,
                ),
            ]
            for index, targets in enumerate(
                alternative_anchor_sets(spec.parameters, selection["high_targets"])
            ):
                jobs.append(
                    (
                        "nonselected_{:02d}_{}".format(index, "_".join(targets)),
                        "pobs_anchor",
                        targets,
                        selected_ratio,
                    )
                )
            for role, algorithm, targets, ratio in jobs:
                for seed in SEEDS:
                    execute(
                        train_command(
                            args,
                            "confirm",
                            prior_protocol,
                            dataset,
                            fold,
                            algorithm,
                            seed,
                            confirm_root(dataset, fold, role, seed),
                            targets,
                            ratio,
                        ),
                        args.dry_run,
                    )


def prediction_paths(dataset: str, fold: int, role: str) -> List[Path]:
    return [
        confirm_root(dataset, fold, role, seed) / "test_predictions_conditions.csv"
        for seed in SEEDS
    ]


def nonselected_paths(dataset: str, fold: int) -> List[Path]:
    root = OUTPUT / "confirm" / dataset / "fold_{}".format(fold)
    paths = []
    for role in sorted(root.glob("nonselected_*")):
        paths.extend(
            role / "seed{}".format(seed) / "test_predictions_conditions.csv"
            for seed in SEEDS
        )
    return paths


def model_summary(
    dataset: str, role: str, frame: pd.DataFrame, parameters: Sequence[str]
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "dataset": dataset,
        "model_role": role,
        "conditions": len(frame),
        "groups": frame["group_id"].nunique(),
        "high_nmae": previous.common.group_equal(frame, "high_nmae"),
    }
    nmaes = []
    for parameter in parameters:
        value = previous.common.group_equal(frame, "nmae_{}".format(parameter))
        row["nmae_{}".format(parameter)] = value
        row["mape_{}".format(parameter)] = previous.common.group_equal(
            frame, "ape_{}".format(parameter)
        )
        nmaes.append(value)
    row["mean_nmae"] = float(np.mean(nmaes))
    return row


def stage_aggregate(args: argparse.Namespace) -> None:
    frozen = verify_frozen()
    prior_protocol = read_json(PREVIOUS_FROZEN)
    frames = []
    run_rows = []
    diagnostic_rows = []
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        for fold in FOLDS:
            selection = fold_selection(prior_protocol, dataset, fold)
            roles: Dict[str, List[Path]] = {
                "domain_shared": previous_confirm_paths(dataset, fold),
                "symmetric_pcgrad": prediction_paths(dataset, fold, "symmetric_pcgrad"),
                "pobs_anchor": prediction_paths(dataset, fold, "pobs_anchor"),
                "nonselected_anchor_mean": nonselected_paths(dataset, fold),
            }
            summary = read_json(
                PREVIOUS_OUTPUT
                / "confirm"
                / dataset
                / "fold_{}".format(fold)
                / "domain_shared"
                / "seed42"
                / "test_metrics.json"
            )
            scales = {
                parameter: float(summary["scaler"][parameter]["maximum"])
                - float(summary["scaler"][parameter]["minimum"])
                for parameter in spec.parameters
            }
            for role, paths in roles.items():
                if not paths:
                    raise RuntimeError("No prediction files for {} fold{} {}".format(dataset, fold, role))
                missing = [path for path in paths if not path.is_file()]
                if missing:
                    raise FileNotFoundError(missing[0])
                frame = previous.common.add_errors(
                    previous.common.average_predictions(paths, spec.parameters),
                    spec.parameters,
                    scales,
                )
                frame["dataset"] = dataset
                frame["fold"] = fold
                frame["model_role"] = role
                frame["high_targets"] = json.dumps(selection["high_targets"])
                frame["high_nmae"] = previous.common.selected_error(
                    frame, profile(prior_protocol, dataset, fold), spec.parameters
                )
                frames.append(frame)
            for role_directory in [
                "symmetric_pcgrad",
                "pobs_anchor",
                *[path.name for path in sorted(
                    (OUTPUT / "confirm" / dataset / "fold_{}".format(fold)).glob("nonselected_*")
                )],
            ]:
                for seed in SEEDS:
                    root = confirm_root(dataset, fold, role_directory, seed)
                    payload = read_json(root / "test_metrics.json")
                    history = pd.read_csv(root / "training_history.csv")
                    run_rows.append(
                        {
                            "dataset": dataset,
                            "fold": fold,
                            "role": role_directory,
                            "seed": seed,
                            "backbone": payload["backbone"],
                            "algorithm": payload["algorithm"],
                            "anchor_targets": json.dumps(payload["anchor_targets"]),
                            "nonanchor_norm_ratio": payload["nonanchor_norm_ratio"],
                            "parameter_count": payload["parameter_count"],
                        }
                    )
                    diagnostic_rows.append(
                        {
                            "dataset": dataset,
                            "fold": fold,
                            "role": role_directory,
                            "seed": seed,
                            "mean_conflict_fraction": history["conflict_fraction"].mean(),
                            "mean_preprojection_cosine": history[
                                "mean_preprojection_cosine"
                            ].mean(),
                            "mean_anchor_gradient_norm": history[
                                "anchor_gradient_norm"
                            ].mean(),
                            "mean_nonanchor_norm_before_cap": history[
                                "nonanchor_norm_before_cap"
                            ].mean(),
                            "mean_nonanchor_norm_after_cap": history[
                                "nonanchor_norm_after_cap"
                            ].mean(),
                        }
                    )
    oof = pd.concat(frames, ignore_index=True)
    oof.to_csv(OUTPUT / "oof_condition_predictions.csv", index=False)
    pd.DataFrame(run_rows).to_csv(OUTPUT / "run_audit.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        OUTPUT / "gradient_diagnostics_summary.csv", index=False
    )

    summaries = []
    comparisons = []
    guardrails = []
    dataset_results = {}
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        data = oof[oof["dataset"].eq(dataset)]
        roles = {
            role: frame.sort_values(["fold", "condition_id"]).reset_index(drop=True)
            for role, frame in data.groupby("model_role", sort=False)
        }
        expected = sum(
            pd.read_csv(spec.split_path(fold, "test"))["condition_id"].astype(str).nunique()
            for fold in FOLDS
        )
        for role, frame in roles.items():
            if len(frame) != expected or frame["condition_id"].nunique() != expected:
                raise RuntimeError("Incomplete OOF coverage for {} {}".format(dataset, role))
            summaries.append(model_summary(dataset, role, frame, spec.parameters))
        candidate = roles["pobs_anchor"]
        keys = ["fold", "condition_id", "group_id", "high_targets"]
        for baseline_role in (
            "domain_shared",
            "symmetric_pcgrad",
            "nonselected_anchor_mean",
        ):
            baseline = roles[baseline_role]
            if not candidate[keys].equals(baseline[keys]):
                raise RuntimeError("Prediction keys differ for {}".format(baseline_role))
            comparison = candidate[keys].copy()
            comparison["gain_high_nmae_pp"] = 100.0 * (
                baseline["high_nmae"].to_numpy(float)
                - candidate["high_nmae"].to_numpy(float)
            )
            statistics = previous.common.cluster_ci(
                comparison,
                "gain_high_nmae_pp",
                args.bootstrap_seed,
                args.bootstrap_iterations,
            )
            comparisons.append(
                {
                    "dataset": dataset,
                    "comparison": "pobs_anchor_vs_{}".format(baseline_role),
                    **statistics,
                    "conditions": len(comparison),
                    "improved_conditions": int(
                        (comparison["gain_high_nmae_pp"] > 0).sum()
                    ),
                    "worse_conditions": int(
                        (comparison["gain_high_nmae_pp"] < 0).sum()
                    ),
                }
            )
        candidate_summary = next(
            row for row in summaries if row["dataset"] == dataset and row["model_role"] == "pobs_anchor"
        )
        shared_summary = next(
            row for row in summaries if row["dataset"] == dataset and row["model_role"] == "domain_shared"
        )
        overall_degradation = 100.0 * (
            candidate_summary["mean_nmae"] - shared_summary["mean_nmae"]
        )
        low_degradations = []
        for parameter in spec.parameters:
            low_mask = candidate["high_targets"].map(
                lambda value: parameter not in json.loads(value)
            )
            if low_mask.any():
                candidate_value = previous.common.group_equal(
                    candidate.loc[low_mask], "nmae_{}".format(parameter)
                )
                shared_value = previous.common.group_equal(
                    roles["domain_shared"].loc[low_mask], "nmae_{}".format(parameter)
                )
                degradation = 100.0 * (candidate_value - shared_value)
                low_degradations.append(degradation)
                guardrails.append(
                    {
                        "dataset": dataset,
                        "guardrail": "low_observability_{}".format(parameter),
                        "candidate_nmae": candidate_value,
                        "domain_shared_nmae": shared_value,
                        "degradation_pp": degradation,
                    }
                )
        guardrails.append(
            {
                "dataset": dataset,
                "guardrail": "overall_mean",
                "candidate_nmae": candidate_summary["mean_nmae"],
                "domain_shared_nmae": shared_summary["mean_nmae"],
                "degradation_pp": overall_degradation,
            }
        )
        current = [row for row in comparisons if row["dataset"] == dataset]
        versus_shared = next(row for row in current if row["comparison"].endswith("domain_shared"))
        versus_symmetric = next(
            row for row in current if row["comparison"].endswith("symmetric_pcgrad")
        )
        versus_placement = next(
            row for row in current if row["comparison"].endswith("nonselected_anchor_mean")
        )
        supported = bool(
            versus_shared["ci_low"] > 0.0
            and versus_symmetric["ci_low"] > 0.0
            and versus_placement["ci_low"] > 0.0
            and overall_degradation <= args.confirm_overall_guardrail_pp
            and (
                not low_degradations
                or max(low_degradations) <= args.confirm_parameter_guardrail_pp
            )
        )
        dataset_results[dataset] = {
            "selected_ratio": frozen["selection_by_dataset"][dataset]["selected_ratio"],
            "pobs_anchor_vs_domain_shared": versus_shared,
            "pobs_anchor_vs_symmetric_pcgrad": versus_symmetric,
            "pobs_anchor_vs_nonselected_anchor_mean": versus_placement,
            "overall_mean_nmae_degradation_vs_domain_shared_pp": overall_degradation,
            "maximum_low_target_degradation_pp": max(low_degradations)
            if low_degradations
            else 0.0,
            "workflow_support_rule_passed": supported,
        }
    pd.DataFrame(summaries).to_csv(OUTPUT / "model_oof_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUTPUT / "paired_group_comparisons.csv", index=False)
    pd.DataFrame(guardrails).to_csv(OUTPUT / "guardrail_summary.csv", index=False)
    write_json(
        OUTPUT / "development_summary.json",
        {
            "scope": "retrospective cross-domain gradient-protection evaluation",
            "not_untouched_external_confirmation": True,
            "dataset_local_models": True,
            "outer_test_used_in_selection": False,
            "datasets": dataset_results,
            "both_datasets_support_workflow": all(
                result["workflow_support_rule_passed"]
                for result in dataset_results.values()
            ),
        },
    )
    print(json.dumps(read_json(OUTPUT / "development_summary.json"), indent=2), flush=True)


def stage_package(args: argparse.Namespace) -> None:
    if not (OUTPUT / "development_summary.json").is_file():
        raise FileNotFoundError("Aggregate before packaging")
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_dir() or path.suffix.lower() in (".pth", ".pt"):
                continue
            archive.add(path, arcname=str(path.relative_to(PROJECT)))
        for path in (
            TRAINER,
            Path(__file__).resolve(),
            PROJECT / "scripts" / "run_5090_observability_gradient_protection_20260806.sh",
            PROJECT / "OBSERVABILITY_GRADIENT_PROTECTION_README_20260806.md",
        ):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(PROJECT)))
    print("[ARCHIVE] {}".format(ARCHIVE), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("preflight", "pilot", "freeze", "confirm", "aggregate", "package")
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    parser.add_argument("--anchor-ratios", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--pilot-epochs", type=int, default=30)
    parser.add_argument("--confirm-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--set-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--validation-overall-guardrail-pp", type=float, default=0.50)
    parser.add_argument("--validation-parameter-guardrail-pp", type=float, default=1.00)
    parser.add_argument("--confirm-overall-guardrail-pp", type=float, default=0.50)
    parser.add_argument("--confirm-parameter-guardrail-pp", type=float, default=1.00)
    parser.add_argument("--pilot-bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    parser.add_argument("--relative-score-threshold", type=float, default=0.80)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(value <= 0.0 for value in args.anchor_ratios):
        raise ValueError("All anchor ratios must be positive")
    if len(set(args.anchor_ratios)) != len(args.anchor_ratios):
        raise ValueError("Anchor ratios must be unique")
    invalid_folds = sorted(set(args.folds) - set(FOLDS))
    if invalid_folds:
        raise ValueError("Invalid folds {}".format(invalid_folds))
    stages = {
        "preflight": stage_preflight,
        "pilot": stage_pilot,
        "freeze": stage_freeze,
        "confirm": stage_confirm,
        "aggregate": stage_aggregate,
        "package": stage_package,
    }
    stages[args.stage](args)


if __name__ == "__main__":
    main()
