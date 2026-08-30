#!/usr/bin/env python3
"""Leakage-controlled multi-target observability experiment orchestration.

DED and NIST are never mixed at the sample, model, scaler, checkpoint or metric
level.  Every outer fold selects its own backbone and algorithm using only that
fold's train/validation split.  The outer-test CSV is first supplied after the
fold-local choices and source hashes have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import train_multitarget_observability_20260806 as trainer  # noqa: E402


TRAINER = PROJECT / "scripts" / "train_multitarget_observability_20260806.py"
OUTPUT = PROJECT / "outputs" / "multitarget_observability_20260806"
FOLDS = list(range(5))
SEEDS = [42, 52, 62]
PILOT_SEED = 42
BACKBONES = list(trainer.BACKBONES)
ALGORITHM_CANDIDATES = [
    "weighted_shared",
    "multi_specialist",
    "multi_specialist_pcgrad",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    parameters: Tuple[str, ...]
    group_column: str
    sample_column: str
    data_root: Path
    observability_root: Path
    split_prefix: str

    def split_path(self, fold: int, split: str) -> Path:
        return self.data_root / "splits" / "fold_{}".format(fold) / "{}_{}.csv".format(
            self.split_prefix, split
        )

    def selection_path(self, fold: int) -> Path:
        return self.observability_root / "fold_{}".format(fold) / "observability" / "selection.json"

    def bootstrap_path(self, fold: int) -> Path:
        return (
            self.observability_root
            / "fold_{}".format(fold)
            / "observability"
            / "bootstrap_rankings.csv"
        )


DATASETS: Dict[str, DatasetSpec] = {
    "ded": DatasetSpec(
        name="ded",
        parameters=("laser_power", "print_speed", "powder_feed_rate"),
        group_column="track_id",
        sample_column="image_path",
        data_root=PROJECT / "data" / "ded_public",
        observability_root=PROJECT / "outputs" / "ded_external_validation_20260801",
        split_prefix="frames",
    ),
    "nist": DatasetSpec(
        name="nist",
        parameters=("spot_diameter", "laser_power", "scan_speed"),
        group_column="group_id",
        sample_column="sample_id",
        data_root=PROJECT / "data" / "nist_mds2_2923",
        observability_root=PROJECT / "outputs" / "nist_mds2_2923_replication_20260802",
        split_prefix="images",
    ),
}


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: Sequence[str], dry_run: bool) -> None:
    print("[CMD]", " ".join(str(value) for value in command), flush=True)
    if not dry_run:
        subprocess.run(list(command), check=True)


def observability_profile(
    spec: DatasetSpec, fold: int, relative_threshold: float
) -> Dict[str, object]:
    payload = read_json(spec.selection_path(fold))
    scores = {parameter: float(payload["scores"][parameter]) for parameter in spec.parameters}
    maximum = max(scores.values())
    high_targets = [
        parameter
        for parameter in spec.parameters
        if scores[parameter] + 1e-12 >= relative_threshold * maximum
    ]
    if not high_targets:
        high_targets = [max(scores, key=scores.get)]
    bootstrap = pd.read_csv(spec.bootstrap_path(fold))
    winner_fraction = {
        parameter: float((bootstrap["winner"].astype(str) == parameter).mean())
        for parameter in spec.parameters
    }
    return {
        "scores": scores,
        "maximum_score": maximum,
        "relative_threshold": relative_threshold,
        "high_targets": high_targets,
        "winner_fraction": winner_fraction,
        "selection_scope": payload.get("selection_scope"),
    }


def trainer_command(
    args: argparse.Namespace,
    phase: str,
    spec: DatasetSpec,
    fold: int,
    backbone: str,
    algorithm: str,
    seed: int,
    destination: Path,
) -> List[str]:
    profile = observability_profile(spec, fold, args.relative_score_threshold)
    command = [
        sys.executable,
        str(TRAINER),
        "--phase",
        phase,
        "--algorithm",
        algorithm,
        "--backbone",
        backbone,
        "--train-csv",
        str(spec.split_path(fold, "train")),
        "--val-csv",
        str(spec.split_path(fold, "val")),
        "--parameters",
        *spec.parameters,
        "--high-targets",
        *profile["high_targets"],
        "--observability-scores",
        *[str(profile["scores"][parameter]) for parameter in spec.parameters],
        "--group-column",
        spec.group_column,
        "--output-dir",
        str(destination),
        "--dataset-id",
        spec.name,
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
        "--selected-priority",
        str(args.selected_priority),
    ]
    if phase == "confirm":
        command.extend(["--test-csv", str(spec.split_path(fold, "test"))])
    return command


def stage_preflight(args: argparse.Namespace) -> None:
    audits: List[Dict[str, object]] = []
    profiles: List[Dict[str, object]] = []
    required_scripts = [
        TRAINER,
        Path(__file__).resolve(),
        PROJECT / "scripts" / "train_condition_set_specialist_20260802.py",
        PROJECT / "scripts" / "train_nist_set_aggregation_upgrade_20260803.py",
        PROJECT / "scripts" / "train_crossdomain_observability_optimizer_20260805.py",
    ]
    for path in required_scripts:
        if not path.is_file():
            raise FileNotFoundError(path)
    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        for fold in FOLDS:
            frames = {}
            for split in ("train", "val", "test"):
                path = spec.split_path(fold, split)
                if not path.is_file():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path, dtype={"condition_id": str, spec.group_column: str})
                required = {
                    "condition_id",
                    spec.group_column,
                    spec.sample_column,
                    "image_path",
                    *spec.parameters,
                }
                missing = sorted(required - set(frame.columns))
                if missing:
                    raise ValueError("{} is missing {}".format(path, missing))
                frames[split] = frame
            overlaps: Dict[str, int] = {}
            for column in (spec.sample_column, "condition_id", spec.group_column):
                for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                    count = len(
                        set(frames[left][column].astype(str))
                        & set(frames[right][column].astype(str))
                    )
                    overlaps["{}_{}_{}".format(column, left, right)] = count
                    if count:
                        raise RuntimeError(
                            "{} fold{} {} overlap {} vs {} = {}".format(
                                dataset_name, fold, column, left, right, count
                            )
                        )
            missing_images = 0
            for split_frame in frames.values():
                for value in split_frame["image_path"].astype(str).drop_duplicates():
                    path = Path(value)
                    if not path.is_absolute():
                        path = PROJECT / path
                    if not path.is_file():
                        missing_images += 1
            if missing_images:
                raise FileNotFoundError(
                    "{} fold{} has {} missing image paths".format(dataset_name, fold, missing_images)
                )
            audits.append(
                {
                    "dataset": dataset_name,
                    "fold": fold,
                    "train_conditions": int(frames["train"]["condition_id"].nunique()),
                    "val_conditions": int(frames["val"]["condition_id"].nunique()),
                    "test_conditions": int(frames["test"]["condition_id"].nunique()),
                    "train_groups": int(frames["train"][spec.group_column].nunique()),
                    "val_groups": int(frames["val"][spec.group_column].nunique()),
                    "test_groups": int(frames["test"][spec.group_column].nunique()),
                    "all_overlap_counts_zero": all(value == 0 for value in overlaps.values()),
                    "missing_images": missing_images,
                }
            )
            profile = observability_profile(spec, fold, args.relative_score_threshold)
            profiles.append(
                {
                    "dataset": dataset_name,
                    "fold": fold,
                    "scores": profile["scores"],
                    "winner_fraction": profile["winner_fraction"],
                    "high_targets": profile["high_targets"],
                    "selection_scope": profile["selection_scope"],
                }
            )
    pd.DataFrame(audits).to_csv(OUTPUT / "preflight_split_audit.csv", index=False)
    write_json(
        OUTPUT / "preflight.json",
        {
            "status": "PASS",
            "dataset_isolation": (
                "DED uses only DED train/validation/test CSVs; NIST uses only NIST "
                "train/validation/test CSVs"
            ),
            "selection_locality": (
                "backbone and algorithm are selected independently inside each "
                "dataset and outer fold"
            ),
            "outer_test_used_in_selection": False,
            "relative_score_threshold": args.relative_score_threshold,
            "multi_target_rule": "select every parameter with score >= threshold * fold maximum",
            "profiles": profiles,
            "split_audits": audits,
        },
    )
    print(json.dumps(read_json(OUTPUT / "preflight.json"), indent=2), flush=True)


def stage_pilot_backbone(args: argparse.Namespace) -> None:
    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        for fold in args.folds:
            for backbone in BACKBONES:
                destination = (
                    OUTPUT
                    / "pilot_backbone"
                    / dataset_name
                    / "fold_{}".format(fold)
                    / backbone
                    / "seed{}".format(PILOT_SEED)
                )
                run(
                    trainer_command(
                        args,
                        "pilot",
                        spec,
                        fold,
                        backbone,
                        "shared_baseline",
                        PILOT_SEED,
                        destination,
                    ),
                    args.dry_run,
                )


def stage_select_backbone(args: argparse.Namespace) -> None:
    rows = []
    selected: Dict[str, Dict[str, str]] = {}
    for dataset_name in args.datasets:
        selected[dataset_name] = {}
        for fold in FOLDS:
            candidates = []
            for backbone in BACKBONES:
                path = (
                    OUTPUT
                    / "pilot_backbone"
                    / dataset_name
                    / "fold_{}".format(fold)
                    / backbone
                    / "seed{}".format(PILOT_SEED)
                    / "validation_summary.json"
                )
                payload = read_json(path)
                if payload.get("outer_test_was_supplied") is not False:
                    raise RuntimeError("Backbone pilot received outer test: {}".format(path))
                row = {
                    "dataset": dataset_name,
                    "fold": fold,
                    "backbone": backbone,
                    "validation_score": float(payload["best_validation_score"]),
                    "validation_high_nmae": float(payload["validation_high_observability_nmae"]),
                    "validation_mean_nmae": float(payload["validation_group_equal"]["mean_nmae"]),
                    "parameter_count": int(payload["parameter_count"]),
                }
                rows.append(row)
                candidates.append(row)
            winner = min(candidates, key=lambda row: (row["validation_score"], row["parameter_count"]))
            selected[dataset_name][str(fold)] = str(winner["backbone"])
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "backbone_inner_validation.csv", index=False)
    write_json(
        OUTPUT / "BACKBONE_SELECTION.json",
        {
            "selection_scope": "dataset-local and outer-fold-local inner validation only",
            "outer_test_used": False,
            "selected_backbone_by_dataset_fold": selected,
        },
    )
    print(json.dumps(read_json(OUTPUT / "BACKBONE_SELECTION.json"), indent=2), flush=True)


def selected_backbone(dataset: str, fold: int) -> str:
    payload = read_json(OUTPUT / "BACKBONE_SELECTION.json")
    return str(payload["selected_backbone_by_dataset_fold"][dataset][str(fold)])


def stage_pilot_algorithm(args: argparse.Namespace) -> None:
    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        for fold in args.folds:
            backbone = selected_backbone(dataset_name, fold)
            for algorithm in ALGORITHM_CANDIDATES:
                destination = (
                    OUTPUT
                    / "pilot_algorithm"
                    / dataset_name
                    / "fold_{}".format(fold)
                    / algorithm
                    / "seed{}".format(PILOT_SEED)
                )
                run(
                    trainer_command(
                        args,
                        "pilot",
                        spec,
                        fold,
                        backbone,
                        algorithm,
                        PILOT_SEED,
                        destination,
                    ),
                    args.dry_run,
                )


def selected_error(
    frame: pd.DataFrame, profile: Dict[str, object], parameters: Sequence[str]
) -> np.ndarray:
    scores = profile["scores"]
    high_targets = profile["high_targets"]
    weights = np.asarray([scores[target] for target in high_targets], dtype=np.float64)
    values = np.column_stack(
        [frame["nmae_{}".format(target)].to_numpy(float) for target in high_targets]
    )
    return np.average(values, axis=1, weights=weights)


def validation_gain_ci(
    baseline_path: Path,
    candidate_path: Path,
    profile: Dict[str, object],
    parameters: Sequence[str],
    seed: int,
    iterations: int,
) -> Dict[str, float]:
    baseline = pd.read_csv(baseline_path).sort_values("condition_id").reset_index(drop=True)
    candidate = pd.read_csv(candidate_path).sort_values("condition_id").reset_index(drop=True)
    keys = ["condition_id", "group_id"]
    if not baseline[keys].equals(candidate[keys]):
        raise RuntimeError("Validation prediction keys differ")
    comparison = baseline[keys].copy()
    comparison["gain_pp"] = 100.0 * (
        selected_error(baseline, profile, parameters)
        - selected_error(candidate, profile, parameters)
    )
    values = comparison.groupby("group_id", sort=False)["gain_pp"].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return {
        "mean_gain_pp": float(values.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "groups": int(len(values)),
    }


def files_to_freeze(args: argparse.Namespace) -> List[Path]:
    paths = [
        TRAINER,
        Path(__file__).resolve(),
        PROJECT / "scripts" / "train_condition_set_specialist_20260802.py",
        PROJECT / "scripts" / "train_nist_set_aggregation_upgrade_20260803.py",
        PROJECT / "scripts" / "train_crossdomain_observability_optimizer_20260805.py",
        OUTPUT / "BACKBONE_SELECTION.json",
        OUTPUT / "backbone_inner_validation.csv",
    ]
    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        for fold in FOLDS:
            paths.extend(
                [
                    spec.split_path(fold, split) for split in ("train", "val", "test")
                ]
            )
            paths.extend([spec.selection_path(fold), spec.bootstrap_path(fold)])
            for backbone in BACKBONES:
                root = (
                    OUTPUT
                    / "pilot_backbone"
                    / dataset_name
                    / "fold_{}".format(fold)
                    / backbone
                    / "seed{}".format(PILOT_SEED)
                )
                paths.extend(
                    [root / "validation_summary.json", root / "validation_predictions_conditions.csv"]
                )
            for algorithm in ALGORITHM_CANDIDATES:
                root = (
                    OUTPUT
                    / "pilot_algorithm"
                    / dataset_name
                    / "fold_{}".format(fold)
                    / algorithm
                    / "seed{}".format(PILOT_SEED)
                )
                paths.extend(
                    [root / "validation_summary.json", root / "validation_predictions_conditions.csv"]
                )
    return paths


def stage_freeze(args: argparse.Namespace) -> None:
    selections: Dict[str, Dict[str, Dict[str, object]]] = {}
    rows = []
    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        selections[dataset_name] = {}
        for fold in FOLDS:
            backbone = selected_backbone(dataset_name, fold)
            profile = observability_profile(spec, fold, args.relative_score_threshold)
            baseline_root = (
                OUTPUT
                / "pilot_backbone"
                / dataset_name
                / "fold_{}".format(fold)
                / backbone
                / "seed{}".format(PILOT_SEED)
            )
            baseline_summary = read_json(baseline_root / "validation_summary.json")
            candidates = [
                {
                    "algorithm": "shared_baseline",
                    "validation_score": float(baseline_summary["best_validation_score"]),
                    "mean_gain_pp": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "eligible": True,
                }
            ]
            for algorithm in ALGORITHM_CANDIDATES:
                candidate_root = (
                    OUTPUT
                    / "pilot_algorithm"
                    / dataset_name
                    / "fold_{}".format(fold)
                    / algorithm
                    / "seed{}".format(PILOT_SEED)
                )
                summary = read_json(candidate_root / "validation_summary.json")
                if summary.get("outer_test_was_supplied") is not False:
                    raise RuntimeError("Algorithm pilot received outer test")
                gain = validation_gain_ci(
                    baseline_root / "validation_predictions_conditions.csv",
                    candidate_root / "validation_predictions_conditions.csv",
                    profile,
                    spec.parameters,
                    seed=args.bootstrap_seed + fold,
                    iterations=args.pilot_bootstrap_iterations,
                )
                candidates.append(
                    {
                        "algorithm": algorithm,
                        "validation_score": float(summary["best_validation_score"]),
                        **gain,
                        "eligible": bool(gain["ci_low"] > 0.0),
                    }
                )
            eligible = [row for row in candidates if row["eligible"] and row["algorithm"] != "shared_baseline"]
            if eligible:
                winner = min(eligible, key=lambda row: row["validation_score"])
            else:
                winner = candidates[0]
            selections[dataset_name][str(fold)] = {
                "backbone": backbone,
                "algorithm": winner["algorithm"],
                "high_targets": profile["high_targets"],
                "observability_scores": profile["scores"],
                "pilot_gain_pp": winner["mean_gain_pp"],
                "pilot_gain_ci": [winner["ci_low"], winner["ci_high"]],
            }
            for candidate in candidates:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "fold": fold,
                        "backbone": backbone,
                        "high_targets": json.dumps(profile["high_targets"]),
                        **candidate,
                        "frozen_winner": candidate["algorithm"] == winner["algorithm"],
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "algorithm_inner_validation.csv", index=False)
    freeze_files = files_to_freeze(args) + [OUTPUT / "algorithm_inner_validation.csv"]
    missing = [path for path in freeze_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing freeze inputs: {}".format(missing[:10]))
    payload = {
        "scope": "exploratory development on previously inspected DED and NIST datasets",
        "not_external_confirmation": True,
        "dataset_training_isolation": True,
        "fold_local_nested_selection": True,
        "outer_test_used_in_backbone_or_algorithm_selection": False,
        "relative_score_threshold": args.relative_score_threshold,
        "multi_target_rule": "score >= threshold * fold maximum",
        "risk_control": "activate nonbaseline algorithm only when paired validation group-bootstrap CI lower bound > 0",
        "confirm_seeds": SEEDS,
        "selection_by_dataset_fold": selections,
        "file_sha256": {
            str(path.relative_to(PROJECT)).replace("\\", "/"): sha256(path)
            for path in freeze_files
        },
    }
    write_json(OUTPUT / "FROZEN_PROTOCOL.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def verify_frozen() -> Dict[str, object]:
    frozen_path = OUTPUT / "FROZEN_PROTOCOL.json"
    if not frozen_path.is_file():
        raise FileNotFoundError("Run freeze before confirm")
    payload = read_json(frozen_path)
    for relative, expected in payload["file_sha256"].items():
        path = PROJECT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError("Frozen input changed: {}".format(path))
    return payload


def stage_confirm(args: argparse.Namespace) -> None:
    frozen = verify_frozen()
    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        for fold in args.folds:
            selection = frozen["selection_by_dataset_fold"][dataset_name][str(fold)]
            backbone = str(selection["backbone"])
            candidate = str(selection["algorithm"])
            jobs = [
                ("baseline", "shared_baseline"),
                ("weighted_control", "weighted_shared"),
            ]
            if candidate not in ("shared_baseline", "weighted_shared"):
                jobs.append(("candidate", candidate))
            for role, algorithm in jobs:
                for seed in SEEDS:
                    destination = (
                        OUTPUT
                        / "confirm"
                        / dataset_name
                        / "fold_{}".format(fold)
                        / role
                        / algorithm
                        / "seed{}".format(seed)
                    )
                    run(
                        trainer_command(
                            args,
                            "confirm",
                            spec,
                            fold,
                            backbone,
                            algorithm,
                            seed,
                            destination,
                        ),
                        args.dry_run,
                    )


def average_predictions(paths: Iterable[Path], parameters: Sequence[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path, dtype={"condition_id": str, "group_id": str}) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    columns = ["condition_id", "group_id"]
    numeric = ["true_{}".format(p) for p in parameters] + ["pred_{}".format(p) for p in parameters]
    return combined[columns + numeric].groupby(columns, as_index=False, sort=True).mean()


def add_errors(
    frame: pd.DataFrame, parameters: Sequence[str], scales: Dict[str, float]
) -> pd.DataFrame:
    result = frame.copy()
    for parameter in parameters:
        error = (result["pred_{}".format(parameter)] - result["true_{}".format(parameter)]).abs()
        result["ae_{}".format(parameter)] = error
        result["nmae_{}".format(parameter)] = error / max(scales[parameter], 1e-6)
        result["ape_{}".format(parameter)] = (
            100.0 * error / result["true_{}".format(parameter)].abs().clip(lower=1e-6)
        )
    return result


def group_equal(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("group_id", sort=False)[column].mean().mean())


def cluster_ci(
    frame: pd.DataFrame, column: str, seed: int, iterations: int
) -> Dict[str, float]:
    values = frame.groupby("group_id", sort=False)[column].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "groups": int(len(values)),
        "bootstrap_iterations": iterations,
    }


def stage_aggregate(args: argparse.Namespace) -> None:
    frozen = verify_frozen()
    all_frames = []
    run_rows = []
    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        for fold in FOLDS:
            selection = frozen["selection_by_dataset_fold"][dataset_name][str(fold)]
            backbone = str(selection["backbone"])
            algorithm = str(selection["algorithm"])
            role_paths: Dict[str, List[Path]] = {
                "shared_baseline": [
                    OUTPUT
                    / "confirm"
                    / dataset_name
                    / "fold_{}".format(fold)
                    / "baseline"
                    / "shared_baseline"
                    / "seed{}".format(seed)
                    / "test_predictions_conditions.csv"
                    for seed in SEEDS
                ],
                "weighted_control": [
                    OUTPUT
                    / "confirm"
                    / dataset_name
                    / "fold_{}".format(fold)
                    / "weighted_control"
                    / "weighted_shared"
                    / "seed{}".format(seed)
                    / "test_predictions_conditions.csv"
                    for seed in SEEDS
                ],
            }
            if algorithm == "shared_baseline":
                role_paths["frozen_candidate"] = role_paths["shared_baseline"]
            elif algorithm == "weighted_shared":
                role_paths["frozen_candidate"] = role_paths["weighted_control"]
            else:
                role_paths["frozen_candidate"] = [
                    OUTPUT
                    / "confirm"
                    / dataset_name
                    / "fold_{}".format(fold)
                    / "candidate"
                    / algorithm
                    / "seed{}".format(seed)
                    / "test_predictions_conditions.csv"
                    for seed in SEEDS
                ]
            baseline_summary_path = role_paths["shared_baseline"][0].parent / "test_metrics.json"
            baseline_summary = read_json(baseline_summary_path)
            scales = {
                parameter: float(baseline_summary["scaler"][parameter]["maximum"])
                - float(baseline_summary["scaler"][parameter]["minimum"])
                for parameter in spec.parameters
            }
            profile = observability_profile(spec, fold, args.relative_score_threshold)
            for role, paths in role_paths.items():
                missing = [path for path in paths if not path.is_file()]
                if missing:
                    raise FileNotFoundError(missing[0])
                frame = add_errors(average_predictions(paths, spec.parameters), spec.parameters, scales)
                frame["dataset"] = dataset_name
                frame["fold"] = fold
                frame["backbone"] = backbone
                frame["frozen_algorithm"] = algorithm
                frame["high_targets"] = json.dumps(profile["high_targets"])
                frame["observability_scores"] = json.dumps(profile["scores"], sort_keys=True)
                frame["model_role"] = role
                frame["high_nmae"] = selected_error(frame, profile, spec.parameters)
                all_frames.append(frame)
            actual_roles = [("baseline", "shared_baseline"), ("weighted_control", "weighted_shared")]
            if algorithm not in ("shared_baseline", "weighted_shared"):
                actual_roles.append(("candidate", algorithm))
            for role, actual_algorithm in actual_roles:
                for seed in SEEDS:
                    summary_path = (
                        OUTPUT
                        / "confirm"
                        / dataset_name
                        / "fold_{}".format(fold)
                        / role
                        / actual_algorithm
                        / "seed{}".format(seed)
                        / "test_metrics.json"
                    )
                    payload = read_json(summary_path)
                    if payload.get("outer_test_was_supplied") is not True:
                        raise RuntimeError("Confirm run lacks outer test")
                    run_rows.append(
                        {
                            "dataset": dataset_name,
                            "fold": fold,
                            "role": role,
                            "algorithm": actual_algorithm,
                            "backbone": backbone,
                            "seed": seed,
                            "high_targets": json.dumps(profile["high_targets"]),
                            "parameter_count": int(payload["parameter_count"]),
                        }
                    )
    oof = pd.concat(all_frames, ignore_index=True)
    oof.to_csv(OUTPUT / "oof_condition_predictions.csv", index=False)
    pd.DataFrame(run_rows).to_csv(OUTPUT / "run_audit.csv", index=False)

    model_rows = []
    comparisons = []
    guardrail_rows = []
    dataset_results = {}
    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        data = oof[oof["dataset"].eq(dataset_name)].copy()
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
                raise RuntimeError("Incomplete OOF coverage for {} {}".format(dataset_name, role))
            row: Dict[str, object] = {
                "dataset": dataset_name,
                "model_role": role,
                "conditions": expected,
                "groups": int(frame["group_id"].nunique()),
                "high_nmae": group_equal(frame, "high_nmae"),
            }
            nmaes = []
            for parameter in spec.parameters:
                value = group_equal(frame, "nmae_{}".format(parameter))
                row["nmae_{}".format(parameter)] = value
                row["mape_{}".format(parameter)] = group_equal(
                    frame, "ape_{}".format(parameter)
                )
                nmaes.append(value)
            row["mean_nmae"] = float(np.mean(nmaes))
            model_rows.append(row)
        candidate = roles["frozen_candidate"]
        keys = ["fold", "condition_id", "group_id", "high_targets"]
        for baseline_role in ("shared_baseline", "weighted_control"):
            baseline = roles[baseline_role]
            if not candidate[keys].equals(baseline[keys]):
                raise RuntimeError("Prediction keys differ")
            comparison = candidate[keys].copy()
            comparison["gain_high_nmae_pp"] = 100.0 * (
                baseline["high_nmae"].to_numpy(float)
                - candidate["high_nmae"].to_numpy(float)
            )
            statistics = cluster_ci(
                comparison,
                "gain_high_nmae_pp",
                args.bootstrap_seed,
                args.bootstrap_iterations,
            )
            comparisons.append(
                {
                    "dataset": dataset_name,
                    "comparison": "frozen_candidate_vs_{}".format(baseline_role),
                    **statistics,
                    "conditions": int(len(comparison)),
                    "improved_conditions": int((comparison["gain_high_nmae_pp"] > 0).sum()),
                    "worse_conditions": int((comparison["gain_high_nmae_pp"] < 0).sum()),
                }
            )
        shared = roles["shared_baseline"]
        candidate_summary = next(
            row
            for row in model_rows
            if row["dataset"] == dataset_name and row["model_role"] == "frozen_candidate"
        )
        shared_summary = next(
            row
            for row in model_rows
            if row["dataset"] == dataset_name and row["model_role"] == "shared_baseline"
        )
        overall_degradation = 100.0 * (
            float(candidate_summary["mean_nmae"]) - float(shared_summary["mean_nmae"])
        )
        low_degradations = []
        for parameter in spec.parameters:
            low_mask = candidate["high_targets"].map(
                lambda value: parameter not in json.loads(value)
            )
            if low_mask.any():
                candidate_low = group_equal(
                    candidate.loc[low_mask], "nmae_{}".format(parameter)
                )
                shared_low = group_equal(shared.loc[low_mask], "nmae_{}".format(parameter))
                degradation = 100.0 * (candidate_low - shared_low)
                low_degradations.append(degradation)
                guardrail_rows.append(
                    {
                        "dataset": dataset_name,
                        "guardrail": "low_observability_{}".format(parameter),
                        "candidate_nmae": candidate_low,
                        "shared_nmae": shared_low,
                        "degradation_pp": degradation,
                    }
                )
        guardrail_rows.append(
            {
                "dataset": dataset_name,
                "guardrail": "overall_mean",
                "candidate_nmae": float(candidate_summary["mean_nmae"]),
                "shared_nmae": float(shared_summary["mean_nmae"]),
                "degradation_pp": overall_degradation,
            }
        )
        current = [row for row in comparisons if row["dataset"] == dataset_name]
        versus_shared = next(row for row in current if row["comparison"].endswith("shared_baseline"))
        versus_weighted = next(row for row in current if row["comparison"].endswith("weighted_control"))
        selections = frozen["selection_by_dataset_fold"][dataset_name]
        specialist_deployed = any(
            selection["algorithm"] in ("multi_specialist", "multi_specialist_pcgrad")
            for selection in selections.values()
        )
        supported = bool(
            versus_shared["ci_low"] > 0.0
            and (not specialist_deployed or versus_weighted["ci_low"] > 0.0)
            and overall_degradation <= 0.5
            and (not low_degradations or max(low_degradations) <= 1.0)
        )
        dataset_results[dataset_name] = {
            "dataset_local_training_validation_testing": True,
            "selection_by_fold": selections,
            "candidate_vs_shared": versus_shared,
            "candidate_vs_weighted_control": versus_weighted,
            "specialist_deployed_in_any_fold": specialist_deployed,
            "overall_mean_nmae_degradation_pp": overall_degradation,
            "maximum_low_observability_degradation_pp": (
                max(low_degradations) if low_degradations else 0.0
            ),
            "development_support_rule_passed": supported,
        }
    pd.DataFrame(model_rows).to_csv(OUTPUT / "model_oof_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUTPUT / "paired_group_comparisons.csv", index=False)
    pd.DataFrame(guardrail_rows).to_csv(OUTPUT / "guardrail_summary.csv", index=False)
    write_json(
        OUTPUT / "development_summary.json",
        {
            "scope": "exploratory development on previously inspected datasets",
            "not_external_confirmation": True,
            "dataset_training_isolation": True,
            "multi_target_selection": True,
            "datasets": dataset_results,
            "cross_dataset_development_rule_passed": all(
                result["development_support_rule_passed"]
                for result in dataset_results.values()
            ),
            "future_confirmation_requirement": (
                "freeze exact protocol and evaluate once on a new untouched dataset"
            ),
        },
    )
    print(json.dumps(read_json(OUTPUT / "development_summary.json"), indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "pilot_backbone",
            "select_backbone",
            "pilot_algorithm",
            "freeze",
            "confirm",
            "aggregate",
        ),
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    parser.add_argument("--relative-score-threshold", type=float, default=0.80)
    parser.add_argument("--pilot-epochs", type=int, default=30)
    parser.add_argument("--confirm-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--set-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--selected-priority", type=float, default=2.0)
    parser.add_argument("--pilot-bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.relative_score_threshold <= 1.0:
        raise ValueError("relative score threshold must be in (0, 1]")
    stages = {
        "preflight": stage_preflight,
        "pilot_backbone": stage_pilot_backbone,
        "select_backbone": stage_select_backbone,
        "pilot_algorithm": stage_pilot_algorithm,
        "freeze": stage_freeze,
        "confirm": stage_confirm,
        "aggregate": stage_aggregate,
    }
    stages[args.stage](args)


if __name__ == "__main__":
    main()
