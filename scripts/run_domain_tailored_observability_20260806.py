#!/usr/bin/env python3
"""Orchestrate leakage-controlled, dataset-tailored external replications.

The common scientific workflow is fixed across DED and NIST:

1. read outer-training-only observability evidence;
2. select one or more high-observability targets by a fixed relative rule;
3. compare a shallow CNN, a domain-tailored shared model, and targeted variants
   using only inner validation;
4. freeze the fold-local choice and source hashes;
5. evaluate on the held-out outer fold and aggregate group-paired OOF results.

The actual representation is allowed to follow the image-formation process:
ordered melt-pool dynamics for DED and unordered cross-sectional geometry for
NIST.  The datasets never share samples, scalers, checkpoints or metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import run_multitarget_observability_20260806 as common  # noqa: E402
from scripts import train_domain_tailored_observability_20260806 as trainer  # noqa: E402


TRAINER = PROJECT / "scripts" / "train_domain_tailored_observability_20260806.py"
OUTPUT = PROJECT / "outputs" / "domain_tailored_observability_20260806"
ARCHIVE = PROJECT / "domain_tailored_observability_20260806_no_weights.tar.gz"
DATASETS = common.DATASETS
FOLDS = list(range(5))
SEEDS = [42, 52, 62]
PILOT_SEED = 42
DOMAIN_BACKBONES = tuple(trainer.BACKBONES)
ALGORITHM_PILOT = ("weighted_shared", "guided_rank", "guided_ordinal_rank")
GUIDED_CANDIDATES = ("guided_rank", "guided_ordinal_rank")


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


def profile(args: argparse.Namespace, dataset: str, fold: int) -> Dict[str, object]:
    return common.observability_profile(
        DATASETS[dataset], fold, args.relative_score_threshold
    )


def pilot_root(
    dataset: str,
    fold: int,
    architecture: str,
    backbone: str,
    algorithm: str,
) -> Path:
    return (
        OUTPUT
        / "pilot"
        / dataset
        / "fold_{}".format(fold)
        / architecture
        / backbone
        / algorithm
        / "seed{}".format(PILOT_SEED)
    )


def backbone_selection_path(dataset: str, fold: int) -> Path:
    return OUTPUT / "pilot" / dataset / "fold_{}".format(fold) / "SELECTED_BACKBONE.json"


def train_command(
    args: argparse.Namespace,
    phase: str,
    dataset: str,
    fold: int,
    architecture: str,
    backbone: str,
    algorithm: str,
    seed: int,
    destination: Path,
    specialist_targets: Sequence[str] = (),
) -> List[str]:
    spec = DATASETS[dataset]
    current = profile(args, dataset, fold)
    command = [
        sys.executable,
        str(TRAINER),
        "--phase",
        phase,
        "--dataset-id",
        dataset,
        "--architecture",
        architecture,
        "--backbone",
        backbone,
        "--algorithm",
        algorithm,
        "--train-csv",
        str(spec.split_path(fold, "train")),
        "--val-csv",
        str(spec.split_path(fold, "val")),
        "--parameters",
        *spec.parameters,
        "--high-targets",
        *current["high_targets"],
        "--observability-scores",
        *[str(current["scores"][parameter]) for parameter in spec.parameters],
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
        "--selected-priority",
        str(args.selected_priority),
        "--rank-weight",
        str(args.rank_weight),
        "--ordinal-weight",
        str(args.ordinal_weight),
    ]
    if specialist_targets:
        command.extend(["--specialist-targets", *specialist_targets])
    if phase == "confirm":
        command.extend(["--test-csv", str(spec.split_path(fold, "test"))])
    return command


def stage_preflight(args: argparse.Namespace) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dependencies = [
        TRAINER,
        Path(__file__).resolve(),
        PROJECT / "scripts" / "train_condition_set_specialist_20260802.py",
        PROJECT / "scripts" / "train_nist_set_aggregation_upgrade_20260803.py",
        PROJECT / "scripts" / "train_crossdomain_observability_optimizer_20260805.py",
        PROJECT / "scripts" / "train_multitarget_observability_20260806.py",
        PROJECT / "scripts" / "run_multitarget_observability_20260806.py",
    ]
    missing_dependencies = [path for path in dependencies if not path.is_file()]
    if missing_dependencies:
        raise FileNotFoundError(missing_dependencies[0])
    rows: List[Dict[str, object]] = []
    profiles = []
    cardinality = []
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        for fold in FOLDS:
            frames: Dict[str, pd.DataFrame] = {}
            for split in ("train", "val", "test"):
                path = spec.split_path(fold, split)
                if not path.is_file():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(
                    path, dtype={"condition_id": str, spec.group_column: str}
                )
                required = {
                    "condition_id",
                    spec.group_column,
                    spec.sample_column,
                    "image_path",
                    *spec.parameters,
                }
                absent = sorted(required - set(frame.columns))
                if absent:
                    raise ValueError("{} is missing {}".format(path, absent))
                frames[split] = frame
            overlap_counts = []
            for column in (spec.sample_column, "condition_id", spec.group_column):
                for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                    overlap_counts.append(
                        len(
                            set(frames[left][column].astype(str))
                            & set(frames[right][column].astype(str))
                        )
                    )
            if any(overlap_counts):
                raise RuntimeError("{} fold{} split overlap".format(dataset, fold))
            missing_images = 0
            for frame in frames.values():
                for value in frame["image_path"].astype(str).drop_duplicates():
                    path = Path(value)
                    if not path.is_absolute():
                        path = PROJECT / path
                    if not path.is_file():
                        missing_images += 1
            if missing_images:
                raise FileNotFoundError(
                    "{} fold{} has {} missing images".format(dataset, fold, missing_images)
                )
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "train_conditions": frames["train"]["condition_id"].nunique(),
                    "val_conditions": frames["val"]["condition_id"].nunique(),
                    "test_conditions": frames["test"]["condition_id"].nunique(),
                    "train_groups": frames["train"][spec.group_column].nunique(),
                    "val_groups": frames["val"][spec.group_column].nunique(),
                    "test_groups": frames["test"][spec.group_column].nunique(),
                    "zero_overlap": True,
                    "missing_images": 0,
                }
            )
            current = profile(args, dataset, fold)
            profiles.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "scores": current["scores"],
                    "high_targets": current["high_targets"],
                    "winner_fraction": current["winner_fraction"],
                    "selection_scope": current["selection_scope"],
                }
            )
            conditions = pd.concat(frames.values(), ignore_index=True).drop_duplicates(
                "condition_id"
            )
            for parameter in spec.parameters:
                cardinality.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "parameter": parameter,
                        "training_unique_values": frames["train"][parameter].nunique(),
                        "all_unique_values": conditions[parameter].nunique(),
                    }
                )
    pd.DataFrame(rows).to_csv(OUTPUT / "preflight_split_audit.csv", index=False)
    pd.DataFrame(cardinality).to_csv(OUTPUT / "target_cardinality_audit.csv", index=False)
    write_json(
        OUTPUT / "preflight.json",
        {
            "status": "PASS",
            "dataset_isolation": True,
            "outer_test_used_for_observability_or_model_selection": False,
            "common_workflow": (
                "outer-training observability -> fixed high-target rule -> inner-validation "
                "model choice -> frozen outer-test evaluation"
            ),
            "ded_model": (
                "centered melt-pool ROI, intensity/shape descriptors, ordered "
                "bidirectional GRU and temporal-difference aggregation"
            ),
            "nist_model": (
                "cross-sectional intensity/shape descriptors and masked gated "
                "set-moment aggregation"
            ),
            "baseline": "four-block shallow CNN with masked mean pooling",
            "profiles": profiles,
        },
    )
    print(json.dumps(read_json(OUTPUT / "preflight.json"), indent=2), flush=True)


def stage_pilot(args: argparse.Namespace) -> None:
    for dataset in args.datasets:
        for fold in args.folds:
            current = profile(args, dataset, fold)
            # First compare the lower-capacity CNN with three candidate visual
            # backbones while keeping the shared prediction head fixed.
            execute(
                train_command(
                    args,
                    "pilot",
                    dataset,
                    fold,
                    "shallow_cnn",
                    "shallow",
                    "shared_baseline",
                    PILOT_SEED,
                    pilot_root(
                        dataset,
                        fold,
                        "shallow_cnn",
                        "shallow",
                        "shared_baseline",
                    ),
                ),
                args.dry_run,
            )
            backbone_rows = []
            for backbone in DOMAIN_BACKBONES:
                root = pilot_root(
                    dataset,
                    fold,
                    "domain_tailored",
                    backbone,
                    "shared_baseline",
                )
                execute(
                    train_command(
                        args,
                        "pilot",
                        dataset,
                        fold,
                        "domain_tailored",
                        backbone,
                        "shared_baseline",
                        PILOT_SEED,
                        root,
                    ),
                    args.dry_run,
                )
                if not args.dry_run:
                    summary = read_json(root / "validation_summary.json")
                    backbone_rows.append(
                        {
                            "backbone": backbone,
                            "validation_score": float(summary["best_validation_score"]),
                            "validation_high_nmae": float(
                                summary["validation_high_observability_nmae"]
                            ),
                            "validation_mean_nmae": float(
                                summary["validation_group_equal"]["mean_nmae"]
                            ),
                            "parameter_count": int(summary["parameter_count"]),
                        }
                    )
            if args.dry_run:
                continue
            winner = min(
                backbone_rows,
                key=lambda value: (value["validation_score"], value["parameter_count"]),
            )
            write_json(
                backbone_selection_path(dataset, fold),
                {
                    "selection_scope": "fold-local inner validation only",
                    "outer_test_used": False,
                    "selected_backbone": winner["backbone"],
                    "candidates": backbone_rows,
                },
            )
            selected_backbone = str(winner["backbone"])
            for algorithm in ALGORITHM_PILOT:
                specialists = (
                    current["high_targets"]
                    if algorithm in trainer.GUIDED_ALGORITHMS
                    else []
                )
                execute(
                    train_command(
                        args,
                        "pilot",
                        dataset,
                        fold,
                        "domain_tailored",
                        selected_backbone,
                        algorithm,
                        PILOT_SEED,
                        pilot_root(
                            dataset,
                            fold,
                            "domain_tailored",
                            selected_backbone,
                            algorithm,
                        ),
                        specialists,
                    ),
                    args.dry_run,
                )


def validation_comparison(
    args: argparse.Namespace,
    dataset: str,
    fold: int,
    baseline: Path,
    candidate: Path,
) -> Dict[str, float]:
    return common.validation_gain_ci(
        baseline,
        candidate,
        profile(args, dataset, fold),
        DATASETS[dataset].parameters,
        seed=args.bootstrap_seed + fold,
        iterations=args.pilot_bootstrap_iterations,
    )


def validation_guardrails(
    dataset: str,
    high_targets: Sequence[str],
    baseline_summary: Dict[str, object],
    candidate_summary: Dict[str, object],
) -> Tuple[float, float]:
    parameters = DATASETS[dataset].parameters
    baseline = baseline_summary["validation_group_equal"]
    candidate = candidate_summary["validation_group_equal"]
    overall = 100.0 * (float(candidate["mean_nmae"]) - float(baseline["mean_nmae"]))
    low = [
        100.0
        * (
            float(candidate["nmae_{}".format(parameter)])
            - float(baseline["nmae_{}".format(parameter)])
        )
        for parameter in parameters
        if parameter not in high_targets
    ]
    return overall, max(low) if low else 0.0


def freeze_inputs() -> List[Path]:
    paths = [
        TRAINER,
        Path(__file__).resolve(),
        PROJECT / "scripts" / "train_condition_set_specialist_20260802.py",
        PROJECT / "scripts" / "train_nist_set_aggregation_upgrade_20260803.py",
        PROJECT / "scripts" / "train_crossdomain_observability_optimizer_20260805.py",
        PROJECT / "scripts" / "train_multitarget_observability_20260806.py",
        PROJECT / "scripts" / "run_multitarget_observability_20260806.py",
        OUTPUT / "preflight.json",
        OUTPUT / "preflight_split_audit.csv",
        OUTPUT / "target_cardinality_audit.csv",
        OUTPUT / "pilot_inner_validation.csv",
    ]
    for dataset, spec in DATASETS.items():
        for fold in FOLDS:
            paths.extend(spec.split_path(fold, split) for split in ("train", "val", "test"))
            paths.extend([spec.selection_path(fold), spec.bootstrap_path(fold)])
            paths.append(backbone_selection_path(dataset, fold))
            roots = [
                pilot_root(
                    dataset, fold, "shallow_cnn", "shallow", "shared_baseline"
                )
            ]
            roots.extend(
                pilot_root(
                    dataset,
                    fold,
                    "domain_tailored",
                    backbone,
                    "shared_baseline",
                )
                for backbone in DOMAIN_BACKBONES
            )
            selected_backbone = str(
                read_json(backbone_selection_path(dataset, fold))["selected_backbone"]
            )
            roots.extend(
                pilot_root(
                    dataset,
                    fold,
                    "domain_tailored",
                    selected_backbone,
                    algorithm,
                )
                for algorithm in ALGORITHM_PILOT
            )
            for root in roots:
                paths.extend(
                    [
                        root / "validation_summary.json",
                        root / "validation_predictions_conditions.csv",
                    ]
                )
    return paths


def stage_freeze(args: argparse.Namespace) -> None:
    selections: Dict[str, Dict[str, Dict[str, object]]] = {}
    rows = []
    for dataset in args.datasets:
        selections[dataset] = {}
        for fold in FOLDS:
            current = profile(args, dataset, fold)
            selected_backbone_payload = read_json(backbone_selection_path(dataset, fold))
            selected_backbone = str(selected_backbone_payload["selected_backbone"])
            simple_root = pilot_root(
                dataset, fold, "shallow_cnn", "shallow", "shared_baseline"
            )
            strong_root = pilot_root(
                dataset,
                fold,
                "domain_tailored",
                selected_backbone,
                "shared_baseline",
            )
            simple_summary = read_json(simple_root / "validation_summary.json")
            strong_summary = read_json(strong_root / "validation_summary.json")
            architecture_gain = validation_comparison(
                args,
                dataset,
                fold,
                simple_root / "validation_predictions_conditions.csv",
                strong_root / "validation_predictions_conditions.csv",
            )
            candidates = []
            for algorithm in GUIDED_CANDIDATES:
                root = pilot_root(
                    dataset,
                    fold,
                    "domain_tailored",
                    selected_backbone,
                    algorithm,
                )
                summary = read_json(root / "validation_summary.json")
                comparison = validation_comparison(
                    args,
                    dataset,
                    fold,
                    strong_root / "validation_predictions_conditions.csv",
                    root / "validation_predictions_conditions.csv",
                )
                overall, worst_low = validation_guardrails(
                    dataset, current["high_targets"], strong_summary, summary
                )
                eligible = bool(
                    comparison["mean_gain_pp"] > 0.0
                    and overall <= args.validation_overall_guardrail_pp
                    and worst_low <= args.validation_parameter_guardrail_pp
                )
                candidate = {
                    "algorithm": algorithm,
                    "validation_score": float(summary["best_validation_score"]),
                    "validation_gain_pp": comparison["mean_gain_pp"],
                    "validation_gain_ci_low": comparison["ci_low"],
                    "validation_gain_ci_high": comparison["ci_high"],
                    "overall_degradation_pp": overall,
                    "worst_low_target_degradation_pp": worst_low,
                    "eligible": eligible,
                }
                candidates.append(candidate)
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "architecture": "domain_tailored",
                        "backbone": selected_backbone,
                        "high_targets": json.dumps(current["high_targets"]),
                        **candidate,
                    }
                )
            eligible = [candidate for candidate in candidates if candidate["eligible"]]
            if eligible:
                winner = min(
                    eligible,
                    key=lambda value: (value["validation_score"], -value["validation_gain_pp"]),
                )
                frozen_algorithm = str(winner["algorithm"])
                reason = "best eligible guided candidate on inner validation"
            else:
                frozen_algorithm = "shared_baseline"
                reason = "no guided candidate improved high targets within validation guardrails"
            selections[dataset][str(fold)] = {
                "high_targets": current["high_targets"],
                "observability_scores": current["scores"],
                "architecture": "domain_tailored",
                "backbone": selected_backbone,
                "algorithm": frozen_algorithm,
                "reason": reason,
                "backbone_candidates": selected_backbone_payload["candidates"],
                "domain_shared_vs_shallow_validation": architecture_gain,
                "guided_candidates": candidates,
            }
    pd.DataFrame(rows).to_csv(OUTPUT / "pilot_inner_validation.csv", index=False)
    required = freeze_inputs()
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing freeze input {}".format(missing[0]))
    payload = {
        "scope": "retrospective cross-domain validation on previously inspected datasets",
        "not_untouched_external_confirmation": True,
        "dataset_isolation": True,
        "outer_test_used_in_selection": False,
        "common_high_target_rule": (
            "select every parameter with observability score >= {:.2f} * fold maximum"
        ).format(args.relative_score_threshold),
        "selection_by_dataset_fold": selections,
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
    return payload


def alternative_placements(
    parameters: Sequence[str], high_targets: Sequence[str]
) -> List[List[str]]:
    low_targets = [parameter for parameter in parameters if parameter not in high_targets]
    if not low_targets or len(high_targets) > len(low_targets):
        return []
    placements = []
    for offset in range(len(low_targets)):
        placement = [
            low_targets[(offset + index) % len(low_targets)]
            for index in range(len(high_targets))
        ]
        if len(set(placement)) == len(placement):
            placements.append(placement)
    unique = []
    for placement in placements:
        if placement not in unique:
            unique.append(placement)
    return unique


def stage_confirm(args: argparse.Namespace) -> None:
    frozen = verify_frozen()
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        for fold in args.folds:
            selection = frozen["selection_by_dataset_fold"][dataset][str(fold)]
            algorithm = str(selection["algorithm"])
            selected_backbone = str(selection["backbone"])
            high_targets = list(selection["high_targets"])
            jobs: List[Tuple[str, str, str, str, Sequence[str]]] = [
                ("shallow_cnn", "shallow_cnn", "shallow", "shared_baseline", []),
                (
                    "domain_shared",
                    "domain_tailored",
                    selected_backbone,
                    "shared_baseline",
                    [],
                ),
                (
                    "weighted_control",
                    "domain_tailored",
                    selected_backbone,
                    "weighted_shared",
                    [],
                ),
            ]
            if algorithm in trainer.GUIDED_ALGORITHMS:
                jobs.append(
                    (
                        "frozen_candidate",
                        "domain_tailored",
                        selected_backbone,
                        algorithm,
                        high_targets,
                    )
                )
                for index, placement in enumerate(
                    alternative_placements(spec.parameters, high_targets)
                ):
                    jobs.append(
                        (
                            "placement_{:02d}_{}".format(index, "_".join(placement)),
                            "domain_tailored",
                            selected_backbone,
                            algorithm,
                            placement,
                        )
                    )
            for role, architecture, backbone, current_algorithm, specialists in jobs:
                for seed in SEEDS:
                    destination = (
                        OUTPUT
                        / "confirm"
                        / dataset
                        / "fold_{}".format(fold)
                        / role
                        / "seed{}".format(seed)
                    )
                    execute(
                        train_command(
                            args,
                            "confirm",
                            dataset,
                            fold,
                            architecture,
                            backbone,
                            current_algorithm,
                            seed,
                            destination,
                            specialists,
                        ),
                        args.dry_run,
                    )


def prediction_paths(dataset: str, fold: int, role: str) -> List[Path]:
    return [
        OUTPUT
        / "confirm"
        / dataset
        / "fold_{}".format(fold)
        / role
        / "seed{}".format(seed)
        / "test_predictions_conditions.csv"
        for seed in SEEDS
    ]


def placement_paths(dataset: str, fold: int) -> List[Path]:
    root = OUTPUT / "confirm" / dataset / "fold_{}".format(fold)
    paths = []
    for directory in sorted(root.glob("placement_*")):
        paths.extend(
            directory / "seed{}".format(seed) / "test_predictions_conditions.csv"
            for seed in SEEDS
        )
    return paths


def stage_aggregate(args: argparse.Namespace) -> None:
    frozen = verify_frozen()
    frames = []
    run_rows = []
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        for fold in FOLDS:
            selection = frozen["selection_by_dataset_fold"][dataset][str(fold)]
            algorithm = str(selection["algorithm"])
            current = profile(args, dataset, fold)
            roles: Dict[str, List[Path]] = {
                "shallow_cnn": prediction_paths(dataset, fold, "shallow_cnn"),
                "domain_shared": prediction_paths(dataset, fold, "domain_shared"),
                "weighted_control": prediction_paths(dataset, fold, "weighted_control"),
            }
            roles["frozen_candidate"] = (
                prediction_paths(dataset, fold, "frozen_candidate")
                if algorithm in trainer.GUIDED_ALGORITHMS
                else roles["domain_shared"]
            )
            placements = placement_paths(dataset, fold)
            # A fold that falls back to the shared model contributes the same
            # predictions to candidate and placement-control roles.  This keeps
            # OOF coverage complete without inventing a targeted comparison in
            # a fold where inner validation rejected targeted refinement.
            roles["nonselected_placement_mean"] = (
                placements if placements else roles["domain_shared"]
            )
            summary_path = roles["domain_shared"][0].parent / "test_metrics.json"
            summary = read_json(summary_path)
            scales = {
                parameter: float(summary["scaler"][parameter]["maximum"])
                - float(summary["scaler"][parameter]["minimum"])
                for parameter in spec.parameters
            }
            for role, paths in roles.items():
                missing = [path for path in paths if not path.is_file()]
                if missing:
                    raise FileNotFoundError(missing[0])
                frame = common.add_errors(
                    common.average_predictions(paths, spec.parameters), spec.parameters, scales
                )
                frame["dataset"] = dataset
                frame["fold"] = fold
                frame["model_role"] = role
                frame["frozen_algorithm"] = algorithm
                frame["high_targets"] = json.dumps(current["high_targets"])
                frame["high_nmae"] = common.selected_error(
                    frame, current, spec.parameters
                )
                frames.append(frame)
            for role in ("shallow_cnn", "domain_shared", "weighted_control"):
                for seed in SEEDS:
                    payload = read_json(
                        OUTPUT
                        / "confirm"
                        / dataset
                        / "fold_{}".format(fold)
                        / role
                        / "seed{}".format(seed)
                        / "test_metrics.json"
                    )
                    run_rows.append(
                        {
                            "dataset": dataset,
                            "fold": fold,
                            "role": role,
                            "seed": seed,
                            "architecture": payload["architecture"],
                            "backbone": payload["backbone"],
                            "algorithm": payload["algorithm"],
                            "specialist_targets": json.dumps(payload["specialist_targets"]),
                            "parameter_count": payload["parameter_count"],
                        }
                    )
            if algorithm in trainer.GUIDED_ALGORITHMS:
                for role_dir in ["frozen_candidate"] + [
                    path.name
                    for path in sorted(
                        (OUTPUT / "confirm" / dataset / "fold_{}".format(fold)).glob(
                            "placement_*"
                        )
                    )
                ]:
                    for seed in SEEDS:
                        payload = read_json(
                            OUTPUT
                            / "confirm"
                            / dataset
                            / "fold_{}".format(fold)
                            / role_dir
                            / "seed{}".format(seed)
                            / "test_metrics.json"
                        )
                        run_rows.append(
                            {
                                "dataset": dataset,
                                "fold": fold,
                                "role": role_dir,
                                "seed": seed,
                                "architecture": payload["architecture"],
                                "backbone": payload["backbone"],
                                "algorithm": payload["algorithm"],
                                "specialist_targets": json.dumps(
                                    payload["specialist_targets"]
                                ),
                                "parameter_count": payload["parameter_count"],
                            }
                        )
    oof = pd.concat(frames, ignore_index=True)
    oof.to_csv(OUTPUT / "oof_condition_predictions.csv", index=False)
    pd.DataFrame(run_rows).to_csv(OUTPUT / "run_audit.csv", index=False)

    model_rows = []
    comparisons = []
    guardrails = []
    dataset_results = {}
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        data = oof[oof["dataset"].eq(dataset)].copy()
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
            row: Dict[str, object] = {
                "dataset": dataset,
                "model_role": role,
                "conditions": expected,
                "groups": frame["group_id"].nunique(),
                "high_nmae": common.group_equal(frame, "high_nmae"),
            }
            nmaes = []
            for parameter in spec.parameters:
                nmae = common.group_equal(frame, "nmae_{}".format(parameter))
                row["nmae_{}".format(parameter)] = nmae
                row["mape_{}".format(parameter)] = common.group_equal(
                    frame, "ape_{}".format(parameter)
                )
                nmaes.append(nmae)
            row["mean_nmae"] = float(np.mean(nmaes))
            model_rows.append(row)

        candidate = roles["frozen_candidate"]
        keys = ["fold", "condition_id", "group_id", "high_targets"]
        baseline_roles = ["shallow_cnn", "domain_shared", "weighted_control"]
        if "nonselected_placement_mean" in roles:
            baseline_roles.append("nonselected_placement_mean")
        for baseline_role in baseline_roles:
            baseline = roles[baseline_role]
            if not candidate[keys].equals(baseline[keys]):
                raise RuntimeError("Prediction keys differ for {}".format(baseline_role))
            comparison = candidate[keys].copy()
            comparison["gain_high_nmae_pp"] = 100.0 * (
                baseline["high_nmae"].to_numpy(float)
                - candidate["high_nmae"].to_numpy(float)
            )
            statistics = common.cluster_ci(
                comparison,
                "gain_high_nmae_pp",
                args.bootstrap_seed,
                args.bootstrap_iterations,
            )
            comparisons.append(
                {
                    "dataset": dataset,
                    "comparison": "frozen_candidate_vs_{}".format(baseline_role),
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
            row
            for row in model_rows
            if row["dataset"] == dataset and row["model_role"] == "frozen_candidate"
        )
        shared_summary = next(
            row
            for row in model_rows
            if row["dataset"] == dataset and row["model_role"] == "domain_shared"
        )
        simple_summary = next(
            row
            for row in model_rows
            if row["dataset"] == dataset and row["model_role"] == "shallow_cnn"
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
                candidate_value = common.group_equal(
                    candidate.loc[low_mask], "nmae_{}".format(parameter)
                )
                shared_value = common.group_equal(
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
        current_comparisons = [row for row in comparisons if row["dataset"] == dataset]
        versus_simple = next(
            row for row in current_comparisons if row["comparison"].endswith("shallow_cnn")
        )
        versus_shared = next(
            row for row in current_comparisons if row["comparison"].endswith("domain_shared")
        )
        versus_placement = next(
            (
                row
                for row in current_comparisons
                if row["comparison"].endswith("nonselected_placement_mean")
            ),
            None,
        )
        guided_deployed = any(
            selection["algorithm"] in trainer.GUIDED_ALGORITHMS
            for selection in frozen["selection_by_dataset_fold"][dataset].values()
        )
        supported = bool(
            guided_deployed
            and versus_simple["ci_low"] > 0.0
            and versus_shared["ci_low"] > 0.0
            and versus_placement is not None
            and versus_placement["ci_low"] > 0.0
            and overall_degradation <= 0.5
            and (not low_degradations or max(low_degradations) <= 1.0)
        )
        dataset_results[dataset] = {
            "high_target_improvement_vs_shallow_cnn": versus_simple,
            "high_target_improvement_vs_domain_shared": versus_shared,
            "high_target_improvement_vs_nonselected_placement": versus_placement,
            "domain_shared_minus_shallow_mean_nmae_pp": 100.0
            * (shared_summary["mean_nmae"] - simple_summary["mean_nmae"]),
            "overall_mean_nmae_degradation_vs_domain_shared_pp": overall_degradation,
            "maximum_low_target_degradation_pp": max(low_degradations)
            if low_degradations
            else 0.0,
            "guided_model_deployed_in_any_fold": guided_deployed,
            "workflow_support_rule_passed": supported,
        }
    pd.DataFrame(model_rows).to_csv(OUTPUT / "model_oof_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUTPUT / "paired_group_comparisons.csv", index=False)
    pd.DataFrame(guardrails).to_csv(OUTPUT / "guardrail_summary.csv", index=False)
    write_json(
        OUTPUT / "development_summary.json",
        {
            "scope": "retrospective cross-domain workflow validation",
            "not_untouched_external_confirmation": True,
            "dataset_local_models": True,
            "common_observability_workflow": True,
            "datasets": dataset_results,
            "both_datasets_support_workflow": all(
                result["workflow_support_rule_passed"]
                for result in dataset_results.values()
            ),
        },
    )
    print(json.dumps(read_json(OUTPUT / "development_summary.json"), indent=2), flush=True)


def stage_package(args: argparse.Namespace) -> None:
    summary = OUTPUT / "development_summary.json"
    if not summary.is_file():
        raise FileNotFoundError("Aggregate before packaging")
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_dir() or path.suffix.lower() in (".pth", ".pt"):
                continue
            archive.add(path, arcname=str(path.relative_to(PROJECT)))
        for path in (
            TRAINER,
            Path(__file__).resolve(),
            PROJECT / "scripts" / "run_5090_domain_tailored_observability_20260806.sh",
            PROJECT / "DOMAIN_TAILORED_OBSERVABILITY_README_20260806.md",
        ):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(PROJECT)))
    print("[ARCHIVE] {}".format(ARCHIVE), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("preflight", "pilot", "freeze", "confirm", "aggregate", "package"),
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS)
    )
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
    parser.add_argument("--rank-weight", type=float, default=0.10)
    parser.add_argument("--ordinal-weight", type=float, default=0.10)
    parser.add_argument("--validation-overall-guardrail-pp", type=float, default=0.50)
    parser.add_argument("--validation-parameter-guardrail-pp", type=float, default=1.00)
    parser.add_argument("--pilot-bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.relative_score_threshold <= 1.0:
        raise ValueError("relative score threshold must be in (0, 1]")
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
