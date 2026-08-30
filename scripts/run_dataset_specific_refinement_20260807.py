#!/usr/bin/env python3
"""Orchestrate dataset-specific DED and NIST refinements on frozen splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import run_multitarget_observability_20260806 as common  # noqa: E402
from scripts import run_observability_gradient_protection_20260806 as previous  # noqa: E402


TRAINER = PROJECT / "scripts" / "train_dataset_specific_refinement_20260807.py"
OUTPUT = PROJECT / "outputs" / "dataset_specific_refinement_20260807"
PRIOR_OUTPUT = PROJECT / "outputs" / "observability_gradient_protection_20260806"
PRIOR_FROZEN = PRIOR_OUTPUT / "FROZEN_PROTOCOL.json"
FROZEN = OUTPUT / "FROZEN_SELECTION.json"
FOLDS = list(range(5))
SEEDS = [42, 52, 62]
PILOT_SEED = 42
DATASETS = common.DATASETS


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute(command: Sequence[str], dry_run: bool) -> None:
    print("[CMD] {}".format(" ".join(str(value) for value in command)), flush=True)
    if not dry_run:
        subprocess.run(list(command), check=True)


def prior_protocol():
    if not PRIOR_FROZEN.is_file():
        raise FileNotFoundError(PRIOR_FROZEN)
    payload = read_json(PRIOR_FROZEN)
    if payload.get("outer_test_used_in_selection") is not False:
        raise RuntimeError("Prior protocol did not isolate outer test")
    return payload


def fold_profile(protocol, dataset: str, fold: int) -> Dict[str, object]:
    return protocol["selection_by_dataset"][dataset]["fold_configuration"][str(fold)]


def run_root(phase: str, dataset: str, fold: int, role: str, seed: int) -> Path:
    return OUTPUT / phase / dataset / "fold_{}".format(fold) / role / "seed{}".format(seed)


def train_command(
    args: argparse.Namespace,
    phase: str,
    dataset: str,
    fold: int,
    role: str,
    seed: int,
    variant: str,
    optimization: str,
    adapter_targets: Sequence[str],
    adapter_rank: int,
    aux_weight: float,
) -> List[str]:
    protocol = prior_protocol()
    profile = fold_profile(protocol, dataset, fold)
    spec = DATASETS[dataset]
    command = [
        sys.executable, str(TRAINER),
        "--phase", phase,
        "--dataset-id", dataset,
        "--variant", variant,
        "--optimization", optimization,
        "--backbone", str(profile["backbone"]),
        "--train-csv", str(spec.split_path(fold, "train")),
        "--val-csv", str(spec.split_path(fold, "val")),
        "--parameters", *spec.parameters,
        "--high-targets", *profile["high_targets"],
        "--observability-scores", *[str(profile["observability_scores"][value]) for value in spec.parameters],
        "--group-column", spec.group_column,
        "--output-dir", str(run_root(phase, dataset, fold, role, seed)),
        "--outer-fold", str(fold),
        "--seed", str(seed),
        "--epochs", str(args.pilot_epochs if phase == "pilot" else args.confirm_epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--set-size", str(args.set_size),
        "--image-size", str(args.image_size),
        "--adapter-rank", str(adapter_rank),
        "--aux-weight", str(aux_weight),
        "--freeze-backbone-epochs", str(args.freeze_backbone_epochs),
    ]
    if adapter_targets:
        command.extend(["--adapter-targets", *adapter_targets])
    if phase == "confirm":
        command.extend(["--test-csv", str(spec.split_path(fold, "test"))])
    return command


def stage_preflight(args: argparse.Namespace) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = prior_protocol()
    checks = []
    for path in (TRAINER, PRIOR_OUTPUT / "oof_condition_predictions.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)
    for dataset, spec in DATASETS.items():
        for fold in FOLDS:
            profile = fold_profile(protocol, dataset, fold)
            for split in ("train", "val", "test"):
                path = spec.split_path(fold, split)
                if not path.is_file():
                    raise FileNotFoundError(path)
            frames = {
                split: pd.read_csv(spec.split_path(fold, split), dtype={"condition_id": str, spec.group_column: str})
                for split in ("train", "val", "test")
            }
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                if set(frames[left]["condition_id"]) & set(frames[right]["condition_id"]):
                    raise RuntimeError("condition overlap {} {} fold{}".format(dataset, left, fold))
                if set(frames[left][spec.group_column]) & set(frames[right][spec.group_column]):
                    raise RuntimeError("group overlap {} {} fold{}".format(dataset, left, fold))
            checks.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "backbone": profile["backbone"],
                    "high_targets": profile["high_targets"],
                    "train_conditions": int(frames["train"]["condition_id"].nunique()),
                    "val_conditions": int(frames["val"]["condition_id"].nunique()),
                    "test_conditions": int(frames["test"]["condition_id"].nunique()),
                }
            )
    pd.DataFrame(checks).to_csv(OUTPUT / "preflight_audit.csv", index=False)
    write_json(
        OUTPUT / "preflight.json",
        {
            "status": "PASS",
            "scope": "retrospective dataset-specific refinement",
            "outer_test_used_in_pilot_or_freeze": False,
            "ded_mechanism": "high-target primary loss + weak auxiliary tasks + late residual adapter",
            "nist_mechanism": "all-task residual adapters + symmetric PCGrad",
            "prior_protocol_sha256": sha256(PRIOR_FROZEN),
        },
    )
    print(json.dumps(read_json(OUTPUT / "preflight.json"), indent=2), flush=True)


def stage_pilot(args: argparse.Namespace) -> None:
    protocol = prior_protocol()
    for fold in args.folds:
        ded = fold_profile(protocol, "ded", fold)
        for aux in args.ded_aux_weights:
            role = "target_primary_aux_{:03d}".format(int(round(aux * 100)))
            execute(
                train_command(
                    args, "pilot", "ded", fold, role, PILOT_SEED,
                    "target_primary_adapter", "standard", ded["high_targets"],
                    args.ded_adapter_rank, aux,
                ),
                args.dry_run,
            )
        nist = fold_profile(protocol, "nist", fold)
        for rank in args.nist_adapter_ranks:
            role = "all_adapter_pcgrad_r{}".format(rank)
            execute(
                train_command(
                    args, "pilot", "nist", fold, role, PILOT_SEED,
                    "all_task_adapters", "symmetric_pcgrad", DATASETS["nist"].parameters,
                    rank, 1.0,
                ),
                args.dry_run,
            )


def stage_freeze(args: argparse.Namespace) -> None:
    if FROZEN.is_file():
        print("[SKIP] frozen {}".format(FROZEN), flush=True)
        return
    protocol = prior_protocol()
    ded_rows, nist_rows = [], []
    for aux in args.ded_aux_weights:
        role = "target_primary_aux_{:03d}".format(int(round(aux * 100)))
        values = []
        for fold in FOLDS:
            path = run_root("pilot", "ded", fold, role, PILOT_SEED) / "validation_summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            values.append(float(read_json(path)["best_validation_score"]))
        ded_rows.append({"role": role, "aux_weight": aux, "fold_mean_score": float(np.mean(values)), "fold_scores": values})
    for rank in args.nist_adapter_ranks:
        role = "all_adapter_pcgrad_r{}".format(rank)
        values = []
        for fold in FOLDS:
            path = run_root("pilot", "nist", fold, role, PILOT_SEED) / "validation_summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            values.append(float(read_json(path)["best_validation_score"]))
        nist_rows.append({"role": role, "adapter_rank": rank, "fold_mean_score": float(np.mean(values)), "fold_scores": values})
    ded_selected = min(ded_rows, key=lambda row: row["fold_mean_score"])
    nist_selected = min(nist_rows, key=lambda row: row["fold_mean_score"])
    payload = {
        "created_before_confirm": True,
        "scope": "retrospective dataset-specific refinement",
        "outer_test_used_in_selection": False,
        "prior_protocol_sha256": sha256(PRIOR_FROZEN),
        "trainer_sha256": sha256(TRAINER),
        "ded": {"selected": ded_selected, "candidate_table": ded_rows},
        "nist": {"selected": nist_selected, "candidate_table": nist_rows},
        "fold_profiles": {
            dataset: {str(fold): fold_profile(protocol, dataset, fold) for fold in FOLDS}
            for dataset in ("ded", "nist")
        },
    }
    write_json(FROZEN, payload)
    print(json.dumps(payload, indent=2), flush=True)


def alternative_targets(profile: Dict[str, object], seed: int) -> List[str]:
    alternatives = [list(value) for value in profile.get("alternative_anchor_sets", [])]
    if not alternatives:
        raise RuntimeError("No non-selected placement control is available")
    return alternatives[SEEDS.index(seed) % len(alternatives)]


def stage_confirm(args: argparse.Namespace) -> None:
    if not FROZEN.is_file():
        raise FileNotFoundError("Freeze before confirm")
    frozen = read_json(FROZEN)
    protocol = prior_protocol()
    ded_aux = float(frozen["ded"]["selected"]["aux_weight"])
    nist_rank = int(frozen["nist"]["selected"]["adapter_rank"])
    for fold in args.folds:
        ded = fold_profile(protocol, "ded", fold)
        nist = fold_profile(protocol, "nist", fold)
        for seed in SEEDS:
            configurations = [
                (
                    "ded", "target_primary_adapter", "target_primary_adapter", "standard",
                    ded["high_targets"], args.ded_adapter_rank, ded_aux,
                ),
                (
                    "ded", "nonselected_adapter", "target_primary_adapter", "standard",
                    alternative_targets(ded, seed), args.ded_adapter_rank, ded_aux,
                ),
                ("ded", "target_only", "target_only", "standard", [], args.ded_adapter_rank, 0.0),
                (
                    "nist", "all_adapter_pcgrad", "all_task_adapters", "symmetric_pcgrad",
                    DATASETS["nist"].parameters, nist_rank, 1.0,
                ),
                (
                    "nist", "all_adapter_standard", "all_task_adapters", "standard",
                    DATASETS["nist"].parameters, nist_rank, 1.0,
                ),
                ("nist", "target_only", "target_only", "standard", [], nist_rank, 0.0),
            ]
            for dataset, role, variant, optimization, adapters, rank, aux in configurations:
                execute(
                    train_command(
                        args, "confirm", dataset, fold, role, seed, variant,
                        optimization, adapters, rank, aux,
                    ),
                    args.dry_run,
                )


def aggregate_role(dataset: str, fold: int, role: str, profile: Dict[str, object]) -> pd.DataFrame:
    spec = DATASETS[dataset]
    paths = [run_root("confirm", dataset, fold, role, seed) / "test_predictions_conditions.csv" for seed in SEEDS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    summary = read_json(run_root("confirm", dataset, fold, role, SEEDS[0]) / "test_metrics.json")
    scales = {
        parameter: float(summary["scaler"][parameter]["maximum"]) - float(summary["scaler"][parameter]["minimum"])
        for parameter in spec.parameters
    }
    frame = common.add_errors(common.average_predictions(paths, spec.parameters), spec.parameters, scales)
    frame["dataset"] = dataset
    frame["fold"] = fold
    frame["model_role"] = role
    frame["high_targets"] = json.dumps(profile["high_targets"])
    # The frozen refinement protocol stores the fold-local image-response
    # weights under ``observability_scores``.  The reusable legacy helper uses
    # the older key ``scores``.  Adapt the display schema here rather than
    # mutating either frozen protocol or historical helper.
    scoring_profile = dict(profile)
    if "scores" not in scoring_profile:
        scoring_profile["scores"] = scoring_profile["observability_scores"]
    frame["high_nmae"] = common.selected_error(frame, scoring_profile, spec.parameters)
    return frame


def stage_aggregate(args: argparse.Namespace) -> None:
    if not FROZEN.is_file():
        raise FileNotFoundError(FROZEN)
    protocol = prior_protocol()
    prior_oof = pd.read_csv(PRIOR_OUTPUT / "oof_condition_predictions.csv")
    frames = [prior_oof[prior_oof["model_role"].isin(["domain_shared", "symmetric_pcgrad"])].copy()]
    run_rows = []
    role_map = {
        "ded": ["target_primary_adapter", "nonselected_adapter", "target_only"],
        "nist": ["all_adapter_pcgrad", "all_adapter_standard", "target_only"],
    }
    for dataset, roles in role_map.items():
        for fold in FOLDS:
            profile = fold_profile(protocol, dataset, fold)
            for role in roles:
                frames.append(aggregate_role(dataset, fold, role, profile))
                for seed in SEEDS:
                    payload = read_json(run_root("confirm", dataset, fold, role, seed) / "test_metrics.json")
                    run_rows.append(
                        {
                            "dataset": dataset, "fold": fold, "role": role, "seed": seed,
                            "variant": payload["variant"], "optimization": payload["optimization"],
                            "backbone": payload["backbone"], "parameter_count": payload["parameter_count"],
                            "adapter_rank": payload["adapter_rank"], "aux_weight": payload["aux_weight"],
                        }
                    )
    oof = pd.concat(frames, ignore_index=True)
    oof.to_csv(OUTPUT / "oof_condition_predictions.csv", index=False)
    pd.DataFrame(run_rows).to_csv(OUTPUT / "run_audit.csv", index=False)

    summaries, comparisons, verdicts = [], [], {}
    for dataset in ("ded", "nist"):
        spec = DATASETS[dataset]
        roles = {
            role: frame.sort_values(["fold", "condition_id"]).reset_index(drop=True)
            for role, frame in oof[oof["dataset"].eq(dataset)].groupby("model_role", sort=False)
        }
        expected = sum(pd.read_csv(spec.split_path(fold, "test"))["condition_id"].astype(str).nunique() for fold in FOLDS)
        for role, frame in roles.items():
            if len(frame) != expected or frame["condition_id"].nunique() != expected:
                raise RuntimeError("Incomplete OOF coverage for {} {}".format(dataset, role))
            summaries.append(previous.model_summary(dataset, role, frame, spec.parameters))
        candidate_role = "target_primary_adapter" if dataset == "ded" else "all_adapter_pcgrad"
        baselines = (
            ["domain_shared", "nonselected_adapter", "target_only"]
            if dataset == "ded"
            else ["domain_shared", "symmetric_pcgrad", "all_adapter_standard", "target_only"]
        )
        candidate = roles[candidate_role]
        keys = ["fold", "condition_id", "group_id", "high_targets"]
        dataset_comparisons = []
        for baseline_role in baselines:
            baseline = roles[baseline_role]
            if not candidate[keys].equals(baseline[keys]):
                raise RuntimeError("Prediction keys differ {} {}".format(dataset, baseline_role))
            comparison = candidate[keys].copy()
            comparison["gain_high_nmae_pp"] = 100.0 * (
                baseline["high_nmae"].to_numpy(float) - candidate["high_nmae"].to_numpy(float)
            )
            statistics = common.cluster_ci(comparison, "gain_high_nmae_pp", args.bootstrap_seed, args.bootstrap_iterations)
            row = {
                "dataset": dataset,
                "comparison": "{}_vs_{}".format(candidate_role, baseline_role),
                **statistics,
                "conditions": len(comparison),
                "improved_conditions": int((comparison["gain_high_nmae_pp"] > 0).sum()),
                "worse_conditions": int((comparison["gain_high_nmae_pp"] < 0).sum()),
            }
            comparisons.append(row)
            dataset_comparisons.append(row)
        candidate_summary = next(row for row in summaries if row["dataset"] == dataset and row["model_role"] == candidate_role)
        shared_summary = next(row for row in summaries if row["dataset"] == dataset and row["model_role"] == "domain_shared")
        overall_degradation = 100.0 * (candidate_summary["mean_nmae"] - shared_summary["mean_nmae"])
        required = [
            row for row in dataset_comparisons
            if row["comparison"].endswith("domain_shared")
            or row["comparison"].endswith("nonselected_adapter")
            or row["comparison"].endswith("symmetric_pcgrad")
            or row["comparison"].endswith("all_adapter_standard")
        ]
        verdicts[dataset] = {
            "candidate_role": candidate_role,
            "comparisons": dataset_comparisons,
            "overall_mean_nmae_degradation_vs_shared_pp": overall_degradation,
            "workflow_support_rule_passed": bool(
                required and all(row["ci_low"] > 0.0 for row in required)
                and overall_degradation <= args.overall_guardrail_pp
            ),
        }
    pd.DataFrame(summaries).to_csv(OUTPUT / "model_oof_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUTPUT / "paired_group_comparisons.csv", index=False)
    write_json(
        OUTPUT / "development_summary.json",
        {
            "scope": "retrospective dataset-specific refinement",
            "outer_test_used_in_selection": False,
            "datasets": verdicts,
            "both_datasets_support_workflow": all(value["workflow_support_rule_passed"] for value in verdicts.values()),
        },
    )
    print(json.dumps(read_json(OUTPUT / "development_summary.json"), indent=2), flush=True)


def stage_package(args: argparse.Namespace) -> None:
    if not (OUTPUT / "development_summary.json").is_file():
        raise FileNotFoundError("Aggregate before package")
    archive_path = PROJECT / "dataset_specific_refinement_20260807_no_weights.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_dir() or path.suffix.lower() in (".pth", ".pt"):
                continue
            archive.add(path, arcname=str(path.relative_to(PROJECT)))
        for path in (TRAINER, Path(__file__).resolve()):
            archive.add(path, arcname=str(path.relative_to(PROJECT)))
    print("[ARCHIVE] {}".format(archive_path), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "pilot", "freeze", "confirm", "aggregate", "package"))
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    parser.add_argument("--ded-aux-weights", nargs="+", type=float, default=[0.10, 0.25, 0.50])
    parser.add_argument("--ded-adapter-rank", type=int, default=64)
    parser.add_argument("--nist-adapter-ranks", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--pilot-epochs", type=int, default=30)
    parser.add_argument("--confirm-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--set-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--overall-guardrail-pp", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    {
        "preflight": stage_preflight,
        "pilot": stage_pilot,
        "freeze": stage_freeze,
        "confirm": stage_confirm,
        "aggregate": stage_aggregate,
        "package": stage_package,
    }[arguments.stage](arguments)
