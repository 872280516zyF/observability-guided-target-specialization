#!/usr/bin/env python3
"""Orchestrate the frozen smallNORB observability-guided replication."""

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

from scripts import prepare_smallnorb_observability_20260807 as prepared  # noqa: E402
from scripts import train_smallnorb_observability_20260807 as trainer  # noqa: E402


TRAINER = PROJECT / "scripts" / "train_smallnorb_observability_20260807.py"
OUTPUT = PROJECT / "outputs" / "smallnorb_observability_20260807"
FROZEN = OUTPUT / "FROZEN_PROTOCOL.json"
FOLDS = list(range(5))
SEEDS = [42, 52, 62]
PILOT_SEED = 42
CANDIDATES = ("selected_spatial_standard", "selected_spatial_pcgrad")


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


def profile_path(fold: int = -1) -> Path:
    return prepared.OBSERVABILITY / (
        "official_train.json" if fold < 0 else "inner_fold_{}.json".format(fold)
    )


def pilot_root(fold: int, role: str) -> Path:
    return OUTPUT / "pilot" / "fold_{}".format(fold) / role / "seed{}".format(PILOT_SEED)


def confirm_root(role: str, seed: int) -> Path:
    return OUTPUT / "confirm" / role / "seed{}".format(seed)


def command(
    args: argparse.Namespace,
    phase: str,
    mode: str,
    optimization: str,
    high_targets: Sequence[str],
    specialist_targets: Sequence[str],
    output: Path,
    seed: int,
    epochs: int,
    fold: int = -1,
) -> List[str]:
    values = [
        sys.executable, str(TRAINER),
        "--phase", phase,
        "--mode", mode,
        "--optimization", optimization,
        "--train-csv", str(
            prepared.ROOT / "official_train.csv"
            if fold < 0
            else prepared.SPLITS / "inner_fold_{}".format(fold) / "train.csv"
        ),
        "--high-targets", *high_targets,
        "--output-dir", str(output),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
    ]
    if specialist_targets:
        values.extend(["--specialist-targets", *specialist_targets])
    if phase == "pilot":
        values.extend(["--val-csv", str(prepared.SPLITS / "inner_fold_{}".format(fold) / "val.csv")])
    else:
        values.extend(["--test-csv", str(prepared.ROOT / "official_test.csv")])
    return values


def stage_preflight(args: argparse.Namespace) -> None:
    required = [
        TRAINER,
        prepared.ROOT / "preflight.json",
        prepared.ROOT / "official_train.csv",
        prepared.ROOT / "official_test.csv",
        profile_path(),
    ]
    for fold in FOLDS:
        required.extend(
            [
                prepared.SPLITS / "inner_fold_{}".format(fold) / "train.csv",
                prepared.SPLITS / "inner_fold_{}".format(fold) / "val.csv",
                profile_path(fold),
            ]
        )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset_audit = read_json(prepared.ROOT / "preflight.json")
    if dataset_audit["status"] != "PASS" or dataset_audit["group_overlap"] != 0:
        raise RuntimeError("smallNORB preparation audit failed")
    final = read_json(profile_path())
    write_json(
        OUTPUT / "preflight.json",
        {
            "status": "PASS",
            "official_test_used_in_pilot_or_freeze": False,
            "official_train_groups": dataset_audit["official_train_groups"],
            "official_test_groups": dataset_audit["official_test_groups"],
            "high_targets_from_official_train_only": final["high_targets"],
            "observability_scores": final["scores"],
            "trainer_sha256": sha256(TRAINER),
        },
    )
    print(json.dumps(read_json(OUTPUT / "preflight.json"), indent=2), flush=True)


def stage_pilot(args: argparse.Namespace) -> None:
    for fold in args.folds:
        profile = read_json(profile_path(fold))
        for role in CANDIDATES:
            optimization = "symmetric_pcgrad" if role.endswith("pcgrad") else "standard"
            execute(
                command(
                    args, "pilot", "selected_spatial", optimization,
                    profile["high_targets"], profile["high_targets"],
                    pilot_root(fold, role), PILOT_SEED, args.pilot_epochs, fold,
                ),
                args.dry_run,
            )


def stage_freeze(args: argparse.Namespace) -> None:
    if FROZEN.is_file():
        print("[SKIP] frozen {}".format(FROZEN), flush=True)
        return
    rows = []
    for role in CANDIDATES:
        scores, epochs = [], []
        for fold in FOLDS:
            path = pilot_root(fold, role) / "validation_summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = read_json(path)
            scores.append(float(payload["selection_score"]))
            epochs.append(int(payload["best_epoch"]))
        rows.append(
            {
                "role": role,
                "fold_mean_score": float(np.mean(scores)),
                "fold_scores": scores,
                "best_epochs": epochs,
                "confirm_epochs": int(max(5, min(30, round(float(np.median(epochs)))))),
            }
        )
    selected = min(rows, key=lambda row: row["fold_mean_score"])
    profile = read_json(profile_path())
    low_targets = [factor for factor in trainer.FACTORS if factor not in set(profile["high_targets"])]
    if not low_targets:
        raise RuntimeError("No non-selected factor remains for placement control")
    nonselected = sorted(low_targets, key=lambda value: profile["scores"][value])[: len(profile["high_targets"])]
    placement_capacity_matched = len(nonselected) == len(profile["high_targets"])
    payload = {
        "created_before_official_test_training": True,
        "official_test_used_in_selection": False,
        "selection_scope": "five inner folds of official training object instances",
        "selected": selected,
        "candidate_table": rows,
        "official_train_observability": profile,
        "nonselected_control_targets": nonselected,
        "placement_capacity_matched": placement_capacity_matched,
        "trainer_sha256": sha256(TRAINER),
        "official_train_csv_sha256": sha256(prepared.ROOT / "official_train.csv"),
        "official_test_csv_sha256": sha256(prepared.ROOT / "official_test.csv"),
    }
    write_json(FROZEN, payload)
    print(json.dumps(payload, indent=2), flush=True)


def stage_confirm(args: argparse.Namespace) -> None:
    if not FROZEN.is_file():
        raise FileNotFoundError("Freeze before official-test confirm")
    frozen = read_json(FROZEN)
    high = frozen["official_train_observability"]["high_targets"]
    nonselected = frozen["nonselected_control_targets"]
    selected_optimization = "symmetric_pcgrad" if frozen["selected"]["role"].endswith("pcgrad") else "standard"
    epochs = int(frozen["selected"]["confirm_epochs"])
    configurations = [
        ("shared_baseline", "shared", "standard", []),
        ("selected_candidate", "selected_spatial", selected_optimization, high),
        ("nonselected_placement", "selected_spatial", selected_optimization, nonselected),
        ("all_spatial_control", "all_spatial", selected_optimization, list(trainer.FACTORS)),
    ]
    for seed in SEEDS:
        for role, mode, optimization, specialists in configurations:
            execute(
                command(
                    args, "confirm", mode, optimization, high, specialists,
                    confirm_root(role, seed), seed, epochs,
                ),
                args.dry_run,
            )


def average_predictions(paths: Sequence[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path).sort_values(["group_id", "row_index"]).reset_index(drop=True) for path in paths]
    keys = ["condition_id", "group_id", "row_index", *["true_{}".format(value) for value in trainer.FACTORS]]
    for frame in frames[1:]:
        if not frames[0][keys].equals(frame[keys]):
            raise RuntimeError("smallNORB prediction keys differ across seeds")
    result = frames[0][keys].copy()
    for factor in trainer.FACTORS:
        probability_columns = ["prob_{}_{}".format(factor, index) for index in range(trainer.CLASS_COUNTS[factor])]
        average = sum(frame[probability_columns].to_numpy(float) for frame in frames) / float(len(frames))
        for index, column in enumerate(probability_columns):
            result[column] = average[:, index]
        result["pred_{}".format(factor)] = np.argmax(average, axis=1)
        result["nmae_{}".format(factor)] = trainer.error_values(
            factor,
            result["true_{}".format(factor)].to_numpy(),
            result["pred_{}".format(factor)].to_numpy(),
        )
        result["correct_{}".format(factor)] = (
            result["true_{}".format(factor)] == result["pred_{}".format(factor)]
        ).astype(float)
    return result


def cluster_ci(frame: pd.DataFrame, column: str, seed: int, iterations: int) -> Dict[str, float]:
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
    if not FROZEN.is_file():
        raise FileNotFoundError(FROZEN)
    frozen = read_json(FROZEN)
    high = frozen["official_train_observability"]["high_targets"]
    roles = {}
    summaries = []
    for role in ("shared_baseline", "selected_candidate", "nonselected_placement", "all_spatial_control"):
        paths = [confirm_root(role, seed) / "test_predictions.csv" for seed in SEEDS]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        frame = average_predictions(paths)
        frame["model_role"] = role
        roles[role] = frame
        values = trainer.metrics(frame, high)
        summaries.append({"model_role": role, **values})
    pd.concat(roles.values(), ignore_index=True).to_csv(OUTPUT / "official_test_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT / "model_summary.csv", index=False)

    candidate = roles["selected_candidate"]
    comparisons = []
    for baseline_role in ("shared_baseline", "nonselected_placement", "all_spatial_control"):
        baseline = roles[baseline_role]
        keys = ["condition_id", "group_id", "row_index"]
        if not candidate[keys].equals(baseline[keys]):
            raise RuntimeError("smallNORB comparison keys differ")
        comparison = candidate[keys].copy()
        baseline_high = np.mean(
            [baseline["nmae_{}".format(factor)].to_numpy(float) for factor in high], axis=0
        )
        candidate_high = np.mean(
            [candidate["nmae_{}".format(factor)].to_numpy(float) for factor in high], axis=0
        )
        comparison["gain_high_nmae_pp"] = 100.0 * (baseline_high - candidate_high)
        comparisons.append(
            {
                "comparison": "selected_candidate_vs_{}".format(baseline_role),
                **cluster_ci(comparison, "gain_high_nmae_pp", args.bootstrap_seed, args.bootstrap_iterations),
                "images": len(comparison),
            }
        )
    pd.DataFrame(comparisons).to_csv(OUTPUT / "paired_group_comparisons.csv", index=False)
    selected_summary = next(row for row in summaries if row["model_role"] == "selected_candidate")
    shared_summary = next(row for row in summaries if row["model_role"] == "shared_baseline")
    overall_degradation = 100.0 * (selected_summary["mean_nmae"] - shared_summary["mean_nmae"])
    required_baselines = ["shared_baseline", "all_spatial_control"]
    if frozen.get("placement_capacity_matched", False):
        required_baselines.append("nonselected_placement")
    required_comparisons = [
        row for row in comparisons
        if row["comparison"].replace("selected_candidate_vs_", "") in required_baselines
    ]
    verdict = {
        "scope": "official held-out-object smallNORB confirmation",
        "official_test_used_in_selection": False,
        "high_targets": high,
        "selected_model": frozen["selected"],
        "placement_capacity_matched": bool(frozen.get("placement_capacity_matched", False)),
        "required_baselines": required_baselines,
        "comparisons": comparisons,
        "overall_mean_nmae_degradation_vs_shared_pp": overall_degradation,
        "workflow_support_rule_passed": bool(
            len(required_comparisons) == len(required_baselines)
            and all(row["ci_low"] > 0.0 for row in required_comparisons)
            and overall_degradation <= args.overall_guardrail_pp
        ),
    }
    write_json(OUTPUT / "development_summary.json", verdict)
    print(json.dumps(verdict, indent=2), flush=True)


def stage_package(args: argparse.Namespace) -> None:
    if not (OUTPUT / "development_summary.json").is_file():
        raise FileNotFoundError("Aggregate before package")
    archive_path = PROJECT / "smallnorb_observability_20260807_no_weights.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_dir() or path.suffix.lower() in (".pth", ".pt"):
                continue
            archive.add(path, arcname=str(path.relative_to(PROJECT)))
        for path in (TRAINER, Path(__file__).resolve(), Path(prepared.__file__).resolve()):
            archive.add(path, arcname=str(path.relative_to(PROJECT)))
    print("[ARCHIVE] {}".format(archive_path), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "pilot", "freeze", "confirm", "aggregate", "package"))
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    parser.add_argument("--pilot-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--overall-guardrail-pp", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    {
        "preflight": stage_preflight,
        "pilot": stage_pilot,
        "freeze": stage_freeze,
        "confirm": stage_confirm,
        "aggregate": stage_aggregate,
        "package": stage_package,
    }[args.stage](args)
