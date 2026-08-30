#!/usr/bin/env python3
"""Two-stage orchestration for ordered and pairwise textile supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dpi_branch_ablation import (  # noqa: E402
    PARAM_NAMES,
    count_trainable_parameters,
)
from scripts.train_textile_ordered_pairwise_20260803 import (  # noqa: E402
    DpiOneFactorPairDataset,
    VARIANTS,
    build_model,
)


FOLDS = [0, 1, 2, 3, 4]
SEEDS = [42, 52, 62]
ENHANCED_VARIANTS = [variant for variant in VARIANTS if variant != "base_regression"]
CONFIRM_GROUPS = [
    "baseline_regression",
    "pairwise_only",
    "ordinal_only",
    "selected_ordered_supervision",
]
DEFAULT_SPLITS = "data/images3/grouped_outer_cv_20260730"
DEFAULT_GROUPED = "outputs/grouped_outer_cv_20260730"
DEFAULT_OUTPUT = "outputs/textile_ordered_pairwise_20260803"
DEFAULT_BEFORE = "data/images3/before"
DEFAULT_AFTER = "data/images3/after"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_dir(args: argparse.Namespace, fold: int) -> Path:
    return args.splits_root / "fold_{}".format(fold)


def target_path(args: argparse.Namespace, fold: int) -> Path:
    return (
        args.grouped_root
        / "fold_{}".format(fold)
        / "observability"
        / "grouped"
        / "selected_observability_target.json"
    )


def selected_target(args: argparse.Namespace, fold: int) -> str:
    path = target_path(args, fold)
    payload = read_json(path)
    expected_scope = "outer-training groups only; outer-test groups were excluded"
    if payload.get("selection_scope") != expected_scope:
        raise RuntimeError("Unexpected observability scope in {}".format(path))
    target = str(payload["selected_target"])
    if target != "dpi":
        raise RuntimeError(
            "Ordered DPI protocol requires dpi in every fold; fold {} selected {}".format(
                fold, target
            )
        )
    return target


def audit_splits(args: argparse.Namespace) -> Dict[str, object]:
    rows = []
    outer_samples: List[str] = []
    outer_groups: List[str] = []
    for fold in FOLDS:
        frames = {}
        for split in ["train", "val", "test"]:
            path = split_dir(args, fold) / "label_{}.csv".format(split)
            frame = pd.read_csv(path)
            required = {"sample_id", "before_id", "pattern_id", *PARAM_NAMES}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError("{} missing {}".format(path, sorted(missing)))
            if frame["sample_id"].astype(str).duplicated().any():
                raise ValueError("Duplicate sample_id in {}".format(path))
            frames[split] = frame
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "samples": int(len(frame)),
                    "initial_image_groups": int(frame["before_id"].nunique()),
                }
            )
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
            sample_overlap = set(frames[left]["sample_id"].astype(str)) & set(
                frames[right]["sample_id"].astype(str)
            )
            group_overlap = set(frames[left]["before_id"].astype(str)) & set(
                frames[right]["before_id"].astype(str)
            )
            if sample_overlap or group_overlap:
                raise RuntimeError(
                    "Fold {} {}/{} overlap: samples={} groups={}".format(
                        fold, left, right, len(sample_overlap), len(group_overlap)
                    )
                )
        outer_samples.extend(frames["test"]["sample_id"].astype(str).tolist())
        outer_groups.extend(frames["test"]["before_id"].astype(str).tolist())
        selected_target(args, fold)
    counts = pd.Series(outer_samples).value_counts()
    if len(counts) != 1240 or not counts.eq(1).all():
        raise RuntimeError("Outer-fold sample coverage is not exactly 1,240 once each")
    if len(set(outer_groups)) != 29:
        raise RuntimeError("Expected 29 initial-image groups")
    if "877" in set(outer_samples):
        raise RuntimeError("Unverified sample 877 must remain excluded")
    return {
        "rows": rows,
        "unique_oof_samples": 1240,
        "unique_initial_image_groups": 29,
        "zero_sample_overlap": True,
        "zero_initial_image_group_overlap": True,
        "sample_877_excluded": True,
    }


def model_args(variant: str) -> argparse.Namespace:
    return argparse.Namespace(
        variant=variant,
        expert_target="dpi",
        no_pretrained=True,
        hidden_dim=256,
        dropout=0.3,
        texture_dim=64,
    )


def model_smoke() -> List[Dict[str, object]]:
    rows = []
    for variant in VARIANTS:
        model = build_model(model_args(variant)).eval()
        with torch.no_grad():
            output = model({"effect": torch.zeros(2, 3, 224, 224)})
        prediction = output[0] if isinstance(output, tuple) else output
        if tuple(prediction.shape) != (2, len(PARAM_NAMES)):
            raise RuntimeError("{} produced shape {}".format(variant, prediction.shape))
        rows.append(
            {
                "variant": variant,
                "parameter_count": count_trainable_parameters(model),
                "has_ordinal_head": bool(isinstance(output, tuple)),
            }
        )
        del model
    return rows


def pair_audit(args: argparse.Namespace) -> List[Dict[str, object]]:
    rows = []
    for fold in FOLDS:
        dataset = DpiOneFactorPairDataset(
            str(split_dir(args, fold) / "label_train.csv"),
            str(args.before_dir),
            str(args.after_dir),
            224,
            "stretch",
            "weak",
        )
        training_ids = set(
            pd.read_csv(split_dir(args, fold) / "label_train.csv")["sample_id"]
            .astype(str)
            .tolist()
        )
        pair_ids = set(dataset.pairs["sample_id_a"].astype(str)) | set(
            dataset.pairs["sample_id_b"].astype(str)
        )
        if not pair_ids <= training_ids:
            raise RuntimeError("Fold {} pair audit crossed training boundary".format(fold))
        rows.append(
            {
                "outer_fold": fold,
                "training_only_pairs": int(len(dataset)),
                "unique_pair_samples": int(len(pair_ids)),
                "initial_image_groups": int(dataset.pairs["before_id"].nunique()),
                "validation_or_test_samples_used": 0,
            }
        )
    return rows


def stage_preflight(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "completed_at": now(),
        "protocol": "subsequent exploratory leakage-controlled grouped internal validation",
        "split_audit": audit_splits(args),
        "training_only_pair_audit": pair_audit(args),
        "model_smoke": model_smoke(),
        "fixed_factors": {
            "input": "processed-image-only",
            "resize": "224 x 224 stretch",
            "augmentation": "weak",
            "backbone": "current two-path P_obs specialist",
            "regression_loss": "Smooth L1 with equal output weights",
            "checkpoint": "inner-validation mean MAPE",
        },
    }
    write_json(args.output_root / "preflight_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def run_command(command: Sequence[str], dry_run: bool) -> None:
    print("[CMD] {}".format(" ".join(str(value) for value in command)), flush=True)
    if not dry_run:
        subprocess.run(list(command), check=True)


def trainer_command(
    args: argparse.Namespace,
    fold: int,
    variant: str,
    model_id: str,
    seed: int,
    epochs: int,
    run_dir: Path,
    include_test: bool,
) -> List[str]:
    directory = split_dir(args, fold)
    command = [
        sys.executable,
        str(args.project_root / "scripts" / "train_textile_ordered_pairwise_20260803.py"),
        "--variant",
        variant,
        "--expert-target",
        "dpi",
        "--train-csv",
        str(directory / "label_train.csv"),
        "--val-csv",
        str(directory / "label_val.csv"),
        "--before-dir",
        str(args.before_dir),
        "--after-dir",
        str(args.after_dir),
        "--run-dir",
        str(run_dir),
        "--run-name",
        "{}_fold{}_seed{}".format(model_id, fold, seed),
        "--outer-fold",
        str(fold),
        "--model-id",
        model_id,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--pair-batch-size",
        str(args.pair_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--img-size",
        "224",
        "--resize-mode",
        "stretch",
        "--augmentation-mode",
        "weak",
        "--selection-metric",
        "val_mean_mape",
    ]
    if include_test:
        command.extend(["--test-csv", str(directory / "label_test.csv")])
    return command


def validation_complete(run_dir: Path) -> bool:
    path = run_dir / "validation_summary.json"
    if not path.exists():
        return False
    payload = read_json(path)
    return (
        payload.get("evaluation_scope") == "inner_validation_only"
        and payload.get("outer_test_was_supplied") is False
        and not (run_dir / "outer_test" / "test_predictions.csv").exists()
    )


def confirm_complete(run_dir: Path, expected_rows: int) -> bool:
    summary = run_dir / "outer_test" / "test_summary.json"
    predictions = run_dir / "outer_test" / "test_predictions.csv"
    if not summary.exists() or not predictions.exists():
        return False
    frame = pd.read_csv(predictions)
    return len(frame) == expected_rows and frame["sample_id"].astype(str).nunique() == expected_rows


def stage_pilot(args: argparse.Namespace) -> None:
    for fold in args.folds:
        selected_target(args, fold)
        for variant in VARIANTS:
            run_dir = args.output_root / "pilot" / "fold_{}".format(fold) / variant / "seed42"
            if args.resume and validation_complete(run_dir):
                print("[SKIP] pilot fold={} variant={}".format(fold, variant))
                continue
            command = trainer_command(
                args, fold, variant, variant, 42, args.pilot_epochs, run_dir, False
            )
            if "--test-csv" in command:
                raise RuntimeError("Pilot command crossed the outer-test boundary")
            run_command(command, args.dry_run)


def pilot_score(payload: Dict[str, object]) -> float:
    metrics = payload["best_eval"]
    parameter = metrics["param_mape_physical"]
    nonselected = [float(value) for name, value in parameter.items() if name != "dpi"]
    return float(parameter["dpi"]) + 0.25 * float(metrics["mape_physical"]) + 0.10 * max(nonselected)


def stage_select(args: argparse.Namespace) -> None:
    rows = []
    winners = {}
    for fold in FOLDS:
        for variant in VARIANTS:
            path = args.output_root / "pilot" / "fold_{}".format(fold) / variant / "seed42" / "validation_summary.json"
            if not path.exists():
                raise FileNotFoundError("Incomplete pilot: {}".format(path))
            payload = read_json(path)
            if payload.get("outer_test_was_supplied") is not False:
                raise RuntimeError("Pilot outer-test boundary violation: {}".format(path))
            rows.append(
                {
                    "outer_fold": fold,
                    "variant": variant,
                    "pilot_score": pilot_score(payload),
                    "val_dpi_mape": payload["best_eval"]["param_mape_physical"]["dpi"],
                    "val_mean_mape": payload["best_eval"]["mape_physical"],
                    "parameter_count": payload["parameter_count"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "pilot_inner_validation.csv", index=False, encoding="utf-8-sig")
    for fold in FOLDS:
        candidates = frame.loc[
            frame["outer_fold"].eq(fold) & frame["variant"].isin(ENHANCED_VARIANTS)
        ].sort_values(["pilot_score", "variant"])
        winners[str(fold)] = str(candidates.iloc[0]["variant"])
    ranking = (
        frame.groupby("variant", as_index=False)
        .agg(mean_pilot_score=("pilot_score", "mean"), sd_pilot_score=("pilot_score", "std"))
        .sort_values(["mean_pilot_score", "variant"])
    )
    ranking.to_csv(args.output_root / "pilot_candidate_ranking.csv", index=False, encoding="utf-8-sig")
    write_json(
        args.output_root / "PILOT_SELECTION.json",
        {
            "selected_at": now(),
            "selection_scope": "fold-specific inner validation only",
            "outer_test_used": False,
            "objective": "DPI MAPE + 0.25*mean MAPE + 0.10*max nonselected MAPE",
            "pilot_seed": 42,
            "winner_by_fold": winners,
            "descriptive_global_ranking_not_used_for_outer_test": ranking.to_dict(orient="records"),
        },
    )
    print("[SELECTED BY FOLD] {}".format(winners))


def frozen_files(args: argparse.Namespace) -> List[Path]:
    files = [
        args.project_root / "scripts" / "train_textile_ordered_pairwise_20260803.py",
        args.project_root / "scripts" / "run_textile_ordered_pairwise_pipeline_20260803.py",
        args.project_root / "scripts" / "aggregate_textile_ordered_pairwise_20260803.py",
        args.output_root / "PILOT_SELECTION.json",
    ]
    for fold in FOLDS:
        directory = split_dir(args, fold)
        files.extend(
            [
                directory / "label_train.csv",
                directory / "label_val.csv",
                directory / "label_test.csv",
                target_path(args, fold),
            ]
        )
    return files


def stage_freeze(args: argparse.Namespace) -> None:
    selection = args.output_root / "PILOT_SELECTION.json"
    if not selection.exists():
        raise FileNotFoundError("Run select before freeze")
    if list((args.output_root / "confirm").glob("**/test_summary.json")):
        raise RuntimeError("Cannot freeze after outer-test results exist")
    files = frozen_files(args)
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen files: {}".format(missing))
    payload = {
        "frozen_at": now(),
        "protocol": "subsequent exploratory leakage-controlled grouped internal validation",
        "outer_test_used_for_candidate_or_hyperparameter_selection": False,
        "pilot": read_json(selection),
        "confirm_groups": CONFIRM_GROUPS,
        "folds": FOLDS,
        "seeds": SEEDS,
        "pilot_epochs": args.pilot_epochs,
        "confirm_epochs": args.confirm_epochs,
        "upgrade_rules": {
            "selected_vs_baseline_DPI_cluster_ci_high_lt_zero": True,
            "overall_mean_APE_delta_le_0_5pp": True,
            "each_nonselected_parameter_delta_le_1pp": True,
            "expected_samples": 1240,
            "expected_initial_image_groups": 29,
        },
        "sha256": {
            str(path.relative_to(args.project_root)): sha256(path) for path in files
        },
    }
    write_json(args.output_root / "FROZEN_SELECTION.json", payload)
    print("[FROZEN] {}".format(args.output_root / "FROZEN_SELECTION.json"))


def confirm_variant(group: str, fold: int, winners: Dict[str, str]) -> str:
    mapping = {
        "baseline_regression": "base_regression",
        "pairwise_only": "pairwise_rank_delta",
        "ordinal_only": "coral_aux",
    }
    if group == "selected_ordered_supervision":
        return winners[str(fold)]
    return mapping[group]


def stage_confirm(args: argparse.Namespace) -> None:
    frozen_path = args.output_root / "FROZEN_SELECTION.json"
    if not frozen_path.exists():
        raise FileNotFoundError("Freeze the protocol before confirm")
    frozen = read_json(frozen_path)
    winners = {str(key): str(value) for key, value in frozen["pilot"]["winner_by_fold"].items()}
    for fold in args.folds:
        expected_rows = len(pd.read_csv(split_dir(args, fold) / "label_test.csv"))
        for group in CONFIRM_GROUPS:
            variant = confirm_variant(group, fold, winners)
            for seed in SEEDS:
                run_dir = args.output_root / "confirm" / "fold_{}".format(fold) / group / "seed{}".format(seed)
                if args.resume and confirm_complete(run_dir, expected_rows):
                    print("[SKIP] confirm fold={} model={} seed={}".format(fold, group, seed))
                    continue
                command = trainer_command(
                    args,
                    fold,
                    variant,
                    group,
                    seed,
                    args.confirm_epochs,
                    run_dir,
                    True,
                )
                run_command(command, args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "pilot", "select", "freeze", "confirm"])
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--splits-root", default=DEFAULT_SPLITS)
    parser.add_argument("--grouped-root", default=DEFAULT_GROUPED)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--before-dir", default=DEFAULT_BEFORE)
    parser.add_argument("--after-dir", default=DEFAULT_AFTER)
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    parser.add_argument("--pilot-epochs", type=int, default=30)
    parser.add_argument("--confirm-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pair-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.project_root = Path(args.project_root).resolve()
    for name in ["splits_root", "grouped_root", "output_root", "before_dir", "after_dir"]:
        value = Path(getattr(args, name))
        if not value.is_absolute():
            value = args.project_root / value
        setattr(args, name, value.resolve())
    if not set(args.folds) <= set(FOLDS):
        raise ValueError("Invalid folds {}".format(args.folds))
    return args


def main() -> None:
    args = parse_args()
    stages = {
        "preflight": stage_preflight,
        "pilot": stage_pilot,
        "select": stage_select,
        "freeze": stage_freeze,
        "confirm": stage_confirm,
    }
    stages[args.stage](args)


if __name__ == "__main__":
    main()
