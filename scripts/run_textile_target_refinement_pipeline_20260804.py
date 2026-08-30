#!/usr/bin/env python3
"""Orchestrate the second, explicitly exploratory target-refinement suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
import torch.nn as nn

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_textile_ordered_pairwise_pipeline_20260803 as prior  # noqa: E402
from scripts.run_dpi_branch_ablation import count_trainable_parameters  # noqa: E402
from scripts.train_textile_target_refinement_20260804 import (  # noqa: E402
    VARIANTS,
    build_model,
    specialist_only_forward,
)


FOLDS = [0, 1, 2, 3, 4]
SEEDS = [42, 52, 62]
ADVANCED_VARIANTS = [
    "isolated_pairwise_coral_pobs",
    "two_stage_coral_pobs",
]
CONFIRM_GROUPS = [
    "mean_selection_baseline",
    "pobs_selection_baseline",
    "pobs_coral",
    "pobs_isolated_pairwise",
    "selected_advanced_refinement",
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def model_args(variant: str) -> argparse.Namespace:
    return argparse.Namespace(
        variant=variant,
        expert_target="dpi",
        no_pretrained=True,
        hidden_dim=256,
        dropout=0.3,
        texture_dim=64,
    )


def bn_state(module: nn.Module) -> List[torch.Tensor]:
    return [
        child.running_mean.detach().clone()
        for child in module.modules()
        if isinstance(child, nn.BatchNorm1d) or isinstance(child, nn.BatchNorm2d)
    ]


def model_and_isolation_smoke() -> Dict[str, object]:
    rows = []
    isolation_passed = False
    for variant in VARIANTS:
        model = build_model(model_args(variant)).train()
        image = torch.zeros(2, 3, 224, 224)
        before = bn_state(model.general_backbone) + bn_state(model.general_aux_encoder) + bn_state(model.general_head)
        if "isolated_pairwise" in variant:
            with torch.no_grad():
                specialist_prediction, _ = specialist_only_forward(model, image)
            after = bn_state(model.general_backbone) + bn_state(model.general_aux_encoder) + bn_state(model.general_head)
            isolation_passed = bool(
                tuple(specialist_prediction.shape) == (2, 1)
                and len(before) == len(after)
                and all(torch.equal(left, right) for left, right in zip(before, after))
            )
            if not isolation_passed:
                raise RuntimeError("Specialist-only pair forward modified general BN state")
        with torch.no_grad():
            output = model({"effect": image})
        prediction = output[0] if isinstance(output, tuple) else output
        if tuple(prediction.shape) != (2, 4):
            raise RuntimeError("{} output shape {}".format(variant, prediction.shape))
        rows.append(
            {
                "variant": variant,
                "parameter_count": count_trainable_parameters(model),
                "has_coral_head": bool(isinstance(output, tuple)),
            }
        )
        del model
    return {
        "models": rows,
        "specialist_only_pair_forward_general_BN_unchanged": isolation_passed,
    }


def stage_preflight(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_at": now(),
        "protocol": (
            "iterative exploratory refinement using previously examined grouped folds; "
            "not independent confirmation"
        ),
        "split_audit": prior.audit_splits(args),
        "training_only_pair_audit": prior.pair_audit(args),
        "model_and_isolation_smoke": model_and_isolation_smoke(),
        "fixed_factors": {
            "input": "processed-image-only",
            "resize": "224 x 224 stretch",
            "augmentation": "weak",
            "architecture": "current P_obs specialist",
            "outer_folds": "unchanged grouped folds",
            "pair_definition": "training-only adjacent one-factor DPI pairs",
        },
    }
    write_json(args.output_root / "preflight_manifest.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


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
    fine_tune_start: int,
    run_dir: Path,
    include_test: bool,
) -> List[str]:
    directory = prior.split_dir(args, fold)
    command = [
        sys.executable,
        str(args.project_root / "scripts" / "train_textile_target_refinement_20260804.py"),
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
        "--fine-tune-start-epoch",
        str(fine_tune_start),
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
        "variant_specific",
    ]
    if include_test:
        command.extend(["--test-csv", str(directory / "label_test.csv")])
    return command


def stage_pilot(args: argparse.Namespace) -> None:
    for fold in args.folds:
        prior.selected_target(args, fold)
        for variant in VARIANTS:
            run_dir = args.output_root / "pilot" / "fold_{}".format(fold) / variant / "seed42"
            if prior.validation_complete(run_dir):
                print("[SKIP] pilot fold={} variant={}".format(fold, variant))
                continue
            command = trainer_command(
                args,
                fold,
                variant,
                variant,
                42,
                args.pilot_epochs,
                args.pilot_fine_tune_start,
                run_dir,
                False,
            )
            if "--test-csv" in command:
                raise RuntimeError("Pilot command crossed test boundary")
            run_command(command, args.dry_run)


def pilot_score(payload: Dict[str, object]) -> float:
    metrics = payload["best_eval"]
    parameters = metrics["param_mape_physical"]
    nonselected = [float(value) for name, value in parameters.items() if name != "dpi"]
    return float(parameters["dpi"]) + 0.25 * float(metrics["mape_physical"]) + 0.10 * max(nonselected)


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
                raise RuntimeError("Pilot used outer test: {}".format(path))
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
            frame["outer_fold"].eq(fold) & frame["variant"].isin(ADVANCED_VARIANTS)
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
            "winner_by_fold": winners,
            "advanced_candidates": ADVANCED_VARIANTS,
            "descriptive_ranking": ranking.to_dict(orient="records"),
        },
    )
    print("[SELECTED ADVANCED BY FOLD] {}".format(winners))


def frozen_files(args: argparse.Namespace) -> List[Path]:
    files = [
        args.project_root / "scripts" / "train_textile_target_refinement_20260804.py",
        args.project_root / "scripts" / "run_textile_target_refinement_pipeline_20260804.py",
        args.project_root / "scripts" / "aggregate_textile_target_refinement_20260804.py",
        args.output_root / "PILOT_SELECTION.json",
    ]
    for fold in FOLDS:
        directory = prior.split_dir(args, fold)
        files.extend(
            [
                directory / "label_train.csv",
                directory / "label_val.csv",
                directory / "label_test.csv",
                prior.target_path(args, fold),
            ]
        )
    return files


def stage_freeze(args: argparse.Namespace) -> None:
    if list((args.output_root / "confirm").glob("**/test_summary.json")):
        raise RuntimeError("Cannot freeze after refinement test results exist")
    files = frozen_files(args)
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen files: {}".format(missing))
    selection = read_json(args.output_root / "PILOT_SELECTION.json")
    payload = {
        "frozen_at": now(),
        "protocol": (
            "iterative exploratory reuse of previously examined grouped folds; "
            "not eligible as independent confirmation"
        ),
        "outer_test_used_for_this_stage_candidate_selection": False,
        "prior_outer_fold_results_had_been_examined_before_design": True,
        "pilot": selection,
        "confirm_groups": CONFIRM_GROUPS,
        "folds": FOLDS,
        "seeds": SEEDS,
        "pilot_epochs": args.pilot_epochs,
        "confirm_epochs": args.confirm_epochs,
        "sha256": {
            str(path.relative_to(args.project_root)): prior.sha256(path) for path in files
        },
    }
    write_json(args.output_root / "FROZEN_SELECTION.json", payload)
    print("[FROZEN EXPLORATORY PROTOCOL]")


def confirm_variant(group: str, fold: int, winners: Dict[str, str]) -> str:
    mapping = {
        "mean_selection_baseline": "base_mean_selection",
        "pobs_selection_baseline": "base_pobs_selection",
        "pobs_coral": "coral_pobs_selection",
        "pobs_isolated_pairwise": "isolated_pairwise_pobs",
    }
    if group == "selected_advanced_refinement":
        return winners[str(fold)]
    return mapping[group]


def stage_confirm(args: argparse.Namespace) -> None:
    frozen_path = args.output_root / "FROZEN_SELECTION.json"
    if not frozen_path.exists():
        raise FileNotFoundError("Freeze before confirm")
    frozen = read_json(frozen_path)
    winners = {str(key): str(value) for key, value in frozen["pilot"]["winner_by_fold"].items()}
    for fold in args.folds:
        expected_rows = len(pd.read_csv(prior.split_dir(args, fold) / "label_test.csv"))
        for group in CONFIRM_GROUPS:
            variant = confirm_variant(group, fold, winners)
            for seed in SEEDS:
                run_dir = args.output_root / "confirm" / "fold_{}".format(fold) / group / "seed{}".format(seed)
                if prior.confirm_complete(run_dir, expected_rows):
                    print("[SKIP] confirm fold={} model={} seed={}".format(fold, group, seed))
                    continue
                command = trainer_command(
                    args,
                    fold,
                    variant,
                    group,
                    seed,
                    args.confirm_epochs,
                    args.confirm_fine_tune_start,
                    run_dir,
                    True,
                )
                run_command(command, args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "pilot", "select", "freeze", "confirm"])
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--splits-root", default=prior.DEFAULT_SPLITS)
    parser.add_argument("--grouped-root", default=prior.DEFAULT_GROUPED)
    parser.add_argument("--output-root", default="outputs/textile_target_refinement_20260804")
    parser.add_argument("--before-dir", default=prior.DEFAULT_BEFORE)
    parser.add_argument("--after-dir", default=prior.DEFAULT_AFTER)
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    parser.add_argument("--pilot-epochs", type=int, default=30)
    parser.add_argument("--pilot-fine-tune-start", type=int, default=20)
    parser.add_argument("--confirm-epochs", type=int, default=50)
    parser.add_argument("--confirm-fine-tune-start", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pair-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
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
