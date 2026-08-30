#!/usr/bin/env python3
"""Run the grouped outer-CV inverse suite with resumable stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PARAMETERS = ["frequency", "pulse_width", "speed", "dpi"]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    runner: str
    variant: str = ""
    expert_target: str = ""


def run_command(command: list[str], dry_run: bool) -> None:
    printable = " ".join(f'"{item}"' if " " in item else item for item in command)
    print(f"[CMD] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def read_selected_target(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = str(payload["selected_target"])
    if target not in PARAMETERS:
        raise ValueError(f"Unexpected selected target {target!r} in {path}")
    return target


def sensitivity_complete(fold_output: Path) -> bool:
    return (
        fold_output
        / "observability"
        / "grouped"
        / "selected_observability_target.json"
    ).exists() and (
        fold_output
        / "observability"
        / "grouped"
        / "leave_one_group_out.csv"
    ).exists()


def run_observability(
    project_root: Path,
    fold: int,
    split_dir: Path,
    fold_output: Path,
    after_dir: Path,
    bootstrap: int,
    seed: int,
    dry_run: bool,
) -> None:
    raw_output = fold_output / "observability" / "pair_metrics"
    grouped_output = fold_output / "observability" / "grouped"
    analyze = project_root / "scripts" / "analyze_parameter_image_sensitivity.py"
    summarize = (
        project_root
        / "scripts"
        / "summarize_grouped_observability_20260730.py"
    )
    pair_metrics = raw_output / "one_factor_pair_image_metrics.csv"
    if not pair_metrics.exists() or dry_run:
        run_command(
            [
                sys.executable,
                str(analyze),
                "--label-csv",
                str(split_dir / "label_outer_train_all.csv"),
                "--after-dir",
                str(after_dir),
                "--output-dir",
                str(raw_output),
                "--img-size",
                "224",
                "--max-pairs-per-param",
                "2000",
                "--balanced-bootstrap-iters",
                "1",
                "--bootstrap-seed",
                str(seed + fold),
            ],
            dry_run,
        )
    run_command(
        [
            sys.executable,
            str(summarize),
            "--pair-metrics",
            str(pair_metrics),
            "--output-dir",
            str(grouped_output),
            "--bootstrap",
            str(bootstrap),
            "--seed",
            str(seed + fold),
        ],
        dry_run,
    )


def model_specs(stage: str, selected_target: str) -> list[ModelSpec]:
    if stage == "core":
        return [
            ModelSpec("plain_cnn", "plain"),
            ModelSpec(
                "shared_head_resnet",
                "branch",
                variant="target_shared4_bn",
            ),
            ModelSpec(
                "selected_texture_expert",
                "branch",
                variant="target_integrated_texture_expert_any_bn",
                expert_target=selected_target,
            ),
            ModelSpec(
                "selected_nonguided_equal_capacity",
                "branch",
                variant="target_integrated_nonguided_control_bn",
                expert_target=selected_target,
            ),
        ]
    if stage == "controls":
        return [
            ModelSpec(
                f"texture_expert_{target}",
                "branch",
                variant="target_integrated_texture_expert_any_bn",
                expert_target=target,
            )
            for target in PARAMETERS
            if target != selected_target
        ]
    if stage == "traditional":
        return [
            ModelSpec("resnet_rf", "traditional", variant="resnet_rf"),
            ModelSpec(
                "resnet_xgboost",
                "traditional",
                variant="resnet_xgboost",
            ),
        ]
    raise ValueError(stage)


def training_command(
    project_root: Path,
    spec: ModelSpec,
    split_dir: Path,
    run_parent: Path,
    run_name: str,
    before_dir: Path,
    after_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    workers: int,
    selection_metric: str,
) -> list[str]:
    if spec.runner == "traditional":
        return [
            sys.executable,
            str(
                project_root
                / "scripts"
                / "run_traditional_inverse_baselines.py"
            ),
            "--train-csv",
            str(split_dir / "label_train.csv"),
            "--val-csv",
            str(split_dir / "label_val.csv"),
            "--test-csv",
            str(split_dir / "label_test.csv"),
            "--before-dir",
            str(before_dir),
            "--after-dir",
            str(after_dir),
            "--output-dir",
            str(run_parent.parent),
            "--variants",
            spec.variant,
            "--seed",
            str(seed),
            "--img-size",
            "224",
            "--batch-size",
            "64",
            "--num-workers",
            str(workers),
            "--device",
            "cuda",
            "--image-role",
            "after",
            "--feature-cache-dir",
            str(run_parent.parent.parent / "traditional_feature_cache"),
            "--grouped-layout",
        ]
    common = [
        "--train-csv",
        str(split_dir / "label_train.csv"),
        "--val-csv",
        str(split_dir / "label_val.csv"),
        "--test-csv",
        str(split_dir / "label_test.csv"),
        "--before-dir",
        str(before_dir),
        "--after-dir",
        str(after_dir),
        "--output-dir",
        str(run_parent),
        "--run-name",
        run_name,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(workers),
        "--img-size",
        "224",
        "--lr",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--selection-metric",
        selection_metric,
    ]
    if spec.runner == "plain":
        return [
            sys.executable,
            str(project_root / "scripts" / "run_plain_cnn_inverse_baseline.py"),
            *common,
            "--width",
            "32",
        ]
    command = [
        sys.executable,
        str(project_root / "scripts" / "run_dpi_branch_ablation.py"),
        "--variant",
        spec.variant,
        *common,
        "--backbone",
        "resnet18",
        "--hidden-dim",
        "256",
        "--dropout",
        "0.3",
        "--loss-type",
        "smooth_l1",
        "--frequency-loss-weight",
        "1.0",
        "--pulse-width-loss-weight",
        "1.0",
        "--speed-loss-weight",
        "1.0",
        "--dpi-loss-weight",
        "1.0",
        "--texture-dim",
        "64",
    ]
    if spec.expert_target:
        command.extend(["--expert-target", spec.expert_target])
    return command


def run_model(
    project_root: Path,
    spec: ModelSpec,
    fold: int,
    selected_target: str,
    split_dir: Path,
    fold_output: Path,
    before_dir: Path,
    after_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    workers: int,
    selection_metric: str,
    calibration_alpha: float,
    resume: bool,
    dry_run: bool,
) -> None:
    run_parent = fold_output / "runs" / spec.model_id
    run_name = f"seed{seed}"
    run_dir = run_parent / run_name
    summary_path = run_dir / "summary.json"
    calibrated_path = run_dir / "calibrated" / "test_predictions.csv"
    if resume and summary_path.exists() and calibrated_path.exists():
        print(f"[SKIP] complete: fold={fold} model={spec.model_id} seed={seed}")
        return
    run_parent.mkdir(parents=True, exist_ok=True)
    run_spec = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "outer_fold": fold,
        "model_id": spec.model_id,
        "runner": spec.runner,
        "variant": spec.variant,
        "expert_target": spec.expert_target,
        "selected_observability_target": selected_target,
        "seed": seed,
        "selection_metric": selection_metric,
        "calibration": "uniform_inner_validation_ridge",
        "calibration_alpha": calibration_alpha,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "grouped_run_spec.json").write_text(
        json.dumps(run_spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not summary_path.exists() or not resume:
        run_command(
            training_command(
                project_root,
                spec,
                split_dir,
                run_parent,
                run_name,
                before_dir,
                after_dir,
                seed,
                epochs,
                batch_size,
                workers,
                selection_metric,
            ),
            dry_run,
        )
    calibrator = (
        project_root
        / "scripts"
        / "calibrate_grouped_outer_run_20260730.py"
    )
    run_command(
        [
            sys.executable,
            str(calibrator),
            "--run-dir",
            str(run_dir),
            "--alpha",
            str(calibration_alpha),
        ],
        dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--splits-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--stage",
        choices=["sensitivity", "core", "controls", "traditional", "all"],
        default="all",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--observability-bootstrap", type=int, default=10000)
    parser.add_argument("--observability-seed", type=int, default=20260730)
    parser.add_argument("--calibration-alpha", type=float, default=0.01)
    parser.add_argument(
        "--selection-metric",
        choices=["val_mean_mape"],
        default="val_mean_mape",
        help="Fixed for all models in the fair grouped suite.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    splits_root = Path(args.splits_root).resolve()
    output_root = Path(args.output_root).resolve()
    before_dir = project_root / "data" / "images3" / "before"
    after_dir = project_root / "data" / "images3" / "after"
    for path in [project_root, splits_root, before_dir, after_dir]:
        if not path.exists():
            raise FileNotFoundError(path)
    if not (splits_root / "split_manifest.json").exists():
        raise FileNotFoundError(splits_root / "split_manifest.json")
    output_root.mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        split_dir = splits_root / f"fold_{fold}"
        fold_output = output_root / f"fold_{fold}"
        if not split_dir.exists():
            raise FileNotFoundError(split_dir)
        if not sensitivity_complete(fold_output):
            run_observability(
                project_root,
                fold,
                split_dir,
                fold_output,
                after_dir,
                args.observability_bootstrap,
                args.observability_seed,
                args.dry_run,
            )
        if args.stage == "sensitivity":
            continue
        selected_path = (
            fold_output
            / "observability"
            / "grouped"
            / "selected_observability_target.json"
        )
        if args.dry_run and not selected_path.exists():
            print(
                f"[DRY-RUN] cannot resolve selected target for fold {fold}; "
                "assuming dpi for command preview."
            )
            selected_target = "dpi"
        else:
            selected_target = read_selected_target(selected_path)
        stages = (
            ["core", "controls", "traditional"]
            if args.stage == "all"
            else [args.stage]
        )
        for stage in stages:
            for spec in model_specs(stage, selected_target):
                for seed in args.seeds:
                    run_model(
                        project_root,
                        spec,
                        fold,
                        selected_target,
                        split_dir,
                        fold_output,
                        before_dir,
                        after_dir,
                        seed,
                        args.epochs,
                        args.batch_size,
                        args.num_workers,
                        args.selection_metric,
                        args.calibration_alpha,
                        args.resume,
                        args.dry_run,
                    )
    print(f"[DONE] grouped outer suite stage={args.stage}: {output_root}")


if __name__ == "__main__":
    main()
