#!/usr/bin/env python3
"""Two-stage orchestration for textile module attribution.

Stages are intentionally separate.  ``pilot`` never receives an outer-test
path.  ``freeze`` hashes code, splits and observability selections before any
``confirm`` stage is accepted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_textile_module_attribution_20260802 import (  # noqa: E402
    PARAM_NAMES,
    PILOT_VARIANTS,
    REFERENCE_FULL_PARAMETER_COUNT,
    build_model,
    count_trainable_parameters,
)


DEFAULT_SPLITS = "data/images3/grouped_outer_cv_20260730"
DEFAULT_GROUPED_RESULTS = "outputs/grouped_outer_cv_20260730"
DEFAULT_OUTPUT = "outputs/textile_module_attribution_20260802"
DEFAULT_BEFORE = "data/images3/before"
DEFAULT_AFTER = "data/images3/after"
FOLDS = [0, 1, 2, 3, 4]
SEEDS = [42, 52, 62]
ATTENTION_TEXTURE_CANDIDATES = [
    "specialist_head_vector_gate",
    "residual_adapter_mtan",
    "residual_adapter_deepten",
    "residual_adapter_mtan_deepten",
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_target_path(grouped_root: Path, fold: int) -> Path:
    return (
        grouped_root
        / "fold_{}".format(fold)
        / "observability"
        / "grouped"
        / "selected_observability_target.json"
    )


def selected_target(grouped_root: Path, fold: int) -> str:
    path = selected_target_path(grouped_root, fold)
    if not path.exists():
        raise FileNotFoundError(
            "Missing fold-contained observability selection: {}".format(path)
        )
    payload = read_json(path)
    if payload.get("selection_scope") != (
        "outer-training groups only; outer-test groups were excluded"
    ):
        raise RuntimeError("Unexpected observability selection scope in {}".format(path))
    target = str(payload["selected_target"])
    if target not in PARAM_NAMES:
        raise ValueError("Unknown selected target {}".format(target))
    return target


def split_dir(splits_root: Path, fold: int) -> Path:
    return splits_root / "fold_{}".format(fold)


def audit_splits(splits_root: Path, grouped_root: Path) -> Dict[str, object]:
    audit_rows = []
    outer_ids: List[str] = []
    outer_groups: List[str] = []
    for fold in FOLDS:
        directory = split_dir(splits_root, fold)
        frames = {}
        for split in ["train", "val", "test"]:
            path = directory / "label_{}.csv".format(split)
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            required = {"sample_id", "before_id"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError("{} missing {}".format(path, sorted(missing)))
            if frame["sample_id"].astype(str).duplicated().any():
                raise ValueError("Duplicate sample_id in {}".format(path))
            frames[split] = frame
            audit_rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "samples": int(len(frame)),
                    "before_groups": int(frame["before_id"].nunique()),
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
                    "Fold {} overlap {}/{}: samples={} groups={}".format(
                        fold, left, right, len(sample_overlap), len(group_overlap)
                    )
                )
        outer_ids.extend(frames["test"]["sample_id"].astype(str).tolist())
        outer_groups.extend(frames["test"]["before_id"].astype(str).tolist())
        selected_target(grouped_root, fold)

    counts = pd.Series(outer_ids).value_counts()
    if len(counts) != 1240 or not counts.eq(1).all():
        raise RuntimeError(
            "Expected 1,240 unique OOF samples exactly once; got {}".format(
                len(counts)
            )
        )
    if len(set(outer_groups)) != 29:
        raise RuntimeError(
            "Expected 29 outer-test before groups; got {}".format(
                len(set(outer_groups))
            )
        )
    if "877" in set(outer_ids):
        raise RuntimeError("Unverified sample 877 must remain excluded")
    return {
        "checked_at": now(),
        "rows": audit_rows,
        "unique_oof_samples": 1240,
        "unique_before_groups": 29,
        "zero_sample_overlap": True,
        "zero_before_group_overlap": True,
        "sample_877_excluded": True,
    }


def model_args(variant: str, target: str) -> argparse.Namespace:
    return argparse.Namespace(
        variant=variant,
        expert_target=target,
        no_pretrained=True,
        hidden_dim=256,
        dropout=0.3,
        texture_dim=64,
        residual_scale=0.25,
    )


def model_smoke(output_root: Path) -> Dict[str, object]:
    from scripts.train_textile_module_attribution_20260802 import build_model

    variants = list(PILOT_VARIANTS) + [
        "current_full_no_attention",
        "capacity_matched_dual_shared",
    ]
    rows = []
    for variant in variants:
        model = build_model(model_args(variant, "dpi"))
        model.eval()
        with torch.no_grad():
            output = model({"effect": torch.zeros(2, 3, 224, 224)})
        if tuple(output.shape) != (2, 4):
            raise RuntimeError("{} output shape {}".format(variant, output.shape))
        if not torch.isfinite(output).all():
            raise RuntimeError("{} produced non-finite output".format(variant))
        count = count_trainable_parameters(model)
        rows.append({"variant": variant, "parameter_count": count})
        if variant == "capacity_matched_dual_shared":
            gap = abs(count - REFERENCE_FULL_PARAMETER_COUNT) / float(
                REFERENCE_FULL_PARAMETER_COUNT
            )
            if gap > 0.005:
                raise RuntimeError("Capacity-matched control exceeds 0.5% gap")
        del model
        gc.collect()
    pd.DataFrame(rows).to_csv(
        output_root / "preflight_parameter_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {"model_forward_smoke": True, "models": rows}


def cuda_parallel_probe() -> Dict[str, object]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "parallel_two_processes_recommended": None,
            "note": "CUDA probe skipped; run on the 5090 before launching jobs.",
        }
    device = torch.device("cuda")
    peaks = []
    probe_error = None
    for variant in [
        "current_full_no_attention",
        "residual_adapter_mtan_deepten",
    ]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = None
        batch = None
        output = None
        try:
            from scripts.train_textile_module_attribution_20260802 import build_model

            model = build_model(model_args(variant, "dpi")).to(device).train()
            batch = {"effect": torch.zeros(32, 3, 224, 224, device=device)}
            output = model(batch)
            output.mean().backward()
            peak = int(torch.cuda.max_memory_allocated(device))
            peaks.append({"variant": variant, "one_step_peak_bytes": peak})
        except torch.cuda.OutOfMemoryError as error:
            probe_error = "{}: {}".format(variant, error)
            break
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            probe_error = "{}: {}".format(variant, error)
            break
        finally:
            del output, batch, model
            torch.cuda.empty_cache()
            gc.collect()
    total = int(torch.cuda.get_device_properties(device).total_memory)
    if probe_error is not None or not peaks:
        return {
            "cuda_available": True,
            "gpu": torch.cuda.get_device_name(0),
            "total_memory_bytes": total,
            "probe": peaks,
            "probe_error": probe_error,
            "parallel_two_processes_recommended": False,
            "note": (
                "A batch-32 probe exceeded available memory. Run one terminal; "
                "do not change the frozen batch size or protocol."
            ),
        }
    worst = max(row["one_step_peak_bytes"] for row in peaks)
    required = int(2.0 * worst * 1.35 + 4 * 1024**3)
    return {
        "cuda_available": True,
        "gpu": torch.cuda.get_device_name(0),
        "total_memory_bytes": total,
        "probe": peaks,
        "two_process_safety_estimate_bytes": required,
        "parallel_two_processes_recommended": bool(required <= total),
        "note": "Conservative estimate; continue to monitor nvidia-smi.",
    }


def run_command(command: Sequence[str], dry_run: bool = False) -> None:
    print("[CMD] {}".format(" ".join(str(item) for item in command)), flush=True)
    if not dry_run:
        subprocess.run(list(command), check=True)


def trainer_command(
    args: argparse.Namespace,
    fold: int,
    variant: str,
    model_id: str,
    expert_target: str,
    seed: int,
    epochs: int,
    run_dir: Path,
    include_test: bool,
) -> List[str]:
    directory = split_dir(args.splits_root, fold)
    command = [
        sys.executable,
        str(args.project_root / "scripts" / "train_textile_module_attribution_20260802.py"),
        "--variant",
        variant,
        "--expert-target",
        expert_target,
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


def validation_complete(path: Path) -> bool:
    summary = path / "validation_summary.json"
    if not summary.exists():
        return False
    payload = read_json(summary)
    return (
        payload.get("evaluation_scope") == "inner_validation_only"
        and payload.get("outer_test_was_supplied") is False
        and int(payload.get("num_val_samples", 0)) > 0
        and not (path / "outer_test" / "test_predictions.csv").exists()
    )


def confirm_complete(path: Path, expected_rows: int) -> bool:
    summary = path / "outer_test" / "test_summary.json"
    predictions = path / "outer_test" / "test_predictions.csv"
    if not summary.exists() or not predictions.exists():
        return False
    payload = read_json(summary)
    if payload.get("evaluation_scope") != "frozen_checkpoint_outer_test":
        return False
    frame = pd.read_csv(predictions)
    return len(frame) == expected_rows and frame["sample_id"].nunique() == expected_rows


def stage_pilot(args: argparse.Namespace) -> None:
    for fold in args.folds:
        target = selected_target(args.grouped_root, fold)
        for variant in PILOT_VARIANTS:
            run_dir = (
                args.output_root
                / "pilot"
                / "fold_{}".format(fold)
                / variant
                / "seed42"
            )
            if args.resume and validation_complete(run_dir):
                print("[SKIP] pilot fold={} variant={}".format(fold, variant))
                continue
            command = trainer_command(
                args,
                fold,
                variant,
                variant,
                target,
                42,
                args.pilot_epochs,
                run_dir,
                include_test=False,
            )
            if "--test-csv" in command:
                raise RuntimeError("Pilot command must not contain --test-csv")
            run_command(command, args.dry_run)


def pilot_score(summary: Dict[str, object], target: str) -> float:
    metrics = summary["best_eval"]
    parameter_metrics = metrics["param_mape_physical"]
    nonselected = [
        float(value)
        for name, value in parameter_metrics.items()
        if name != target
    ]
    return (
        float(parameter_metrics[target])
        + 0.25 * float(metrics["mape_physical"])
        + 0.10 * max(nonselected)
    )


def stage_select(args: argparse.Namespace) -> None:
    rows = []
    for fold in FOLDS:
        target = selected_target(args.grouped_root, fold)
        for variant in PILOT_VARIANTS:
            path = (
                args.output_root
                / "pilot"
                / "fold_{}".format(fold)
                / variant
                / "seed42"
                / "validation_summary.json"
            )
            if not path.exists():
                raise FileNotFoundError("Incomplete pilot: {}".format(path))
            payload = read_json(path)
            if payload.get("outer_test_was_supplied") is not False:
                raise RuntimeError("Pilot outer-test boundary violation: {}".format(path))
            rows.append(
                {
                    "outer_fold": fold,
                    "variant": variant,
                    "selected_target": target,
                    "pilot_seed": 42,
                    "pilot_score": pilot_score(payload, target),
                    "val_selected_target_mape": payload["best_eval"][
                        "param_mape_physical"
                    ][target],
                    "val_mean_mape": payload["best_eval"]["mape_physical"],
                    "parameter_count": payload["parameter_count"],
                }
            )
    frame = pd.DataFrame(rows)
    table_path = args.output_root / "pilot_inner_validation.csv"
    frame.to_csv(table_path, index=False, encoding="utf-8-sig")
    fold_equal = (
        frame.groupby("variant", as_index=False)
        .agg(
            mean_pilot_score=("pilot_score", "mean"),
            sd_pilot_score=("pilot_score", "std"),
            folds=("outer_fold", "nunique"),
        )
        .sort_values(["mean_pilot_score", "variant"])
    )
    candidate_rows = fold_equal.loc[
        fold_equal["variant"].isin(ATTENTION_TEXTURE_CANDIDATES)
    ]
    if len(candidate_rows) != len(ATTENTION_TEXTURE_CANDIDATES):
        raise RuntimeError("Pilot candidate table is incomplete")
    global_descriptive_winner = str(candidate_rows.iloc[0]["variant"])
    winner_by_fold = {}
    for fold in FOLDS:
        candidates = frame.loc[
            frame["outer_fold"].eq(fold)
            & frame["variant"].isin(ATTENTION_TEXTURE_CANDIDATES)
        ].sort_values(["pilot_score", "variant"])
        if len(candidates) != len(ATTENTION_TEXTURE_CANDIDATES):
            raise RuntimeError("Fold {} pilot candidates are incomplete".format(fold))
        winner_by_fold[str(fold)] = str(candidates.iloc[0]["variant"])
    fold_equal.to_csv(
        args.output_root / "pilot_candidate_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(
        args.output_root / "PILOT_SELECTION.json",
        {
            "selected_at": now(),
            "selection_scope": "fold-specific inner validation only",
            "outer_test_used": False,
            "objective": (
                "selected_target_mape + 0.25*mean_mape + "
                "0.10*max_nonselected_mape"
            ),
            "pilot_seed": 42,
            "winner_by_fold": winner_by_fold,
            "global_descriptive_winner_not_used_for_outer_test": (
                global_descriptive_winner
            ),
            "global_descriptive_ranking": fold_equal.to_dict(orient="records"),
        },
    )
    print("[SELECTED BY FOLD] {}".format(winner_by_fold))
    print(
        "[DESCRIPTIVE GLOBAL RANK 1; NOT USED FOR OUTER TEST] {}".format(
            global_descriptive_winner
        )
    )


def frozen_files(args: argparse.Namespace) -> List[Path]:
    files = [
        args.project_root / "scripts" / "train_textile_module_attribution_20260802.py",
        args.project_root / "scripts" / "run_textile_module_attribution_pipeline_20260802.py",
        args.project_root / "scripts" / "aggregate_textile_module_attribution_20260802.py",
        args.output_root / "PILOT_SELECTION.json",
    ]
    for fold in FOLDS:
        directory = split_dir(args.splits_root, fold)
        files.extend(
            [
                directory / "label_train.csv",
                directory / "label_val.csv",
                directory / "label_test.csv",
                selected_target_path(args.grouped_root, fold),
            ]
        )
    return files


def stage_freeze(args: argparse.Namespace) -> None:
    selection_path = args.output_root / "PILOT_SELECTION.json"
    if not selection_path.exists():
        raise FileNotFoundError("Run select before freeze")
    if list((args.output_root / "confirm").glob("**/test_summary.json")):
        raise RuntimeError("Cannot freeze after outer-test results already exist")
    files = frozen_files(args)
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen inputs: {}".format(missing))
    selection = read_json(selection_path)
    payload = {
        "frozen_at": now(),
        "protocol": "subsequent exploratory leakage-controlled internal validation",
        "outer_test_inspected_for_candidate_selection": False,
        "pilot": selection,
        "fixed_confirm_model_groups": [
            "specialist_head_plain",
            "residual_adapter_core",
            "selected_literature_winner",
            "current_full_no_attention",
            "capacity_matched_dual_shared",
            "winner_nonselected_placement",
        ],
        "folds": FOLDS,
        "seeds": SEEDS,
        "pilot_epochs": args.pilot_epochs,
        "confirm_epochs": args.confirm_epochs,
        "batch_size": args.batch_size,
        "checkpoint_selection": "inner-validation mean MAPE",
        "primary_output": "raw direct model output",
        "upgrade_rules": {
            "candidate_vs_current_full_pobs_ci_high_lt_zero": True,
            "candidate_vs_nonselected_pobs_ci_high_lt_zero": True,
            "overall_mean_ape_noninferiority_margin_pp": 0.5,
            "each_nonselected_mape_margin_pp": 1.0,
            "expected_samples": 1240,
            "expected_before_groups": 29,
        },
        "sha256": {str(path.relative_to(args.project_root)): sha256(path) for path in files},
    }
    write_json(args.output_root / "FROZEN_SELECTION.json", payload)
    print("[FROZEN] {}".format(args.output_root / "FROZEN_SELECTION.json"))


def verify_frozen(args: argparse.Namespace) -> Dict[str, object]:
    path = args.output_root / "FROZEN_SELECTION.json"
    if not path.exists():
        raise FileNotFoundError("Run freeze before confirm")
    payload = read_json(path)
    for relative, expected in payload["sha256"].items():
        source = args.project_root / relative
        actual = sha256(source)
        if actual != expected:
            raise RuntimeError(
                "Frozen input changed: {} expected={} actual={}".format(
                    source, expected, actual
                )
            )
    return payload


def nonselected_target(selected: str, seed: int) -> str:
    candidates = [name for name in PARAM_NAMES if name != selected]
    seed_index = {42: 0, 52: 1, 62: 2}[int(seed)]
    return candidates[seed_index]


def confirm_specs(winner: str, selected: str, seed: int) -> List[Dict[str, str]]:
    return [
        {
            "model_id": "specialist_head_plain",
            "variant": "specialist_head_plain",
            "expert_target": selected,
        },
        {
            "model_id": "residual_adapter_core",
            "variant": "residual_adapter_core",
            "expert_target": selected,
        },
        {
            "model_id": "selected_literature_winner",
            "variant": winner,
            "expert_target": selected,
        },
        {
            "model_id": "current_full_no_attention",
            "variant": "current_full_no_attention",
            "expert_target": selected,
        },
        {
            "model_id": "capacity_matched_dual_shared",
            "variant": "capacity_matched_dual_shared",
            "expert_target": selected,
        },
        {
            "model_id": "winner_nonselected_placement",
            "variant": winner,
            "expert_target": nonselected_target(selected, seed),
        },
    ]


def stage_confirm(args: argparse.Namespace) -> None:
    frozen = verify_frozen(args)
    winners = frozen["pilot"]["winner_by_fold"]
    for fold in args.folds:
        winner = str(winners[str(fold)])
        selected = selected_target(args.grouped_root, fold)
        expected_rows = len(
            pd.read_csv(split_dir(args.splits_root, fold) / "label_test.csv")
        )
        for seed in SEEDS:
            for spec in confirm_specs(winner, selected, seed):
                run_dir = (
                    args.output_root
                    / "confirm"
                    / "fold_{}".format(fold)
                    / spec["model_id"]
                    / "seed{}".format(seed)
                )
                if args.resume and confirm_complete(run_dir, expected_rows):
                    print(
                        "[SKIP] confirm fold={} model={} seed={}".format(
                            fold, spec["model_id"], seed
                        )
                    )
                    continue
                command = trainer_command(
                    args,
                    fold,
                    spec["variant"],
                    spec["model_id"],
                    spec["expert_target"],
                    seed,
                    args.confirm_epochs,
                    run_dir,
                    include_test=True,
                )
                run_command(command, args.dry_run)


def stage_preflight(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit = audit_splits(args.splits_root, args.grouped_root)
    write_json(args.output_root / "preflight_split_audit.json", audit)
    smoke = model_smoke(args.output_root)
    memory = cuda_parallel_probe()
    write_json(
        args.output_root / "preflight_manifest.json",
        {
            "completed_at": now(),
            "split_audit": audit,
            "model_smoke": smoke,
            "cuda_parallel_probe": memory,
        },
    )
    print(json.dumps(memory, ensure_ascii=False, indent=2))
    print("[PASS] preflight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["preflight", "pilot", "select", "freeze", "confirm"]
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--splits-root", default=DEFAULT_SPLITS)
    parser.add_argument("--grouped-root", default=DEFAULT_GROUPED_RESULTS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--before-dir", default=DEFAULT_BEFORE)
    parser.add_argument("--after-dir", default=DEFAULT_AFTER)
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    parser.add_argument("--pilot-epochs", type=int, default=30)
    parser.add_argument("--confirm-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
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
    invalid_folds = sorted(set(args.folds) - set(FOLDS))
    if invalid_folds:
        raise ValueError("Invalid folds: {}".format(invalid_folds))
    return args


def main() -> None:
    args = parse_args()
    if args.stage == "preflight":
        stage_preflight(args)
    elif args.stage == "pilot":
        stage_pilot(args)
    elif args.stage == "select":
        stage_select(args)
    elif args.stage == "freeze":
        stage_freeze(args)
    elif args.stage == "confirm":
        stage_confirm(args)


if __name__ == "__main__":
    main()
