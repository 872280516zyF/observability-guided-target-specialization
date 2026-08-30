#!/usr/bin/env python3
"""Grouped outer-CV factorial ablation for inverse-model inputs and augmentation.

The eight conditions are fixed in advance: four input representations crossed
with no augmentation versus weak paired augmentation. Training sees only the
inner train/validation split. Each frozen checkpoint is evaluated by a separate
process on the corresponding outer-test groups.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DEFAULT = CURRENT_DIR.parent
PARAMETERS = ["frequency", "pulse_width", "speed", "dpi"]


@dataclass(frozen=True)
class Condition:
    condition_id: str
    input_family: str
    input_mode: str
    fusion: str
    augmentation_mode: str


def make_conditions() -> list[Condition]:
    base = [
        ("after_only", "after_only", "concat_diff"),
        ("before_only", "before_only", "concat_diff"),
        ("absdiff_only", "dual", "input_absdiff"),
        ("dual_shared_absdiff", "dual", "concat_absdiff"),
    ]
    return [
        Condition(
            condition_id=f"{family}_{'noaug' if aug == 'none' else aug}",
            input_family=family,
            input_mode=input_mode,
            fusion=fusion,
            augmentation_mode=aug,
        )
        for family, input_mode, fusion in base
        for aug in ("none", "weak")
    ]


CONDITIONS = make_conditions()
CONDITION_MAP = {item.condition_id: item for item in CONDITIONS}


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["audit", "run", "aggregate", "package", "all"],
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT_DEFAULT))
    parser.add_argument(
        "--splits-root",
        default="data/images3/grouped_outer_cv_20260730",
    )
    parser.add_argument("--before-dir", default="data/images3/before")
    parser.add_argument("--after-dir", default="data/images3/after")
    parser.add_argument(
        "--output-root",
        default="outputs/grouped_input_augmentation_ablation_20260731",
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_command(command: list[str], dry_run: bool) -> None:
    print("[CMD]", " ".join(str(item) for item in command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def run_parent(output_root: Path, fold: int, condition_id: str, seed: int) -> Path:
    return output_root / f"fold_{fold}" / "runs" / condition_id / f"seed{seed}"


def audit_splits(args: argparse.Namespace, root: Path, output_root: Path) -> None:
    splits_root = resolve(root, args.splits_root)
    audit_rows = []
    all_fold_universes: list[set[str]] = []
    outer_test_ids: list[str] = []
    for fold in parse_int_list(args.folds):
        split_dir = splits_root / f"fold_{fold}"
        frames = {
            split: pd.read_csv(split_dir / f"label_{split}.csv")
            for split in ("train", "val", "test")
        }
        for split, frame in frames.items():
            missing = {"sample_id", "before_id"} - set(frame.columns)
            if missing:
                raise ValueError(f"fold {fold} {split} missing columns: {missing}")
            if frame["sample_id"].astype(str).duplicated().any():
                raise ValueError(f"fold {fold} {split} has duplicate sample_id")

        sample_sets = {
            name: set(frame["sample_id"].astype(str))
            for name, frame in frames.items()
        }
        group_sets = {
            name: set(frame["before_id"].astype(str))
            for name, frame in frames.items()
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if sample_sets[left] & sample_sets[right]:
                raise ValueError(f"sample overlap: fold {fold} {left}/{right}")
            if group_sets[left] & group_sets[right]:
                raise ValueError(f"before_id overlap: fold {fold} {left}/{right}")

        universe = set().union(*sample_sets.values())
        all_fold_universes.append(universe)
        outer_test_ids.extend(frames["test"]["sample_id"].astype(str).tolist())
        audit_rows.append(
            {
                "outer_fold": fold,
                "train_samples": len(frames["train"]),
                "val_samples": len(frames["val"]),
                "test_samples": len(frames["test"]),
                "train_before_groups": len(group_sets["train"]),
                "val_before_groups": len(group_sets["val"]),
                "test_before_groups": len(group_sets["test"]),
                "sample_overlap": 0,
                "before_id_overlap": 0,
            }
        )

    reference = all_fold_universes[0]
    if any(universe != reference for universe in all_fold_universes[1:]):
        raise ValueError("Fold universes are not identical")
    if len(outer_test_ids) != len(set(outer_test_ids)):
        raise ValueError("An outer-test sample occurs in more than one fold")
    if set(outer_test_ids) != reference:
        raise ValueError("Outer-test folds do not cover the full sample universe")

    audit_dir = output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(
        audit_dir / "split_audit.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "status": "PASS",
        "selection_scope": "inner train/validation only",
        "outer_test_scope": "evaluation-only after checkpoint freezing",
        "paired_augmentation": True,
        "conditions": [asdict(item) for item in CONDITIONS],
        "num_outer_folds": len(parse_int_list(args.folds)),
        "num_unique_samples": len(reference),
        "outer_test_coverage_once": True,
        "sample_overlap": 0,
        "before_id_overlap": 0,
        "caveat": (
            "The grouped outer splits were examined in earlier development; "
            "this is leakage-safe programmatically but is secondary evidence."
        ),
    }
    (audit_dir / "split_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_all(args: argparse.Namespace, root: Path, output_root: Path) -> None:
    splits_root = resolve(root, args.splits_root)
    before_dir = resolve(root, args.before_dir)
    after_dir = resolve(root, args.after_dir)
    train_script = root / "scripts" / "run_inverse_experiment.py"
    eval_script = root / "scripts" / "evaluate_input_condition_checkpoint_20260731.py"

    for fold in parse_int_list(args.folds):
        split_dir = splits_root / f"fold_{fold}"
        for condition in CONDITIONS:
            for seed in parse_int_list(args.seeds):
                parent = run_parent(output_root, fold, condition.condition_id, seed)
                train_dir = parent / "train"
                checkpoint = train_dir / "best_checkpoint.pth"
                train_summary = train_dir / "summary.json"
                if not (checkpoint.exists() and train_summary.exists()):
                    command = [
                        sys.executable,
                        str(train_script),
                        "--train-csv", str(split_dir / "label_train.csv"),
                        "--val-csv", str(split_dir / "label_val.csv"),
                        "--before-dir", str(before_dir),
                        "--after-dir", str(after_dir),
                        "--output-dir", str(parent),
                        "--run-name", "train",
                        "--seed", str(seed),
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--num-workers", str(args.num_workers),
                        "--img-size", str(args.img_size),
                        "--lr", str(args.lr),
                        "--weight-decay", str(args.weight_decay),
                        "--backbone", "resnet18",
                        "--input-mode", condition.input_mode,
                        "--fusion", condition.fusion,
                        "--head-type", "bn",
                        # The paired images are not pixel-registered and often
                        # have different aspect ratios. Stretching both to the
                        # same grid avoids condition-specific padding artifacts
                        # in the difference-only representation.
                        "--resize-mode", "stretch",
                        "--augmentation-mode", condition.augmentation_mode,
                        "--selection-metric", "val_mean_mape",
                    ]
                    # Deliberately no --test-csv here.
                    run_command(command, args.dry_run)
                else:
                    print(f"[SKIP train] {train_dir}", flush=True)

                test_dir = parent / "outer_test"
                test_summary = test_dir / "test_summary.json"
                if not test_summary.exists():
                    command = [
                        sys.executable,
                        str(eval_script),
                        "--checkpoint", str(checkpoint),
                        "--test-csv", str(split_dir / "label_test.csv"),
                        "--before-dir", str(before_dir),
                        "--after-dir", str(after_dir),
                        "--output-dir", str(test_dir),
                        "--outer-fold", str(fold),
                        "--seed", str(seed),
                        "--condition-id", condition.condition_id,
                        "--batch-size", str(args.batch_size),
                        "--num-workers", str(args.num_workers),
                    ]
                    run_command(command, args.dry_run)
                else:
                    print(f"[SKIP eval] {test_dir}", flush=True)


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    result = {f"mape_{name}": float(frame[f"ape_{name}"].mean()) for name in PARAMETERS}
    result["mean_mape"] = float(frame["mean_ape"].mean())
    return result


def median_across_seeds(frame: pd.DataFrame, expected_seeds: int) -> pd.DataFrame:
    rows = []
    metadata = ["sample_id", "before_id", "pattern_id", "batch_id", "outer_fold"]
    for sample_id, group in frame.groupby("sample_id", sort=False):
        if group["seed"].nunique() != expected_seeds:
            raise ValueError(f"sample {sample_id} missing one or more seeds")
        item = {column: group.iloc[0][column] for column in metadata if column in group}
        for name in PARAMETERS:
            true_values = group[f"true_{name}"].to_numpy(float)
            if not np.allclose(true_values, true_values[0]):
                raise ValueError(f"inconsistent true_{name} for sample {sample_id}")
            true_value = float(true_values[0])
            pred_value = float(np.median(group[f"pred_{name}"].to_numpy(float)))
            abs_error = abs(pred_value - true_value)
            item[f"true_{name}"] = true_value
            item[f"pred_{name}"] = pred_value
            item[f"abs_err_{name}"] = abs_error
            item[f"ape_{name}"] = abs_error / max(abs(true_value), 1e-6) * 100.0
        item["mean_ape"] = float(np.mean([item[f"ape_{name}"] for name in PARAMETERS]))
        rows.append(item)
    return pd.DataFrame(rows)


def bootstrap_ci(delta: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(delta)
    estimates = np.empty(iterations, dtype=float)
    for idx in range(iterations):
        estimates[idx] = float(delta[rng.integers(0, n, size=n)].mean())
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[original_index]))
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()


def paired_comparisons(
    median_frames: dict[str, pd.DataFrame], bootstrap_iters: int
) -> pd.DataFrame:
    contrasts: list[tuple[str, str, str]] = []
    for family in ("after_only", "before_only", "absdiff_only", "dual_shared_absdiff"):
        contrasts.append((f"{family}: no augmentation - weak", f"{family}_noaug", f"{family}_weak"))
    for aug in ("noaug", "weak"):
        for family in ("before_only", "absdiff_only", "dual_shared_absdiff"):
            contrasts.append((f"{family} - after_only ({aug})", f"{family}_{aug}", f"after_only_{aug}"))

    rows = []
    for contrast_index, (label, left_id, right_id) in enumerate(contrasts):
        left = median_frames[left_id].copy()
        right = median_frames[right_id].copy()
        left["sample_id"] = left["sample_id"].astype(str)
        right["sample_id"] = right["sample_id"].astype(str)
        left = left.set_index("sample_id")
        right = right.set_index("sample_id")
        if set(left.index) != set(right.index):
            raise ValueError(f"sample mismatch for contrast: {label}")
        order = sorted(left.index)
        left = left.loc[order]
        right = right.loc[order]
        for metric in ["mean_ape", *[f"ape_{name}" for name in PARAMETERS]]:
            delta = left[metric].to_numpy(float) - right[metric].to_numpy(float)
            low, high = bootstrap_ci(delta, bootstrap_iters, 20260731 + contrast_index)
            try:
                from scipy.stats import wilcoxon

                p_value = float(wilcoxon(delta, zero_method="zsplit").pvalue)
            except Exception:
                p_value = float("nan")
            rows.append(
                {
                    "contrast": label,
                    "left_condition": left_id,
                    "right_condition": right_id,
                    "metric": metric,
                    "n_paired_samples": len(delta),
                    "mean_delta_left_minus_right": float(delta.mean()),
                    "median_delta_left_minus_right": float(np.median(delta)),
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    "wilcoxon_p_raw": p_value,
                    "negative_delta_favors": left_id,
                    "positive_delta_favors": right_id,
                }
            )
    result = pd.DataFrame(rows)
    finite_mask = np.isfinite(result["wilcoxon_p_raw"].to_numpy(float))
    result["wilcoxon_p_holm"] = np.nan
    if finite_mask.any():
        result.loc[finite_mask, "wilcoxon_p_holm"] = holm_adjust(
            result.loc[finite_mask, "wilcoxon_p_raw"].astype(float).tolist()
        )
    return result


def aggregate(args: argparse.Namespace, output_root: Path) -> None:
    seeds = parse_int_list(args.seeds)
    folds = parse_int_list(args.folds)
    summary_dir = output_root / "summary"
    median_dir = summary_dir / "seed_median_predictions"
    median_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    median_rows = []
    median_frames: dict[str, pd.DataFrame] = {}
    for condition in CONDITIONS:
        condition_frames = []
        for seed in seeds:
            seed_frames = []
            for fold in folds:
                path = run_parent(output_root, fold, condition.condition_id, seed) / "outer_test" / "test_predictions.csv"
                if not path.exists():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path)
                seed_frames.append(frame)
                condition_frames.append(frame)
            oof = pd.concat(seed_frames, ignore_index=True)
            if oof["sample_id"].astype(str).duplicated().any():
                raise ValueError(f"duplicate OOF sample: {condition.condition_id} seed{seed}")
            per_seed_rows.append(
                {
                    **asdict(condition),
                    "seed": seed,
                    "n_oof_samples": len(oof),
                    **prediction_metrics(oof),
                }
            )

        combined = pd.concat(condition_frames, ignore_index=True)
        median = median_across_seeds(combined, expected_seeds=len(seeds))
        median["condition_id"] = condition.condition_id
        median["input_family"] = condition.input_family
        median["augmentation_mode"] = condition.augmentation_mode
        median.to_csv(
            median_dir / f"{condition.condition_id}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        median_frames[condition.condition_id] = median
        median_rows.append(
            {
                **asdict(condition),
                "aggregation": "samplewise_median_across_seeds",
                "n_oof_samples": len(median),
                **prediction_metrics(median),
            }
        )

    per_seed = pd.DataFrame(per_seed_rows)
    median_summary = pd.DataFrame(median_rows).sort_values("mean_mape")
    comparisons = paired_comparisons(median_frames, args.bootstrap_iters)
    per_seed.to_csv(summary_dir / "per_seed_oof_summary.csv", index=False, encoding="utf-8-sig")
    median_summary.to_csv(summary_dir / "seed_median_oof_summary.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(summary_dir / "paired_comparisons.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "status": "complete",
        "design": "4 input families x 2 augmentation modes x grouped 5-fold outer CV x 3 seeds",
        "conditions": [asdict(item) for item in CONDITIONS],
        "folds": folds,
        "seeds": seeds,
        "training_runs": len(CONDITIONS) * len(folds) * len(seeds),
        "selection_metric": "inner-validation mean MAPE",
        "outer_test_use": "evaluation-only; all prespecified conditions reported",
        "seed_aggregation": "samplewise median prediction before MAPE calculation",
        "paired_augmentation": True,
        "resize_mode": "stretch for every condition",
        "primary_outputs": {
            "seed_median_oof_summary": str(summary_dir / "seed_median_oof_summary.csv"),
            "per_seed_oof_summary": str(summary_dir / "per_seed_oof_summary.csv"),
            "paired_comparisons": str(summary_dir / "paired_comparisons.csv"),
        },
        "interpretation_caveat": (
            "These outer groups were available during earlier manuscript development. "
            "Treat this rerun as secondary evidence until confirmed on newly acquired groups."
        ),
    }
    (summary_dir / "aggregate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(median_summary.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def package_results(root: Path, output_root: Path) -> Path:
    archive = root / "grouped_input_augmentation_ablation_20260731_no_weights.tar.gz"
    script_names = [
        "run_inverse_experiment.py",
        "evaluate_input_condition_checkpoint_20260731.py",
        "run_grouped_input_augmentation_ablation_20260731.py",
        "run_5090_grouped_input_augmentation_ablation_20260731.sh",
    ]

    def filter_weights(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        suffix = Path(info.name).suffix.lower()
        return None if suffix in {".pth", ".pt", ".ckpt"} else info

    with tarfile.open(archive, "w:gz") as handle:
        handle.add(
            output_root,
            arcname=str(output_root.relative_to(root)),
            filter=filter_weights,
        )
        for name in script_names:
            path = root / "scripts" / name
            if path.exists():
                handle.add(path, arcname=f"scripts/{name}")
        readme = root / "GROUPED_INPUT_AUGMENTATION_ABLATION_5090_README_20260731.md"
        if readme.exists():
            handle.add(readme, arcname=readme.name)
    print(f"[ARCHIVE] {archive}")
    return archive


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    output_root = resolve(root, args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.stage in {"audit", "all"}:
        audit_splits(args, root, output_root)
    if args.stage in {"run", "all"}:
        run_all(args, root, output_root)
    if args.stage in {"aggregate", "all"}:
        aggregate(args, output_root)
    if args.stage in {"package", "all"}:
        package_results(root, output_root)


if __name__ == "__main__":
    main()
