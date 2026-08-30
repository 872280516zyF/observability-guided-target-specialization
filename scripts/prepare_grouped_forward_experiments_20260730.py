#!/usr/bin/env python3
"""Prepare Pix2PixHD configs for before-image-grouped outer folds."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.metadata import load_table  # noqa: E402


PARAMETER_ORDER = ["frequency", "pulse_width", "speed", "dpi"]


def clean_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def identifier_set(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame.columns:
        raise ValueError(f"Missing grouped identifier column {column!r}")
    return set(frame[column].map(clean_id))


def validate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    output = {}
    for identifier in ["sample_id", "before_id"]:
        sets = {
            "train": identifier_set(train, identifier),
            "val": identifier_set(val, identifier),
            "test": identifier_set(test, identifier),
        }
        for left, right in [
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        ]:
            overlap = sets[left] & sets[right]
            output[f"{identifier}_{left}_{right}_overlap"] = len(overlap)
            if overlap:
                raise RuntimeError(
                    f"{identifier} overlap in {left}/{right}: "
                    f"{sorted(overlap)[:10]}"
                )
    if "877" in (
        identifier_set(train, "sample_id")
        | identifier_set(val, "sample_id")
        | identifier_set(test, "sample_id")
    ):
        raise RuntimeError("Unverified sample 877 is present.")
    return output


def build_stats(
    train: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    stats = {}
    for column in PARAMETER_ORDER:
        if column not in train.columns:
            raise ValueError(f"Missing parameter column {column!r}")
        values = pd.to_numeric(train[column], errors="raise")
        standard_deviation = float(values.std(ddof=0))
        if standard_deviation <= 0:
            raise ValueError(f"Non-positive training std for {column}")
        stats[column] = {
            "mean": float(values.mean()),
            "std": standard_deviation,
        }
    return stats


def relative(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root)).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="configs/forward_model_5090_l1_proxy.yaml",
    )
    parser.add_argument("--splits-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="42,52,62")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_config_path = (
        Path(args.base_config)
        if Path(args.base_config).is_absolute()
        else project_root / args.base_config
    ).resolve()
    splits_root = (
        Path(args.splits_root)
        if Path(args.splits_root).is_absolute()
        else project_root / args.splits_root
    ).resolve()
    output_dir = (
        Path(args.output_dir)
        if Path(args.output_dir).is_absolute()
        else project_root / args.output_dir
    ).resolve()
    run_root = (
        Path(args.run_root)
        if Path(args.run_root).is_absolute()
        else project_root / args.run_root
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    with base_config_path.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    if str(base_config.get("model", {}).get("type", "")).lower() != "pix2pixhd":
        raise ValueError("Grouped forward interpretation requires Pix2PixHD.")

    configurations = []
    fold_rows = []
    for fold in folds:
        split_dir = splits_root / f"fold_{fold}"
        paths = {
            "train": split_dir / "label_train.csv",
            "val": split_dir / "label_val.csv",
            "test": split_dir / "label_test.csv",
        }
        tables = {name: load_table(path) for name, path in paths.items()}
        overlap = validate(tables["train"], tables["val"], tables["test"])
        stats = build_stats(tables["train"])
        fold_config_dir = output_dir / f"fold_{fold}"
        fold_config_dir.mkdir(parents=True, exist_ok=True)
        stats_path = fold_config_dir / "label_stats_train_only.yaml"
        stats_path.write_text(
            yaml.safe_dump(stats, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        fold_rows.append(
            {
                "outer_fold": fold,
                "train_samples": len(tables["train"]),
                "val_samples": len(tables["val"]),
                "test_samples": len(tables["test"]),
                "train_groups": tables["train"]["before_id"].nunique(),
                "val_groups": tables["val"]["before_id"].nunique(),
                "test_groups": tables["test"]["before_id"].nunique(),
                **overlap,
            }
        )
        for seed in seeds:
            config = deepcopy(base_config)
            config["seed"] = seed
            config["data"]["train_manifest"] = relative(
                paths["train"], project_root
            )
            config["data"]["val_manifest"] = relative(
                paths["val"], project_root
            )
            config["data"]["test_manifest"] = relative(
                paths["test"], project_root
            )
            config["data"]["label_stats"] = relative(
                stats_path, project_root
            )
            checkpoint_dir = (
                run_root / f"fold_{fold}" / f"seed{seed}" / "checkpoints"
            )
            config["trainer"]["save_dir"] = relative(
                checkpoint_dir, project_root
            )
            config["trainer"].pop("pretrained_generator_path", None)
            config["grouped_outer_protocol"] = {
                "policy": "before_image_grouped_outer_cv",
                "outer_fold": fold,
                "group_column": "before_id",
                "sample_877_excluded": True,
                "test_is_used_during_training": False,
                "label_stats_source": "inner_training_partition_only",
            }
            config_path = (
                fold_config_dir
                / f"forward_pix2pixhd_grouped_fold{fold}_seed{seed}.yaml"
            )
            config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            configurations.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "config": str(config_path),
                    "checkpoint_dir": str(checkpoint_dir),
                }
            )
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "grouped_forward_fold_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_config": str(base_config_path),
        "splits_root": str(splits_root),
        "run_root": str(run_root),
        "folds": folds,
        "seeds": seeds,
        "configurations": configurations,
        "note": (
            "Forward outputs are model-based consistency evidence. They are "
            "not physical validation."
        ),
    }
    (output_dir / "grouped_forward_config_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
