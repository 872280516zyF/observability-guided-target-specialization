#!/usr/bin/env python3
"""Freeze provenance for a validation-selected locked-forward checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.metadata import load_table


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path_like: str | Path, project_root: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (project_root / path).resolve()


def clean_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def ids(path: Path, column: str = "sample_id") -> set[str]:
    table = load_table(path)
    if column not in table.columns:
        if column == "sample_id":
            column = table.columns[0]
        else:
            raise ValueError(f"Missing required split column {column!r}: {path}")
    return set(table[column].map(clean_id))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--start-manifest", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = resolve(args.config, project_root)
    checkpoint_path = resolve(args.checkpoint, project_root)
    start_manifest_path = resolve(args.start_manifest, project_root)
    training_log_path = resolve(args.training_log, project_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not start_manifest_path.exists():
        raise FileNotFoundError(start_manifest_path)
    if not training_log_path.exists():
        raise FileNotFoundError(training_log_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    protocol_name = (
        "grouped_outer_protocol"
        if "grouped_outer_protocol" in config
        else "locked_protocol"
    )
    protocol = config.get(protocol_name, {})
    accepted_policies = {
        "train_only_fit_val_only_selection_test_once",
        "before_image_grouped_outer_cv",
    }
    if protocol.get("policy") not in accepted_policies:
        raise RuntimeError(
            "Config is not a prepared locked/grouped-forward protocol config."
        )
    start_manifest = json.loads(start_manifest_path.read_text(encoding="utf-8"))
    if start_manifest.get("status") != "training_started_before_checkpoint_creation":
        raise RuntimeError("Invalid training-start manifest status.")
    if Path(start_manifest.get("expected_checkpoint", "")).resolve() != checkpoint_path.resolve():
        raise RuntimeError("Training-start manifest names a different checkpoint.")
    if (
        start_manifest.get("inputs", {})
        .get("config", {})
        .get("sha256")
        != sha256(config_path)
    ):
        raise RuntimeError("Config changed after the training-start manifest was written.")
    checkpoint_mtime_ns = checkpoint_path.stat().st_mtime_ns
    if checkpoint_mtime_ns < int(start_manifest["created_at_unix_ns"]):
        raise RuntimeError(
            "Checkpoint predates the training-start manifest; refusing to certify it."
        )
    data = config["data"]
    train_path = resolve(data["train_manifest"], project_root)
    val_path = resolve(data["val_manifest"], project_root)
    test_path = resolve(data["test_manifest"], project_root)
    label_stats_path = resolve(data["label_stats"], project_root)
    start_inputs = start_manifest.get("inputs", {})
    for name, path in {
        "train_csv": train_path,
        "val_csv": val_path,
        "test_csv": test_path,
        "label_stats": label_stats_path,
    }.items():
        if start_inputs.get(name, {}).get("sha256") != sha256(path):
            raise RuntimeError(f"{name} changed after training began.")
    if start_manifest.get("seed") != config.get("seed"):
        raise RuntimeError("Seed differs from the training-start manifest.")
    train_ids = ids(train_path)
    val_ids = ids(val_path)
    test_ids = ids(test_path)
    partition_audit = {
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "test_samples": len(test_ids),
        "overlap_train_val": len(train_ids & val_ids),
        "overlap_train_test": len(train_ids & test_ids),
        "overlap_val_test": len(val_ids & test_ids),
    }
    if protocol.get("policy") == "before_image_grouped_outer_cv":
        group_column = protocol.get("group_column", "before_id")
        train_groups = ids(train_path, group_column)
        val_groups = ids(val_path, group_column)
        test_groups = ids(test_path, group_column)
        partition_audit.update(
            {
                "group_column": group_column,
                "train_groups": len(train_groups),
                "val_groups": len(val_groups),
                "test_groups": len(test_groups),
                "group_overlap_train_val": len(train_groups & val_groups),
                "group_overlap_train_test": len(train_groups & test_groups),
                "group_overlap_val_test": len(val_groups & test_groups),
            }
        )
        overlap_fields = [
            "overlap_train_val",
            "overlap_train_test",
            "overlap_val_test",
            "group_overlap_train_val",
            "group_overlap_train_test",
            "group_overlap_val_test",
        ]
        if any(partition_audit[field] != 0 for field in overlap_fields):
            raise RuntimeError(
                f"Grouped outer partition audit failed: {partition_audit}"
            )
    elif partition_audit != {
        "train_samples": 871,
        "val_samples": 185,
        "test_samples": 185,
        "overlap_train_val": 0,
        "overlap_train_test": 0,
        "overlap_val_test": 0,
    }:
        raise RuntimeError(f"Locked partition audit failed: {partition_audit}")

    output_path = (
        resolve(args.output, project_root)
        if args.output is not None
        else checkpoint_path.parent / "locked_training_manifest.json"
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "checkpoint_frozen_before_test_evaluation",
        "selection_partition": "validation",
        "test_used_during_training": False,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_mtime_ns": checkpoint_mtime_ns,
        "training_start_manifest": str(start_manifest_path),
        "training_start_manifest_sha256": sha256(start_manifest_path),
        "training_log": str(training_log_path),
        "training_log_sha256": sha256(training_log_path),
        "train_csv": str(train_path),
        "train_csv_sha256": sha256(train_path),
        "val_csv": str(val_path),
        "val_csv_sha256": sha256(val_path),
        "test_csv": str(test_path),
        "test_csv_sha256": sha256(test_path),
        "label_stats": str(label_stats_path),
        "label_stats_sha256": sha256(label_stats_path),
        "partition_audit": partition_audit,
        "seed": config.get("seed"),
        "model": config.get("model"),
        "trainer": config.get("trainer"),
        "protocol_name": protocol_name,
        "protocol": protocol,
        protocol_name: protocol,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
