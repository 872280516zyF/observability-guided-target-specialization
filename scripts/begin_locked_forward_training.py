#!/usr/bin/env python3
"""Create an immutable start record before a locked-forward training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path_like: str | Path, project_root: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (project_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = resolve(args.config, project_root)
    checkpoint_path = resolve(args.expected_checkpoint, project_root)
    output_path = resolve(args.output, project_root)
    if checkpoint_path.exists():
        raise RuntimeError(
            "Refusing to start: the expected checkpoint already exists. "
            "Use a new run directory so an old checkpoint cannot be certified."
        )
    if output_path.exists():
        raise RuntimeError(
            "Refusing to replace an existing training-start manifest. "
            "Use a new run directory."
        )

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

    data = config["data"]
    files = {
        "config": config_path,
        "train_csv": resolve(data["train_manifest"], project_root),
        "val_csv": resolve(data["val_manifest"], project_root),
        "test_csv": resolve(data["test_manifest"], project_root),
        "label_stats": resolve(data["label_stats"], project_root),
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing locked-forward inputs: {missing}")

    now_ns = time.time_ns()
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "created_at_unix_ns": now_ns,
        "status": "training_started_before_checkpoint_creation",
        "expected_checkpoint": str(checkpoint_path),
        "seed": config.get("seed"),
        "protocol_name": protocol_name,
        "protocol": protocol,
        protocol_name: protocol,
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in files.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
