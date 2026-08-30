#!/usr/bin/env python3
"""Audit exact and perceptual image duplicates across grouped outer splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PARTITION_PAIRS = [
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
]


def clean_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def image_index(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            key = clean_id(path.stem)
            if key in index:
                raise RuntimeError(
                    f"Multiple image files resolve to identifier {key}: "
                    f"{index[key]}, {path}"
                )
            index[key] = path
    return index


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(path: Path) -> int:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), resampling)
    pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return value


def audit_type(
    fold: int,
    image_type: str,
    identifier_column: str,
    tables: dict[str, pd.DataFrame],
    index: dict[str, Path],
    cache: dict[str, dict[str, object]],
    threshold: int,
) -> tuple[list[dict], dict]:
    identifiers = {
        split: sorted(
            set(table[identifier_column].map(clean_id))
        )
        for split, table in tables.items()
    }
    missing = sorted(
        {
            identifier
            for values in identifiers.values()
            for identifier in values
            if identifier not in index
        }
    )
    if missing:
        raise FileNotFoundError(
            f"Missing {image_type} images for identifiers: {missing[:20]}"
        )
    for identifier in {
        identifier for values in identifiers.values() for identifier in values
    }:
        key = f"{image_type}:{identifier}"
        if key not in cache:
            path = index[identifier]
            cache[key] = {
                "path": str(path),
                "sha256": sha256(path),
                "dhash64": difference_hash(path),
            }

    rows = []
    for split_a, split_b in PARTITION_PAIRS:
        for identifier_a in identifiers[split_a]:
            item_a = cache[f"{image_type}:{identifier_a}"]
            for identifier_b in identifiers[split_b]:
                item_b = cache[f"{image_type}:{identifier_b}"]
                distance = bin(
                    int(item_a["dhash64"]) ^ int(item_b["dhash64"])
                ).count("1")
                exact = item_a["sha256"] == item_b["sha256"]
                if exact or distance <= threshold:
                    rows.append(
                        {
                            "outer_fold": fold,
                            "image_type": image_type,
                            "split_a": split_a,
                            "split_b": split_b,
                            "id_a": identifier_a,
                            "id_b": identifier_b,
                            "path_a": item_a["path"],
                            "path_b": item_b["path"],
                            "exact_sha256_duplicate": bool(exact),
                            "dhash64_hamming_distance": distance,
                            "near_duplicate_threshold": threshold,
                        }
                    )
    summary = {
        "outer_fold": fold,
        "image_type": image_type,
        "identifier_column": identifier_column,
        "train_images": len(identifiers["train"]),
        "val_images": len(identifiers["val"]),
        "test_images": len(identifiers["test"]),
        "cross_partition_exact_duplicates": sum(
            row["exact_sha256_duplicate"] for row in rows
        ),
        "cross_partition_perceptual_candidates": len(rows),
        "dhash64_threshold": threshold,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-root", required=True)
    parser.add_argument("--before-dir", required=True)
    parser.add_argument("--after-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--near-threshold", type=int, default=2)
    parser.add_argument("--fail-on-exact", action="store_true")
    args = parser.parse_args()

    splits_root = Path(args.splits_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    indexes = {
        "before": image_index(Path(args.before_dir)),
        "after": image_index(Path(args.after_dir)),
    }
    cache: dict[str, dict[str, object]] = {}
    candidate_rows: list[dict] = []
    summary_rows: list[dict] = []
    for fold in folds:
        split_dir = splits_root / f"fold_{fold}"
        tables = {
            split: pd.read_csv(split_dir / f"label_{split}.csv")
            for split in ["train", "val", "test"]
        }
        for image_type, column in [
            ("before", "before_id"),
            ("after", "sample_id"),
        ]:
            rows, summary = audit_type(
                fold,
                image_type,
                column,
                tables,
                indexes[image_type],
                cache,
                args.near_threshold,
            )
            candidate_rows.extend(rows)
            summary_rows.append(summary)

    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        candidates = pd.DataFrame(
            columns=[
                "outer_fold",
                "image_type",
                "split_a",
                "split_b",
                "id_a",
                "id_b",
                "path_a",
                "path_b",
                "exact_sha256_duplicate",
                "dhash64_hamming_distance",
                "near_duplicate_threshold",
            ]
        )
    summary = pd.DataFrame(summary_rows)
    candidates.to_csv(
        output / "cross_partition_duplicate_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output / "duplicate_audit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "splits_root": str(splits_root),
        "folds": folds,
        "exact_method": "SHA-256 of encoded image file",
        "perceptual_method": "64-bit horizontal difference hash",
        "near_duplicate_threshold": args.near_threshold,
        "candidate_pairs": int(len(candidates)),
        "exact_duplicate_pairs": int(
            candidates["exact_sha256_duplicate"].astype(bool).sum()
        ),
        "note": (
            "Perceptual candidates are an audit list for review, not automatic "
            "proof of duplicated experimental units."
        ),
        "summary": summary_rows,
    }
    (output / "duplicate_audit_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.fail_on_exact and payload["exact_duplicate_pairs"] > 0:
        raise RuntimeError(
            "Cross-partition exact image duplicates were detected."
        )


if __name__ == "__main__":
    main()
