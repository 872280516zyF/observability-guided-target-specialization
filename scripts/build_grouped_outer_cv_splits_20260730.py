#!/usr/bin/env python3
"""Build deterministic before-image-group outer/inner folds.

The outer test groups are never used for observability screening, checkpoint
selection, calibration, or model fitting. Sample 877 is excluded by default
because its DPI label is not independently verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "sample_id",
    "frequency",
    "pulse_width",
    "speed",
    "dpi",
    "pattern_id",
    "before_id",
]


def clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def canonicalize(path: Path) -> pd.DataFrame:
    frame = None
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if frame is None:
        raise RuntimeError(f"Unable to decode {path}: {last_error}")
    aliases = {
        "sample_id": ["sample_id", "编号"],
        "frequency": ["frequency", "频率"],
        "pulse_width": ["pulse_width", "脉宽"],
        "speed": ["speed", "速度"],
        "dpi": ["dpi", "DPI"],
        "pattern_id": ["pattern_id"],
        "before_id": ["before_id"],
    }
    fallback = {
        "sample_id": 0,
        "frequency": 1,
        "pulse_width": 2,
        "speed": 3,
        "dpi": 4,
        "pattern_id": 5,
        "before_id": 7,
    }
    output = pd.DataFrame()
    for canonical, candidates in aliases.items():
        source = next((item for item in candidates if item in frame.columns), None)
        if source is None:
            index = fallback[canonical]
            if index >= len(frame.columns):
                raise ValueError(f"{path}: missing {canonical}")
            source = frame.columns[index]
        output[canonical] = frame[source]
    for column in ["sample_id", "pattern_id", "before_id"]:
        output[column] = output[column].map(clean_id)
    for column in ["frequency", "pulse_width", "speed", "dpi"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["source_csv"] = str(path)
    return output.dropna(subset=["frequency", "pulse_width", "speed", "dpi"])


def assign_balanced_groups(
    frame: pd.DataFrame,
    group_column: str,
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    counts = (
        frame.groupby(group_column, as_index=False)
        .agg(n_samples=("sample_id", "size"))
    )
    if len(counts) < n_folds:
        raise ValueError(
            f"Only {len(counts)} groups are available for {n_folds} folds."
        )
    rng = np.random.default_rng(seed)
    counts["_tie"] = rng.random(len(counts))
    counts = counts.sort_values(
        ["n_samples", "_tie"], ascending=[False, True]
    ).reset_index(drop=True)
    fold_sizes = np.zeros(n_folds, dtype=int)
    fold_groups = np.zeros(n_folds, dtype=int)
    assignment: dict[str, int] = {}
    tie_order = rng.permutation(n_folds)
    tie_rank = {int(fold): index for index, fold in enumerate(tie_order)}
    for row in counts.itertuples():
        candidate = min(
            range(n_folds),
            key=lambda fold: (
                int(fold_sizes[fold]),
                int(fold_groups[fold]),
                tie_rank[fold],
            ),
        )
        assignment[str(getattr(row, group_column))] = int(candidate)
        fold_sizes[candidate] += int(row.n_samples)
        fold_groups[candidate] += 1
    return assignment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_partition(frame: pd.DataFrame, fold: int, partition: str) -> dict:
    result = {
        "outer_fold": fold,
        "partition": partition,
        "n_samples": int(len(frame)),
        "n_before_groups": int(frame["before_id"].nunique()),
        "n_patterns": int(frame["pattern_id"].nunique()),
    }
    for parameter in ["frequency", "pulse_width", "speed", "dpi"]:
        values = frame[parameter].astype(float)
        result[f"{parameter}_min"] = float(values.min())
        result[f"{parameter}_median"] = float(values.median())
        result[f"{parameter}_max"] = float(values.max())
    return result


def validate_fold(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    all_samples: set[str],
) -> dict:
    partitions = {"train": train, "val": val, "test": test}
    overlap = {}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        for identifier in ["sample_id", "before_id"]:
            shared = set(partitions[left][identifier]) & set(
                partitions[right][identifier]
            )
            overlap[f"{identifier}_{left}_{right}"] = len(shared)
            if shared:
                raise RuntimeError(
                    f"{identifier} overlap in {left}/{right}: "
                    f"{sorted(shared)[:10]}"
                )
    union = set(train["sample_id"]) | set(val["sample_id"]) | set(test["sample_id"])
    if union != all_samples:
        raise RuntimeError("Fold does not cover the full canonical sample set.")
    return overlap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-column", default="before_id")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--exclude-sample-id",
        action="append",
        default=["877"],
        help="Repeatable; sample 877 is excluded by default.",
    )
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources = [Path(item) for item in args.label_csv]
    combined = pd.concat([canonicalize(path) for path in sources], ignore_index=True)
    rows_before_dedup = len(combined)
    duplicate_count = int(combined.duplicated("sample_id", keep="first").sum())
    combined = combined.drop_duplicates("sample_id", keep="first").copy()
    excluded = {clean_id(item) for item in args.exclude_sample_id}
    excluded_rows = combined.loc[combined["sample_id"].isin(excluded)].copy()
    combined = combined.loc[~combined["sample_id"].isin(excluded)].copy()
    combined = combined.sort_values(
        ["before_id", "pattern_id", "sample_id"]
    ).reset_index(drop=True)
    if combined["before_id"].eq("").any():
        raise ValueError("Empty before_id values are not permitted.")
    canonical_path = output / "canonical_labels_excluding_unverified.csv"
    combined[CANONICAL_COLUMNS + ["source_csv"]].to_csv(
        canonical_path, index=False, encoding="utf-8-sig"
    )
    excluded_rows.to_csv(
        output / "excluded_samples.csv", index=False, encoding="utf-8-sig"
    )

    outer_assignment = assign_balanced_groups(
        combined, args.group_column, args.outer_folds, args.seed
    )
    combined["outer_fold"] = combined[args.group_column].map(outer_assignment)
    all_samples = set(combined["sample_id"])
    partition_rows: list[dict] = []
    fold_manifest: list[dict] = []
    oof_counts = {sample_id: 0 for sample_id in all_samples}

    for outer_fold in range(args.outer_folds):
        fold_dir = output / f"fold_{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        test = combined.loc[combined["outer_fold"] == outer_fold].copy()
        outer_train = combined.loc[combined["outer_fold"] != outer_fold].copy()
        inner_assignment = assign_balanced_groups(
            outer_train,
            args.group_column,
            args.inner_folds,
            args.seed + 1000 + outer_fold,
        )
        outer_train["inner_fold"] = outer_train[args.group_column].map(
            inner_assignment
        )
        validation_inner_fold = outer_fold % args.inner_folds
        val = outer_train.loc[
            outer_train["inner_fold"] == validation_inner_fold
        ].copy()
        train = outer_train.loc[
            outer_train["inner_fold"] != validation_inner_fold
        ].copy()
        for sample_id in test["sample_id"]:
            oof_counts[sample_id] += 1

        train["split"] = "train"
        val["split"] = "val"
        test["split"] = "test"
        outer_train_all = pd.concat([train, val], ignore_index=True)
        outer_train_all["split"] = "outer_train"
        output_columns = CANONICAL_COLUMNS + ["split", "outer_fold"]
        paths = {
            "train": fold_dir / "label_train.csv",
            "val": fold_dir / "label_val.csv",
            "test": fold_dir / "label_test.csv",
            "outer_train_all": fold_dir / "label_outer_train_all.csv",
        }
        train[output_columns].to_csv(
            paths["train"], index=False, encoding="utf-8-sig"
        )
        val[output_columns].to_csv(
            paths["val"], index=False, encoding="utf-8-sig"
        )
        test[output_columns].to_csv(
            paths["test"], index=False, encoding="utf-8-sig"
        )
        outer_train_all[output_columns].to_csv(
            paths["outer_train_all"], index=False, encoding="utf-8-sig"
        )

        overlap = validate_fold(train, val, test, all_samples)
        for name, frame in [("train", train), ("val", val), ("test", test)]:
            partition_rows.append(summarize_partition(frame, outer_fold, name))
        fold_manifest.append(
            {
                "outer_fold": outer_fold,
                "validation_inner_fold": validation_inner_fold,
                "train_samples": int(len(train)),
                "val_samples": int(len(val)),
                "test_samples": int(len(test)),
                "train_groups": int(train[args.group_column].nunique()),
                "val_groups": int(val[args.group_column].nunique()),
                "test_groups": int(test[args.group_column].nunique()),
                "test_group_ids": sorted(test[args.group_column].unique()),
                "overlap": overlap,
                "files": {
                    key: {"path": str(path), "sha256": sha256(path)}
                    for key, path in paths.items()
                },
            }
        )

    if set(oof_counts.values()) != {1}:
        bad = {
            sample_id: count
            for sample_id, count in oof_counts.items()
            if count != 1
        }
        raise RuntimeError(f"OOF coverage failure: {list(bad.items())[:10]}")

    pd.DataFrame(partition_rows).to_csv(
        output / "fold_partition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "before_id": group,
                "outer_fold": fold,
                "n_samples": int(
                    (combined[args.group_column] == group).sum()
                ),
            }
            for group, fold in sorted(outer_assignment.items())
        ]
    ).to_csv(
        output / "outer_group_assignment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "before_image_grouped_outer_cv_with_grouped_inner_validation",
        "source_csvs": [str(path) for path in sources],
        "rows_before_dedup": rows_before_dedup,
        "duplicate_rows_removed": duplicate_count,
        "excluded_sample_ids": sorted(excluded),
        "excluded_rows_found": int(len(excluded_rows)),
        "canonical_samples": int(len(combined)),
        "before_groups": int(combined[args.group_column].nunique()),
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "group_column": args.group_column,
        "seed": args.seed,
        "oof_coverage_exactly_once": True,
        "folds": fold_manifest,
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
