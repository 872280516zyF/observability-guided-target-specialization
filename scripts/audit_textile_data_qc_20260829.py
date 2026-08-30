#!/usr/bin/env python3
"""Reproduce the textile-label and image-integrity QC counts used in the SI.

This script is read-only with respect to the dataset.  It distinguishes row
deduplication by canonical sample identifier from the later cross-partition
image-similarity audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PARAMETERS = ["frequency", "pulse_width", "speed", "dpi"]
IDENTIFIERS = ["sample_id", "pattern_id", "before_id"]
EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def read_canonical(path: Path) -> pd.DataFrame:
    frame = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if frame is None:
        raise RuntimeError("Unable to decode {}".format(path))
    aliases = {
        "sample_id": ["sample_id", "缂栧彿"],
        "frequency": ["frequency", "棰戠巼"],
        "pulse_width": ["pulse_width", "鑴夊"],
        "speed": ["speed", "閫熷害"],
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
        source = next((name for name in candidates if name in frame.columns), None)
        if source is None:
            source = frame.columns[fallback[canonical]]
        output[canonical] = frame[source]
    for column in IDENTIFIERS:
        output[column] = output[column].map(clean_id)
    for column in PARAMETERS:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["source_csv"] = str(path)
    output["source_row"] = range(2, len(output) + 2)
    return output.dropna(subset=PARAMETERS)


def image_exists(directory: Path, identifier: str) -> bool:
    return any((directory / (identifier + suffix)).exists() for suffix in EXTENSIONS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-csv", action="append", required=True)
    parser.add_argument("--before-dir", required=True)
    parser.add_argument("--after-dir", required=True)
    parser.add_argument("--duplicate-audit-manifest", required=True)
    parser.add_argument("--duplicate-candidates-csv", required=True)
    parser.add_argument("--exclude-sample-id", default="877")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    sources = [Path(item) for item in args.label_csv]
    combined = pd.concat([read_canonical(path) for path in sources], ignore_index=True)
    raw_rows = int(len(combined))
    duplicate_mask = combined.duplicated("sample_id", keep="first")
    duplicate_rows_removed = int(duplicate_mask.sum())
    duplicate_ids = sorted(combined.loc[duplicate_mask, "sample_id"].unique())

    conflict_rows = []
    compare_columns = PARAMETERS + ["pattern_id", "before_id"]
    for sample_id, group in combined.groupby("sample_id", sort=True):
        if len(group) < 2:
            continue
        unique = group[compare_columns].drop_duplicates()
        if len(unique) > 1:
            conflict_rows.append(
                {
                    "sample_id": sample_id,
                    "occurrences": int(len(group)),
                    "distinct_label_rows": int(len(unique)),
                    "source_rows": "; ".join(
                        "{}:{}".format(row.source_csv, row.source_row)
                        for row in group.itertuples()
                    ),
                }
            )

    deduplicated = combined.loc[~duplicate_mask].copy()
    excluded_id = clean_id(args.exclude_sample_id)
    excluded_mask = deduplicated["sample_id"].eq(excluded_id)
    entry_error_rows_removed = int(excluded_mask.sum())
    canonical = deduplicated.loc[~excluded_mask].copy()

    before_dir = Path(args.before_dir)
    after_dir = Path(args.after_dir)
    missing_before = sorted(
        identifier
        for identifier in canonical["before_id"].drop_duplicates()
        if not image_exists(before_dir, identifier)
    )
    missing_after = sorted(
        identifier
        for identifier in canonical["sample_id"].drop_duplicates()
        if not image_exists(after_dir, identifier)
    )

    manifest_path = Path(args.duplicate_audit_manifest)
    image_audit = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = pd.read_csv(args.duplicate_candidates_csv)
    unique_candidates = set()
    if not candidates.empty:
        for row in candidates.itertuples():
            unique_candidates.add(
                (str(row.image_type),) + tuple(sorted((clean_id(row.id_a), clean_id(row.id_b))))
            )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    conflicts = pd.DataFrame(conflict_rows)
    conflicts.to_csv(output / "duplicate_id_label_conflicts.csv", index=False, encoding="utf-8-sig")

    payload = {
        "source_csvs_in_concatenation_order": [str(path) for path in sources],
        "raw_rows_after_numeric_validity_filter": raw_rows,
        "duplicate_id_rule": (
            "canonicalize sample_id by trimming whitespace and a terminal .0; "
            "keep the first occurrence in source-table concatenation order"
        ),
        "duplicate_id_rows_removed": duplicate_rows_removed,
        "duplicate_ids_with_later_occurrences": int(len(duplicate_ids)),
        "duplicate_ids_with_conflicting_parameter_or_group_fields": int(len(conflict_rows)),
        "rows_after_id_deduplication": int(len(deduplicated)),
        "excluded_unverified_entry_id": excluded_id,
        "entry_error_rows_removed": entry_error_rows_removed,
        "final_samples": int(len(canonical)),
        "final_unique_sample_ids": int(canonical["sample_id"].nunique()),
        "final_initial_image_groups": int(canonical["before_id"].nunique()),
        "missing_initial_images": int(len(missing_before)),
        "missing_processed_images": int(len(missing_after)),
        "image_duplicate_audit": {
            "role": "cross-partition audit only; not the sample-ID deduplication rule",
            "exact_method": image_audit["exact_method"],
            "exact_cross_partition_pairs_foldwise": int(image_audit["exact_duplicate_pairs"]),
            "perceptual_method": image_audit["perceptual_method"],
            "perceptual_hamming_threshold": int(image_audit["near_duplicate_threshold"]),
            "perceptual_candidate_instances_foldwise": int(image_audit["candidate_pairs"]),
            "perceptual_candidate_pairs_unique": int(len(unique_candidates)),
            "candidate_policy": image_audit["note"],
        },
        "source_hashes": {
            str(path): sha256(path) for path in sources
        },
    }
    (output / "textile_data_qc_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
