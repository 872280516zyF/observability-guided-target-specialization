#!/usr/bin/env python3
"""Download and prepare smallNORB for leakage-controlled factor inversion.

Only the official training instances enter observability analysis and pilot
selection. The official testing instances remain unavailable to those stages.
Images are not rewritten; the trainer memory-maps the official binary matrices.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import struct
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "data" / "smallnorb_20260807"
RAW = ROOT / "raw"
SPLITS = ROOT / "splits"
OBSERVABILITY = ROOT / "observability"
BASE_URL = "https://cs.nyu.edu/~yann/data/norb-v1.0-small/"
PREFIXES = {
    "training": "smallnorb-5x46789x9x18x6x2x96x96-training",
    "testing": "smallnorb-5x01235x9x18x6x2x96x96-testing",
}
KINDS = ("dat", "cat", "info")
FACTORS = ("elevation", "azimuth", "lighting")
MAGIC_DTYPES = {
    0x1E3D4C51: np.dtype("<f4"),
    0x1E3D4C53: np.dtype("<f8"),
    0x1E3D4C54: np.dtype("<i4"),
    0x1E3D4C55: np.dtype("u1"),
    0x1E3D4C56: np.dtype("<i2"),
}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_path(split: str, kind: str, compressed: bool = False) -> Path:
    suffix = ".mat.gz" if compressed else ".mat"
    return RAW / "{}-{}{}".format(PREFIXES[split], kind, suffix)


def resumable_download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print("[SKIP DOWNLOAD] {}".format(destination), flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", "bytes={}-".format(offset))
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        append = offset > 0 and status == 206
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        total_header = response.headers.get("Content-Length")
        total = offset + int(total_header) if total_header else None
        downloaded = offset
        with partial.open(mode) as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                if total:
                    print(
                        "[DOWNLOAD] {} {:.1f}%".format(destination.name, 100.0 * downloaded / total),
                        end="\r", flush=True,
                    )
    print("", flush=True)
    partial.replace(destination)


def stage_download(args: argparse.Namespace) -> None:
    manifest = []
    for split, prefix in PREFIXES.items():
        for kind in KINDS:
            destination = matrix_path(split, kind, compressed=True)
            resumable_download(BASE_URL + destination.name, destination)
            manifest.append(
                {"split": split, "kind": kind, "file": destination.name,
                 "bytes": destination.stat().st_size, "sha256": sha256(destination)}
            )
    write_json(ROOT / "download_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def decompress(source: Path, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print("[SKIP DECOMPRESS] {}".format(destination), flush=True)
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    with gzip.open(str(source), "rb") as incoming, temporary.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
    temporary.replace(destination)


def matrix_memmap(path: Path) -> Tuple[np.memmap, Tuple[int, ...], int]:
    with path.open("rb") as handle:
        magic, ndim = struct.unpack("<ii", handle.read(8))
        dimension_count = max(3, ndim)
        dimensions = struct.unpack("<{}i".format(dimension_count), handle.read(4 * dimension_count))
    if magic not in MAGIC_DTYPES:
        raise ValueError("Unsupported matrix magic {} in {}".format(hex(magic), path))
    shape = tuple(int(value) for value in dimensions[:ndim])
    offset = 8 + 4 * dimension_count
    values = np.memmap(str(path), dtype=MAGIC_DTYPES[magic], mode="r", offset=offset, shape=shape)
    return values, shape, offset


def metadata_frame(split: str) -> pd.DataFrame:
    categories, cat_shape, _ = matrix_memmap(matrix_path(split, "cat"))
    information, info_shape, _ = matrix_memmap(matrix_path(split, "info"))
    if len(cat_shape) != 1 or info_shape != (cat_shape[0], 4):
        raise ValueError("Unexpected metadata matrix shape for {}".format(split))
    frame = pd.DataFrame(
        {
            "source_split": split,
            "row_index": np.arange(cat_shape[0], dtype=np.int64),
            "category": np.asarray(categories, dtype=np.int64),
            "instance": np.asarray(information[:, 0], dtype=np.int64),
            "elevation": np.asarray(information[:, 1], dtype=np.int64),
            "azimuth": np.asarray(information[:, 2], dtype=np.int64) // 2,
            "lighting": np.asarray(information[:, 3], dtype=np.int64),
        }
    )
    frame["group_id"] = frame.apply(
        lambda row: "category{}_instance{}".format(int(row["category"]), int(row["instance"])), axis=1
    )
    return frame


def one_factor_group_scores(training: pd.DataFrame) -> pd.DataFrame:
    cache = ROOT / "one_factor_group_scores.csv"
    if cache.is_file():
        return pd.read_csv(cache)
    images, shape, _ = matrix_memmap(matrix_path("training", "dat"))
    if shape != (24300, 2, 96, 96):
        raise ValueError("Unexpected training image shape {}".format(shape))
    rows = []
    for group_id, group in training.groupby("group_id", sort=True):
        for factor in FACTORS:
            other = [value for value in ("category", "instance", *FACTORS) if value != factor]
            values: List[float] = []
            for _, subset in group.groupby(other, sort=False):
                subset = subset.sort_values(factor)
                indices = subset["row_index"].to_numpy(np.int64)
                if len(indices) < 2:
                    continue
                current = np.asarray(images[indices, 0], dtype=np.int16)
                differences = np.abs(current[1:] - current[:-1]).mean(axis=(1, 2)) / 255.0
                values.extend(float(value) for value in differences)
            if not values:
                raise RuntimeError("No one-factor pairs for {} {}".format(group_id, factor))
            rows.append(
                {"group_id": group_id, "factor": factor, "mean_response": float(np.mean(values)), "n_pairs": len(values)}
            )
    result = pd.DataFrame(rows)
    result.to_csv(cache, index=False)
    return result


def observability_payload(
    scores: pd.DataFrame,
    groups: Sequence[str],
    bootstrap_seed: int,
    bootstrap_iterations: int,
    relative_threshold: float,
    max_high_targets: int,
) -> Dict[str, object]:
    filtered = scores[scores["group_id"].isin(set(groups))]
    matrix = filtered.pivot(index="group_id", columns="factor", values="mean_response")[list(FACTORS)]
    if matrix.shape[0] != len(set(groups)):
        raise RuntimeError("Incomplete group-factor response matrix")
    raw = matrix.mean(0)
    normalized = raw / max(float(raw.max()), 1e-12)
    maximum = float(normalized.max())
    threshold_eligible = [
        factor for factor in FACTORS
        if float(normalized[factor]) + 1e-12 >= relative_threshold * maximum
    ]
    if not threshold_eligible:
        raise RuntimeError("No factor satisfies the high-observability rule")
    high = sorted(
        threshold_eligible, key=lambda factor: (-float(normalized[factor]), FACTORS.index(factor))
    )[:max_high_targets]
    rng = np.random.default_rng(bootstrap_seed)
    data = matrix.to_numpy(float)
    wins = {factor: 0 for factor in FACTORS}
    for _ in range(bootstrap_iterations):
        indices = rng.integers(0, len(data), size=len(data))
        winner = FACTORS[int(np.argmax(data[indices].mean(0)))]
        wins[winner] += 1
    return {
        "selection_scope": "official-training object instances only",
        "groups": list(matrix.index),
        "raw_scores": {factor: float(raw[factor]) for factor in FACTORS},
        "scores": {factor: float(normalized[factor]) for factor in FACTORS},
        "relative_threshold": relative_threshold,
        "max_high_targets": max_high_targets,
        "threshold_eligible_targets": threshold_eligible,
        "selection_cap_applied": len(threshold_eligible) > max_high_targets,
        "high_targets": high,
        "bootstrap_rank1_fraction": {factor: wins[factor] / float(bootstrap_iterations) for factor in FACTORS},
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
    }


def stage_prepare(args: argparse.Namespace) -> None:
    if not 1 <= args.max_high_targets < len(FACTORS):
        raise ValueError("max_high_targets must be between 1 and {}".format(len(FACTORS) - 1))
    for split in PREFIXES:
        for kind in KINDS:
            source = matrix_path(split, kind, compressed=True)
            if not source.is_file():
                raise FileNotFoundError("Download first: {}".format(source))
            decompress(source, matrix_path(split, kind))
    training = metadata_frame("training")
    testing = metadata_frame("testing")
    if set(training["group_id"]) & set(testing["group_id"]):
        raise RuntimeError("Official train/test object groups overlap")
    ROOT.mkdir(parents=True, exist_ok=True)
    training.to_csv(ROOT / "official_train.csv", index=False)
    testing.to_csv(ROOT / "official_test.csv", index=False)
    train_instances = sorted(training["instance"].unique().tolist())
    if len(train_instances) != 5:
        raise RuntimeError("Expected five official training instances per category")
    audit = []
    for fold, instance in enumerate(train_instances):
        validation_groups = set(training.loc[training["instance"].eq(instance), "group_id"])
        train = training[~training["group_id"].isin(validation_groups)].copy()
        validation = training[training["group_id"].isin(validation_groups)].copy()
        destination = SPLITS / "inner_fold_{}".format(fold)
        destination.mkdir(parents=True, exist_ok=True)
        train.to_csv(destination / "train.csv", index=False)
        validation.to_csv(destination / "val.csv", index=False)
        audit.append(
            {"fold": fold, "validation_instance": int(instance), "train_images": len(train),
             "val_images": len(validation), "train_groups": train["group_id"].nunique(),
             "val_groups": validation["group_id"].nunique()}
        )
    pd.DataFrame(audit).to_csv(ROOT / "inner_fold_audit.csv", index=False)

    group_scores = one_factor_group_scores(training)
    OBSERVABILITY.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        train = pd.read_csv(SPLITS / "inner_fold_{}".format(fold) / "train.csv")
        payload = observability_payload(
            group_scores, sorted(train["group_id"].unique()), args.bootstrap_seed + fold,
            args.bootstrap_iterations, args.relative_threshold, args.max_high_targets,
        )
        write_json(OBSERVABILITY / "inner_fold_{}.json".format(fold), payload)
    final = observability_payload(
        group_scores, sorted(training["group_id"].unique()), args.bootstrap_seed + 100,
        args.bootstrap_iterations, args.relative_threshold, args.max_high_targets,
    )
    write_json(OBSERVABILITY / "official_train.json", final)
    write_json(
        ROOT / "preflight.json",
        {
            "status": "PASS",
            "official_train_images": len(training),
            "official_test_images": len(testing),
            "official_train_groups": training["group_id"].nunique(),
            "official_test_groups": testing["group_id"].nunique(),
            "group_overlap": 0,
            "input": "left 96x96 grayscale image only",
            "targets": list(FACTORS),
            "final_observability": final,
        },
    )
    print(json.dumps(json.loads((ROOT / "preflight.json").read_text()), indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("download", "prepare"))
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--relative-threshold", type=float, default=0.80)
    parser.add_argument("--max-high-targets", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    {"download": stage_download, "prepare": stage_prepare}[args.stage](args)
