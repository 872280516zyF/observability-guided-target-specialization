from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .metadata import infer_metadata_columns, load_table


GROUP_SPLIT_MODES = {
    "cross_pattern": "pattern_id",
    "cross_batch": "batch_id",
    "cross_fabric": "fabric_id",
    "cross_device": "device_id",
}


@dataclass
class SplitResult:
    dataframe: pd.DataFrame
    split_column: str = "split"

    def subset(self, split_name: str) -> pd.DataFrame:
        return self.dataframe[self.dataframe[self.split_column] == split_name].copy()


def _assign_random_split(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    n_total = len(indices)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_test = min(n_test, max(n_total - 2, 0))
    n_val = min(n_val, max(n_total - n_test - 1, 0))

    split = np.full(n_total, "train", dtype=object)
    split[indices[:n_test]] = "test"
    split[indices[n_test : n_test + n_val]] = "val"
    return pd.Series(split, index=df.index)


def _choose_groups(groups: np.ndarray, ratio: float, rng: np.random.Generator) -> set[str]:
    groups = np.asarray(groups, dtype=object)
    if len(groups) == 0 or ratio <= 0:
        return set()
    n_groups = max(1, int(math.ceil(len(groups) * ratio)))
    chosen = rng.choice(groups, size=min(n_groups, len(groups)), replace=False)
    return {str(item) for item in chosen}


def _assign_group_split(
    df: pd.DataFrame,
    group_column: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.Series:
    if group_column not in df.columns:
        raise ValueError(f"Column '{group_column}' is required for grouped split")

    group_values = df[group_column].fillna("").astype(str)
    valid_groups = np.array(sorted({value for value in group_values if value}), dtype=object)
    if len(valid_groups) < 2:
        raise ValueError(f"Column '{group_column}' does not have enough groups for grouped split")

    rng = np.random.default_rng(seed)
    test_groups = _choose_groups(valid_groups, test_ratio, rng)
    remaining = np.array([group for group in valid_groups if group not in test_groups], dtype=object)
    val_groups = _choose_groups(remaining, val_ratio, rng)

    split = []
    for value in group_values:
        if value in test_groups:
            split.append("test")
        elif value in val_groups:
            split.append("val")
        else:
            split.append("train")
    return pd.Series(split, index=df.index)


def build_split_dataframe(
    annotation_path: str | Path,
    split_mode: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitResult:
    df = infer_metadata_columns(load_table(annotation_path))
    split_mode = split_mode.lower()

    if split_mode == "random":
        split = _assign_random_split(df, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    elif split_mode in GROUP_SPLIT_MODES:
        split = _assign_group_split(
            df,
            group_column=GROUP_SPLIT_MODES[split_mode],
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    out = df.copy()
    out["split"] = split
    out["split_mode"] = split_mode
    out["split_seed"] = seed
    return SplitResult(dataframe=out)


def write_split_manifests(split_result: SplitResult, output_dir: str | Path, stem: str) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split_name in ("train", "val", "test"):
        path = output_dir / f"{stem}_{split_name}.csv"
        split_result.subset(split_name).to_csv(path, index=False, encoding="utf-8-sig")
        paths[split_name] = path

    full_path = output_dir / f"{stem}_all.csv"
    split_result.dataframe.to_csv(full_path, index=False, encoding="utf-8-sig")
    paths["all"] = full_path
    return paths


def summarize_split(split_result: SplitResult) -> pd.DataFrame:
    df = split_result.dataframe
    summary_rows = []
    for split_name in ("train", "val", "test"):
        subset = df[df["split"] == split_name]
        row = {
            "split": split_name,
            "samples": int(len(subset)),
        }
        for column in ("pattern_id", "batch_id", "fabric_id", "device_id"):
            if column in subset.columns:
                row[f"unique_{column}"] = int(subset[column].replace("", np.nan).dropna().nunique())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)
