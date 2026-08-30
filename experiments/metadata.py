from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

PARAM_COLUMN_ALIASES = {
    "棰戠巼": ["棰戠巼", "频率"],
    "鑴夊": ["鑴夊", "脉宽"],
    "閫熷害": ["閫熷害", "速度"],
    "DPI": ["DPI", "dpi"],
}

SAMPLE_ID_COLUMNS = ["缂栧彿", "编号", "sample_id", "id", "ID"]


DEFAULT_PARAM_COLUMNS = ["频率", "脉宽", "速度", "DPI"]


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        for encoding in ("utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError("csv", b"", 0, 1, f"unable to read {path}")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {suffix}")


def normalize_identifier(series: pd.Series) -> pd.Series:
    return series.map(lambda x: str(x).strip() if pd.notna(x) else "")


def infer_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    for canonical, aliases in PARAM_COLUMN_ALIASES.items():
        if canonical not in enriched.columns:
            for alias in aliases:
                if alias in enriched.columns:
                    enriched[canonical] = enriched[alias]
                    break

    sample_col = next((column for column in SAMPLE_ID_COLUMNS if column in enriched.columns), None)
    if sample_col is not None:
        enriched["sample_id"] = normalize_identifier(enriched[sample_col])

    if "编号" in enriched.columns:
        enriched["sample_id"] = normalize_identifier(enriched["编号"])
    elif "sample_id" not in enriched.columns:
        enriched["sample_id"] = enriched.index.map(str)
    else:
        enriched["sample_id"] = normalize_identifier(enriched["sample_id"])

    if "before_id" in enriched.columns:
        enriched["before_id"] = normalize_identifier(enriched["before_id"])

    if "pattern_id" in enriched.columns:
        enriched["pattern_id"] = normalize_identifier(enriched["pattern_id"])

    # Current dataset does not expose batch_id explicitly.
    # Use before_id as a stable proxy so cross-batch split is possible now.
    if "batch_id" not in enriched.columns and "before_id" in enriched.columns:
        enriched["batch_id"] = enriched["before_id"]
    elif "batch_id" in enriched.columns:
        enriched["batch_id"] = normalize_identifier(enriched["batch_id"])

    for optional_col in ("fabric_id", "device_id"):
        if optional_col in enriched.columns:
            enriched[optional_col] = normalize_identifier(enriched[optional_col])

    return enriched


def ensure_parameter_columns(
    df: pd.DataFrame,
    param_columns: Optional[Iterable[str]] = None,
) -> list[str]:
    columns = list(param_columns or DEFAULT_PARAM_COLUMNS)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing parameter columns: {missing}")
    return columns
