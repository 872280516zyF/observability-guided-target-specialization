from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .evaluation import build_rerank_sample_table, paired_improvement_summary, parameter_summary


PARAM_NAMES_EN = ["frequency", "pulse_width", "speed", "dpi"]


def save_rerank_reports(results: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_summary = pd.concat(
        [
            parameter_summary(results, stage="base", param_names=PARAM_NAMES_EN),
            parameter_summary(results, stage="reranked", param_names=PARAM_NAMES_EN),
        ],
        ignore_index=True,
    )
    improvement_summary = pd.concat(
        [
            paired_improvement_summary(results, "mean_ape"),
            paired_improvement_summary(results, "image_score"),
            paired_improvement_summary(results, "total_score"),
        ],
        ignore_index=True,
    )
    sample_table = build_rerank_sample_table(results)

    stage_summary_path = output_dir / "rerank_stage_summary.csv"
    improvement_summary_path = output_dir / "rerank_improvement_summary.csv"
    sample_table_path = output_dir / "rerank_sample_analysis.csv"
    json_path = output_dir / "rerank_report.json"

    stage_summary.to_csv(stage_summary_path, index=False, encoding="utf-8-sig")
    improvement_summary.to_csv(improvement_summary_path, index=False, encoding="utf-8-sig")
    sample_table.to_csv(sample_table_path, index=False, encoding="utf-8-sig")

    payload = {
        "stage_summary": stage_summary.to_dict(orient="records"),
        "improvement_summary": improvement_summary.to_dict(orient="records"),
        "num_samples": int(len(results)),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return {
        "stage_summary": stage_summary_path,
        "improvement_summary": improvement_summary_path,
        "sample_analysis": sample_table_path,
        "json": json_path,
    }

