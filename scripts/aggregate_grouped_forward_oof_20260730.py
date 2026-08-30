#!/usr/bin/env python3
"""Aggregate grouped outer-fold forward fidelity and counterfactual evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_METRICS = ["rgb_mae", "one_minus_ssim", "edge_mae", "roi_mae"]
FIDELITY_METRICS = [
    "image_mae",
    "ssim",
    "psnr_db",
    "edge_mae",
    "roi_mae",
    "background_mae",
]


def normalize_summary(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    normalized_columns = []
    for metric in SCORE_METRICS:
        values = output[metric].astype(float)
        denominator = float(values.max() - values.min())
        column = f"{metric}_normalized"
        output[column] = (
            (values - values.min()) / denominator
            if denominator > 0
            else values * 0
        )
        normalized_columns.append(column)
    output["composite_response_score"] = output[normalized_columns].mean(axis=1)
    output["rank"] = output["composite_response_score"].rank(
        method="min", ascending=False
    ).astype(int)
    return output.sort_values(["rank", "parameter"]).reset_index(drop=True)


def validate_coverage(
    frame: pd.DataFrame,
    expected_samples: set[str],
    multiplier: int,
    label: str,
) -> list[dict]:
    rows = []
    for seed, group in frame.groupby("seed"):
        counts = group.groupby("sample_id").size()
        complete = (
            set(counts.index.astype(str)) == expected_samples
            and counts.eq(multiplier).all()
        )
        rows.append(
            {
                "evidence": label,
                "seed": int(seed),
                "n_rows": int(len(group)),
                "n_samples": int(len(counts)),
                "expected_rows_per_sample": multiplier,
                "complete_oof": bool(complete),
            }
        )
        if not complete:
            raise RuntimeError(
                f"Incomplete {label} OOF coverage for seed {seed}"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    args = parser.parse_args()

    splits_root = Path(args.splits_root)
    run_root = Path(args.run_root)
    aggregate = run_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_csv(
        splits_root / "canonical_labels_excluding_unverified.csv"
    )
    canonical["sample_id"] = canonical["sample_id"].astype(str)
    expected_samples = set(canonical["sample_id"].astype(str))
    before_id_by_sample = dict(
        zip(canonical["sample_id"], canonical["before_id"].astype(str))
    )
    fidelity_parts = []
    counterfactual_parts = []
    inventory = []

    for fold in args.folds:
        expected_fold = set(
            pd.read_csv(
                splits_root / f"fold_{fold}" / "label_test.csv"
            )["sample_id"].astype(str)
        )
        for seed in args.seeds:
            evidence = (
                run_root / f"fold_{fold}" / f"seed{seed}" / "outer_test_evidence"
            )
            fidelity_path = evidence / "forward_fidelity_per_sample.csv"
            counterfactual_path = (
                evidence / "forward_counterfactual_per_sample.csv"
            )
            if not fidelity_path.exists() or not counterfactual_path.exists():
                raise FileNotFoundError(
                    f"Missing forward evidence for fold={fold} seed={seed}"
                )
            fidelity = pd.read_csv(fidelity_path)
            counterfactual = pd.read_csv(counterfactual_path)
            fidelity["sample_id"] = fidelity["sample_id"].astype(str)
            counterfactual["sample_id"] = counterfactual["sample_id"].astype(str)
            if "before_id" not in counterfactual.columns:
                counterfactual["before_id"] = counterfactual["sample_id"].map(
                    before_id_by_sample
                )
            if set(fidelity["sample_id"]) != expected_fold:
                raise RuntimeError(f"Fidelity sample mismatch: fold={fold} seed={seed}")
            if set(counterfactual["sample_id"]) != expected_fold:
                raise RuntimeError(
                    f"Counterfactual sample mismatch: fold={fold} seed={seed}"
                )
            fidelity["outer_fold"] = fold
            fidelity["seed"] = seed
            counterfactual["outer_fold"] = fold
            counterfactual["seed"] = seed
            fidelity_parts.append(fidelity)
            counterfactual_parts.append(counterfactual)
            inventory.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "fidelity_path": str(fidelity_path),
                    "counterfactual_path": str(counterfactual_path),
                    "n_test_samples": len(expected_fold),
                }
            )

    fidelity_all = pd.concat(fidelity_parts, ignore_index=True)
    counterfactual_all = pd.concat(counterfactual_parts, ignore_index=True)
    coverage = [
        *validate_coverage(
            fidelity_all, expected_samples, 1, "forward_fidelity"
        ),
        *validate_coverage(
            counterfactual_all, expected_samples, 4, "counterfactual"
        ),
    ]

    fidelity_seed_rows = []
    for seed, group in fidelity_all.groupby("seed"):
        row = {
            "seed": int(seed),
            "n_samples": int(group["sample_id"].nunique()),
        }
        for metric in FIDELITY_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_median"] = float(group[metric].median())
        fidelity_seed_rows.append(row)
    fidelity_by_seed = pd.DataFrame(fidelity_seed_rows)

    counterfactual_seed_rows = []
    for seed, seed_frame in counterfactual_all.groupby("seed"):
        summary = (
            seed_frame.groupby("parameter", as_index=False)
            .agg(
                samples=("sample_id", "nunique"),
                rgb_mae=("rgb_mae", "mean"),
                one_minus_ssim=("one_minus_ssim", "mean"),
                edge_mae=("edge_mae", "mean"),
                roi_mae=("roi_mae", "mean"),
                background_mae=("background_mae", "mean"),
                roi_background_mae_ratio=(
                    "roi_background_mae_ratio",
                    "median",
                ),
            )
        )
        summary = normalize_summary(summary)
        summary["seed"] = int(seed)
        counterfactual_seed_rows.append(summary)
    counterfactual_by_seed = pd.concat(
        counterfactual_seed_rows, ignore_index=True
    )

    counterfactual_seed_averaged = (
        counterfactual_all.groupby(
            ["sample_id", "before_id", "parameter"], as_index=False
        )
        .agg(
            **{
                metric: (metric, "mean")
                for metric in [
                    *SCORE_METRICS,
                    "background_mae",
                    "roi_background_mae_ratio",
                ]
            }
        )
    )
    aggregate_summary = (
        counterfactual_seed_averaged.groupby("parameter", as_index=False)
        .agg(
            samples=("sample_id", "nunique"),
            rgb_mae=("rgb_mae", "mean"),
            one_minus_ssim=("one_minus_ssim", "mean"),
            edge_mae=("edge_mae", "mean"),
            roi_mae=("roi_mae", "mean"),
            background_mae=("background_mae", "mean"),
            roi_background_mae_ratio=(
                "roi_background_mae_ratio",
                "median",
            ),
        )
    )
    aggregate_summary = normalize_summary(aggregate_summary)

    outputs = {
        "forward_run_inventory.csv": pd.DataFrame(inventory),
        "forward_oof_coverage_audit.csv": pd.DataFrame(coverage),
        "forward_fidelity_oof_all_seeds.csv": fidelity_all,
        "forward_fidelity_oof_by_seed.csv": fidelity_by_seed,
        "forward_counterfactual_oof_all_seeds.csv": counterfactual_all,
        "forward_counterfactual_oof_by_seed.csv": counterfactual_by_seed,
        "forward_counterfactual_oof_seed_averaged_by_sample.csv": (
            counterfactual_seed_averaged
        ),
        "forward_counterfactual_oof_summary.csv": aggregate_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(
            aggregate / filename, index=False, encoding="utf-8-sig"
        )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation": "before-image-grouped forward outer-fold OOF",
        "folds": args.folds,
        "seeds": args.seeds,
        "samples_per_seed": len(expected_samples),
        "sample_877_excluded": True,
        "oof_complete": bool(
            pd.DataFrame(coverage)["complete_oof"].all()
        ),
        "interpretation": (
            "model-based consistency analysis only; not physical or causal "
            "validation"
        ),
        "outputs": {
            filename: str(aggregate / filename)
            for filename in outputs
        },
    }
    (aggregate / "GROUPED_FORWARD_OOF_COMPLETE.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
