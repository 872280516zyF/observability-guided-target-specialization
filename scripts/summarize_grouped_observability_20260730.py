#!/usr/bin/env python3
"""Summarize one-factor image responses with before-image-group resampling."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TEXTURE_METRICS = [
    "edge_mae_per_norm_step",
    "edge_density_delta_per_norm_step",
    "high_freq_energy_delta_per_norm_step",
    "local_contrast_delta_per_norm_step",
    "changed_area_ratio_per_norm_step",
]
EXPECTED_PARAMETERS = ["frequency", "pulse_width", "speed", "dpi"]


def minmax_scores(values: np.ndarray) -> np.ndarray:
    minimum = values.min(axis=0, keepdims=True)
    maximum = values.max(axis=0, keepdims=True)
    denominator = maximum - minimum
    normalized = np.divide(
        values - minimum,
        denominator,
        out=np.zeros_like(values),
        where=denominator > 0,
    )
    return normalized.mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pair_metrics)
    required = {
        "varied_param",
        "before_id",
        "pattern_id",
        "sample_id_a",
        *TEXTURE_METRICS,
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Missing pair-metric columns: {missing}")
    found_parameters = sorted(pairs["varied_param"].astype(str).unique())
    missing_parameters = sorted(set(EXPECTED_PARAMETERS) - set(found_parameters))
    if missing_parameters:
        raise RuntimeError(
            "At least one outer-training fold lacks one-factor evidence for "
            f"{missing_parameters}. The fold cannot select P^obs."
        )

    group_means = (
        pairs.groupby(["varied_param", "before_id"], as_index=False)[
            TEXTURE_METRICS
        ]
        .mean()
    )
    counts = (
        pairs.groupby("varied_param")
        .agg(
            n_pairs=("sample_id_a", "size"),
            n_before_groups=("before_id", "nunique"),
            n_pattern_groups=("pattern_id", "nunique"),
        )
        .reset_index()
    )
    counts.to_csv(
        output / "group_counts.csv", index=False, encoding="utf-8-sig"
    )
    group_means.to_csv(
        output / "group_mean_metrics.csv", index=False, encoding="utf-8-sig"
    )

    parameters = sorted(group_means["varied_param"].unique())
    arrays = {
        parameter: group_means.loc[
            group_means["varied_param"] == parameter, TEXTURE_METRICS
        ].to_numpy(float)
        for parameter in parameters
    }
    balanced_groups = min(len(values) for values in arrays.values())
    if balanced_groups < 2:
        raise RuntimeError(
            "Fewer than two before-image groups are available for at least one "
            "parameter; grouped observability is not estimable."
        )

    full_values = np.stack(
        [arrays[parameter].mean(axis=0) for parameter in parameters]
    )
    full_scores = minmax_scores(full_values)
    full_order = np.argsort(-full_scores)
    full_ranking = pd.DataFrame(
        [
            {
                "varied_param": parameters[index],
                "score": float(full_scores[index]),
                "rank": rank + 1,
            }
            for rank, index in enumerate(full_order)
        ]
    )
    full_ranking.to_csv(
        output / "group_mean_ranking.csv", index=False, encoding="utf-8-sig"
    )
    selected_target = str(full_ranking.iloc[0]["varied_param"])
    selected_score = float(full_ranking.iloc[0]["score"])
    runner_up = str(full_ranking.iloc[1]["varied_param"])
    margin = selected_score - float(full_ranking.iloc[1]["score"])

    logo_rows = []
    for omitted_group in sorted(
        group_means["before_id"].astype(str).unique()
    ):
        retained = group_means.loc[
            group_means["before_id"].astype(str) != omitted_group
        ]
        retained_arrays = {
            parameter: retained.loc[
                retained["varied_param"] == parameter, TEXTURE_METRICS
            ].to_numpy(float)
            for parameter in parameters
        }
        if any(len(values) == 0 for values in retained_arrays.values()):
            logo_rows.append(
                {
                    "omitted_before_id": omitted_group,
                    "status": "not_estimable",
                    "selected_target": "",
                    "selected_score": np.nan,
                    "runner_up": "",
                    "score_margin": np.nan,
                }
            )
            continue
        values = np.stack(
            [retained_arrays[parameter].mean(axis=0) for parameter in parameters]
        )
        scores = minmax_scores(values)
        ranking = np.argsort(-scores)
        logo_rows.append(
            {
                "omitted_before_id": omitted_group,
                "status": "ok",
                "selected_target": parameters[ranking[0]],
                "selected_score": float(scores[ranking[0]]),
                "runner_up": parameters[ranking[1]],
                "score_margin": float(
                    scores[ranking[0]] - scores[ranking[1]]
                ),
            }
        )
    logo = pd.DataFrame(logo_rows)
    logo.to_csv(
        output / "leave_one_group_out.csv",
        index=False,
        encoding="utf-8-sig",
    )
    estimable_logo = logo.loc[logo["status"] == "ok"]
    logo_selected_fraction = float(
        (estimable_logo["selected_target"] == selected_target).mean()
    )
    logo_target_counts = {
        str(key): int(value)
        for key, value in estimable_logo["selected_target"]
        .value_counts()
        .to_dict()
        .items()
    }

    rng = np.random.default_rng(args.seed)
    sampled_means = np.stack(
        [
            values[
                rng.integers(
                    0,
                    len(values),
                    size=(args.bootstrap, balanced_groups),
                )
            ].mean(axis=1)
            for values in arrays.values()
        ],
        axis=1,
    )
    bootstrap_scores = np.stack(
        [minmax_scores(sampled_means[index]) for index in range(args.bootstrap)]
    )
    order = np.argsort(-bootstrap_scores, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(args.bootstrap)[:, None], order] = np.arange(
        1, len(parameters) + 1
    )
    bootstrap_summary = pd.DataFrame(
        [
            {
                "varied_param": parameter,
                "rank1_fraction": float((ranks[:, index] == 1).mean()),
                "mean_rank": float(ranks[:, index].mean()),
                "mean_score": float(bootstrap_scores[:, index].mean()),
                "groups_sampled_per_parameter": balanced_groups,
            }
            for index, parameter in enumerate(parameters)
        ]
    ).sort_values(["mean_rank", "rank1_fraction"], ascending=[True, False])
    bootstrap_summary.to_csv(
        output / "group_bootstrap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pair_metrics": str(Path(args.pair_metrics)),
        "bootstrap_unit": "before_id",
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "balanced_groups_per_parameter": balanced_groups,
        "selected_target": selected_target,
        "selected_score": selected_score,
        "runner_up": runner_up,
        "score_margin": margin,
        "selected_rank1_fraction": float(
            bootstrap_summary.loc[
                bootstrap_summary["varied_param"] == selected_target,
                "rank1_fraction",
            ].iloc[0]
        ),
        "leave_one_group_out_total": int(len(logo)),
        "leave_one_group_out_estimable": int(len(estimable_logo)),
        "leave_one_group_out_selected_fraction": logo_selected_fraction,
        "leave_one_group_out_target_counts": logo_target_counts,
        "group_counts": counts.to_dict(orient="records"),
        "ranking": full_ranking.to_dict(orient="records"),
        "selection_scope": (
            "outer-training groups only; outer-test groups were excluded"
        ),
    }
    (output / "selected_observability_target.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
