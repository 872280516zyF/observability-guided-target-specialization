from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None


def _safe_mean(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def bootstrap_ci(values: Iterable[float], n_bootstrap: int = 2000, seed: int = 42) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[~np.isnan(array)]
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(array, size=len(array), replace=True)
        means.append(np.mean(sample))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def parameter_summary(results: pd.DataFrame, stage: str, param_names: list[str]) -> pd.DataFrame:
    rows = []
    per_sample_mapes = []
    for param_name in param_names:
        true = results[f"true_{param_name}"].to_numpy(dtype=float)
        pred = results[f"{stage}_{param_name}"].to_numpy(dtype=float)
        error = pred - true
        abs_error = np.abs(error)
        ape = abs_error / np.maximum(np.abs(true), 1e-6) * 100.0
        per_sample_mapes.append(ape)
        ci_low, ci_high = bootstrap_ci(ape)
        rows.append(
            {
                "stage": stage,
                "parameter": param_name,
                "mae": _safe_mean(abs_error),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mape": _safe_mean(ape),
                "mape_std": _safe_std(ape),
                "mape_ci95_low": ci_low,
                "mape_ci95_high": ci_high,
            }
        )

    overall = np.mean(np.stack(per_sample_mapes, axis=1), axis=1)
    ci_low, ci_high = bootstrap_ci(overall)
    rows.append(
        {
            "stage": stage,
            "parameter": "overall_mean",
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": _safe_mean(overall),
            "mape_std": _safe_std(overall),
            "mape_ci95_low": ci_low,
            "mape_ci95_high": ci_high,
        }
    )
    return pd.DataFrame(rows)


def paired_improvement_summary(results: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    before = results[f"base_{metric_column}"].to_numpy(dtype=float)
    after = results[f"reranked_{metric_column}"].to_numpy(dtype=float)
    delta = after - before

    improved_mask = delta < 0
    worsened_mask = delta > 0
    ci_low, ci_high = bootstrap_ci(delta)

    row = {
        "metric": metric_column,
        "num_samples": int(len(delta)),
        "before_mean": _safe_mean(before),
        "after_mean": _safe_mean(after),
        "mean_delta": _safe_mean(delta),
        "delta_std": _safe_std(delta),
        "delta_ci95_low": ci_low,
        "delta_ci95_high": ci_high,
        "improved_count": int(improved_mask.sum()),
        "worsened_count": int(worsened_mask.sum()),
        "unchanged_count": int((delta == 0).sum()),
        "improved_ratio": float(improved_mask.mean()) if len(delta) else float("nan"),
        "worsened_ratio": float(worsened_mask.mean()) if len(delta) else float("nan"),
        "mean_improvement_on_improved": _safe_mean((-delta[improved_mask])) if improved_mask.any() else float("nan"),
        "mean_worsening_on_worsened": _safe_mean(delta[worsened_mask]) if worsened_mask.any() else float("nan"),
    }

    if stats is not None and len(delta) >= 2:
        try:
            t_stat, t_p = stats.ttest_rel(before, after)
            row["paired_t_stat"] = float(t_stat)
            row["paired_t_pvalue"] = float(t_p)
        except Exception:
            row["paired_t_stat"] = float("nan")
            row["paired_t_pvalue"] = float("nan")
        try:
            w_stat, w_p = stats.wilcoxon(before, after, zero_method="wilcox", alternative="two-sided")
            row["wilcoxon_stat"] = float(w_stat)
            row["wilcoxon_pvalue"] = float(w_p)
        except Exception:
            row["wilcoxon_stat"] = float("nan")
            row["wilcoxon_pvalue"] = float("nan")
    else:
        row["paired_t_stat"] = float("nan")
        row["paired_t_pvalue"] = float("nan")
        row["wilcoxon_stat"] = float("nan")
        row["wilcoxon_pvalue"] = float("nan")

    return pd.DataFrame([row])


def build_rerank_sample_table(results: pd.DataFrame) -> pd.DataFrame:
    sample_table = results.copy()
    sample_table["mean_ape_delta"] = sample_table["reranked_mean_ape"] - sample_table["base_mean_ape"]
    sample_table["image_score_delta"] = sample_table["reranked_image_score"] - sample_table["base_image_score"]
    sample_table["total_score_delta"] = sample_table["reranked_total_score"] - sample_table["base_total_score"]
    sample_table["mean_ape_improved"] = sample_table["mean_ape_delta"] < 0
    sample_table["image_score_improved"] = sample_table["image_score_delta"] < 0
    return sample_table

