#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.metadata import load_table  # noqa: E402


PARAMS = [
    ("frequency", ["frequency", "freq", "频率", "棰戠巼"], 1),
    ("pulse_width", ["pulse_width", "pulse", "脉宽", "鑴夊"], 2),
    ("speed", ["speed", "速度", "閫熷害"], 3),
    ("dpi", ["dpi", "DPI"], 4),
]
ID_ALIASES = ["sample_id", "编号", "缂栧彿", "id", "ID"]


PARAM_RANGES = {
    "frequency": (20.0, 95.0),
    "pulse_width": (25.0, 100.0),
    "speed": (30000.0, 50000.0),
    "dpi": (25.0, 175.0),
}
RAW_METRICS = ["rgb_mae", "one_minus_ssim", "edge_mae", "roi_mae"]
TEXTURE_METRICS = ["edge_mae", "edge_density_delta", "high_freq_energy_delta", "local_contrast_delta", "changed_area_ratio"]


def first_col(df: pd.DataFrame, aliases: Iterable[str], fallback_index: int | None = None) -> str:
    for alias in aliases:
        if alias in df.columns:
            return alias
    if fallback_index is not None and fallback_index < len(df.columns):
        return str(df.columns[fallback_index])
    raise ValueError(f"Missing required column from aliases: {list(aliases)}")


def clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def canonicalize_labels(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        df = load_table(path)
        out = pd.DataFrame()
        out["sample_id"] = df[first_col(df, ID_ALIASES, 0)].map(clean_id)
        for canonical, aliases, fallback in PARAMS:
            out[canonical] = pd.to_numeric(df[first_col(df, aliases, fallback)], errors="coerce")
        out["before_id"] = df["before_id"].map(clean_id) if "before_id" in df.columns else out["sample_id"]
        out["pattern_id"] = df["pattern_id"].map(clean_id) if "pattern_id" in df.columns else ""
        out["source_csv"] = str(path)
        rows.append(out)
    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined = combined.dropna(subset=[name for name, _, _ in PARAMS]).copy()
    combined = combined.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)
    return combined


def image_array(path: Path, img_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    return np.asarray(img, dtype=np.float32) / 255.0


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    return np.sqrt(gx * gx + gy * gy)


def laplacian_energy(gray: np.ndarray) -> float:
    center = gray[1:-1, 1:-1]
    lap = (
        4.0 * center
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    return float(np.mean(np.abs(lap)))


def local_contrast(gray: np.ndarray) -> float:
    padded = np.pad(gray, 1, mode="edge")
    local_mean = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0
    return float(np.mean(np.abs(gray - local_mean)))


def global_ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = float(gray_a.mean())
    mu_b = float(gray_b.mean())
    var_a = float(gray_a.var())
    var_b = float(gray_b.var())
    cov = float(((gray_a - mu_a) * (gray_b - mu_b)).mean())
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return float(num / den) if den else 1.0


def compare_images(path_a: Path, path_b: Path, img_size: int) -> dict[str, float]:
    a = image_array(path_a, img_size)
    b = image_array(path_b, img_size)
    diff = b - a
    abs_diff = np.abs(diff)
    gray_a = a.mean(axis=2)
    gray_b = b.mean(axis=2)
    mse = float(np.mean(diff * diff))
    psnr = float(20.0 * math.log10(1.0 / math.sqrt(max(mse, 1e-12))))
    ssim = global_ssim(gray_a, gray_b)
    grad_a = gradient_magnitude(gray_a)
    grad_b = gradient_magnitude(gray_b)
    gray_diff = abs_diff.mean(axis=2)
    edge_mae = float(np.mean(np.abs(grad_b - grad_a)))
    mask = gray_diff > max(0.03, float(np.percentile(gray_diff, 75)))
    roi_mae = float(gray_diff[mask].mean()) if np.any(mask) else float(gray_diff.mean())
    edge_density_a = float(np.mean(grad_a > 0.05))
    edge_density_b = float(np.mean(grad_b > 0.05))
    local_contrast_a = local_contrast(gray_a)
    local_contrast_b = local_contrast(gray_b)
    high_freq_a = laplacian_energy(gray_a)
    high_freq_b = laplacian_energy(gray_b)
    return {
        "rgb_mae": float(abs_diff.mean()),
        "rgb_rmse": float(math.sqrt(mse)),
        "psnr": psnr,
        "global_ssim": ssim,
        "one_minus_ssim": float(1.0 - ssim),
        "edge_mae": edge_mae,
        "roi_mae": roi_mae,
        "changed_area_ratio": float(np.mean(gray_diff > 0.05)),
        "edge_density_delta": abs(edge_density_b - edge_density_a),
        "high_freq_energy_delta": abs(high_freq_b - high_freq_a),
        "local_contrast_delta": abs(local_contrast_b - local_contrast_a),
    }


def build_pairs(df: pd.DataFrame, after_dir: Path, max_pairs_per_param: int) -> pd.DataFrame:
    pair_rows = []
    param_names = [name for name, _, _ in PARAMS]
    for varied in param_names:
        fixed = [p for p in param_names if p != varied]
        group_cols = ["before_id", "pattern_id"] + fixed
        for _, group in df.groupby(group_cols, dropna=False):
            values = (
                group.sort_values(["before_id", "pattern_id", varied, "sample_id"])
                .drop_duplicates(subset=[varied], keep="first")
                .reset_index(drop=True)
            )
            if values[varied].nunique() < 2:
                continue
            for idx in range(len(values) - 1):
                a = values.iloc[idx]
                b = values.iloc[idx + 1]
                path_a = after_dir / f"{a['sample_id']}.jpg"
                path_b = after_dir / f"{b['sample_id']}.jpg"
                if not path_a.exists() or not path_b.exists():
                    continue
                lower, upper = PARAM_RANGES[varied]
                delta_param = float(b[varied] - a[varied])
                delta_param_abs = abs(delta_param)
                if delta_param_abs <= 0:
                    continue
                delta_param_norm = delta_param_abs / max(upper - lower, 1e-12)
                pair_rows.append(
                    {
                        "varied_param": varied,
                        "sample_id_a": a["sample_id"],
                        "sample_id_b": b["sample_id"],
                        "before_id": a["before_id"],
                        "pattern_id": a["pattern_id"],
                        "value_a": float(a[varied]),
                        "value_b": float(b[varied]),
                        "delta_param": delta_param,
                        "delta_param_abs": delta_param_abs,
                        "delta_param_norm": delta_param_norm,
                        **{f"fixed_{p}": float(a[p]) for p in fixed},
                        "after_a": str(path_a),
                        "after_b": str(path_b),
                    }
                )
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        return pairs
    if max_pairs_per_param > 0:
        parts = []
        for _, group in pairs.groupby("varied_param"):
            if len(group) > max_pairs_per_param:
                parts.append(group.sample(n=max_pairs_per_param, random_state=20260627))
            else:
                parts.append(group)
        pairs = pd.concat(parts, ignore_index=True)
    return pairs


def add_normalized_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    denom = pd.to_numeric(scored["delta_param_norm"], errors="coerce").clip(lower=1e-9)
    for metric in sorted(set(RAW_METRICS + TEXTURE_METRICS + ["rgb_rmse"])):
        if metric in scored.columns:
            scored[f"{metric}_per_norm_step"] = pd.to_numeric(scored[metric], errors="coerce") / denom
    return scored


def rank_by_metrics(summary: pd.DataFrame, metrics: list[str], score_name: str) -> pd.DataFrame:
    if summary.empty:
        return summary
    rank = summary.copy()
    score = np.zeros(len(rank), dtype=float)
    used = []
    for metric in metrics:
        if metric not in rank.columns:
            continue
        vals = rank[metric].to_numpy(dtype=float)
        denom = vals.max() - vals.min()
        score += (vals - vals.min()) / denom if denom > 0 else 0
        used.append(metric)
    rank[score_name] = score / max(len(used), 1)
    rank["rank_metrics"] = ",".join(used)
    rank = rank.sort_values(score_name, ascending=False).reset_index(drop=True)
    rank["sensitivity_rank"] = np.arange(1, len(rank) + 1)
    return rank


def balanced_bootstrap_rank(scored: pd.DataFrame, metrics: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    if scored.empty or n_boot <= 0:
        return pd.DataFrame()
    groups = {param: group.reset_index(drop=True) for param, group in scored.groupby("varied_param")}
    if not groups:
        return pd.DataFrame()
    n_balanced = min(len(group) for group in groups.values())
    if n_balanced <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    records = []
    for boot_idx in range(n_boot):
        rows = []
        for param, group in groups.items():
            sampled = group.iloc[rng.integers(0, len(group), size=n_balanced)]
            row = {"varied_param": param, "bootstrap_iter": boot_idx}
            for metric in metrics:
                if metric in sampled.columns:
                    row[metric] = float(pd.to_numeric(sampled[metric], errors="coerce").mean())
            rows.append(row)
        boot = pd.DataFrame(rows)
        ranked = rank_by_metrics(boot, metrics, "bootstrap_composite_score")
        for _, row in ranked.iterrows():
            records.append(
                {
                    "bootstrap_iter": boot_idx,
                    "varied_param": row["varied_param"],
                    "bootstrap_rank": int(row["sensitivity_rank"]),
                    "bootstrap_composite_score": float(row["bootstrap_composite_score"]),
                    "n_balanced_pairs_per_param": n_balanced,
                }
            )
    detail = pd.DataFrame(records)
    summary_rows = []
    for param, group in detail.groupby("varied_param"):
        summary_rows.append(
            {
                "varied_param": param,
                "n_bootstrap": int(group["bootstrap_iter"].nunique()),
                "n_balanced_pairs_per_param": int(group["n_balanced_pairs_per_param"].iloc[0]),
                "mean_bootstrap_rank": float(group["bootstrap_rank"].mean()),
                "median_bootstrap_rank": float(group["bootstrap_rank"].median()),
                "rank1_fraction": float(np.mean(group["bootstrap_rank"] == 1)),
                "mean_bootstrap_composite_score": float(group["bootstrap_composite_score"].mean()),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(["mean_bootstrap_rank", "rank1_fraction"], ascending=[True, False]).reset_index(drop=True)


def summarize(scored: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_cols = sorted(set(RAW_METRICS + TEXTURE_METRICS + ["rgb_rmse"]))
    normalized_metric_cols = [f"{metric}_per_norm_step" for metric in metric_cols]
    summary_rows = []
    for param, group in scored.groupby("varied_param"):
        row = {"varied_param": param, "n_pairs": int(len(group))}
        row["delta_param_norm_mean"] = float(pd.to_numeric(group["delta_param_norm"], errors="coerce").mean())
        row["delta_param_norm_median"] = float(pd.to_numeric(group["delta_param_norm"], errors="coerce").median())
        for metric in metric_cols + normalized_metric_cols:
            if metric not in group.columns:
                continue
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary, summary, summary, summary, summary
    raw_rank = rank_by_metrics(summary, [f"{metric}_mean" for metric in RAW_METRICS], "raw_composite_sensitivity_score")
    normalized_rank = rank_by_metrics(
        summary,
        [f"{metric}_per_norm_step_mean" for metric in RAW_METRICS],
        "normalized_composite_sensitivity_score",
    )
    texture_rank = rank_by_metrics(
        summary,
        [f"{metric}_per_norm_step_mean" for metric in TEXTURE_METRICS],
        "texture_normalized_sensitivity_score",
    )
    bootstrap_rank = balanced_bootstrap_rank(
        scored,
        [f"{metric}_per_norm_step" for metric in TEXTURE_METRICS],
        n_boot=n_boot,
        seed=seed,
    )
    return summary, raw_rank, normalized_rank, texture_rank, bootstrap_rank


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    raw_rank: pd.DataFrame,
    normalized_rank: pd.DataFrame,
    texture_rank: pd.DataFrame,
    bootstrap_rank: pd.DataFrame,
) -> None:
    lines = [
        "# Parameter image-sensitivity analysis",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Pairs are one-factor comparisons: same before_id, pattern_id, and three fixed parameters; one laser parameter varies.",
        "",
    ]
    if summary.empty:
        lines.append("No valid one-factor image pairs were found.")
    else:
        best_norm = normalized_rank.iloc[0] if not normalized_rank.empty else None
        best_texture = texture_rank.iloc[0] if not texture_rank.empty else None
        best_boot = bootstrap_rank.iloc[0] if not bootstrap_rank.empty else None
        lines += [
            "Fairness upgrades in this version:",
            "",
            "1. image change is normalized by parameter-step size;",
            "2. each parameter is compared under balanced bootstrap resampling;",
            "3. DPI-relevant local texture/density metrics are reported separately.",
            "",
            f"Top normalized global-response parameter: `{best_norm['varied_param'] if best_norm is not None else ''}`.",
            f"Top normalized texture-response parameter: `{best_texture['varied_param'] if best_texture is not None else ''}`.",
            f"Top balanced-bootstrap texture parameter: `{best_boot['varied_param'] if best_boot is not None else ''}`.",
            "",
            "## Raw Global Rank",
            "",
            md_table(raw_rank),
            "",
            "## Normalized Global Rank",
            "",
            md_table(normalized_rank),
            "",
            "## Normalized Texture Rank",
            "",
            md_table(texture_rank),
            "",
            "## Balanced Bootstrap Texture Rank",
            "",
            md_table(bootstrap_rank),
            "",
            "## Summary",
            "",
            md_table(summary),
        ]
    (output_dir / "parameter_sensitivity_dpi_branch_justification.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-factor image sensitivity analysis for laser parameters.")
    parser.add_argument("--label-csv", action="append", required=True, help="One or more label CSV/XLSX files.")
    parser.add_argument("--after-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--max-pairs-per-param", type=int, default=2000)
    parser.add_argument("--balanced-bootstrap-iters", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260627)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = canonicalize_labels([Path(p) for p in args.label_csv])
    pairs = build_pairs(labels, Path(args.after_dir), args.max_pairs_per_param)
    scored_rows = []
    for row in pairs.to_dict("records"):
        scored_rows.append({**row, **compare_images(Path(row["after_a"]), Path(row["after_b"]), args.img_size)})
    scored = pd.DataFrame(scored_rows)
    scored = add_normalized_metrics(scored) if not scored.empty else scored
    if not scored.empty:
        summary, raw_rank, normalized_rank, texture_rank, bootstrap_rank = summarize(
            scored,
            n_boot=args.balanced_bootstrap_iters,
            seed=args.bootstrap_seed,
        )
    else:
        summary = raw_rank = normalized_rank = texture_rank = bootstrap_rank = pd.DataFrame()
    labels.to_csv(output_dir / "canonical_labels_used.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(output_dir / "one_factor_pairs.csv", index=False, encoding="utf-8-sig")
    scored.to_csv(output_dir / "one_factor_pair_image_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "parameter_image_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    raw_rank.to_csv(output_dir / "parameter_sensitivity_rank_raw_global.csv", index=False, encoding="utf-8-sig")
    normalized_rank.to_csv(output_dir / "parameter_sensitivity_rank.csv", index=False, encoding="utf-8-sig")
    normalized_rank.to_csv(output_dir / "parameter_sensitivity_rank_normalized_global.csv", index=False, encoding="utf-8-sig")
    texture_rank.to_csv(output_dir / "parameter_sensitivity_rank_normalized_texture.csv", index=False, encoding="utf-8-sig")
    bootstrap_rank.to_csv(output_dir / "parameter_sensitivity_balanced_bootstrap_texture_rank.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir, summary, raw_rank, normalized_rank, texture_rank, bootstrap_rank)
    print(f"Parameter image sensitivity complete: {output_dir}")


if __name__ == "__main__":
    main()
