"""Utilities for saving training-curve figures from CSV logs.

The functions are intentionally lightweight so training scripts can call them
after optimization without changing the training logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_read_csv(path):
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            pass
    return pd.read_csv(path)


def _numeric_columns(frame):
    output = []
    for column in frame.columns:
        try:
            pd.to_numeric(frame[column])
            output.append(column)
        except Exception:
            pass
    return output


def plot_forward_training_log(log_file, out_dir=None, model_name=None):
    """Plot forward-generation training curves."""

    log_file = Path(log_file)
    if out_dir is None:
        out_dir = log_file.parent / (log_file.stem + "_figures")
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    frame = safe_read_csv(log_file)
    if "epoch" not in frame.columns:
        return []

    plt = _setup_matplotlib()
    saved = []
    title_prefix = model_name or log_file.stem
    loss_groups = [
        [
            column
            for column in (
                "train_loss", "val_loss", "train_g_loss", "val_g_loss",
                "train_d_loss", "val_d_loss",
            )
            if column in frame.columns
        ],
        [
            column
            for column in (
                "train_l1", "val_l1", "train_l1_loss", "val_l1_loss",
                "train_supervised_loss", "val_supervised_loss",
            )
            if column in frame.columns
        ],
        [
            column
            for column in (
                "train_ssim", "val_ssim", "train_perceptual", "val_perceptual",
                "train_adv_loss", "val_adv_loss",
            )
            if column in frame.columns
        ],
    ]
    loss_groups = [group for group in loss_groups if group]
    if loss_groups:
        figure, axes = plt.subplots(
            len(loss_groups), 1, figsize=(9, 3.6 * len(loss_groups)), sharex=True
        )
        if len(loss_groups) == 1:
            axes = [axes]
        for axis, group in zip(axes, loss_groups):
            for column in group:
                axis.plot(frame["epoch"], frame[column], marker="o", linewidth=1.8, label=column)
            axis.set_ylabel("Loss")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        axes[-1].set_xlabel("Epoch")
        figure.suptitle("%s loss curves" % title_prefix, fontsize=14, fontweight="bold")
        output = out_dir / "loss_curves.png"
        figure.savefig(str(output), dpi=300, bbox_inches="tight")
        figure.savefig(str(out_dir / "loss_curves.pdf"), bbox_inches="tight")
        plt.close(figure)
        saved.append(str(output))

    metric_columns = [
        column for column in ("mae", "mse", "psnr", "ssim", "r2") if column in frame.columns
    ]
    if metric_columns:
        figure, axes = plt.subplots(1, len(metric_columns), figsize=(4.0 * len(metric_columns), 3.8))
        if len(metric_columns) == 1:
            axes = [axes]
        for axis, column in zip(axes, metric_columns):
            plot_frame = frame.copy()
            if column == "r2":
                plot_frame = plot_frame[(plot_frame[column] != 0) | (plot_frame["epoch"] == frame["epoch"].max())]
            axis.plot(plot_frame["epoch"], plot_frame[column], marker="o", linewidth=2)
            axis.set_title(column.upper())
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
        figure.suptitle("%s evaluation metrics" % title_prefix, fontsize=14, fontweight="bold")
        output = out_dir / "metric_curves.png"
        figure.savefig(str(output), dpi=300, bbox_inches="tight")
        figure.savefig(str(out_dir / "metric_curves.pdf"), bbox_inches="tight")
        plt.close(figure)
        saved.append(str(output))
    return saved


def plot_inverse_training_log(log_file, out_dir=None, model_name=None):
    """Plot inverse-parameter training curves."""

    log_file = Path(log_file)
    if out_dir is None:
        out_dir = log_file.parent / (log_file.stem + "_figures")
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    frame = safe_read_csv(log_file)
    if "epoch" not in frame.columns:
        return []

    plt = _setup_matplotlib()
    saved = []
    title_prefix = model_name or log_file.stem
    basic_columns = [
        column
        for column in ("train_loss", "val_loss", "val_mae", "val_mape", "lr")
        if column in frame.columns
    ]
    if basic_columns:
        figure, axes = plt.subplots(1, len(basic_columns), figsize=(4.0 * len(basic_columns), 3.8))
        if len(basic_columns) == 1:
            axes = [axes]
        for axis, column in zip(axes, basic_columns):
            axis.plot(frame["epoch"], frame[column], marker="o", linewidth=2)
            axis.set_title(column)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
        figure.suptitle("%s training summary" % title_prefix, fontsize=14, fontweight="bold")
        output = out_dir / "training_summary.png"
        figure.savefig(str(output), dpi=300, bbox_inches="tight")
        figure.savefig(str(out_dir / "training_summary.pdf"), bbox_inches="tight")
        plt.close(figure)
        saved.append(str(output))

    validation_columns = [
        column
        for column in frame.columns
        if column.startswith("val_") and column.endswith("_loss") and column != "val_loss"
    ]
    training_columns = [
        column
        for column in frame.columns
        if column.startswith("train_") and column.endswith("_loss") and column != "train_loss"
    ]
    columns = training_columns + validation_columns
    if columns:
        figure, axis = plt.subplots(figsize=(9, 5))
        for column in columns:
            axis.plot(frame["epoch"], frame[column], marker="o", linewidth=1.7, label=column)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Normalized parameter loss")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
        axis.set_title("%s parameter-wise losses" % title_prefix, fontsize=13, fontweight="bold")
        output = out_dir / "parameter_loss_curves.png"
        figure.savefig(str(output), dpi=300, bbox_inches="tight")
        figure.savefig(str(out_dir / "parameter_loss_curves.pdf"), bbox_inches="tight")
        plt.close(figure)
        saved.append(str(output))
    return saved


def summarize_logs(log_paths, task, out_dir, figure_prefix):
    """Create a compact cross-model comparison from training logs."""

    rows = []
    for path in log_paths:
        path = Path(path)
        if not path.exists():
            continue
        frame = safe_read_csv(path)
        if len(frame) == 0 or "epoch" not in frame.columns:
            continue
        numeric_columns = _numeric_columns(frame)
        model_name = path.parent.parent.name if path.parent.name == "logs" else path.stem
        final = frame.iloc[-1]
        best = final
        if task == "forward" and "mae" in frame.columns:
            best = frame.loc[pd.to_numeric(frame["mae"]).idxmin()]
        elif "val_loss" in frame.columns:
            best = frame.loc[pd.to_numeric(frame["val_loss"]).idxmin()]
        elif "val_g_loss" in frame.columns:
            best = frame.loc[pd.to_numeric(frame["val_g_loss"]).idxmin()]
        row = {"model": model_name, "log_file": str(path), "best_epoch": int(best["epoch"])}
        for column in numeric_columns:
            if column == "epoch":
                continue
            row["best_" + column] = float(best[column])
            row["final_" + column] = float(final[column])
        rows.append(row)

    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    summary = pd.DataFrame(rows)
    csv_path = out_dir / ("%s_summary.csv" % figure_prefix)
    summary.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
    if len(summary) == 0:
        return str(csv_path), []

    plt = _setup_matplotlib()
    saved = []
    if task == "forward":
        columns = [
            column
            for column in ("best_mae", "best_psnr", "best_ssim", "best_r2")
            if column in summary.columns
        ]
    else:
        columns = [
            column
            for column in ("best_val_loss", "best_val_mae", "best_val_mape")
            if column in summary.columns
        ]
    if columns:
        figure, axes = plt.subplots(1, len(columns), figsize=(4.2 * len(columns), 4.2))
        if len(columns) == 1:
            axes = [axes]
        x_values = np.arange(len(summary))
        for axis, column in zip(axes, columns):
            axis.bar(x_values, summary[column].values, color="#4E79A7", alpha=0.85)
            axis.set_xticks(x_values)
            axis.set_xticklabels(summary["model"].values, rotation=35, ha="right")
            axis.set_title(column.replace("best_", ""))
            axis.grid(axis="y", alpha=0.25)
        figure.suptitle("%s cross-model comparison" % task.capitalize(), fontsize=14, fontweight="bold")
        output = out_dir / ("%s_comparison.png" % figure_prefix)
        figure.savefig(str(output), dpi=300, bbox_inches="tight")
        figure.savefig(str(out_dir / ("%s_comparison.pdf" % figure_prefix)), bbox_inches="tight")
        plt.close(figure)
        saved.append(str(output))
    return str(csv_path), saved
