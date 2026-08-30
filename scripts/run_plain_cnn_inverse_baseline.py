#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for p in (CURRENT_DIR, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from run_inverse_experiment import (
    PARAM_SPECS,
    build_loader,
    count_trainable_parameters,
    evaluate,
    export_predictions,
    run_epoch,
    write_run_manifest,
)
from utils.experiment_plots import plot_inverse_training_log
from utils.seed import set_seed


class PlainCNNInverseNet(nn.Module):
    """A deliberately plain single-image CNN baseline without pretrained modules."""

    def __init__(self, width: int = 32, image_key: str = "effect"):
        super().__init__()
        self.image_key = image_key
        self.features = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 4, width * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 8, len(PARAM_SPECS)),
            nn.Sigmoid(),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.regressor(self.features(batch[self.image_key]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plain single-image CNN inverse baseline.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--before-dir", required=True)
    parser.add_argument("--after-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument(
        "--selection-metric",
        choices=["val_loss", "val_mean_mape", "val_dpi_mape"],
        default="val_mean_mape",
        help="Checkpoint metric. Use val_mean_mape for the grouped fair-comparison suite.",
    )
    return parser.parse_args()


def selection_score(result, metric: str) -> float:
    if metric == "val_loss":
        return float(result.loss)
    if metric == "val_mean_mape":
        return float(result.mape_physical)
    if metric == "val_dpi_mape":
        return float(result.param_mape_physical["dpi"])
    raise ValueError(metric)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name or f"plain_cnn_baseline_seed{args.seed}"
    run_dir = Path(args.output_dir) / run_name
    checkpoints_dir = run_dir / "checkpoints"
    predictions_dir = run_dir / "predictions"
    logs_dir = run_dir / "logs"
    for path in (checkpoints_dir, predictions_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"run_dir: {run_dir}")

    train_loader = build_loader(args.train_csv, args.before_dir, args.after_dir, args.img_size, args.batch_size, args.num_workers, True)
    val_loader = build_loader(args.val_csv, args.before_dir, args.after_dir, args.img_size, args.batch_size, args.num_workers, False)
    test_loader = build_loader(args.test_csv, args.before_dir, args.after_dir, args.img_size, args.batch_size, args.num_workers, False)

    model = PlainCNNInverseNet(width=args.width).to(device)
    parameter_count = count_trainable_parameters(model)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "device": str(device),
            "run_name": run_name,
            "architecture": "plain_single_image_cnn",
            "input_mode": "after_only",
            "pretrained": False,
            "normalization": "same_image_transform_as_inverse_experiment",
            "selection_metric": args.selection_metric,
            "loss_type": "smooth_l1",
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_manifest(
        run_dir,
        {
            "run_name": run_name,
            "config": config_payload,
            "parameter_count": parameter_count,
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "test_csv": args.test_csv,
        },
    )

    history_rows = []
    best_state = None
    best_result = None
    best_selection_score = float("inf")
    log_file = logs_dir / f"{run_name}_history.csv"

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result.loss,
            "val_mae": val_result.mae_norm,
            "val_mape": val_result.mape_physical,
            "lr": scheduler.get_last_lr()[0],
        }
        for key, value in val_result.param_mae_norm.items():
            row[f"val_{key}_loss"] = value
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(log_file, index=False, encoding="utf-8-sig")

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_result.loss:.6f} | "
            f"val_mae={val_result.mae_norm:.6f} | val_mape={val_result.mape_physical:.4f}"
        )
        current_selection_score = selection_score(val_result, args.selection_metric)
        if current_selection_score < best_selection_score:
            best_selection_score = current_selection_score
            best_result = deepcopy(val_result.__dict__)
            best_state = deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": best_state,
                    "optimizer": optimizer.state_dict(),
                    "best_selection_score": best_selection_score,
                    "selection_metric": args.selection_metric,
                    "run_name": run_name,
                    "config": config_payload,
                },
                checkpoints_dir / "best_model.pth",
            )
            shutil.copy2(checkpoints_dir / "best_model.pth", run_dir / "best_checkpoint.pth")

    if best_state is None or best_result is None:
        raise RuntimeError("Training finished without a valid checkpoint")

    model.load_state_dict(best_state)
    val_start = time.perf_counter()
    val_predictions = export_predictions(model, val_loader, device, predictions_dir / "val_predictions.csv", run_name, "val")
    val_inference_seconds = time.perf_counter() - val_start
    test_start = time.perf_counter()
    test_predictions = export_predictions(model, test_loader, device, predictions_dir / "test_predictions.csv", run_name, "test")
    test_inference_seconds = time.perf_counter() - test_start

    summary = {
        "run_name": run_name,
        "parameter_count": parameter_count,
        "best_selection_score": best_selection_score,
        "best_val_loss": float(best_result["loss"]),
        "best_eval": best_result,
        "num_train_samples": int(len(train_loader.dataset)),
        "num_val_samples": int(len(val_loader.dataset)),
        "num_test_samples": int(len(test_loader.dataset)),
        "val_prediction_file": str(predictions_dir / "val_predictions.csv"),
        "test_prediction_file": str(predictions_dir / "test_predictions.csv"),
        "mean_val_prediction_mape": float(val_predictions["mean_ape"].mean()),
        "mean_test_prediction_mape": float(test_predictions["mean_ape"].mean()),
        "val_inference_seconds": val_inference_seconds,
        "test_inference_seconds": test_inference_seconds,
        "val_inference_ms_per_sample": float(val_inference_seconds / max(len(val_predictions), 1) * 1000.0),
        "test_inference_ms_per_sample": float(test_inference_seconds / max(len(test_predictions), 1) * 1000.0),
        "selection_metric": args.selection_metric,
        "loss_type": "smooth_l1",
        "frequency_loss_weight": 1.0,
        "pulse_width_loss_weight": 1.0,
        "speed_loss_weight": 1.0,
        "dpi_loss_weight": 1.0,
        "selection_mean_weight": 0.25,
        "selection_non_dpi_max_weight": 0.0,
        "architecture": "plain_single_image_cnn",
        "pretrained": False,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        shutil.copy2(log_file, run_dir / "train_log.csv")
        plot_inverse_training_log(log_file, logs_dir / f"{run_name}_figures", model_name=run_name)
    except Exception as exc:
        print(f"[WARN] failed to plot training curves: {exc}")

    print(f"best checkpoint: {checkpoints_dir / 'best_model.pth'}")
    print(f"val predictions: {predictions_dir / 'val_predictions.csv'}")
    print(f"test predictions: {predictions_dir / 'test_predictions.csv'}")
    print(f"summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
