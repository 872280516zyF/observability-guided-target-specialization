#!/usr/bin/env python3
"""Evaluation-only runner for the grouped input/augmentation ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_inverse_experiment import (
    build_loader,
    build_model,
    evaluate,
    export_predictions,
)
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--before-dir", required=True)
    parser.add_argument("--after-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def config_with_defaults(config: dict) -> dict:
    defaults = {
        "backbone": "resnet18",
        "fusion": "concat_diff",
        "input_mode": "dual",
        "hidden_dim": 256,
        "dropout": 0.3,
        "head_type": "simple",
        "no_pretrained": False,
        "resize_mode": "letterbox",
        "augmentation_mode": "none",
        "img_size": 224,
        "selection_metric": "val_mean_mape",
        "seed": 42,
    }
    defaults.update(config)
    return defaults


def main() -> None:
    cli = parse_args()
    checkpoint_path = Path(cli.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in payload or "config" not in payload:
        raise ValueError("Checkpoint must contain state_dict and config")

    config = config_with_defaults(dict(payload["config"]))
    train_args = argparse.Namespace(**config)
    set_seed(int(train_args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Every parameter is overwritten by the checkpoint. Avoid a network lookup.
    pretrained_flag = bool(train_args.no_pretrained)
    train_args.no_pretrained = True
    model = build_model(train_args, device)
    train_args.no_pretrained = pretrained_flag
    model.load_state_dict(payload["state_dict"], strict=True)

    loader = build_loader(
        csv_path=cli.test_csv,
        before_dir=cli.before_dir,
        after_dir=cli.after_dir,
        img_size=int(train_args.img_size),
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        is_train=False,
        resize_mode=train_args.resize_mode,
        augmentation_mode="none",
    )
    criterion = nn.SmoothL1Loss()
    metrics = evaluate(model, loader, criterion, device)

    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = export_predictions(
        model,
        loader,
        device,
        output_dir / "test_predictions.csv",
        cli.condition_id,
        "test",
    )
    predictions["outer_fold"] = int(cli.outer_fold)
    predictions["seed"] = int(cli.seed)
    predictions["condition_id"] = cli.condition_id
    predictions["input_mode"] = train_args.input_mode
    predictions["fusion"] = train_args.fusion
    predictions["augmentation_mode"] = train_args.augmentation_mode
    predictions["resize_mode"] = train_args.resize_mode
    predictions.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "evaluation_scope": "outer_test_evaluation_only",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "best_selection_score": payload.get("best_selection_score"),
        "selection_metric": train_args.selection_metric,
        "condition_id": cli.condition_id,
        "outer_fold": int(cli.outer_fold),
        "seed": int(cli.seed),
        "input_mode": train_args.input_mode,
        "fusion": train_args.fusion,
        "augmentation_mode": train_args.augmentation_mode,
        "resize_mode": train_args.resize_mode,
        "num_test_samples": int(len(loader.dataset)),
        "test_metrics": {
            "loss": metrics.loss,
            "mae_norm": metrics.mae_norm,
            "mape_physical": metrics.mape_physical,
            "param_mae_norm": metrics.param_mae_norm,
            "param_mape_physical": metrics.param_mape_physical,
        },
        "mean_test_prediction_mape": float(predictions["mean_ape"].mean()),
    }
    (output_dir / "test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
