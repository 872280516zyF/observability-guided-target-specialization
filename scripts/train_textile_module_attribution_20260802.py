#!/usr/bin/env python3
"""Train one leakage-controlled textile module-attribution run.

The script deliberately reuses the canonical dataset, normalization, physical
error and checkpoint-selection helpers from ``run_dpi_branch_ablation.py``.
Pilot calls omit ``--test-csv`` entirely.  Confirmatory calls add the frozen
outer-test CSV only after architecture selection has been written to disk.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dpi_branch_ablation import (  # noqa: E402
    PARAM_NAMES,
    TargetedIntegratedExpertBNNet,
    build_loader,
    build_output_weights,
    count_trainable_parameters,
    evaluate,
    export_predictions,
    move_tensor_batch,
    selection_score,
    train_one_epoch,
)
from scripts.run_inverse_experiment import (  # noqa: E402
    BNRegressionHead,
    PARAM_SPECS,
    get_backbone,
)
from utils.seed import set_seed  # noqa: E402


PILOT_VARIANTS = [
    "specialist_head_plain",
    "specialist_head_vector_gate",
    "residual_adapter_core",
    "residual_adapter_mtan",
    "residual_adapter_deepten",
    "residual_adapter_mtan_deepten",
]
CONTROL_VARIANTS = [
    "current_full_no_attention",
    "capacity_matched_dual_shared",
]
ALL_VARIANTS = PILOT_VARIANTS + CONTROL_VARIANTS
REFERENCE_FULL_PARAMETER_COUNT = 23_347_844


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def make_resnet18_feature_map(pretrained: bool) -> nn.Module:
    if pretrained:
        try:
            network = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            network = models.resnet18(pretrained=True)
    else:
        try:
            network = models.resnet18(weights=None)
        except TypeError:
            network = models.resnet18(pretrained=False)
    return nn.Sequential(*list(network.children())[:-2])


def assemble_outputs(
    general: torch.Tensor,
    specialist: torch.Tensor,
    target_index: int,
) -> torch.Tensor:
    columns = []
    general_index = 0
    for parameter_index in range(len(PARAM_NAMES)):
        if parameter_index == int(target_index):
            columns.append(specialist)
        else:
            columns.append(general[:, general_index : general_index + 1])
            general_index += 1
    return torch.cat(columns, dim=1)


class SpecialistHeadNet(nn.Module):
    """Shared encoder with one separately supervised target head."""

    def __init__(
        self,
        target_index: int,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        vector_gate: bool,
    ) -> None:
        super().__init__()
        self.target_index = int(target_index)
        self.backbone, feature_dim = get_backbone("resnet18", pretrained)
        self.general_head = BNRegressionHead(
            feature_dim, hidden_dim, dropout, num_params=3
        )
        self.vector_gate = bool(vector_gate)
        if self.vector_gate:
            self.gate = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = nn.Identity()
        self.specialist_head = BNRegressionHead(
            feature_dim, hidden_dim, dropout, num_params=1
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.backbone(batch["effect"]).flatten(1)
        general = self.general_head(features)
        specialist_features = (
            features * self.gate(features) if self.vector_gate else features
        )
        specialist = self.specialist_head(specialist_features)
        return assemble_outputs(general, specialist, self.target_index)


class MTANLite(nn.Module):
    """Task-specific soft mask over the final convolutional feature map."""

    def __init__(self, channels: int = 512, bottleneck: int = 64) -> None:
        super().__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map * self.mask(feature_map)


class DeepTENLite(nn.Module):
    """Small end-to-end residual encoding layer for orderless texture cues."""

    def __init__(
        self, in_channels: int = 512, encoding_dim: int = 64, codewords: int = 8
    ) -> None:
        super().__init__()
        self.encoding_dim = int(encoding_dim)
        self.codeword_count = int(codewords)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, encoding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(encoding_dim),
            nn.ReLU(inplace=True),
        )
        self.codewords = nn.Parameter(
            torch.empty(self.codeword_count, self.encoding_dim)
        )
        self.log_scale = nn.Parameter(torch.zeros(self.codeword_count))
        nn.init.normal_(self.codewords, mean=0.0, std=0.05)
        self.output_dim = self.codeword_count * self.encoding_dim

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        projected = self.project(feature_map)
        samples = projected.flatten(2).transpose(1, 2)
        residuals = samples.unsqueeze(2) - self.codewords.view(
            1, 1, self.codeword_count, self.encoding_dim
        )
        squared_distance = residuals.square().sum(dim=-1)
        scale = F.softplus(self.log_scale).view(1, 1, -1) + 1e-4
        assignment = torch.softmax(-scale * squared_distance, dim=2)
        encoded = (assignment.unsqueeze(-1) * residuals).sum(dim=1)
        encoded = F.normalize(encoded, p=2, dim=-1)
        encoded = encoded.flatten(1)
        return F.normalize(encoded, p=2, dim=1)


class ResidualAdapterNet(nn.Module):
    """Shared four-output model with a routed zero-initialized residual."""

    def __init__(
        self,
        target_index: int,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        use_mtan: bool,
        use_deepten: bool,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.target_index = int(target_index)
        self.residual_scale = float(residual_scale)
        self.feature_map = make_resnet18_feature_map(pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.base_head = BNRegressionHead(
            512, hidden_dim, dropout, num_params=len(PARAM_NAMES)
        )
        self.mtan = MTANLite(512, 64) if use_mtan else nn.Identity()
        self.texture = DeepTENLite(512, 64, 8) if use_deepten else None
        adapter_dim = 512 + (self.texture.output_dim if self.texture else 0)
        self.adapter = nn.Sequential(
            nn.Linear(adapter_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        final_linear = self.adapter[-2]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        feature_map = self.feature_map(batch["effect"])
        pooled = self.pool(feature_map).flatten(1)
        base = self.base_head(pooled)
        specialist_map = self.mtan(feature_map)
        parts = [self.pool(specialist_map).flatten(1)]
        if self.texture is not None:
            parts.append(self.texture(specialist_map))
        adapter_input = torch.cat(parts, dim=1)
        residual = self.adapter(adapter_input) * self.residual_scale
        prediction = base.clone()
        column = prediction[:, self.target_index : self.target_index + 1]
        prediction[:, self.target_index : self.target_index + 1] = (
            column + residual
        ).clamp(0.0, 1.0)
        return prediction


class CapacityMatchedDualShared(nn.Module):
    """Two active ResNet paths with one non-specialized four-output head."""

    def __init__(
        self,
        pretrained: bool,
        dropout: float,
        target_parameter_count: int,
    ) -> None:
        super().__init__()
        self.encoder_a, feature_dim = get_backbone("resnet18", pretrained)
        self.encoder_b, feature_dim_b = get_backbone("resnet18", pretrained)
        if feature_dim != feature_dim_b:
            raise RuntimeError("Dual-shared encoder dimensions differ")

        backbone_parameters = sum(
            parameter.numel()
            for module in (self.encoder_a, self.encoder_b)
            for parameter in module.parameters()
            if parameter.requires_grad
        )

        def head_parameters(width: int) -> int:
            # Linear(2d,w), BN(w), Linear(w,4), including biases.
            return (2 * feature_dim * width + width) + 2 * width + (
                width * len(PARAM_NAMES) + len(PARAM_NAMES)
            )

        widths = range(64, 2049)
        self.fusion_width = min(
            widths,
            key=lambda width: abs(
                backbone_parameters
                + head_parameters(width)
                - int(target_parameter_count)
            ),
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim * 2, self.fusion_width),
            nn.BatchNorm1d(self.fusion_width),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_width, len(PARAM_NAMES)),
            nn.Sigmoid(),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = batch["effect"]
        feature_a = self.encoder_a(image).flatten(1)
        feature_b = self.encoder_b(image).flatten(1)
        return self.head(torch.cat([feature_a, feature_b], dim=1))


class UnitFeatureGate(nn.Module):
    """Return unit weights so ``feature * gate(feature)`` is unchanged."""

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(feature)


def build_model(args: argparse.Namespace) -> nn.Module:
    target_index = PARAM_NAMES.index(args.expert_target)
    pretrained = not args.no_pretrained
    if args.variant == "specialist_head_plain":
        model = SpecialistHeadNet(
            target_index, pretrained, args.hidden_dim, args.dropout, False
        )
    elif args.variant == "specialist_head_vector_gate":
        model = SpecialistHeadNet(
            target_index, pretrained, args.hidden_dim, args.dropout, True
        )
    elif args.variant.startswith("residual_adapter_"):
        model = ResidualAdapterNet(
            target_index=target_index,
            pretrained=pretrained,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            use_mtan="mtan" in args.variant,
            use_deepten="deepten" in args.variant,
            residual_scale=args.residual_scale,
        )
    elif args.variant == "current_full_no_attention":
        model = TargetedIntegratedExpertBNNet(
            backbone_name="resnet18",
            pretrained=pretrained,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            texture_dim=args.texture_dim,
            expert_target=args.expert_target,
            texture_guided=True,
        )
        # TargetedIntegratedExpertBNNet multiplies features by the returned
        # gate.  Identity would therefore square the feature vector rather
        # than disable attention; an all-ones gate is the true ablation.
        model.specialist_attention = UnitFeatureGate()
    elif args.variant == "capacity_matched_dual_shared":
        model = CapacityMatchedDualShared(
            pretrained=pretrained,
            dropout=args.dropout,
            target_parameter_count=REFERENCE_FULL_PARAMETER_COUNT,
        )
    else:
        raise ValueError(args.variant)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=ALL_VARIANTS, required=True)
    parser.add_argument("--expert-target", choices=PARAM_NAMES, required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", default="")
    parser.add_argument("--before-dir", required=True)
    parser.add_argument("--after-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--resize-mode", choices=["stretch"], default="stretch")
    parser.add_argument(
        "--augmentation-mode", choices=["weak"], default="weak"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--texture-dim", type=int, default=64)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--selection-metric", default="val_mean_mape")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def add_shared_training_args(args: argparse.Namespace) -> None:
    # Imported canonical helpers expect these legacy CLI attributes.
    args.loss_type = "smooth_l1"
    args.frequency_loss_weight = 1.0
    args.pulse_width_loss_weight = 1.0
    args.speed_loss_weight = 1.0
    args.dpi_loss_weight = 1.0
    args.dpi_aux_weight = 0.0
    args.dpi_ordinal_bins = 8
    args.expert_ordinal_sigma = 0.75
    args.selection_mean_weight = 0.25
    args.selection_non_dpi_max_weight = 0.10
    args.group_balanced_sampler = False


def main() -> None:
    args = parse_args()
    add_shared_training_args(args)
    set_seed(args.seed)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = run_dir / "checkpoints"
    predictions = run_dir / "predictions"
    outer_test = run_dir / "outer_test"
    logs = run_dir / "logs"
    for path in (checkpoints, predictions, logs):
        path.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or "{}_fold{}_seed{}".format(
        args.model_id, args.outer_fold, args.seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_loader(
        args.train_csv,
        args.before_dir,
        args.after_dir,
        args.img_size,
        args.batch_size,
        args.num_workers,
        True,
        resize_mode=args.resize_mode,
        augmentation_mode=args.augmentation_mode,
        group_balanced_sampler=False,
    )
    val_loader = build_loader(
        args.val_csv,
        args.before_dir,
        args.after_dir,
        args.img_size,
        args.batch_size,
        args.num_workers,
        False,
        resize_mode=args.resize_mode,
        augmentation_mode="weak",
        group_balanced_sampler=False,
    )
    test_loader = None
    if args.test_csv:
        test_loader = build_loader(
            args.test_csv,
            args.before_dir,
            args.after_dir,
            args.img_size,
            args.batch_size,
            args.num_workers,
            False,
            resize_mode=args.resize_mode,
            augmentation_mode="weak",
            group_balanced_sampler=False,
        )

    model = build_model(args).to(device)
    parameter_count = count_trainable_parameters(model)
    if args.variant == "capacity_matched_dual_shared":
        relative_gap = abs(
            parameter_count - REFERENCE_FULL_PARAMETER_COUNT
        ) / float(REFERENCE_FULL_PARAMETER_COUNT)
        if relative_gap > 0.005:
            raise RuntimeError(
                "Dual-shared parameter gap exceeds 0.5%: {} vs {}".format(
                    parameter_count, REFERENCE_FULL_PARAMETER_COUNT
                )
            )

    output_weights = build_output_weights(args, device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    config = vars(args).copy()
    config.update(
        {
            "run_name": run_name,
            "input_mode": "after_only",
            "parameter_count": parameter_count,
            "reference_full_parameter_count": REFERENCE_FULL_PARAMETER_COUNT,
            "device": str(device),
            "evaluation_scope": (
                "inner_validation_and_outer_test"
                if args.test_csv
                else "inner_validation_only"
            ),
        }
    )
    write_json(run_dir / "config.json", config)
    write_json(
        run_dir / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
            ),
            "config": config,
        },
    )

    best_score = float("inf")
    best_state = None
    best_eval = None
    best_epoch = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, output_weights, optimizer, device, args
        )
        val_metrics = evaluate(model, val_loader, output_weights, device, args)
        scheduler.step()
        score = selection_score(val_metrics, args)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mean_mape": val_metrics["mape_physical"],
            "selection_score": score,
            "lr": scheduler.get_last_lr()[0],
        }
        for name, value in val_metrics["param_mape_physical"].items():
            row["val_{}_mape".format(name)] = value
        history.append(row)
        pd.DataFrame(history).to_csv(
            logs / "train_history.csv", index=False, encoding="utf-8-sig"
        )
        print(
            "epoch {:03d}/{:03d} train={:.6f} val_mean={:.4f}".format(
                epoch, args.epochs, train_loss, score
            ),
            flush=True,
        )
        if score < best_score:
            best_score = float(score)
            best_state = deepcopy(model.state_dict())
            best_eval = val_metrics
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": best_state,
                    "best_selection_score": best_score,
                    "best_eval": best_eval,
                    "config": config,
                },
                checkpoints / "best_model.pth",
            )

    if best_state is None or best_eval is None:
        raise RuntimeError("No validation-selected checkpoint was produced")
    model.load_state_dict(best_state)
    shutil.copy2(checkpoints / "best_model.pth", run_dir / "best_checkpoint.pth")
    val_start = time.perf_counter()
    val_predictions = export_predictions(
        model,
        val_loader,
        device,
        predictions / "val_predictions.csv",
        run_name,
        "val",
    )
    val_seconds = time.perf_counter() - val_start
    validation_summary = {
        "run_name": run_name,
        "model_id": args.model_id,
        "variant": args.variant,
        "outer_fold": args.outer_fold,
        "seed": args.seed,
        "expert_target": args.expert_target,
        "evaluation_scope": "inner_validation_only",
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_eval": best_eval,
        "parameter_count": parameter_count,
        "num_train_samples": len(train_loader.dataset),
        "num_val_samples": len(val_loader.dataset),
        "mean_val_prediction_mape": float(val_predictions["mean_ape"].mean()),
        "val_inference_ms_per_sample": float(
            val_seconds / max(len(val_predictions), 1) * 1000.0
        ),
        "outer_test_was_supplied": bool(args.test_csv),
    }
    write_json(run_dir / "validation_summary.json", validation_summary)

    test_metrics = None
    if test_loader is not None:
        outer_test.mkdir(parents=True, exist_ok=True)
        test_start = time.perf_counter()
        test_predictions = export_predictions(
            model,
            test_loader,
            device,
            outer_test / "test_predictions.csv",
            run_name,
            "test",
        )
        test_seconds = time.perf_counter() - test_start
        test_metrics = evaluate(model, test_loader, output_weights, device, args)
        test_summary = {
            **validation_summary,
            "evaluation_scope": "frozen_checkpoint_outer_test",
            "test_metrics": test_metrics,
            "num_test_samples": len(test_loader.dataset),
            "mean_test_prediction_mape": float(
                test_predictions["mean_ape"].mean()
            ),
            "test_inference_ms_per_sample": float(
                test_seconds / max(len(test_predictions), 1) * 1000.0
            ),
        }
        write_json(outer_test / "test_summary.json", test_summary)

    write_json(
        run_dir / "summary.json",
        {
            **validation_summary,
            "test_metrics": test_metrics,
            "num_test_samples": len(test_loader.dataset) if test_loader else 0,
        },
    )
    print("[DONE] {}".format(run_dir), flush=True)


if __name__ == "__main__":
    main()
