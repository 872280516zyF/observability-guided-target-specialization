#!/usr/bin/env python3
"""Train one dataset-local, fold-local multi-target observability model.

The caller supplies CSV files from one dataset and one outer fold.  Pilot mode
rejects an outer-test CSV.  Confirm mode requires one.  Multiple high-
observability targets may receive score-weighted loss and independent
zero-initialized residual heads.  Backbone and algorithm selection are handled
outside this trainer using only the corresponding fold's inner validation set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import train_condition_set_specialist_20260802 as base  # noqa: E402
from scripts import train_nist_set_aggregation_upgrade_20260803 as sets  # noqa: E402
from scripts import train_crossdomain_observability_optimizer_20260805 as prior  # noqa: E402


BACKBONES = ("resnet18", "efficientnet_b0", "convnext_tiny")
ALGORITHMS = (
    "shared_baseline",
    "weighted_shared",
    "multi_specialist",
    "multi_specialist_pcgrad",
)


def make_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    if name == "resnet18":
        try:
            network = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
        except AttributeError:
            network = models.resnet18(pretrained=pretrained)
        return nn.Sequential(*list(network.children())[:-2]), 512
    if name == "efficientnet_b0":
        try:
            network = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
        except AttributeError:
            network = models.efficientnet_b0(pretrained=pretrained)
        return network.features, 1280
    if name == "convnext_tiny":
        try:
            network = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            )
        except AttributeError:
            network = models.convnext_tiny(pretrained=pretrained)
        return network.features, 768
    raise ValueError("Unknown backbone {}".format(name))


class MultiTargetObservabilityModel(nn.Module):
    """Shared condition encoder with optional high-observability residual heads."""

    def __init__(
        self,
        parameter_count: int,
        high_indices: Sequence[int],
        algorithm: str,
        backbone: str,
        pretrained: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if algorithm not in ALGORITHMS:
            raise ValueError("Unknown algorithm {}".format(algorithm))
        self.algorithm = algorithm
        self.high_indices = tuple(int(index) for index in high_indices)
        self.encoder, feature_dim = make_backbone(backbone, pretrained)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        gate_hidden = min(256, max(64, feature_dim // 4))
        self.frame_gate = nn.Sequential(
            nn.Linear(feature_dim, gate_hidden), nn.Tanh(), nn.Linear(gate_hidden, 1)
        )
        self.moment_project = nn.Sequential(
            nn.Linear(feature_dim * 3, 512),
            nn.LayerNorm(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.base_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, parameter_count),
        )
        specialist_enabled = algorithm in ("multi_specialist", "multi_specialist_pcgrad")
        self.specialists = nn.ModuleDict()
        if specialist_enabled:
            for index in self.high_indices:
                head = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.SiLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(128, 1),
                )
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
                self.specialists[str(index)] = head

    def _encode_valid(self, image_sets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, count, channels, height, width = image_sets.shape
        flat = image_sets.reshape(batch * count, channels, height, width)
        valid = mask.reshape(-1)
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise RuntimeError("A batch contains no valid images")
        maps = self.encoder(flat.index_select(0, indices))
        valid_features = self.spatial_pool(maps).flatten(1)
        features = valid_features.new_zeros((batch * count, valid_features.shape[1]))
        features = features.index_copy(0, indices, valid_features)
        return features.reshape(batch, count, valid_features.shape[1])

    def _aggregate(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.frame_gate(features).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        gated = (weights * features).sum(1)
        standard_deviation = sets.masked_std(features, mask, gated)
        maximum = sets.masked_max(features, mask)
        return self.moment_project(torch.cat([gated, standard_deviation, maximum], dim=1))

    def forward(self, image_sets: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self._aggregate(self._encode_valid(image_sets, mask), mask)
        logits = self.base_head(pooled)
        if self.specialists:
            logits = logits.clone()
            for key, head in self.specialists.items():
                index = int(key)
                logits[:, index] = logits[:, index] + head(pooled).squeeze(1)
        return {"prediction": torch.sigmoid(logits)}


def weighted_objectives(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    high_indices: Sequence[int],
    score_weights: torch.Tensor,
    selected_priority: float,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    per_parameter = F.smooth_l1_loss(prediction, targets, reduction="none").mean(0)
    high_index_tensor = torch.as_tensor(high_indices, dtype=torch.long, device=targets.device)
    high_weights = score_weights.index_select(0, high_index_tensor)
    high_loss = (
        per_parameter.index_select(0, high_index_tensor) * high_weights
    ).sum() / high_weights.sum().clamp_min(1e-8)
    low_indices = [index for index in range(targets.shape[1]) if index not in high_indices]
    if not low_indices:
        return high_loss, high_loss, None
    low_loss = per_parameter[low_indices].mean()
    total = (selected_priority * high_loss + low_loss) / (selected_priority + 1.0)
    return total, selected_priority * high_loss / (selected_priority + 1.0), low_loss / (
        selected_priority + 1.0
    )


def high_validation_score(
    metrics: Dict[str, float],
    parameters: Sequence[str],
    high_targets: Sequence[str],
    observability_scores: Sequence[float],
) -> Tuple[float, float, float]:
    score_by_parameter = dict(zip(parameters, observability_scores))
    selected_weights = np.asarray(
        [score_by_parameter[target] for target in high_targets], dtype=np.float64
    )
    selected_values = np.asarray(
        [metrics["nmae_{}".format(target)] for target in high_targets], dtype=np.float64
    )
    high_nmae = float(np.average(selected_values, weights=selected_weights))
    low_values = [
        float(metrics["nmae_{}".format(parameter)])
        for parameter in parameters
        if parameter not in high_targets
    ]
    worst_low = max(low_values) if low_values else high_nmae
    score = high_nmae + 0.25 * float(metrics["mean_nmae"]) + 0.10 * worst_low
    return float(score), high_nmae, float(worst_low)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "confirm"), required=True)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", default="")
    parser.add_argument("--parameters", nargs="+", required=True)
    parser.add_argument("--high-targets", nargs="+", required=True)
    parser.add_argument("--observability-scores", nargs="+", type=float, required=True)
    parser.add_argument("--group-column", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--set-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--selected-priority", type=float, default=2.0)
    parser.add_argument("--max-gradient-norm", type=float, default=5.0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "pilot" and args.test_csv:
        raise ValueError("Pilot mode must not receive an outer-test CSV")
    if args.phase == "confirm" and not args.test_csv:
        raise ValueError("Confirm mode requires --test-csv")
    parameters = list(args.parameters)
    high_targets = list(args.high_targets)
    if len(args.observability_scores) != len(parameters):
        raise ValueError("observability scores must align with parameters")
    if not high_targets or not set(high_targets).issubset(parameters):
        raise ValueError("high targets must be a non-empty subset of parameters")
    if len(set(high_targets)) != len(high_targets):
        raise ValueError("high targets contain duplicates")
    high_indices = [parameters.index(target) for target in high_targets]

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_name = "validation_summary.json" if args.phase == "pilot" else "test_metrics.json"
    summary_path = output_directory / summary_name
    if summary_path.is_file():
        print("[SKIP] completed {}".format(output_directory), flush=True)
        return

    base.seed_everything(args.seed)
    train_frame = pd.read_csv(args.train_csv)
    scaler = base.ParameterScaler(train_frame, parameters)
    loader_common = (
        scaler,
        parameters,
        args.group_column,
        args.set_size,
        args.image_size,
        False,
    )
    _, train_loader = sets.make_loader(
        Path(args.train_csv), *loader_common, True, args.batch_size, args.num_workers, args.seed
    )
    _, validation_loader = sets.make_loader(
        Path(args.val_csv), *loader_common, False, args.batch_size, args.num_workers, args.seed
    )
    test_loader = None
    if args.test_csv:
        _, test_loader = sets.make_loader(
            Path(args.test_csv), *loader_common, False, args.batch_size, args.num_workers, args.seed
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTargetObservabilityModel(
        len(parameters),
        high_indices,
        args.algorithm,
        args.backbone,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    score_weights = torch.as_tensor(
        args.observability_scores, dtype=torch.float32, device=device
    ).clamp_min(1e-6)
    best_score, best_epoch, stale = math.inf, -1, 0
    checkpoint = output_directory / "best_model.pth"
    history: List[Dict[str, object]] = []

    for epoch in range(args.epochs):
        encoder_trainable = epoch >= args.freeze_backbone_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(encoder_trainable)
        model.train()
        if not encoder_trainable:
            model.encoder.eval()
        losses = []
        for images, mask, targets, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            prediction = model(images, mask)["prediction"]
            if args.algorithm == "shared_baseline":
                total_loss = F.smooth_l1_loss(prediction, targets)
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    args.max_gradient_norm,
                )
                optimizer.step()
            else:
                total_loss, high_loss, low_loss = weighted_objectives(
                    prediction,
                    targets,
                    high_indices,
                    score_weights,
                    args.selected_priority,
                )
                if args.algorithm == "multi_specialist_pcgrad" and low_loss is not None:
                    prior.pcgrad_step(
                        high_loss,
                        low_loss,
                        optimizer,
                        list(model.parameters()),
                        args.max_gradient_norm,
                    )
                else:
                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        args.max_gradient_norm,
                    )
                    optimizer.step()
            losses.append(float(total_loss.detach().cpu()))
        scheduler.step()

        validation_predictions = prior.predictions(
            model, validation_loader, scaler, parameters, device
        )
        metrics = base.group_equal_metrics(validation_predictions, parameters)
        selection_score, high_nmae, worst_low = high_validation_score(
            metrics, parameters, high_targets, args.observability_scores
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_selection_score": selection_score,
                "val_high_observability_nmae": high_nmae,
                "val_mean_nmae": float(metrics["mean_nmae"]),
                "val_worst_low_observability_nmae": worst_low,
                "encoder_trainable": encoder_trainable,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            "[{} {} fold{} {} {} seed{}] epoch={:03d} loss={:.6f} val_score={:.6f}".format(
                args.phase,
                args.dataset_id,
                args.outer_fold,
                args.backbone,
                args.algorithm,
                args.seed,
                epoch + 1,
                float(np.mean(losses)),
                selection_score,
            ),
            flush=True,
        )
        if selection_score < best_score - 1e-8:
            best_score, best_epoch, stale = selection_score, epoch + 1, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "dataset_id": args.dataset_id,
                    "backbone": args.backbone,
                    "algorithm": args.algorithm,
                    "parameters": parameters,
                    "high_targets": high_targets,
                    "observability_scores": args.observability_scores,
                    "scaler": scaler.as_dict(),
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    validation_predictions = prior.predictions(
        model, validation_loader, scaler, parameters, device
    )
    validation_predictions.to_csv(
        output_directory / "validation_predictions_conditions.csv", index=False
    )
    pd.DataFrame(history).to_csv(output_directory / "training_history.csv", index=False)
    validation_metrics = base.group_equal_metrics(validation_predictions, parameters)
    final_score, high_nmae, worst_low = high_validation_score(
        validation_metrics, parameters, high_targets, args.observability_scores
    )
    payload: Dict[str, object] = {
        "phase": args.phase,
        "dataset_id": args.dataset_id,
        "outer_fold": args.outer_fold,
        "backbone": args.backbone,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "parameters": parameters,
        "high_targets": high_targets,
        "observability_scores": dict(zip(parameters, args.observability_scores)),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_score": final_score,
        "validation_high_observability_nmae": high_nmae,
        "validation_worst_low_observability_nmae": worst_low,
        "validation_group_equal": validation_metrics,
        "selection_metric": (
            "score-weighted high-observability NMAE + 0.25 mean NMAE + "
            "0.10 worst low-observability NMAE"
        ),
        "outer_test_was_supplied": bool(args.test_csv),
        "condition_aggregation": "masked gated mean + standard deviation + maximum",
        "selected_priority": None if args.algorithm == "shared_baseline" else args.selected_priority,
        "pcgrad": args.algorithm == "multi_specialist_pcgrad",
        "augmentation": "none",
        "set_size": args.set_size,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "scaler": scaler.as_dict(),
    }
    if test_loader is not None:
        test_predictions = prior.predictions(model, test_loader, scaler, parameters, device)
        test_predictions.to_csv(
            output_directory / "test_predictions_conditions.csv", index=False
        )
        payload["group_equal"] = base.group_equal_metrics(test_predictions, parameters)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
