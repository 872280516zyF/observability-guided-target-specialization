#!/usr/bin/env python3
"""Train one dataset-local domain-tailored observability model.

The experimental unit is a process condition represented by up to eight
images.  DED uses ordered coaxial frames and a temporal encoder.  NIST uses an
unordered set of cross-sectional micrographs and a geometry-aware set encoder.
The shallow-CNN baseline consumes the identical image sets and splits.

Pilot mode rejects outer-test input.  Confirm mode accepts it only after the
orchestration script has frozen the fold-local model and target selection.
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import train_condition_set_specialist_20260802 as base  # noqa: E402
from scripts import train_nist_set_aggregation_upgrade_20260803 as sets  # noqa: E402
from scripts import train_crossdomain_observability_optimizer_20260805 as prior  # noqa: E402
from scripts.train_multitarget_observability_20260806 import (  # noqa: E402
    BACKBONES,
    high_validation_score,
    make_backbone,
)


ARCHITECTURES = ("shallow_cnn", "domain_tailored")
ALGORITHMS = (
    "shared_baseline",
    "weighted_shared",
    "guided_rank",
    "guided_ordinal_rank",
)
GUIDED_ALGORITHMS = ("guided_rank", "guided_ordinal_rank")


class ShallowFrameEncoder(nn.Module):
    """Small four-block CNN used as a transparent lower-capacity baseline."""

    output_dim = 256

    def __init__(self) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        channels = [3, 32, 64, 128, 256]
        for incoming, outgoing in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(outgoing),
                    nn.SiLU(inplace=True),
                ]
            )
        self.network = nn.Sequential(*blocks)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def gray_from_imagenet_tensor(images: torch.Tensor) -> torch.Tensor:
    """Approximately invert the normalization applied by the shared loader."""

    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images * std + mean).mean(1, keepdim=True).clamp(0.0, 1.0)


def image_descriptors(images: torch.Tensor) -> torch.Tensor:
    """Differentiable intensity, shape and edge descriptors (10 values/image)."""

    gray = gray_from_imagenet_tensor(images)
    count, _, height, width = gray.shape
    flat = gray.flatten(2)
    mean = flat.mean(2)
    standard_deviation = flat.std(2, unbiased=False).clamp_min(1e-6)
    standardized = (gray - mean.view(count, 1, 1, 1)) / standard_deviation.view(
        count, 1, 1, 1
    )
    foreground = torch.sigmoid(3.0 * standardized)
    mass = foreground.sum((2, 3)).clamp_min(1e-6)
    x = torch.linspace(-1.0, 1.0, width, device=gray.device, dtype=gray.dtype).view(
        1, 1, 1, width
    )
    y = torch.linspace(-1.0, 1.0, height, device=gray.device, dtype=gray.dtype).view(
        1, 1, height, 1
    )
    center_x = (foreground * x).sum((2, 3)) / mass
    center_y = (foreground * y).sum((2, 3)) / mass
    spread_x = (
        foreground * (x - center_x.view(count, 1, 1, 1)).square()
    ).sum((2, 3)) / mass
    spread_y = (
        foreground * (y - center_y.view(count, 1, 1, 1)).square()
    ).sum((2, 3)) / mass
    gradient_x = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs().mean((2, 3))
    gradient_y = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs().mean((2, 3))
    maximum = flat.max(2).values
    return torch.cat(
        [
            mean,
            standard_deviation,
            foreground.mean((2, 3)),
            center_x,
            center_y,
            spread_x,
            spread_y,
            gradient_x,
            gradient_y,
            maximum,
        ],
        dim=1,
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype).unsqueeze(-1)
    return (values * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def masked_temporal_difference(
    values: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if values.shape[1] < 2:
        return values.new_zeros((values.shape[0], values.shape[2]))
    difference = (values[:, 1:] - values[:, :-1]).abs()
    valid = mask[:, 1:] & mask[:, :-1]
    return masked_mean(difference, valid)


class DomainTailoredObservabilityModel(nn.Module):
    """Shared prediction with DED-temporal or NIST-geometry representation."""

    def __init__(
        self,
        dataset_id: str,
        architecture: str,
        backbone: str,
        parameter_count: int,
        specialist_indices: Sequence[int],
        algorithm: str,
        ordinal_specs: Sequence[Tuple[np.ndarray, np.ndarray]],
        pretrained: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if dataset_id not in ("ded", "nist"):
            raise ValueError("Unsupported dataset {}".format(dataset_id))
        if architecture not in ARCHITECTURES:
            raise ValueError("Unsupported architecture {}".format(architecture))
        if algorithm not in ALGORITHMS:
            raise ValueError("Unsupported algorithm {}".format(algorithm))
        self.dataset_id = dataset_id
        self.architecture = architecture
        self.backbone = backbone
        self.algorithm = algorithm
        self.specialist_indices = tuple(int(value) for value in specialist_indices)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        if architecture == "shallow_cnn":
            self.encoder = ShallowFrameEncoder()
            self.feature_dim = self.encoder.output_dim
            self.representation_dim = 256
            self.shallow_project = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.LayerNorm(256),
                nn.SiLU(inplace=True),
            )
            self.descriptor_project = None
            self.frame_project = None
            self.temporal = None
            self.frame_gate = None
            self.condition_project = None
        else:
            self.encoder, self.feature_dim = make_backbone(backbone, pretrained)
            self.representation_dim = 512
            self.descriptor_project = nn.Sequential(
                nn.Linear(10, 64), nn.LayerNorm(64), nn.SiLU(inplace=True)
            )
            self.frame_project = nn.Sequential(
                nn.Linear(self.feature_dim + 64, 384),
                nn.LayerNorm(384),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.frame_gate = nn.Sequential(
                nn.Linear(384, 96), nn.Tanh(), nn.Linear(96, 1)
            )
            if dataset_id == "ded":
                self.temporal = nn.GRU(
                    input_size=384,
                    hidden_size=192,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
                condition_input = 384 * 4
            else:
                self.temporal = None
                condition_input = 384 * 3
            self.condition_project = nn.Sequential(
                nn.Linear(condition_input, 512),
                nn.LayerNorm(512),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.shallow_project = None

        self.base_head = nn.Sequential(
            nn.Linear(self.representation_dim, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, parameter_count),
        )
        self.specialists = nn.ModuleList()
        if algorithm in GUIDED_ALGORITHMS:
            for _ in self.specialist_indices:
                head = nn.Sequential(
                    nn.Linear(self.representation_dim, 128),
                    nn.SiLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(128, 1),
                )
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
                self.specialists.append(head)

        self.ordinal_heads = nn.ModuleList()
        if algorithm == "guided_ordinal_rank":
            if len(ordinal_specs) != len(self.specialist_indices):
                raise ValueError("Ordinal specifications do not match specialists")
            for slot, (thresholds, centers) in enumerate(ordinal_specs):
                if len(thresholds) < 1 or len(centers) != len(thresholds) + 1:
                    raise ValueError("Invalid ordinal specification at slot {}".format(slot))
                self.register_buffer(
                    "ordinal_thresholds_{}".format(slot),
                    torch.as_tensor(thresholds, dtype=torch.float32),
                )
                self.register_buffer(
                    "ordinal_centers_{}".format(slot),
                    torch.as_tensor(centers, dtype=torch.float32),
                )
                self.ordinal_heads.append(
                    nn.Sequential(
                        nn.Linear(self.representation_dim, 128),
                        nn.SiLU(inplace=True),
                        nn.Dropout(dropout),
                        nn.Linear(128, len(thresholds)),
                    )
                )

    def _prepare_valid_images(
        self, image_sets: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        batch, count, channels, height, width = image_sets.shape
        flat = image_sets.reshape(batch * count, channels, height, width)
        valid = mask.reshape(-1)
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise RuntimeError("A batch contains no valid images")
        images = flat.index_select(0, indices)
        if self.dataset_id == "ded" and self.architecture == "domain_tailored":
            crop_y = max(1, height // 8)
            crop_x = max(1, width // 8)
            images = images[:, :, crop_y : height - crop_y, crop_x : width - crop_x]
            images = F.interpolate(
                images, size=(height, width), mode="bilinear", align_corners=False
            )
        return images, indices, batch, count

    def _frame_features(
        self, image_sets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        images, indices, batch, count = self._prepare_valid_images(image_sets, mask)
        maps = self.encoder(images)
        visual = self.spatial_pool(maps).flatten(1)
        if self.architecture == "domain_tailored":
            descriptors = self.descriptor_project(image_descriptors(images))
            visual = self.frame_project(torch.cat([visual, descriptors], dim=1))
        output = visual.new_zeros((batch * count, visual.shape[1]))
        output = output.index_copy(0, indices, visual)
        return output.reshape(batch, count, visual.shape[1])

    def _representation(
        self, image_sets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        features = self._frame_features(image_sets, mask)
        if self.architecture == "shallow_cnn":
            return self.shallow_project(masked_mean(features, mask))
        if self.temporal is not None:
            lengths = mask.sum(1).to(torch.long).clamp_min(1)
            packed = pack_padded_sequence(
                features,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.temporal(packed)
            features, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=mask.shape[1],
            )
        scores = self.frame_gate(features).squeeze(-1).masked_fill(~mask, -1e4)
        gated = (torch.softmax(scores, dim=1).unsqueeze(-1) * features).sum(1)
        standard_deviation = sets.masked_std(features, mask, gated)
        maximum = sets.masked_max(features, mask)
        pieces = [gated, standard_deviation, maximum]
        if self.dataset_id == "ded":
            pieces.append(masked_temporal_difference(features, mask))
        return self.condition_project(torch.cat(pieces, dim=1))

    @staticmethod
    def _ordinal_expectation(
        logits: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        exceedance = torch.cummin(torch.sigmoid(logits), dim=1).values
        probabilities = torch.cat(
            [
                1.0 - exceedance[:, :1],
                exceedance[:, :-1] - exceedance[:, 1:],
                exceedance[:, -1:],
            ],
            dim=1,
        ).clamp_min(0.0)
        probabilities = probabilities / probabilities.sum(1, keepdim=True).clamp_min(1e-8)
        return (probabilities * centers.unsqueeze(0)).sum(1)

    def forward(self, image_sets: torch.Tensor, mask: torch.Tensor) -> Dict[str, object]:
        representation = self._representation(image_sets, mask)
        logits = self.base_head(representation)
        if self.specialists:
            logits = logits.clone()
            for index, head in zip(self.specialist_indices, self.specialists):
                logits[:, index] = logits[:, index] + head(representation).squeeze(1)
        prediction = torch.sigmoid(logits)
        ordinal_logits: List[torch.Tensor] = []
        if self.ordinal_heads:
            prediction = prediction.clone()
            for slot, (index, head) in enumerate(
                zip(self.specialist_indices, self.ordinal_heads)
            ):
                current = head(representation)
                ordinal_logits.append(current)
                centers = getattr(self, "ordinal_centers_{}".format(slot))
                expected = self._ordinal_expectation(current, centers)
                prediction[:, index] = 0.5 * prediction[:, index] + 0.5 * expected
        return {
            "prediction": prediction,
            "ordinal_logits": ordinal_logits,
            "representation": representation,
        }


def ranking_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    indices: Sequence[int],
    minimum_difference: float,
    temperature: float,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for index in indices:
        truth_difference = targets[:, index].unsqueeze(1) - targets[:, index].unsqueeze(0)
        prediction_difference = (
            prediction[:, index].unsqueeze(1) - prediction[:, index].unsqueeze(0)
        )
        upper = torch.triu(
            torch.ones_like(truth_difference, dtype=torch.bool), diagonal=1
        )
        valid = upper & (truth_difference.abs() >= minimum_difference)
        if valid.any():
            signed_margin = (
                truth_difference[valid].sign() * prediction_difference[valid]
            )
            losses.append(F.softplus(-signed_margin / temperature).mean())
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def training_objective(
    output: Dict[str, object],
    targets: torch.Tensor,
    model: DomainTailoredObservabilityModel,
    score_weights: torch.Tensor,
    high_indices: Sequence[int],
    selected_priority: float,
    rank_weight: float,
    ordinal_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    prediction = output["prediction"]
    per_parameter = F.smooth_l1_loss(prediction, targets, reduction="none").mean(0)
    base_loss = per_parameter.mean()
    if model.algorithm == "shared_baseline":
        total = base_loss
    elif model.algorithm == "weighted_shared":
        weights = torch.ones_like(per_parameter)
        index_tensor = torch.as_tensor(high_indices, device=targets.device, dtype=torch.long)
        selected_weights = score_weights.index_select(0, index_tensor)
        selected_weights = selected_weights / selected_weights.mean().clamp_min(1e-8)
        weights.index_copy_(0, index_tensor, 1.0 + selected_priority * selected_weights)
        total = (per_parameter * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        specialist_indices = list(model.specialist_indices)
        specialist_loss = per_parameter[specialist_indices].mean()
        rank = ranking_loss(
            prediction,
            targets,
            specialist_indices,
            minimum_difference=0.03,
            temperature=0.10,
        )
        ordinal = prediction.sum() * 0.0
        ordinal_logits = output["ordinal_logits"]
        if ordinal_logits:
            ordinal_losses = []
            for slot, (index, logits) in enumerate(
                zip(specialist_indices, ordinal_logits)
            ):
                thresholds = getattr(model, "ordinal_thresholds_{}".format(slot))
                labels = (targets[:, index].unsqueeze(1) > thresholds.unsqueeze(0)).to(
                    logits.dtype
                )
                ordinal_losses.append(F.binary_cross_entropy_with_logits(logits, labels))
            ordinal = torch.stack(ordinal_losses).mean()
        total = (
            base_loss
            + max(0.0, selected_priority - 1.0) * specialist_loss
            + rank_weight * rank
            + ordinal_weight * ordinal
        )
    return total, {
        "base": float(base_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def normalized_ordinal_specs(
    train_frame: pd.DataFrame,
    scaler: base.ParameterScaler,
    parameters: Sequence[str],
    specialist_targets: Sequence[str],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    result = []
    for target in specialist_targets:
        index = parameters.index(target)
        thresholds, centers = prior.ordinal_spec(
            pd.to_numeric(train_frame[target], errors="raise").to_numpy(float),
            maximum_classes=20,
        )
        scale = max(float(scaler.maximum[index] - scaler.minimum[index]), 1e-12)
        result.append(
            (
                ((thresholds - scaler.minimum[index]) / scale).astype(np.float32),
                ((centers - scaler.minimum[index]) / scale).astype(np.float32),
            )
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "confirm"), required=True)
    parser.add_argument("--dataset-id", choices=("ded", "nist"), required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument(
        "--backbone", choices=("shallow", *BACKBONES), default="resnet18"
    )
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", default="")
    parser.add_argument("--parameters", nargs="+", required=True)
    parser.add_argument("--high-targets", nargs="+", required=True)
    parser.add_argument("--specialist-targets", nargs="*", default=[])
    parser.add_argument("--observability-scores", nargs="+", type=float, required=True)
    parser.add_argument("--group-column", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--set-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--selected-priority", type=float, default=2.0)
    parser.add_argument("--rank-weight", type=float, default=0.10)
    parser.add_argument("--ordinal-weight", type=float, default=0.10)
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
    specialist_targets = list(args.specialist_targets)
    if not set(high_targets).issubset(parameters):
        raise ValueError("High targets must be parameters")
    if len(args.observability_scores) != len(parameters):
        raise ValueError("Observability scores must align with parameters")
    if args.algorithm in GUIDED_ALGORITHMS:
        if not specialist_targets or not set(specialist_targets).issubset(parameters):
            raise ValueError("Guided algorithms require valid specialist targets")
        if len(set(specialist_targets)) != len(specialist_targets):
            raise ValueError("Specialist targets must be unique")
    elif specialist_targets:
        raise ValueError("Only guided algorithms accept specialist targets")
    if args.architecture == "shallow_cnn" and args.backbone != "shallow":
        raise ValueError("The shallow CNN must use --backbone shallow")
    if args.architecture == "domain_tailored" and args.backbone == "shallow":
        raise ValueError("The domain-tailored model requires an ImageNet backbone")

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

    specialist_indices = [parameters.index(target) for target in specialist_targets]
    ordinal_specs = (
        normalized_ordinal_specs(train_frame, scaler, parameters, specialist_targets)
        if args.algorithm == "guided_ordinal_rank"
        else []
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DomainTailoredObservabilityModel(
        args.dataset_id,
        args.architecture,
        args.backbone,
        len(parameters),
        specialist_indices,
        args.algorithm,
        ordinal_specs,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    score_weights = torch.as_tensor(
        args.observability_scores, dtype=torch.float32, device=device
    ).clamp_min(1e-6)
    high_indices = [parameters.index(target) for target in high_targets]
    best_score, best_epoch, stale = math.inf, -1, 0
    checkpoint = output_directory / "best_model.pth"
    history: List[Dict[str, object]] = []

    for epoch in range(args.epochs):
        # The shallow CNN is randomly initialized and must train from epoch 1.
        # Only the ImageNet-initialized domain encoder uses a short frozen warm-up.
        encoder_trainable = (
            args.architecture == "shallow_cnn"
            or epoch >= args.freeze_backbone_epochs
        )
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(encoder_trainable)
        model.train()
        if not encoder_trainable:
            model.encoder.eval()
        epoch_losses = []
        for images, mask, targets, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(images, mask)
            loss, components = training_objective(
                output,
                targets,
                model,
                score_weights,
                high_indices,
                args.selected_priority,
                args.rank_weight,
                args.ordinal_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.max_gradient_norm,
            )
            optimizer.step()
            epoch_losses.append(components["total"])
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
                "train_loss": float(np.mean(epoch_losses)),
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
                args.architecture,
                args.algorithm,
                args.seed,
                epoch + 1,
                float(np.mean(epoch_losses)),
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
                    "architecture": args.architecture,
                    "backbone": args.backbone,
                    "algorithm": args.algorithm,
                    "parameters": parameters,
                    "high_targets": high_targets,
                    "specialist_targets": specialist_targets,
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
        "architecture": args.architecture,
        "backbone": args.backbone,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "parameters": parameters,
        "high_targets": high_targets,
        "specialist_targets": specialist_targets,
        "observability_scores": dict(zip(parameters, args.observability_scores)),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_score": final_score,
        "validation_high_observability_nmae": high_nmae,
        "validation_worst_low_observability_nmae": worst_low,
        "validation_group_equal": validation_metrics,
        "outer_test_was_supplied": bool(args.test_csv),
        "set_size": args.set_size,
        "augmentation": "none",
        "domain_adaptation": (
            "centered melt-pool ROI + ordered bidirectional GRU + temporal dynamics"
            if args.dataset_id == "ded" and args.architecture == "domain_tailored"
            else "cross-sectional intensity/shape descriptors + masked gated set moments"
            if args.dataset_id == "nist" and args.architecture == "domain_tailored"
            else "four-block shallow CNN + masked mean pooling"
        ),
        "selection_metric": (
            "score-weighted high-observability NMAE + 0.25 mean NMAE + "
            "0.10 worst low-observability NMAE"
        ),
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
