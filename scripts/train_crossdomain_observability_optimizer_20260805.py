#!/usr/bin/env python3
"""Train one leakage-controlled cross-domain observability candidate.

The runner is intentionally dataset agnostic.  Each CSV supplies image paths,
condition identifiers, an independent grouping column and an ordered list of
numeric targets.  Pilot mode never accepts an outer-test CSV.  Confirm mode
uses a model variant that has already been frozen by the orchestration script.

The candidate family separates three questions that were confounded by the
earlier whole-expert experiments:

* does a lightweight residual head help the selected target;
* does a label-structure-aware ordinal objective help;
* does removing destructive gradients between the selected and nonselected
  objectives help.

All targeted variants use the same condition-level masked gated-moment
aggregator.  The same frozen variant is later routed to every target, providing
the required placement control without changing its capacity.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import train_condition_set_specialist_20260802 as base  # noqa: E402
from scripts import train_nist_set_aggregation_upgrade_20260803 as sets  # noqa: E402


VARIANTS = (
    "shared_baseline",
    "target_residual",
    "target_ordinal",
    "target_uncertainty",
    "target_pcgrad",
    "target_combined",
)
TARGETED_VARIANTS = tuple(name for name in VARIANTS if name != "shared_baseline")


def uses_ordinal(variant: str) -> bool:
    return variant in ("target_ordinal", "target_combined")


def uses_uncertainty(variant: str) -> bool:
    return variant in ("target_uncertainty", "target_combined")


def uses_pcgrad(variant: str) -> bool:
    return variant in ("target_pcgrad", "target_combined")


def ordinal_spec(values: np.ndarray, maximum_classes: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Derive ordered cut points and class representatives from training only."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    unique = np.unique(values)
    if len(unique) < 2:
        raise ValueError("The selected target needs at least two training values")
    if len(unique) <= maximum_classes:
        centers = unique
        thresholds = 0.5 * (unique[:-1] + unique[1:])
    else:
        quantiles = np.linspace(0.0, 1.0, maximum_classes + 1)[1:-1]
        thresholds = np.unique(np.quantile(values, quantiles))
        if len(thresholds) < 1:
            thresholds = np.asarray([float(np.median(values))], dtype=np.float64)
        bins = np.digitize(values, thresholds, right=False)
        centers_list = []
        lower, upper = float(values.min()), float(values.max())
        for index in range(len(thresholds) + 1):
            members = values[bins == index]
            if len(members):
                centers_list.append(float(members.mean()))
            else:
                left = lower if index == 0 else float(thresholds[index - 1])
                right = upper if index == len(thresholds) else float(thresholds[index])
                centers_list.append(0.5 * (left + right))
        centers = np.asarray(centers_list, dtype=np.float64)
    return thresholds.astype(np.float32), centers.astype(np.float32)


class ObservabilityConstrainedModel(nn.Module):
    """Shared condition encoder plus optional target-specific refinement."""

    def __init__(
        self,
        parameter_count: int,
        routing_index: int,
        variant: str,
        thresholds: Sequence[float],
        ordinal_centers: Sequence[float],
        pretrained: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("Unknown variant {}".format(variant))
        self.variant = variant
        self.routing_index = int(routing_index)
        self.encoder = base.make_resnet18_feature_map(pretrained)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.frame_gate = nn.Sequential(
            nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.moment_project = nn.Sequential(
            nn.Linear(1536, 512),
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
        targeted = variant != "shared_baseline"
        self.specialist = (
            nn.Sequential(
                nn.Linear(512, 256),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )
            if targeted
            else None
        )
        if self.specialist is not None:
            nn.init.zeros_(self.specialist[-1].weight)
            nn.init.zeros_(self.specialist[-1].bias)

        self.ordinal_head = None
        if uses_ordinal(variant):
            if len(thresholds) < 1 or len(ordinal_centers) != len(thresholds) + 1:
                raise ValueError("Invalid ordinal specification")
            self.ordinal_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, len(thresholds)),
            )
        self.log_variance_head = None
        if uses_uncertainty(variant):
            self.log_variance_head = nn.Sequential(
                nn.Linear(512, 128), nn.SiLU(inplace=True), nn.Linear(128, 1)
            )
            nn.init.zeros_(self.log_variance_head[-1].weight)
            nn.init.zeros_(self.log_variance_head[-1].bias)

        self.register_buffer(
            "ordinal_thresholds", torch.as_tensor(thresholds, dtype=torch.float32)
        )
        self.register_buffer(
            "ordinal_centers", torch.as_tensor(ordinal_centers, dtype=torch.float32)
        )

    def _encode_valid(
        self, image_sets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, count, channels, height, width = image_sets.shape
        flat = image_sets.reshape(batch * count, channels, height, width)
        valid = mask.reshape(-1)
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise RuntimeError("A batch contains no valid images")
        valid_maps = self.encoder(flat.index_select(0, indices))
        valid_features = self.spatial_pool(valid_maps).flatten(1)
        features = valid_features.new_zeros((batch * count, 512))
        features = features.index_copy(0, indices, valid_features)
        return features.reshape(batch, count, 512)

    def _aggregate(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.frame_gate(features).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        gated = (weights * features).sum(1)
        standard_deviation = sets.masked_std(features, mask, gated)
        maximum = sets.masked_max(features, mask)
        return self.moment_project(
            torch.cat([gated, standard_deviation, maximum], dim=1)
        )

    def _ordinal_expectation(self, logits: torch.Tensor) -> torch.Tensor:
        exceedance = torch.sigmoid(logits)
        # Independent cumulative logits are monotonized only for the expected
        # value calculation.  BCE remains defined on the original logits.
        exceedance = torch.cummin(exceedance, dim=1).values
        probabilities = torch.cat(
            [
                1.0 - exceedance[:, :1],
                exceedance[:, :-1] - exceedance[:, 1:],
                exceedance[:, -1:],
            ],
            dim=1,
        ).clamp_min(0.0)
        probabilities = probabilities / probabilities.sum(1, keepdim=True).clamp_min(1e-8)
        return (probabilities * self.ordinal_centers.unsqueeze(0)).sum(1)

    def forward(self, image_sets: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self._encode_valid(image_sets, mask)
        pooled = self._aggregate(features, mask)
        logits = self.base_head(pooled)
        if self.specialist is not None:
            residual = self.specialist(pooled).squeeze(1)
            logits = logits.clone()
            logits[:, self.routing_index] = logits[:, self.routing_index] + residual
        prediction = torch.sigmoid(logits)
        ordinal_logits = None
        if self.ordinal_head is not None:
            ordinal_logits = self.ordinal_head(pooled)
            ordinal_prediction = self._ordinal_expectation(ordinal_logits)
            prediction = prediction.clone()
            prediction[:, self.routing_index] = 0.5 * (
                prediction[:, self.routing_index] + ordinal_prediction
            )
        log_variance = None
        if self.log_variance_head is not None:
            log_variance = self.log_variance_head(pooled).squeeze(1).clamp(-5.0, 5.0)
        return {
            "prediction": prediction,
            "ordinal_logits": ordinal_logits,
            "log_variance": log_variance,
        }


def objectives(
    output: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    model: ObservabilityConstrainedModel,
    selected_priority: float,
    ordinal_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = output["prediction"]
    per_parameter = F.smooth_l1_loss(prediction, targets, reduction="none").mean(0)
    selected = per_parameter[model.routing_index]
    if output["log_variance"] is not None:
        error = prediction[:, model.routing_index] - targets[:, model.routing_index]
        log_variance = output["log_variance"]
        gaussian_nll = 0.5 * (
            torch.exp(-log_variance) * error.square() + log_variance
        ).mean()
        selected = 0.5 * selected + 0.5 * gaussian_nll
    if output["ordinal_logits"] is not None:
        ordinal_truth = (
            targets[:, model.routing_index].unsqueeze(1)
            > model.ordinal_thresholds.unsqueeze(0)
        ).to(targets.dtype)
        selected = selected + ordinal_weight * F.binary_cross_entropy_with_logits(
            output["ordinal_logits"], ordinal_truth
        )
    other_indices = [index for index in range(targets.shape[1]) if index != model.routing_index]
    nonselected = per_parameter[other_indices].mean()
    denominator = selected_priority + 1.0
    return (
        (selected_priority * selected + nonselected) / denominator,
        selected_priority * selected / denominator,
        nonselected / denominator,
    )


def pcgrad_step(
    selected_loss: torch.Tensor,
    nonselected_loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[nn.Parameter],
    maximum_gradient_norm: float,
) -> None:
    """Two-objective symmetric PCGrad without using validation/test feedback."""
    active = [parameter for parameter in parameters if parameter.requires_grad]
    selected_gradients = torch.autograd.grad(
        selected_loss, active, retain_graph=True, allow_unused=True
    )
    other_gradients = torch.autograd.grad(
        nonselected_loss, active, allow_unused=True
    )
    dot = selected_loss.new_zeros(())
    selected_norm = selected_loss.new_zeros(())
    other_norm = selected_loss.new_zeros(())
    for selected_gradient, other_gradient in zip(selected_gradients, other_gradients):
        if selected_gradient is not None:
            selected_norm = selected_norm + selected_gradient.detach().square().sum()
        if other_gradient is not None:
            other_norm = other_norm + other_gradient.detach().square().sum()
        if selected_gradient is not None and other_gradient is not None:
            dot = dot + (selected_gradient.detach() * other_gradient.detach()).sum()
    conflict = bool(dot.item() < 0.0)
    selected_coefficient = dot / other_norm.clamp_min(1e-12) if conflict else dot.new_zeros(())
    other_coefficient = dot / selected_norm.clamp_min(1e-12) if conflict else dot.new_zeros(())
    optimizer.zero_grad(set_to_none=True)
    for parameter, selected_gradient, other_gradient in zip(
        active, selected_gradients, other_gradients
    ):
        if selected_gradient is None and other_gradient is None:
            continue
        if selected_gradient is None:
            combined = other_gradient
        elif other_gradient is None:
            combined = selected_gradient
        elif conflict:
            combined = (
                selected_gradient - selected_coefficient * other_gradient
                + other_gradient - other_coefficient * selected_gradient
            )
        else:
            combined = selected_gradient + other_gradient
        parameter.grad = combined.detach().clone()
    torch.nn.utils.clip_grad_norm_(active, maximum_gradient_norm)
    optimizer.step()


def predictions(
    model: ObservabilityConstrainedModel,
    loader,
    scaler: base.ParameterScaler,
    parameters: Sequence[str],
    device: torch.device,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for images, mask, targets, condition_ids, group_ids in loader:
            output = model(
                images.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            )
            estimate = output["prediction"].cpu().numpy()
            truth = targets.numpy()
            estimate_raw = scaler.inverse(estimate)
            truth_raw = scaler.inverse(truth)
            for row_index in range(len(condition_ids)):
                row: Dict[str, object] = {
                    "condition_id": str(condition_ids[row_index]),
                    "group_id": str(group_ids[row_index]),
                }
                for parameter_index, parameter in enumerate(parameters):
                    true_value = float(truth_raw[row_index, parameter_index])
                    predicted_value = float(estimate_raw[row_index, parameter_index])
                    absolute = abs(predicted_value - true_value)
                    row["true_{}".format(parameter)] = true_value
                    row["pred_{}".format(parameter)] = predicted_value
                    row["ae_{}".format(parameter)] = absolute
                    row["nmae_{}".format(parameter)] = absolute / float(
                        scaler.scale[parameter_index]
                    )
                    row["ape_{}".format(parameter)] = (
                        100.0 * absolute / max(abs(true_value), 1e-6)
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "confirm"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", default="")
    parser.add_argument("--parameters", nargs="+", required=True)
    parser.add_argument("--group-column", required=True)
    parser.add_argument("--routing-target", required=True)
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
    parser.add_argument("--selected-priority", type=float, default=1.5)
    parser.add_argument("--ordinal-weight", type=float, default=0.2)
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
    if args.routing_target not in parameters:
        raise ValueError("routing target must be one of {}".format(parameters))
    if args.variant == "shared_baseline" and args.selected_priority != 1.5:
        # The value is recorded but intentionally unused by the shared model.
        print("[NOTE] selected priority is ignored by shared_baseline", flush=True)

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_name = "validation_summary.json" if args.phase == "pilot" else "test_metrics.json"
    summary_path = output_directory / summary_name
    if summary_path.is_file():
        print("[SKIP] completed {}".format(output_directory))
        return

    base.seed_everything(args.seed)
    train_frame = pd.read_csv(args.train_csv)
    scaler = base.ParameterScaler(train_frame, parameters)
    routing_index = parameters.index(args.routing_target)
    scaled_selected = scaler.transform(
        train_frame[parameters].to_numpy(dtype=np.float32)
    )[:, routing_index]
    thresholds, centers = ordinal_spec(scaled_selected)
    if not uses_ordinal(args.variant):
        # Non-ordinal variants retain empty buffers and no unused trainable head.
        thresholds = np.asarray([], dtype=np.float32)
        centers = np.asarray([], dtype=np.float32)

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
    model = ObservabilityConstrainedModel(
        len(parameters),
        routing_index,
        args.variant,
        thresholds,
        centers,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_score, best_epoch, stale = math.inf, -1, 0
    history: List[Dict[str, object]] = []
    checkpoint = output_directory / "best_model.pth"

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
            output = model(images, mask)
            if args.variant == "shared_baseline":
                total_loss = F.smooth_l1_loss(output["prediction"], targets)
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.max_gradient_norm
                )
                optimizer.step()
            else:
                total_loss, selected_loss, other_loss = objectives(
                    output,
                    targets,
                    model,
                    args.selected_priority,
                    args.ordinal_weight,
                )
                if uses_pcgrad(args.variant):
                    pcgrad_step(
                        selected_loss,
                        other_loss,
                        optimizer,
                        list(model.parameters()),
                        args.max_gradient_norm,
                    )
                else:
                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.max_gradient_norm
                    )
                    optimizer.step()
            losses.append(float(total_loss.detach().cpu()))
        scheduler.step()

        validation_predictions = predictions(
            model, validation_loader, scaler, parameters, device
        )
        validation_metrics = base.group_equal_metrics(validation_predictions, parameters)
        selected_nmae = float(
            validation_metrics["nmae_{}".format(args.routing_target)]
        )
        nonselected_nmae = [
            float(validation_metrics["nmae_{}".format(parameter)])
            for parameter in parameters
            if parameter != args.routing_target
        ]
        score = float(
            selected_nmae
            + 0.25 * float(validation_metrics["mean_nmae"])
            + 0.10 * max(nonselected_nmae)
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_selection_score": score,
                "val_selected_nmae": selected_nmae,
                "val_mean_nmae": float(validation_metrics["mean_nmae"]),
                "val_worst_nonselected_nmae": max(nonselected_nmae),
                "encoder_trainable": encoder_trainable,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            "[{} {} fold{} {} seed{}] epoch={:03d} loss={:.6f} val_score={:.6f}".format(
                args.phase,
                args.dataset_id,
                args.outer_fold,
                args.variant,
                args.seed,
                epoch + 1,
                float(np.mean(losses)),
                score,
            ),
            flush=True,
        )
        if score < best_score - 1e-8:
            best_score, best_epoch, stale = score, epoch + 1, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "dataset_id": args.dataset_id,
                    "variant": args.variant,
                    "routing_target": args.routing_target,
                    "parameters": parameters,
                    "scaler": scaler.as_dict(),
                    "ordinal_thresholds": thresholds.tolist(),
                    "ordinal_centers": centers.tolist(),
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    validation_predictions = predictions(
        model, validation_loader, scaler, parameters, device
    )
    validation_predictions.to_csv(
        output_directory / "validation_predictions_conditions.csv", index=False
    )
    pd.DataFrame(history).to_csv(output_directory / "training_history.csv", index=False)
    payload: Dict[str, object] = {
        "phase": args.phase,
        "dataset_id": args.dataset_id,
        "outer_fold": args.outer_fold,
        "variant": args.variant,
        "routing_target": args.routing_target,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "trainable_parameter_count_at_initialization": trainable_parameter_count,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "validation_group_equal": base.group_equal_metrics(
            validation_predictions, parameters
        ),
        "selection_metric": (
            "selected-target NMAE + 0.25 mean NMAE + "
            "0.10 worst nonselected NMAE"
        ),
        "outer_test_was_supplied": bool(args.test_csv),
        "condition_aggregation": "masked gated mean + standard deviation + maximum",
        "selected_priority": None if args.variant == "shared_baseline" else args.selected_priority,
        "ordinal_classes": int(len(centers)) if uses_ordinal(args.variant) else 0,
        "pcgrad": uses_pcgrad(args.variant),
        "heteroscedastic_selected_loss": uses_uncertainty(args.variant),
        "augmentation": "none",
        "set_size": args.set_size,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "scaler": scaler.as_dict(),
    }
    if test_loader is not None:
        test_predictions = predictions(model, test_loader, scaler, parameters, device)
        test_predictions.to_csv(
            output_directory / "test_predictions_conditions.csv", index=False
        )
        payload["group_equal"] = base.group_equal_metrics(test_predictions, parameters)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
