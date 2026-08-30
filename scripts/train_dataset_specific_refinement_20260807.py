#!/usr/bin/env python3
"""Train one frozen-split DED/NIST refinement run.

DED uses a target-primary loss with weak auxiliary tasks and a late residual
adapter, because prior symmetric gradient surgery did not help. NIST uses
capacity-matched adapters for every task and compares ordinary optimization
with task-symmetric PCGrad, because the prior PCGrad diagnostic improved the
high-observability target. A target-only role is included only as a diagnostic
for negative transfer. Pilot mode cannot receive an outer-test CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import train_condition_set_specialist_20260802 as metrics_base  # noqa: E402
from scripts import train_crossdomain_observability_optimizer_20260805 as prediction_base  # noqa: E402
from scripts import train_domain_tailored_observability_20260806 as domain  # noqa: E402
from scripts import train_nist_set_aggregation_upgrade_20260803 as set_data  # noqa: E402
from scripts import train_observability_gradient_protection_20260806 as gradient  # noqa: E402


VARIANTS = ("target_primary_adapter", "all_task_adapters", "target_only")
OPTIMIZERS = ("standard", "symmetric_pcgrad")


class ResidualAdapter(nn.Module):
    def __init__(self, width: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, rank),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(rank, width),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class DatasetSpecificRefinement(nn.Module):
    """Frozen domain representation plus lightweight task-specific adapters."""

    def __init__(
        self,
        dataset_id: str,
        backbone: str,
        parameter_count: int,
        adapter_indices: Sequence[int],
        adapter_rank: int,
        dropout: float,
        pretrained: bool,
    ) -> None:
        super().__init__()
        self.core = domain.DomainTailoredObservabilityModel(
            dataset_id=dataset_id,
            architecture="domain_tailored",
            backbone=backbone,
            parameter_count=parameter_count,
            specialist_indices=(),
            algorithm="shared_baseline",
            ordinal_specs=(),
            pretrained=pretrained,
            dropout=dropout,
        )
        width = self.core.representation_dim
        self.adapter_indices = tuple(int(value) for value in adapter_indices)
        self.adapters = nn.ModuleDict(
            {str(index): ResidualAdapter(width, adapter_rank, dropout) for index in self.adapter_indices}
        )
        self.adapter_heads = nn.ModuleDict(
            {str(index): nn.Linear(width, 1) for index in self.adapter_indices}
        )
        for head in self.adapter_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @property
    def encoder(self) -> nn.Module:
        return self.core.encoder

    def forward(self, image_sets: torch.Tensor, mask: torch.Tensor) -> Dict[str, object]:
        representation = self.core._representation(image_sets, mask)
        logits = self.core.base_head(representation)
        if self.adapters:
            logits = logits.clone()
            for index in self.adapter_indices:
                key = str(index)
                adapted = self.adapters[key](representation)
                logits[:, index] = logits[:, index] + self.adapter_heads[key](adapted).squeeze(1)
        return {"prediction": torch.sigmoid(logits), "representation": representation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "confirm"), required=True)
    parser.add_argument("--dataset-id", choices=("ded", "nist"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--optimization", choices=OPTIMIZERS, default="standard")
    parser.add_argument("--backbone", choices=domain.BACKBONES, required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", default="")
    parser.add_argument("--parameters", nargs="+", required=True)
    parser.add_argument("--high-targets", nargs="+", required=True)
    parser.add_argument("--adapter-targets", nargs="*", default=[])
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
    parser.add_argument("--adapter-rank", type=int, default=64)
    parser.add_argument("--aux-weight", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--max-gradient-norm", type=float, default=5.0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def task_weights(
    parameters: Sequence[str], high_targets: Sequence[str], variant: str, aux_weight: float
) -> torch.Tensor:
    high = set(high_targets)
    if variant == "target_only":
        return torch.as_tensor([1.0 if value in high else 0.0 for value in parameters])
    if variant == "target_primary_adapter":
        return torch.as_tensor([1.0 if value in high else aux_weight for value in parameters])
    return torch.ones(len(parameters), dtype=torch.float32)


def main() -> None:
    args = parse_args()
    if args.phase == "pilot" and args.test_csv:
        raise ValueError("Pilot mode must not receive an outer-test CSV")
    if args.phase == "confirm" and not args.test_csv:
        raise ValueError("Confirm mode requires --test-csv")
    if args.dataset_id == "ded" and args.optimization != "standard":
        raise ValueError("DED refinement intentionally uses standard target-primary optimization")
    if args.dataset_id == "nist" and args.variant == "target_primary_adapter":
        raise ValueError("NIST uses capacity-matched all-task adapters")
    if args.variant == "target_only" and args.adapter_targets:
        raise ValueError("Target-only diagnostic does not use adapters")
    if args.variant == "all_task_adapters" and set(args.adapter_targets) != set(args.parameters):
        raise ValueError("all_task_adapters requires every parameter as an adapter target")

    parameters = list(args.parameters)
    high_targets = list(args.high_targets)
    adapter_targets = list(args.adapter_targets)
    if not set(high_targets).issubset(parameters):
        raise ValueError("Unknown high target")
    if not set(adapter_targets).issubset(parameters):
        raise ValueError("Unknown adapter target")
    if len(args.observability_scores) != len(parameters):
        raise ValueError("Observability scores must align with parameters")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_name = "validation_summary.json" if args.phase == "pilot" else "test_metrics.json"
    summary_path = output / summary_name
    if summary_path.is_file():
        print("[SKIP] {}".format(output), flush=True)
        return

    metrics_base.seed_everything(args.seed)
    train_frame = pd.read_csv(args.train_csv)
    scaler = metrics_base.ParameterScaler(train_frame, parameters)
    loader_common = (
        scaler, parameters, args.group_column, args.set_size, args.image_size, False
    )
    _, train_loader = set_data.make_loader(
        Path(args.train_csv), *loader_common, True, args.batch_size, args.num_workers, args.seed
    )
    _, val_loader = set_data.make_loader(
        Path(args.val_csv), *loader_common, False, args.batch_size, args.num_workers, args.seed
    )
    test_loader = None
    if args.test_csv:
        _, test_loader = set_data.make_loader(
            Path(args.test_csv), *loader_common, False, args.batch_size, args.num_workers, args.seed
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter_indices = [parameters.index(value) for value in adapter_targets]
    model = DatasetSpecificRefinement(
        args.dataset_id,
        args.backbone,
        len(parameters),
        adapter_indices,
        args.adapter_rank,
        args.dropout,
        pretrained=not args.no_pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    weights = task_weights(parameters, high_targets, args.variant, args.aux_weight).to(device)
    high_indices = [parameters.index(value) for value in high_targets]
    gradient_generator = torch.Generator(device="cpu")
    gradient_generator.manual_seed(args.seed + 20260807)
    best_score, best_epoch, stale = math.inf, -1, 0
    checkpoint = output / "best_model.pth"
    history: List[Dict[str, object]] = []

    for epoch in range(args.epochs):
        encoder_trainable = epoch >= args.freeze_backbone_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(encoder_trainable)
        model.train()
        if not encoder_trainable:
            model.encoder.eval()
        losses, conflict_values, cosine_values = [], [], []
        for images, mask, targets, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            prediction = model(images, mask)["prediction"]
            per_sample = F.smooth_l1_loss(prediction, targets, reduction="none")
            per_task = [per_sample[:, index].mean() * weights[index] for index in range(len(parameters))]
            active = [value for index, value in enumerate(per_task) if float(weights[index]) > 0.0]
            optimizer.zero_grad(set_to_none=True)
            if args.optimization == "symmetric_pcgrad" and len(active) > 1:
                diagnostics = gradient.apply_gradient_rule(
                    model,
                    active,
                    "symmetric_pcgrad",
                    [],
                    1.0,
                    gradient_generator,
                )
                torch.nn.utils.clip_grad_norm_(
                    [value for value in model.parameters() if value.requires_grad],
                    args.max_gradient_norm,
                )
                optimizer.step()
                conflict_values.append(float(diagnostics["conflict_fraction"]))
                cosine_values.append(float(diagnostics["mean_preprojection_cosine"]))
                total = torch.stack(active).mean().detach()
            else:
                total = torch.stack(active).sum() / weights.sum().clamp_min(1e-8)
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    [value for value in model.parameters() if value.requires_grad],
                    args.max_gradient_norm,
                )
                optimizer.step()
            losses.append(float(total.cpu()))
        scheduler.step()

        val_predictions = prediction_base.predictions(model, val_loader, scaler, parameters, device)
        val_metrics = metrics_base.group_equal_metrics(val_predictions, parameters)
        score, high_nmae, worst_low = domain.high_validation_score(
            val_metrics, parameters, high_targets, args.observability_scores
        )
        if args.variant == "target_only":
            score = high_nmae
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_selection_score": score,
                "val_high_nmae": high_nmae,
                "val_mean_nmae": float(val_metrics["mean_nmae"]),
                "val_worst_low_nmae": worst_low,
                "conflict_fraction": float(np.mean(conflict_values)) if conflict_values else np.nan,
                "mean_preprojection_cosine": float(np.mean(cosine_values)) if cosine_values else np.nan,
                "encoder_trainable": encoder_trainable,
            }
        )
        print(
            "[{} {} fold{} {} {} seed{}] epoch={:03d} loss={:.6f} val={:.6f}".format(
                args.phase, args.dataset_id, args.outer_fold, args.variant,
                args.optimization, args.seed, epoch + 1, float(np.mean(losses)), score
            ),
            flush=True,
        )
        if score < best_score - 1e-8:
            best_score, best_epoch, stale = score, epoch + 1, 0
            torch.save({"model": model.state_dict()}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    val_predictions = prediction_base.predictions(model, val_loader, scaler, parameters, device)
    val_predictions.to_csv(output / "validation_predictions_conditions.csv", index=False)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    val_metrics = metrics_base.group_equal_metrics(val_predictions, parameters)
    score, high_nmae, worst_low = domain.high_validation_score(
        val_metrics, parameters, high_targets, args.observability_scores
    )
    if args.variant == "target_only":
        score = high_nmae
    payload: Dict[str, object] = {
        "phase": args.phase,
        "dataset_id": args.dataset_id,
        "outer_fold": args.outer_fold,
        "variant": args.variant,
        "optimization": args.optimization,
        "backbone": args.backbone,
        "seed": args.seed,
        "parameters": parameters,
        "high_targets": high_targets,
        "adapter_targets": adapter_targets,
        "adapter_rank": args.adapter_rank,
        "aux_weight": args.aux_weight,
        "observability_scores": dict(zip(parameters, args.observability_scores)),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_score": score,
        "validation_high_nmae": high_nmae,
        "validation_worst_low_nmae": worst_low,
        "validation_group_equal": val_metrics,
        "outer_test_was_supplied": bool(args.test_csv),
        "selection_metric": "high NMAE + 0.25 mean NMAE + 0.10 worst-low NMAE",
        "scaler": scaler.as_dict(),
    }
    if test_loader is not None:
        test_predictions = prediction_base.predictions(model, test_loader, scaler, parameters, device)
        test_predictions.to_csv(output / "test_predictions_conditions.csv", index=False)
        payload["group_equal"] = metrics_base.group_equal_metrics(test_predictions, parameters)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
