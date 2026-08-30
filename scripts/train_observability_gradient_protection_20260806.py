#!/usr/bin/env python3
"""Train one dataset-local gradient-protection experiment.

This runner keeps the frozen domain-tailored representation and changes only
the multi-task optimization rule.  ``symmetric_pcgrad`` is the published
task-symmetric gradient-surgery control.  ``pobs_anchor`` is an asymmetric,
observability-guided variant: the summed gradient of the outer-training-only
high-observability targets is left unchanged, while conflicting components of
the remaining task gradients are removed.  The total non-anchor gradient norm
is capped relative to the anchor norm.

Pilot mode rejects outer-test input.  Confirm mode requires it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import train_condition_set_specialist_20260802 as base  # noqa: E402
from scripts import train_crossdomain_observability_optimizer_20260805 as prior  # noqa: E402
from scripts import train_domain_tailored_observability_20260806 as domain  # noqa: E402
from scripts import train_nist_set_aggregation_upgrade_20260803 as sets  # noqa: E402
from scripts.train_multitarget_observability_20260806 import (  # noqa: E402
    BACKBONES,
    high_validation_score,
)


ALGORITHMS = ("symmetric_pcgrad", "pobs_anchor")


def tensor_dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> torch.Tensor:
    return sum((a * b).sum() for a, b in zip(left, right))


def tensor_norm(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(tensor_dot(values, values).clamp_min(1e-24))


def add_gradients(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> List[torch.Tensor]:
    return [a + b for a, b in zip(left, right)]


def scale_gradients(values: Sequence[torch.Tensor], scale: torch.Tensor) -> List[torch.Tensor]:
    return [value * scale for value in values]


def mean_gradients(items: Sequence[Sequence[torch.Tensor]]) -> List[torch.Tensor]:
    if not items:
        raise ValueError("Cannot average an empty gradient collection")
    count = float(len(items))
    return [sum(parts) / count for parts in zip(*items)]


def project_if_conflicting(
    gradient: Sequence[torch.Tensor], reference: Sequence[torch.Tensor]
) -> Tuple[List[torch.Tensor], bool, float]:
    dot = tensor_dot(gradient, reference)
    denominator = tensor_dot(reference, reference).clamp_min(1e-24)
    left_norm = tensor_norm(gradient)
    right_norm = tensor_norm(reference)
    cosine = float((dot / (left_norm * right_norm).clamp_min(1e-24)).detach().cpu())
    if float(dot.detach().cpu()) >= 0.0:
        return [value.clone() for value in gradient], False, cosine
    coefficient = dot / denominator
    return [value - coefficient * anchor for value, anchor in zip(gradient, reference)], True, cosine


def task_gradients(
    losses: Sequence[torch.Tensor], parameters: Sequence[torch.nn.Parameter]
) -> List[List[torch.Tensor]]:
    result: List[List[torch.Tensor]] = []
    for index, loss in enumerate(losses):
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=index < len(losses) - 1,
            allow_unused=True,
        )
        result.append(
            [
                torch.zeros_like(parameter) if gradient is None else gradient.detach()
                for parameter, gradient in zip(parameters, gradients)
            ]
        )
    return result


def symmetric_pcgrad(
    gradients: Sequence[Sequence[torch.Tensor]], generator: torch.Generator
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    projected: List[List[torch.Tensor]] = []
    conflict_count = 0
    comparisons = 0
    cosines: List[float] = []
    for task_index, gradient in enumerate(gradients):
        current = [value.clone() for value in gradient]
        order = torch.randperm(len(gradients), generator=generator).tolist()
        for other_index in order:
            if other_index == task_index:
                continue
            current, conflicted, cosine = project_if_conflicting(
                current, gradients[other_index]
            )
            comparisons += 1
            conflict_count += int(conflicted)
            cosines.append(cosine)
        projected.append(current)
    return mean_gradients(projected), {
        "conflict_fraction": float(conflict_count / max(comparisons, 1)),
        "mean_preprojection_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "anchor_norm": float("nan"),
        "nonanchor_norm_before_cap": float("nan"),
        "nonanchor_norm_after_cap": float("nan"),
    }


def pobs_anchor_gradients(
    gradients: Sequence[Sequence[torch.Tensor]],
    anchor_indices: Sequence[int],
    nonanchor_norm_ratio: float,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    anchor_set = set(int(index) for index in anchor_indices)
    if not anchor_set or max(anchor_set) >= len(gradients):
        raise ValueError("Invalid anchor-task indices")
    anchor_items = [gradient for index, gradient in enumerate(gradients) if index in anchor_set]
    nonanchor_items = [
        gradient for index, gradient in enumerate(gradients) if index not in anchor_set
    ]
    anchor_sum = [sum(parts) for parts in zip(*anchor_items)]
    projected_low: List[List[torch.Tensor]] = []
    conflict_count = 0
    cosines: List[float] = []
    for gradient in nonanchor_items:
        projected, conflicted, cosine = project_if_conflicting(gradient, anchor_sum)
        projected_low.append(projected)
        conflict_count += int(conflicted)
        cosines.append(cosine)
    if projected_low:
        low_sum = [sum(parts) for parts in zip(*projected_low)]
    else:
        low_sum = [torch.zeros_like(value) for value in anchor_sum]
    anchor_norm = tensor_norm(anchor_sum)
    low_norm_before = tensor_norm(low_sum)
    maximum_low_norm = anchor_norm * float(nonanchor_norm_ratio)
    cap = torch.minimum(
        torch.ones_like(low_norm_before),
        maximum_low_norm / low_norm_before.clamp_min(1e-24),
    )
    low_sum = scale_gradients(low_sum, cap)
    low_norm_after = tensor_norm(low_sum)
    # Division by the number of tasks keeps the step scale comparable to the
    # ordinary mean-loss update.  The anchor direction itself is never edited.
    combined = scale_gradients(
        add_gradients(anchor_sum, low_sum),
        anchor_norm.new_tensor(1.0 / float(len(gradients))),
    )
    return combined, {
        "conflict_fraction": float(conflict_count / max(len(nonanchor_items), 1)),
        "mean_preprojection_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "anchor_norm": float(anchor_norm.detach().cpu()),
        "nonanchor_norm_before_cap": float(low_norm_before.detach().cpu()),
        "nonanchor_norm_after_cap": float(low_norm_after.detach().cpu()),
    }


def apply_gradient_rule(
    model: torch.nn.Module,
    per_task_losses: Sequence[torch.Tensor],
    algorithm: str,
    anchor_indices: Sequence[int],
    nonanchor_norm_ratio: float,
    generator: torch.Generator,
) -> Dict[str, float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = task_gradients(per_task_losses, parameters)
    if algorithm == "symmetric_pcgrad":
        combined, diagnostics = symmetric_pcgrad(gradients, generator)
    elif algorithm == "pobs_anchor":
        combined, diagnostics = pobs_anchor_gradients(
            gradients, anchor_indices, nonanchor_norm_ratio
        )
    else:
        raise ValueError("Unsupported algorithm {}".format(algorithm))
    for parameter, gradient in zip(parameters, combined):
        parameter.grad = gradient
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "confirm"), required=True)
    parser.add_argument("--dataset-id", choices=("ded", "nist"), required=True)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", default="")
    parser.add_argument("--parameters", nargs="+", required=True)
    parser.add_argument("--high-targets", nargs="+", required=True)
    parser.add_argument("--anchor-targets", nargs="*", default=[])
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
    parser.add_argument("--nonanchor-norm-ratio", type=float, default=1.0)
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
    anchor_targets = list(args.anchor_targets)
    if not set(high_targets).issubset(parameters):
        raise ValueError("High targets must be model parameters")
    if args.algorithm == "pobs_anchor":
        if not anchor_targets or not set(anchor_targets).issubset(parameters):
            raise ValueError("pobs_anchor requires valid --anchor-targets")
        if len(set(anchor_targets)) != len(anchor_targets):
            raise ValueError("Anchor targets must be unique")
        if args.nonanchor_norm_ratio <= 0.0:
            raise ValueError("nonanchor norm ratio must be positive")
    elif anchor_targets:
        raise ValueError("symmetric_pcgrad does not accept anchor targets")
    if len(args.observability_scores) != len(parameters):
        raise ValueError("Observability scores must align with parameters")

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
        Path(args.train_csv),
        *loader_common,
        True,
        args.batch_size,
        args.num_workers,
        args.seed,
    )
    _, validation_loader = sets.make_loader(
        Path(args.val_csv),
        *loader_common,
        False,
        args.batch_size,
        args.num_workers,
        args.seed,
    )
    test_loader = None
    if args.test_csv:
        _, test_loader = sets.make_loader(
            Path(args.test_csv),
            *loader_common,
            False,
            args.batch_size,
            args.num_workers,
            args.seed,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = domain.DomainTailoredObservabilityModel(
        args.dataset_id,
        "domain_tailored",
        args.backbone,
        len(parameters),
        [],
        "shared_baseline",
        [],
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    high_indices = [parameters.index(target) for target in high_targets]
    anchor_indices = [parameters.index(target) for target in anchor_targets]
    best_score, best_epoch, stale = math.inf, -1, 0
    checkpoint = output_directory / "best_model.pth"
    history: List[Dict[str, object]] = []
    pcgrad_generator = torch.Generator(device="cpu")
    pcgrad_generator.manual_seed(args.seed + 700001)

    for epoch in range(args.epochs):
        encoder_trainable = epoch >= args.freeze_backbone_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(encoder_trainable)
        model.train()
        if not encoder_trainable:
            model.encoder.eval()
        epoch_losses: List[float] = []
        diagnostics_rows: List[Dict[str, float]] = []
        for images, mask, targets, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(images, mask)
            per_example = F.smooth_l1_loss(
                output["prediction"], targets, reduction="none"
            )
            losses = [per_example[:, index].mean() for index in range(len(parameters))]
            optimizer.zero_grad(set_to_none=True)
            diagnostics = apply_gradient_rule(
                model,
                losses,
                args.algorithm,
                anchor_indices,
                args.nonanchor_norm_ratio,
                pcgrad_generator,
            )
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.max_gradient_norm,
            )
            optimizer.step()
            epoch_losses.append(float(torch.stack(losses).mean().detach().cpu()))
            diagnostics_rows.append(diagnostics)
        scheduler.step()

        validation_predictions = prior.predictions(
            model, validation_loader, scaler, parameters, device
        )
        metrics = base.group_equal_metrics(validation_predictions, parameters)
        selection_score, high_nmae, worst_low = high_validation_score(
            metrics, parameters, high_targets, args.observability_scores
        )
        finite_anchor = [
            row["anchor_norm"] for row in diagnostics_rows if math.isfinite(row["anchor_norm"])
        ]
        finite_low_before = [
            row["nonanchor_norm_before_cap"]
            for row in diagnostics_rows
            if math.isfinite(row["nonanchor_norm_before_cap"])
        ]
        finite_low_after = [
            row["nonanchor_norm_after_cap"]
            for row in diagnostics_rows
            if math.isfinite(row["nonanchor_norm_after_cap"])
        ]
        history.append(
            {
                "epoch": epoch + 1,
                "train_mean_task_loss": float(np.mean(epoch_losses)),
                "val_selection_score": selection_score,
                "val_high_observability_nmae": high_nmae,
                "val_mean_nmae": float(metrics["mean_nmae"]),
                "val_worst_low_observability_nmae": worst_low,
                "conflict_fraction": float(
                    np.mean([row["conflict_fraction"] for row in diagnostics_rows])
                ),
                "mean_preprojection_cosine": float(
                    np.mean([row["mean_preprojection_cosine"] for row in diagnostics_rows])
                ),
                "anchor_gradient_norm": float(np.mean(finite_anchor)) if finite_anchor else np.nan,
                "nonanchor_norm_before_cap": float(np.mean(finite_low_before))
                if finite_low_before
                else np.nan,
                "nonanchor_norm_after_cap": float(np.mean(finite_low_after))
                if finite_low_after
                else np.nan,
                "encoder_trainable": encoder_trainable,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            "[{} {} fold{} {} seed{}] epoch={:03d} loss={:.6f} val_score={:.6f} conflict={:.3f}".format(
                args.phase,
                args.dataset_id,
                args.outer_fold,
                args.algorithm,
                args.seed,
                epoch + 1,
                float(np.mean(epoch_losses)),
                selection_score,
                history[-1]["conflict_fraction"],
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
                    "anchor_targets": anchor_targets,
                    "observability_scores": args.observability_scores,
                    "nonanchor_norm_ratio": args.nonanchor_norm_ratio,
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
        "architecture": "domain_tailored",
        "backbone": args.backbone,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "parameters": parameters,
        "high_targets": high_targets,
        "anchor_targets": anchor_targets,
        "observability_scores": dict(zip(parameters, args.observability_scores)),
        "nonanchor_norm_ratio": args.nonanchor_norm_ratio,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_score": final_score,
        "validation_high_observability_nmae": high_nmae,
        "validation_worst_low_observability_nmae": worst_low,
        "validation_group_equal": validation_metrics,
        "outer_test_was_supplied": bool(args.test_csv),
        "set_size": args.set_size,
        "augmentation": "none",
        "optimization_rule": (
            "task-symmetric PCGrad"
            if args.algorithm == "symmetric_pcgrad"
            else "observability-anchored asymmetric gradient projection with non-anchor norm cap"
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
