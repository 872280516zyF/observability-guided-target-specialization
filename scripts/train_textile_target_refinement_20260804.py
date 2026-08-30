#!/usr/bin/env python3
"""Train one target-refinement run on the frozen textile architecture.

This second exploratory stage fixes the pairwise BatchNorm-contamination path,
tests target-aligned checkpoint selection and optionally performs a final
specialist-only fine-tuning phase.  It does not edit historical runners.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.train_textile_module_attribution_20260802 as base  # noqa: E402
import scripts.train_textile_ordered_pairwise_20260803 as ordered  # noqa: E402
from scripts.run_dpi_branch_ablation import (  # noqa: E402
    PARAM_NAMES,
    TargetedIntegratedExpertBNNet,
    move_tensor_batch,
    unpack_model_output,
)


VARIANTS = [
    "base_mean_selection",
    "base_pobs_selection",
    "coral_pobs_selection",
    "isolated_pairwise_pobs",
    "isolated_pairwise_coral_pobs",
    "two_stage_coral_pobs",
]
DPI_INDEX = PARAM_NAMES.index("dpi")
PAIR_LOADER: Optional[DataLoader] = None
EPOCH_COUNTER = 0
GENERAL_FROZEN = False


def uses_coral(variant: str) -> bool:
    return "coral" in variant


def uses_pairwise(variant: str) -> bool:
    return "isolated_pairwise" in variant


def uses_two_stage(variant: str) -> bool:
    return variant == "two_stage_coral_pobs"


def build_model(args: argparse.Namespace) -> nn.Module:
    if uses_coral(args.variant):
        return ordered.CurrentExpertWithOrdinalAux(args)
    return TargetedIntegratedExpertBNNet(
        backbone_name="resnet18",
        pretrained=not args.no_pretrained,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        texture_dim=args.texture_dim,
        expert_target=args.expert_target,
        texture_guided=True,
    )


def specialist_only_forward(
    model: nn.Module, image: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run only the P_obs path; the general branch is never called or updated."""
    specialist_feature = model.specialist_backbone(image).squeeze(-1).squeeze(-1)
    specialist_aux = model.specialist_descriptor_mlp(model.specialist_descriptor(image))
    specialist_context = torch.cat(
        [
            specialist_feature * model.specialist_attention(specialist_feature),
            specialist_aux,
        ],
        dim=1,
    )
    specialist_prediction = model.specialist_head(specialist_context)
    return specialist_prediction, specialist_context


def freeze_general_branch(model: nn.Module) -> None:
    global GENERAL_FROZEN
    modules = [model.general_backbone, model.general_aux_encoder, model.general_head]
    for module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    GENERAL_FROZEN = True


def set_training_mode(model: nn.Module, args: argparse.Namespace) -> None:
    model.train()
    if uses_two_stage(args.variant) and EPOCH_COUNTER > args.fine_tune_start_epoch:
        if not GENERAL_FROZEN:
            freeze_general_branch(model)
        else:
            model.general_backbone.eval()
            model.general_aux_encoder.eval()
            model.general_head.eval()


def pobs_selection_score(metrics: Dict[str, object], args: argparse.Namespace) -> float:
    if args.variant == "base_mean_selection":
        return float(metrics["mape_physical"])
    parameter_metrics = metrics["param_mape_physical"]
    nonselected = [
        float(value) for name, value in parameter_metrics.items() if name != "dpi"
    ]
    return (
        float(parameter_metrics["dpi"])
        + 0.25 * float(metrics["mape_physical"])
        + 0.10 * max(nonselected)
    )


def train_one_epoch(
    model: nn.Module,
    loader,
    output_weights: torch.Tensor,
    optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    global EPOCH_COUNTER
    EPOCH_COUNTER += 1
    set_training_mode(model, args)
    total_loss = 0.0
    pair_iterator = iter(PAIR_LOADER) if PAIR_LOADER is not None else None
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_tensor_batch(batch, device)
        optimizer.zero_grad()
        output = model(batch)
        prediction, auxiliary = unpack_model_output(output)
        loss = ordered.regression_training_loss(
            prediction,
            batch["params"],
            batch["params_raw"],
            output_weights,
            False,
        )
        if uses_coral(args.variant):
            loss = loss + args.coral_weight * ordered.coral_training_loss(
                auxiliary["coral_logits"], batch["params"]
            )

        if uses_pairwise(args.variant):
            if PAIR_LOADER is None or pair_iterator is None:
                raise RuntimeError("Isolated pairwise variant has no pair loader")
            pair, pair_iterator = ordered.next_pair_batch(pair_iterator, PAIR_LOADER)
            effect = torch.cat([pair["effect_a"], pair["effect_b"]], dim=0).to(device)
            specialist_prediction, _ = specialist_only_forward(model, effect)
            half = pair["effect_a"].shape[0]
            predicted_delta = (
                specialist_prediction[half:, 0] - specialist_prediction[:half, 0]
            )
            true_delta = pair["delta_norm"].to(device)
            margin = torch.minimum(
                0.5 * true_delta, torch.full_like(true_delta, 0.05)
            )
            rank_loss = F.relu(margin - predicted_delta).mean()
            delta_loss = F.smooth_l1_loss(predicted_delta, true_delta)
            loss = (
                loss
                + args.pair_rank_weight * rank_loss
                + args.pair_delta_weight * delta_loss
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch["params"].size(0)
    return float(total_loss / max(len(loader.dataset), 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
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
    parser.add_argument("--fine-tune-start-epoch", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pair-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--resize-mode", choices=["stretch"], default="stretch")
    parser.add_argument("--augmentation-mode", choices=["weak"], default="weak")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--texture-dim", type=int, default=64)
    parser.add_argument("--selection-metric", default="variant_specific")
    parser.add_argument("--coral-weight", type=float, default=0.005)
    parser.add_argument("--pair-rank-weight", type=float, default=0.05)
    parser.add_argument("--pair-delta-weight", type=float, default=0.5)
    parser.add_argument("--lds-sigma", type=float, default=1.5)
    parser.add_argument("--lds-max-weight", type=float, default=3.0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if args.expert_target != "dpi":
        raise ValueError("Target refinement is pre-specified for DPI")
    if not 0 < args.fine_tune_start_epoch < args.epochs:
        raise ValueError("fine_tune_start_epoch must fall inside training")
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
    return args


def main() -> None:
    global PAIR_LOADER, EPOCH_COUNTER, GENERAL_FROZEN
    args = parse_args()
    EPOCH_COUNTER = 0
    GENERAL_FROZEN = False
    statistics = ordered.prepare_training_statistics(args)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {
        "variant": args.variant,
        "selection_rule": (
            "validation mean MAPE"
            if args.variant == "base_mean_selection"
            else "validation DPI MAPE + 0.25 mean MAPE + 0.10 max nonselected MAPE"
        ),
        "general_branch_called_for_pairwise_batch": False,
        "general_branch_batchnorm_updated_by_pairwise_batch": False,
        "fine_tune_start_epoch": (
            args.fine_tune_start_epoch if uses_two_stage(args.variant) else None
        ),
        "fine_tune_scope": (
            "specialist branch and ordinal head only"
            if uses_two_stage(args.variant)
            else None
        ),
        **statistics,
    }
    if uses_pairwise(args.variant):
        pair_dataset = ordered.DpiOneFactorPairDataset(
            args.train_csv,
            args.before_dir,
            args.after_dir,
            args.img_size,
            args.resize_mode,
            args.augmentation_mode,
        )
        PAIR_LOADER = DataLoader(
            pair_dataset,
            batch_size=args.pair_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,
        )
        manifest["training_only_pair_count"] = int(len(pair_dataset))
        pair_dataset.pairs.to_csv(
            run_dir / "training_only_dpi_pairs.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        PAIR_LOADER = None
        manifest["training_only_pair_count"] = 0
    (run_dir / "target_refinement_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    base.ALL_VARIANTS = list(VARIANTS)
    base.parse_args = lambda: args
    base.build_model = build_model
    base.train_one_epoch = train_one_epoch
    base.selection_score = pobs_selection_score
    base.main()


if __name__ == "__main__":
    main()
