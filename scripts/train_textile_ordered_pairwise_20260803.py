#!/usr/bin/env python3
"""Train one leakage-controlled ordered/pairwise textile inverse run.

The image encoder and regression architecture remain the current
processed-image-only P_obs specialist.  Experimental variants alter only the
training objective using outer-training data:

* density-smoothed DPI regression weights;
* a rank-consistent cumulative ordinal auxiliary head; and/or
* synchronized one-factor DPI pair supervision.

Pilot invocations omit ``--test-csv``.  The shared audited trainer still owns
checkpoint selection, prediction export and physical-unit metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.train_textile_module_attribution_20260802 as base  # noqa: E402
from scripts.run_dpi_branch_ablation import (  # noqa: E402
    PARAM_NAMES,
    TargetedIntegratedExpertBNNet,
    move_tensor_batch,
    unpack_model_output,
)
from scripts.run_inverse_experiment import (  # noqa: E402
    InverseExperimentDataset,
    PARAM_SPECS,
)


VARIANTS = [
    "base_regression",
    "lds_regression",
    "coral_aux",
    "pairwise_rank_delta",
    "pairwise_coral",
    "pairwise_coral_lds",
]
DPI_INDEX = PARAM_NAMES.index("dpi")
DPI_LEVELS = 31
DPI_MIN = 25.0
DPI_STEP = 5.0

CURRENT_ARGS: Optional[argparse.Namespace] = None
PAIR_LOADER: Optional[DataLoader] = None
LDS_LEVEL_WEIGHTS: Optional[torch.Tensor] = None
CORAL_POS_WEIGHTS: Optional[torch.Tensor] = None


class RankConsistentOrdinalHead(nn.Module):
    """Cumulative ordinal head with shared score and ordered thresholds."""

    def __init__(self, input_dim: int, levels: int = DPI_LEVELS) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be at least 2")
        self.levels = int(levels)
        self.score = nn.Linear(input_dim, 1)
        nn.init.normal_(self.score.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.score.bias)
        self.first_threshold = nn.Parameter(torch.tensor(-3.0))
        target_gap = 6.0 / float(self.levels - 2) if self.levels > 2 else 1.0
        inverse_softplus = math.log(math.expm1(target_gap))
        self.raw_gaps = nn.Parameter(
            torch.full((self.levels - 2,), float(inverse_softplus))
        )

    def thresholds(self) -> torch.Tensor:
        if self.raw_gaps.numel() == 0:
            return self.first_threshold.view(1)
        gaps = F.softplus(self.raw_gaps).clamp_min(1e-4)
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=gaps.dtype, device=gaps.device),
                torch.cumsum(gaps, dim=0),
            ]
        )
        return self.first_threshold + offsets

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score(features) - self.thresholds().view(1, -1)


class CurrentExpertWithOrdinalAux(TargetedIntegratedExpertBNNet):
    """Current specialist architecture plus a small ordinal training head."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            backbone_name="resnet18",
            pretrained=not args.no_pretrained,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            texture_dim=args.texture_dim,
            expert_target=args.expert_target,
            texture_guided=True,
        )
        input_dim = int(self.specialist_head.head[0].in_features)
        self.ordinal_head = RankConsistentOrdinalHead(input_dim, DPI_LEVELS)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image = batch["effect"]
        general_feat = self.general_backbone(image).squeeze(-1).squeeze(-1)
        general_aux = self.general_aux_encoder(image)
        general = self.general_head(torch.cat([general_feat, general_aux], dim=1))

        specialist_feat = self.specialist_backbone(image).squeeze(-1).squeeze(-1)
        specialist_aux = self.specialist_descriptor_mlp(
            self.specialist_descriptor(image)
        )
        specialist_context = torch.cat(
            [
                specialist_feat * self.specialist_attention(specialist_feat),
                specialist_aux,
            ],
            dim=1,
        )
        specialist = self.specialist_head(specialist_context)
        columns: List[torch.Tensor] = []
        general_column = 0
        for parameter_index in range(len(PARAM_NAMES)):
            if parameter_index == self.specialist_index:
                columns.append(specialist)
            else:
                columns.append(general[:, general_column : general_column + 1])
                general_column += 1
        prediction = torch.cat(columns, dim=1)
        return prediction, {"coral_logits": self.ordinal_head(specialist_context)}


class DpiOneFactorPairDataset(Dataset):
    """Adjacent DPI pairs with all other factors and image identity fixed."""

    def __init__(
        self,
        train_csv: str,
        before_dir: str,
        after_dir: str,
        image_size: int,
        resize_mode: str,
        augmentation_mode: str,
    ) -> None:
        self.base = InverseExperimentDataset(
            train_csv,
            before_dir,
            after_dir,
            img_size=image_size,
            is_train=True,
            resize_mode=resize_mode,
            augmentation_mode=augmentation_mode,
        )
        self.after_dir = Path(after_dir)
        frame = self.base.df.copy().reset_index(drop=True)
        required = {
            "sample_id",
            "before_id",
            "pattern_id",
            "frequency",
            "pulse_width",
            "speed",
            "dpi",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError("Training CSV is missing {}".format(sorted(missing)))
        group_columns = [
            "before_id",
            "pattern_id",
            "frequency",
            "pulse_width",
            "speed",
        ]
        pairs: List[Dict[str, object]] = []
        for _, group in frame.groupby(group_columns, dropna=False):
            ordered = (
                group.sort_values(["dpi", "sample_id"])
                .drop_duplicates(subset=["dpi"], keep="first")
                .reset_index(drop=True)
            )
            for index in range(len(ordered) - 1):
                lower = ordered.iloc[index]
                upper = ordered.iloc[index + 1]
                delta = float(upper["dpi"] - lower["dpi"])
                if delta <= 0:
                    continue
                pairs.append(
                    {
                        "sample_id_a": str(lower["sample_id"]),
                        "sample_id_b": str(upper["sample_id"]),
                        "before_id": str(lower["before_id"]),
                        "pattern_id": str(lower["pattern_id"]),
                        "dpi_a": float(lower["dpi"]),
                        "dpi_b": float(upper["dpi"]),
                        "delta_norm": delta / 150.0,
                    }
                )
        if not pairs:
            raise RuntimeError("No training-only one-factor DPI pairs were found")
        self.pairs = pd.DataFrame(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _capture_rng():
        return torch.get_rng_state(), np.random.get_state(), random.getstate()

    @staticmethod
    def _restore_rng(state) -> None:
        torch.set_rng_state(state[0])
        np.random.set_state(state[1])
        random.setstate(state[2])

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.pairs.iloc[index]
        image_a = self.base._open_rgb(
            self.after_dir / "{}.jpg".format(row["sample_id_a"])
        )
        image_b = self.base._open_rgb(
            self.after_dir / "{}.jpg".format(row["sample_id_b"])
        )
        state = self._capture_rng()
        tensor_a = self.base.transform(image_a)
        self._restore_rng(state)
        tensor_b = self.base.transform(image_b)
        return {
            "effect_a": tensor_a,
            "effect_b": tensor_b,
            "delta_norm": torch.tensor(float(row["delta_norm"]), dtype=torch.float32),
            "sample_id_a": str(row["sample_id_a"]),
            "sample_id_b": str(row["sample_id_b"]),
        }


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
    parser.add_argument("--selection-metric", default="val_mean_mape")
    parser.add_argument("--coral-weight", type=float, default=0.005)
    parser.add_argument("--pair-rank-weight", type=float, default=0.05)
    parser.add_argument("--pair-delta-weight", type=float, default=0.5)
    parser.add_argument("--lds-sigma", type=float, default=1.5)
    parser.add_argument("--lds-max-weight", type=float, default=3.0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if args.expert_target != "dpi":
        raise ValueError("This ordered-grid experiment is pre-specified for DPI")
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


def uses_pairwise(variant: str) -> bool:
    return variant.startswith("pairwise_")


def uses_coral(variant: str) -> bool:
    return "coral" in variant


def uses_lds(variant: str) -> bool:
    return "lds" in variant


def build_model(args: argparse.Namespace) -> nn.Module:
    if uses_coral(args.variant):
        return CurrentExpertWithOrdinalAux(args)
    return TargetedIntegratedExpertBNNet(
        backbone_name="resnet18",
        pretrained=not args.no_pretrained,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        texture_dim=args.texture_dim,
        expert_target=args.expert_target,
        texture_guided=True,
    )


def level_indices(raw_dpi: np.ndarray) -> np.ndarray:
    return np.rint((raw_dpi.astype(float) - DPI_MIN) / DPI_STEP).astype(int).clip(0, DPI_LEVELS - 1)


def prepare_training_statistics(args: argparse.Namespace) -> Dict[str, object]:
    global LDS_LEVEL_WEIGHTS, CORAL_POS_WEIGHTS
    dataset = InverseExperimentDataset(
        args.train_csv,
        args.before_dir,
        args.after_dir,
        img_size=args.img_size,
        is_train=False,
        resize_mode=args.resize_mode,
        augmentation_mode="weak",
    )
    dpi = pd.to_numeric(dataset.df["dpi"], errors="raise").to_numpy(float)
    indices = level_indices(dpi)
    counts = np.bincount(indices, minlength=DPI_LEVELS).astype(float)

    radius = int(math.ceil(3.0 * args.lds_sigma))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / max(args.lds_sigma, 1e-6)) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(counts, kernel, mode="same")
    sample_density = np.maximum(smoothed[indices], 1e-6)
    sample_weights = 1.0 / np.sqrt(sample_density)
    sample_weights /= sample_weights.mean()
    level_weights = np.ones(DPI_LEVELS, dtype=np.float32)
    for level in range(DPI_LEVELS):
        members = sample_weights[indices == level]
        if len(members):
            level_weights[level] = float(members[0])
    level_weights = np.clip(level_weights, 0.5, args.lds_max_weight)
    level_weights /= float(np.mean(level_weights[indices]))
    LDS_LEVEL_WEIGHTS = torch.tensor(level_weights, dtype=torch.float32)

    ranks = torch.tensor(indices, dtype=torch.long)
    thresholds = torch.arange(DPI_LEVELS - 1).view(1, -1)
    targets = ranks.view(-1, 1) > thresholds
    positives = targets.sum(dim=0).float()
    negatives = float(len(ranks)) - positives
    pos_weight = negatives / positives.clamp_min(1.0)
    pos_weight = pos_weight.clamp(0.5, 5.0)
    pos_weight[positives.eq(0)] = 1.0
    CORAL_POS_WEIGHTS = pos_weight
    return {
        "train_samples": int(len(indices)),
        "dpi_level_counts": {str(level): int(value) for level, value in enumerate(counts)},
        "dpi_physical_level_counts": {
            str(int(DPI_MIN + DPI_STEP * level)): int(value)
            for level, value in enumerate(counts)
        },
        "lds_level_weights": [float(value) for value in level_weights],
        "coral_positive_weights": [float(value) for value in pos_weight.tolist()],
        "unseen_dpi_levels_in_training": [
            int(DPI_MIN + DPI_STEP * level)
            for level, value in enumerate(counts)
            if value == 0
        ],
    }


def regression_training_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    output_weights: torch.Tensor,
    use_density_weights: bool,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    if use_density_weights:
        if LDS_LEVEL_WEIGHTS is None:
            raise RuntimeError("LDS weights were not prepared")
        indices = torch.round((target_raw[:, DPI_INDEX] - DPI_MIN) / DPI_STEP).long()
        indices = indices.clamp(0, DPI_LEVELS - 1)
        weights = LDS_LEVEL_WEIGHTS.to(target.device)[indices]
        density_matrix = torch.ones_like(loss)
        density_matrix[:, DPI_INDEX] = weights
        loss = loss * density_matrix
    return (loss * output_weights.view(1, -1)).mean()


def coral_training_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if CORAL_POS_WEIGHTS is None:
        raise RuntimeError("CORAL weights were not prepared")
    ranks = torch.round(target[:, DPI_INDEX].clamp(0.0, 1.0) * (DPI_LEVELS - 1)).long()
    thresholds = torch.arange(DPI_LEVELS - 1, device=target.device).view(1, -1)
    binary = (ranks.view(-1, 1) > thresholds).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(
        logits,
        binary,
        pos_weight=CORAL_POS_WEIGHTS.to(target.device, dtype=logits.dtype),
    )


def next_pair_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_one_epoch(
    model: nn.Module,
    loader,
    output_weights: torch.Tensor,
    optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    total_loss = 0.0
    pair_iterator = iter(PAIR_LOADER) if PAIR_LOADER is not None else None
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_tensor_batch(batch, device)
        optimizer.zero_grad()
        output = model(batch)
        pred, aux = unpack_model_output(output)
        loss = regression_training_loss(
            pred,
            batch["params"],
            batch["params_raw"],
            output_weights,
            uses_lds(args.variant),
        )
        if uses_coral(args.variant):
            loss = loss + args.coral_weight * coral_training_loss(
                aux["coral_logits"], batch["params"]
            )

        if uses_pairwise(args.variant):
            if PAIR_LOADER is None or pair_iterator is None:
                raise RuntimeError("Pairwise variant has no training pair loader")
            pair, pair_iterator = next_pair_batch(pair_iterator, PAIR_LOADER)
            effect = torch.cat([pair["effect_a"], pair["effect_b"]], dim=0).to(device)
            pair_output = model({"effect": effect})
            pair_pred, _ = unpack_model_output(pair_output)
            half = pair["effect_a"].shape[0]
            predicted_delta = (
                pair_pred[half:, DPI_INDEX] - pair_pred[:half, DPI_INDEX]
            )
            true_delta = pair["delta_norm"].to(device)
            margin = torch.minimum(0.5 * true_delta, torch.full_like(true_delta, 0.05))
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


def main() -> None:
    global CURRENT_ARGS, PAIR_LOADER
    args = parse_args()
    CURRENT_ARGS = args
    statistics = prepare_training_statistics(args)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    pair_manifest: Dict[str, object] = {
        "construction_scope": "outer-training CSV only; validation and outer test excluded",
        "definition": (
            "same before_id, pattern_id, frequency, pulse_width and speed; "
            "adjacent distinct DPI levels"
        ),
        "synchronized_pair_augmentation": True,
        "variant": args.variant,
        **statistics,
    }
    if uses_pairwise(args.variant):
        pair_dataset = DpiOneFactorPairDataset(
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
        pair_manifest.update(
            {
                "pair_count": int(len(pair_dataset)),
                "unique_pair_samples": int(
                    len(
                        set(pair_dataset.pairs["sample_id_a"].astype(str))
                        | set(pair_dataset.pairs["sample_id_b"].astype(str))
                    )
                ),
                "unique_pair_initial_image_groups": int(
                    pair_dataset.pairs["before_id"].nunique()
                ),
            }
        )
        pair_dataset.pairs.to_csv(
            run_dir / "training_only_dpi_pairs.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        PAIR_LOADER = None
        pair_manifest["pair_count"] = 0
    (run_dir / "ordered_supervision_manifest.json").write_text(
        json.dumps(pair_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    base.ALL_VARIANTS = list(VARIANTS)
    base.parse_args = lambda: args
    base.build_model = build_model
    base.train_one_epoch = train_one_epoch
    base.main()


if __name__ == "__main__":
    main()
