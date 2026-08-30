#!/usr/bin/env python3
"""Train one exploratory NIST condition-set aggregation model.

This runner keeps the frozen NIST condition-disjoint folds, normalized targets,
loss, checkpoint rule and ResNet-18 encoder used by the 20260802 replication.
It varies only how the replicate micrographs belonging to one physical
condition are represented before the shared head and scalar specialist.

Pilot mode never accepts or reads an outer-test CSV.  Confirm mode evaluates a
frozen fold-specific aggregator on the outer test only after inner-validation
selection has been written by the orchestration script.
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import train_condition_set_specialist_20260802 as base  # noqa: E402


VARIANTS = (
    "legacy_repeat_attention",
    "masked_attention",
    "masked_mean_std_max",
    "masked_gated_moments",
    "masked_set_transformer_lite",
    "masked_spatial_gated_moments",
)


class MaskedConditionImageSets(Dataset):
    """Condition sets with explicit padding masks instead of repeated frames."""

    def __init__(
        self,
        csv_path: Path,
        scaler: base.ParameterScaler,
        parameters: Sequence[str],
        group_col: str,
        set_size: int,
        image_size: int,
        repeat_to_fill: bool,
    ) -> None:
        source = base.ConditionImageSets(
            csv_path, scaler, parameters, group_col, set_size, image_size
        )
        self.records = source.records
        self.parameters = source.parameters
        self.group_col = source.group_col
        self.scaler = source.scaler
        self.set_size = source.set_size
        self.transform = source.transform
        self.repeat_to_fill = bool(repeat_to_fill)

    def __len__(self) -> int:
        return len(self.records)

    def _chosen(self, paths: Sequence[str]) -> List[str]:
        if self.repeat_to_fill:
            if len(paths) >= self.set_size:
                indices = np.linspace(0, len(paths) - 1, self.set_size)
                return [paths[int(round(index))] for index in indices]
            return [paths[index % len(paths)] for index in range(self.set_size)]
        if len(paths) <= self.set_size:
            return list(paths)
        indices = np.linspace(0, len(paths) - 1, self.set_size)
        return [paths[int(round(index))] for index in indices]

    def __getitem__(self, index: int):
        record = self.records[index]
        tensors: List[torch.Tensor] = []
        chosen = self._chosen(record["paths"])
        for relative in chosen:
            path = Path(str(relative))
            if not path.is_absolute():
                path = PROJECT / path
            with Image.open(path) as image:
                tensors.append(self.transform(image.convert("L")))
        valid = len(tensors)
        if not self.repeat_to_fill:
            blank = torch.zeros_like(tensors[0])
            tensors.extend(blank.clone() for _ in range(self.set_size - valid))
        mask = torch.zeros(self.set_size, dtype=torch.bool)
        mask[:valid] = True
        values = np.asarray(record["truth"], dtype=np.float32)
        return (
            torch.stack(tensors, dim=0),
            mask,
            torch.from_numpy(self.scaler.transform(values)),
            str(record["condition_id"]),
            str(record["group_id"]),
        )


def make_loader(
    csv_path: Path,
    scaler: base.ParameterScaler,
    parameters: Sequence[str],
    group_col: str,
    set_size: int,
    image_size: int,
    repeat_to_fill: bool,
    train: bool,
    batch_size: int,
    workers: int,
    seed: int,
) -> Tuple[MaskedConditionImageSets, DataLoader]:
    dataset = MaskedConditionImageSets(
        csv_path,
        scaler,
        parameters,
        group_col,
        set_size,
        image_size,
        repeat_to_fill,
    )
    sampler = None
    if train:
        counts: Dict[str, int] = {}
        for record in dataset.records:
            group = str(record["group_id"])
            counts[group] = counts.get(group, 0) + 1
        weights = torch.as_tensor(
            [1.0 / counts[str(record["group_id"])] for record in dataset.records],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype).unsqueeze(-1)
    return (values * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def masked_std(values: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype).unsqueeze(-1)
    variance = ((values - mean.unsqueeze(1)).square() * weight).sum(1)
    variance = variance / weight.sum(1).clamp_min(1.0)
    return torch.sqrt(variance.clamp_min(1e-8))


def masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    floor = torch.finfo(values.dtype).min
    return values.masked_fill(~mask.unsqueeze(-1), floor).max(1).values


class SetTransformerLite(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.project = nn.Linear(512, 256)
        self.norm1 = nn.LayerNorm(256)
        self.self_attention = nn.MultiheadAttention(
            256, 4, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(256)
        self.feed_forward = nn.Sequential(
            nn.Linear(256, 512), nn.SiLU(inplace=True), nn.Dropout(dropout), nn.Linear(512, 256)
        )
        self.query = nn.Parameter(torch.zeros(1, 1, 256))
        nn.init.normal_(self.query, std=0.02)
        self.query_attention = nn.MultiheadAttention(
            256, 4, dropout=dropout, batch_first=True
        )
        self.out = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 512))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.project(values)
        attended, _ = self.self_attention(
            self.norm1(encoded), self.norm1(encoded), self.norm1(encoded),
            key_padding_mask=~mask,
            need_weights=False,
        )
        encoded = encoded + attended
        encoded = encoded + self.feed_forward(self.norm2(encoded))
        query = self.query.expand(encoded.shape[0], -1, -1)
        pooled, _ = self.query_attention(
            query, encoded, encoded, key_padding_mask=~mask, need_weights=False
        )
        return self.out(pooled[:, 0])


class NISTSetAggregationModel(nn.Module):
    def __init__(
        self,
        parameter_count: int,
        routing_index: int,
        variant: str,
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
        needs_attention = variant in (
            "legacy_repeat_attention",
            "masked_attention",
            "masked_gated_moments",
            "masked_spatial_gated_moments",
        )
        self.frame_attention = (
            nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
            if needs_attention else None
        )
        needs_moments = variant in (
            "masked_mean_std_max",
            "masked_gated_moments",
            "masked_spatial_gated_moments",
        )
        self.moment_project = (
            nn.Sequential(
                nn.Linear(1536, 512), nn.LayerNorm(512),
                nn.SiLU(inplace=True), nn.Dropout(dropout),
            )
            if needs_moments else None
        )
        self.spatial_project = (
            nn.Sequential(
                nn.Linear(1024, 512), nn.LayerNorm(512), nn.SiLU(inplace=True)
            )
            if variant == "masked_spatial_gated_moments" else None
        )
        self.set_transformer = (
            SetTransformerLite(dropout)
            if variant == "masked_set_transformer_lite" else None
        )
        self.base_head = nn.Sequential(
            nn.Linear(512, 256), nn.SiLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, parameter_count)
        )
        self.specialist = nn.Sequential(
            nn.Linear(512, 256), nn.SiLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, 1)
        )
        nn.init.zeros_(self.specialist[-1].weight)
        nn.init.zeros_(self.specialist[-1].bias)

    def _encode_valid(self, image_sets: torch.Tensor, mask: torch.Tensor):
        batch, count, channels, height, width = image_sets.shape
        flat = image_sets.reshape(batch * count, channels, height, width)
        valid = mask.reshape(-1)
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        valid_maps = self.encoder(flat.index_select(0, valid_indices))
        map_shape = valid_maps.shape[1:]
        maps = valid_maps.new_zeros((batch * count, *map_shape))
        maps = maps.index_copy(0, valid_indices, valid_maps)
        maps = maps.reshape(batch, count, *map_shape)
        features = self.spatial_pool(maps.reshape(batch * count, *map_shape)).flatten(1)
        return maps, features.reshape(batch, count, 512)

    def _attention_mean(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.frame_attention(features).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (weights * features).sum(1)

    def aggregate(self, maps: torch.Tensor, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.variant in ("legacy_repeat_attention", "masked_attention"):
            return self._attention_mean(features, mask)
        if self.variant == "masked_mean_std_max":
            mean = masked_mean(features, mask)
            return self.moment_project(torch.cat([mean, masked_std(features, mask, mean), masked_max(features, mask)], dim=1))
        if self.variant == "masked_gated_moments":
            gated = self._attention_mean(features, mask)
            return self.moment_project(torch.cat([gated, masked_std(features, mask, gated), masked_max(features, mask)], dim=1))
        if self.variant == "masked_set_transformer_lite":
            return self.set_transformer(features, mask)
        if self.variant == "masked_spatial_gated_moments":
            spatial_std = maps.flatten(-2).std(-1, unbiased=False)
            per_frame = self.spatial_project(torch.cat([features, spatial_std], dim=-1))
            gated = self._attention_mean(per_frame, mask)
            return self.moment_project(torch.cat([gated, masked_std(per_frame, mask, gated), masked_max(per_frame, mask)], dim=1))
        raise RuntimeError(self.variant)

    def forward(self, image_sets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        maps, features = self._encode_valid(image_sets, mask)
        pooled = self.aggregate(maps, features, mask)
        logits = self.base_head(pooled)
        residual = self.specialist(pooled).squeeze(1)
        routed = logits.clone()
        routed[:, self.routing_index] = routed[:, self.routing_index] + residual
        return torch.sigmoid(routed)


def predictions(model, loader, scaler, parameters, device) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for images, mask, targets, condition_ids, group_ids in loader:
            pred = model(images.to(device), mask.to(device)).cpu().numpy()
            true = targets.numpy()
            pred_raw = scaler.inverse(pred)
            true_raw = scaler.inverse(true)
            for i in range(len(condition_ids)):
                row: Dict[str, object] = {"condition_id": condition_ids[i], "group_id": group_ids[i]}
                for j, parameter in enumerate(parameters):
                    truth = float(true_raw[i, j])
                    estimate = float(pred_raw[i, j])
                    absolute = abs(estimate - truth)
                    row["true_{}".format(parameter)] = truth
                    row["pred_{}".format(parameter)] = estimate
                    row["ae_{}".format(parameter)] = absolute
                    row["nmae_{}".format(parameter)] = absolute / float(scaler.scale[j])
                    row["ape_{}".format(parameter)] = 100.0 * absolute / max(abs(truth), 1e-6)
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "confirm"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv")
    parser.add_argument("--parameters", nargs="+", required=True)
    parser.add_argument("--group-column", default="group_id")
    parser.add_argument("--routing-target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--set-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if args.phase == "pilot" and args.test_csv:
        raise ValueError("Pilot mode must not receive an outer-test CSV")
    if args.phase == "confirm" and not args.test_csv:
        raise ValueError("Confirm mode requires --test-csv")
    parameters = list(args.parameters)
    if args.routing_target not in parameters:
        raise ValueError("routing target must be one of {}".format(parameters))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_name = "validation_summary.json" if args.phase == "pilot" else "test_metrics.json"
    summary_path = output / summary_name
    if summary_path.exists():
        print("[SKIP] completed {}".format(output))
        return

    base.seed_everything(args.seed)
    train_frame = pd.read_csv(args.train_csv)
    scaler = base.ParameterScaler(train_frame, parameters)
    repeat = args.variant == "legacy_repeat_attention"
    loader_args = (scaler, parameters, args.group_column, args.set_size, args.image_size, repeat)
    _, train_loader = make_loader(Path(args.train_csv), *loader_args, True, args.batch_size, args.num_workers, args.seed)
    _, val_loader = make_loader(Path(args.val_csv), *loader_args, False, args.batch_size, args.num_workers, args.seed)
    test_loader = None
    if args.test_csv:
        _, test_loader = make_loader(Path(args.test_csv), *loader_args, False, args.batch_size, args.num_workers, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NISTSetAggregationModel(
        len(parameters), parameters.index(args.routing_target), args.variant,
        pretrained=not args.no_pretrained, dropout=args.dropout,
    ).to(device)
    count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.SmoothL1Loss()
    best_score, best_epoch, stale = math.inf, -1, 0
    history = []
    checkpoint = output / "best_model.pth"

    for epoch in range(args.epochs):
        encoder_trainable = epoch >= args.freeze_backbone_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(encoder_trainable)
        model.train()
        if not encoder_trainable:
            model.encoder.eval()
        losses = []
        for images, mask, targets, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            estimate = model(images.to(device, non_blocking=True), mask.to(device, non_blocking=True))
            loss = criterion(estimate, targets.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_predictions = predictions(model, val_loader, scaler, parameters, device)
        val_metrics = base.group_equal_metrics(val_predictions, parameters)
        selected = val_metrics["nmae_{}".format(args.routing_target)]
        nonselected = [val_metrics["nmae_{}".format(p)] for p in parameters if p != args.routing_target]
        score = float(selected + 0.25 * val_metrics["mean_nmae"] + 0.10 * max(nonselected))
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_selection_score": score,
            "val_mean_nmae": val_metrics["mean_nmae"],
            "val_selected_nmae": selected,
            "encoder_trainable": encoder_trainable,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        print("[{} fold{} {} seed{}] epoch={:03d} loss={:.6f} val_score={:.6f}".format(
            args.phase, args.outer_fold, args.variant, args.seed, epoch + 1, float(np.mean(losses)), score
        ), flush=True)
        if score < best_score - 1e-8:
            best_score, best_epoch, stale = score, epoch + 1, 0
            torch.save({"model": model.state_dict(), "variant": args.variant}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    val_predictions = predictions(model, val_loader, scaler, parameters, device)
    val_predictions.to_csv(output / "validation_predictions_conditions.csv", index=False)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    payload = {
        "phase": args.phase,
        "outer_fold": args.outer_fold,
        "variant": args.variant,
        "routing_target": args.routing_target,
        "seed": args.seed,
        "parameter_count": count,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "validation_group_equal": base.group_equal_metrics(val_predictions, parameters),
        "selection_metric": "selected-target NMAE + 0.25 mean NMAE + 0.10 worst nonselected NMAE",
        "explicit_padding_mask": not repeat,
        "set_size": args.set_size,
        "augmentation": "none",
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "scaler": scaler.as_dict(),
    }
    if test_loader is not None:
        test_predictions = predictions(model, test_loader, scaler, parameters, device)
        test_predictions.to_csv(output / "test_predictions_conditions.csv", index=False)
        payload["group_equal"] = base.group_equal_metrics(test_predictions, parameters)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
