#!/usr/bin/env python3
"""Train a leakage-controlled condition-set inverse model.

The model fixes two weaknesses of the historical DED comparator:

1. images from one physical condition are encoded jointly instead of being
   treated as independent training examples;
2. every placement uses one shared encoder/base predictor and the same
   scalar residual specialist.  Only the target receiving that specialist is
   changed, so all placement controls have identical parameters and effective
   specialist supervision.

The script is intentionally generic: a CSV supplies ``image_path``,
``condition_id``, a grouping column, and any ordered list of numeric targets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ParameterScaler:
    def __init__(self, frame: pd.DataFrame, parameters: Sequence[str]):
        self.parameters = list(parameters)
        self.minimum = frame[self.parameters].min().to_numpy(np.float32)
        self.maximum = frame[self.parameters].max().to_numpy(np.float32)
        self.scale = np.maximum(self.maximum - self.minimum, 1e-6)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.minimum) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.minimum

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "minimum": float(self.minimum[index]),
                "maximum": float(self.maximum[index]),
            }
            for index, name in enumerate(self.parameters)
        }


class ConditionImageSets(Dataset):
    def __init__(
        self,
        csv_path: Path,
        scaler: ParameterScaler,
        parameters: Sequence[str],
        group_col: str,
        set_size: int,
        image_size: int,
    ):
        frame = pd.read_csv(
            csv_path,
            dtype={"condition_id": str, group_col: str, "image_path": str},
        )
        required = {"condition_id", "image_path", group_col, *parameters}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError("{} is missing columns {}".format(csv_path, missing))
        self.parameters = list(parameters)
        self.group_col = group_col
        self.scaler = scaler
        self.set_size = int(set_size)
        self.records: List[Dict[str, object]] = []
        for condition_id, group in frame.groupby("condition_id", sort=True):
            target_values = group[self.parameters].apply(
                pd.to_numeric, errors="raise"
            ).to_numpy(dtype=np.float64)
            if not np.isfinite(target_values).all():
                raise ValueError(
                    "condition {} has non-finite target rows".format(condition_id)
                )
            canonical_truth = np.median(target_values, axis=0)
            consistent = np.allclose(
                target_values,
                canonical_truth.reshape(1, -1),
                rtol=1e-9,
                atol=1e-9,
            )
            if not consistent:
                spans = {
                    parameter: float(
                        target_values[:, index].max()
                        - target_values[:, index].min()
                    )
                    for index, parameter in enumerate(self.parameters)
                }
                raise ValueError(
                    "condition {} has materially inconsistent target rows: {}"
                    .format(condition_id, spans)
                )
            groups = group[group_col].astype(str).drop_duplicates()
            if len(groups) != 1:
                raise ValueError(
                    "condition {} spans multiple {} values".format(condition_id, group_col)
                )
            paths = sorted(group["image_path"].astype(str).drop_duplicates().tolist())
            if not paths:
                raise ValueError("condition {} has no images".format(condition_id))
            self.records.append(
                {
                    "condition_id": str(condition_id),
                    "group_id": str(groups.iloc[0]),
                    "paths": paths,
                    "truth": canonical_truth.astype(np.float32),
                }
            )
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _select_paths(self, paths: Sequence[str]) -> List[str]:
        if len(paths) >= self.set_size:
            indices = np.linspace(0, len(paths) - 1, self.set_size)
            return [paths[int(round(index))] for index in indices]
        return [paths[index % len(paths)] for index in range(self.set_size)]

    def __getitem__(self, index: int):
        record = self.records[index]
        tensors = []
        for relative in self._select_paths(record["paths"]):
            path = Path(str(relative))
            if not path.is_absolute():
                path = project_root() / path
            with Image.open(path) as image:
                tensors.append(self.transform(image.convert("L")))
        values = np.asarray(record["truth"], dtype=np.float32)
        return (
            torch.stack(tensors, dim=0),
            torch.from_numpy(self.scaler.transform(values)),
            str(record["condition_id"]),
            str(record["group_id"]),
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


class ConditionSetSpecialist(nn.Module):
    """Shared predictor plus one target-routed, zero-initialized specialist."""

    def __init__(
        self,
        parameter_count: int,
        routing_index: int,
        pretrained: bool,
        dropout: float,
    ):
        super().__init__()
        self.routing_index = int(routing_index)
        self.encoder = make_resnet18_feature_map(pretrained)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.frame_attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.base_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, parameter_count),
        )
        self.specialist = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        # The specialist starts as an exact no-op; any nonzero contribution
        # must be earned from the routed target's inner-training loss.
        nn.init.zeros_(self.specialist[-1].weight)
        nn.init.zeros_(self.specialist[-1].bias)

    def forward(self, image_sets: torch.Tensor) -> torch.Tensor:
        batch, images_per_condition, channels, height, width = image_sets.shape
        maps = self.encoder(
            image_sets.reshape(
                batch * images_per_condition, channels, height, width
            )
        )
        frame_features = self.spatial_pool(maps).flatten(1)
        frame_features = frame_features.reshape(batch, images_per_condition, 512)
        weights = torch.softmax(self.frame_attention(frame_features), dim=1)
        pooled = torch.sum(weights * frame_features, dim=1)
        logits = self.base_head(pooled)
        residual = self.specialist(pooled).squeeze(1)
        routed = logits.clone()
        routed[:, self.routing_index] = (
            routed[:, self.routing_index] + residual
        )
        return torch.sigmoid(routed)


def condition_predictions(
    model: nn.Module,
    loader: DataLoader,
    scaler: ParameterScaler,
    parameters: Sequence[str],
    device: torch.device,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for images, targets, condition_ids, group_ids in loader:
            prediction = model(images.to(device)).cpu().numpy()
            truth = targets.numpy()
            prediction_raw = scaler.inverse(prediction)
            truth_raw = scaler.inverse(truth)
            for row_index in range(len(condition_ids)):
                row: Dict[str, object] = {
                    "condition_id": condition_ids[row_index],
                    "group_id": group_ids[row_index],
                }
                for parameter_index, parameter in enumerate(parameters):
                    true_value = float(truth_raw[row_index, parameter_index])
                    predicted = float(prediction_raw[row_index, parameter_index])
                    absolute = abs(predicted - true_value)
                    row["true_{}".format(parameter)] = true_value
                    row["pred_{}".format(parameter)] = predicted
                    row["ae_{}".format(parameter)] = absolute
                    row["nmae_{}".format(parameter)] = absolute / float(
                        scaler.scale[parameter_index]
                    )
                    row["ape_{}".format(parameter)] = (
                        100.0 * absolute / max(abs(true_value), 1e-6)
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def group_equal_metrics(
    frame: pd.DataFrame, parameters: Sequence[str]
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    grouped = frame.groupby("group_id", sort=False)
    nmae_values = []
    for parameter in parameters:
        result["mape_{}".format(parameter)] = float(
            grouped["ape_{}".format(parameter)].mean().mean()
        )
        result["mae_{}".format(parameter)] = float(
            grouped["ae_{}".format(parameter)].mean().mean()
        )
        nmae = float(grouped["nmae_{}".format(parameter)].mean().mean())
        result["nmae_{}".format(parameter)] = nmae
        nmae_values.append(nmae)
    result["mean_nmae"] = float(np.mean(nmae_values))
    result["mean_mape"] = float(
        np.mean([result["mape_{}".format(name)] for name in parameters])
    )
    result["groups"] = int(frame["group_id"].nunique())
    result["conditions"] = int(frame["condition_id"].nunique())
    return result


def make_loader(
    csv_path: Path,
    scaler: ParameterScaler,
    parameters: Sequence[str],
    group_col: str,
    set_size: int,
    image_size: int,
    train: bool,
    batch_size: int,
    workers: int,
    seed: int,
) -> Tuple[ConditionImageSets, DataLoader]:
    dataset = ConditionImageSets(
        csv_path,
        scaler,
        parameters,
        group_col,
        set_size,
        image_size,
    )
    sampler = None
    if train:
        group_counts: Dict[str, int] = {}
        for record in dataset.records:
            group = str(record["group_id"])
            group_counts[group] = group_counts.get(group, 0) + 1
        weights = torch.as_tensor(
            [1.0 / group_counts[str(record["group_id"])] for record in dataset.records],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return dataset, loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--parameters", nargs="+", required=True)
    parser.add_argument("--group-column", default="track_id")
    parser.add_argument("--routing-target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
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

    parameters = list(args.parameters)
    if args.routing_target not in parameters:
        raise ValueError("routing target must be one of {}".format(parameters))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "test_metrics.json"
    if metrics_path.exists():
        print("[SKIP] completed {}".format(output))
        return

    seed_everything(args.seed)
    train_frame = pd.read_csv(args.train_csv)
    scaler = ParameterScaler(train_frame, parameters)
    _, train_loader = make_loader(
        Path(args.train_csv), scaler, parameters, args.group_column,
        args.set_size, args.image_size, True, args.batch_size,
        args.num_workers, args.seed,
    )
    _, val_loader = make_loader(
        Path(args.val_csv), scaler, parameters, args.group_column,
        args.set_size, args.image_size, False, args.batch_size,
        args.num_workers, args.seed,
    )
    _, test_loader = make_loader(
        Path(args.test_csv), scaler, parameters, args.group_column,
        args.set_size, args.image_size, False, args.batch_size,
        args.num_workers, args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionSetSpecialist(
        len(parameters),
        parameters.index(args.routing_target),
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.SmoothL1Loss()
    best_score = math.inf
    best_epoch = -1
    stale = 0
    history = []
    checkpoint = output / "best_model.pth"

    for epoch in range(args.epochs):
        encoder_trainable = epoch >= args.freeze_backbone_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(encoder_trainable)
        model.train()
        if not encoder_trainable:
            # Keep ImageNet batch-normalization statistics fixed while the
            # newly initialized pooling/head modules warm up.
            model.encoder.eval()
        losses = []
        for images, targets, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(images)
            loss = criterion(prediction, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        validation = condition_predictions(
            model, val_loader, scaler, parameters, device
        )
        validation_metrics = group_equal_metrics(validation, parameters)
        score = float(validation_metrics["mean_nmae"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_group_equal_mean_nmae": score,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "encoder_trainable": encoder_trainable,
            }
        )
        print(
            "[adapter_{} seed{}] epoch={:03d} loss={:.6f} val_nmae={:.6f}".format(
                args.routing_target,
                args.seed,
                epoch + 1,
                float(np.mean(losses)),
                score,
            ),
            flush=True,
        )
        if score < best_score - 1e-8:
            best_score = score
            best_epoch = epoch + 1
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "parameters": parameters,
                    "routing_target": args.routing_target,
                    "seed": args.seed,
                    "scaler": scaler.as_dict(),
                    "parameter_count": parameter_count,
                    "architecture": "shared_resnet18_condition_attention_zero_init_residual_adapter",
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    test_predictions = condition_predictions(
        model, test_loader, scaler, parameters, device
    )
    test_predictions.to_csv(output / "test_predictions_conditions.csv", index=False)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    metrics = {
        "routing": "adapter_{}".format(args.routing_target),
        "routing_target": args.routing_target,
        "parameters": parameters,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "selection_metric": "inner-validation group-equal mean normalized MAE",
        "best_validation_score": best_score,
        "architecture": "shared_resnet18_condition_attention_zero_init_residual_adapter",
        "parameter_count": parameter_count,
        "pretrained": not args.no_pretrained,
        "augmentation": "none",
        "bounded_output": "sigmoid in train-fold min-max space",
        "set_size": args.set_size,
        "condition_target_consistency": (
            "canonical median after finite-value and allclose checks "
            "(rtol=1e-9, atol=1e-9)"
        ),
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "train_groups": int(train_frame[args.group_column].astype(str).nunique()),
        "train_conditions": int(train_frame["condition_id"].astype(str).nunique()),
        "train_images": int(train_frame["image_path"].astype(str).nunique()),
        "group_equal": group_equal_metrics(test_predictions, parameters),
        "scaler": scaler.as_dict(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
