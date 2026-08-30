#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageOps, ImageStat
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from experiments.metadata import infer_metadata_columns, load_table
from utils.experiment_plots import plot_inverse_training_log
from utils.seed import set_seed


PARAM_SPECS = [
    ("频率", "frequency", 20.0, 95.0),
    ("脉宽", "pulse_width", 25.0, 100.0),
    ("速度", "speed", 30000.0, 50000.0),
    ("DPI", "dpi", 25.0, 175.0),
]
PARAM_COLUMN_ALIASES = {
    "频率": ("频率", "frequency", "freq", "棰戠巼"),
    "脉宽": ("脉宽", "pulse_width", "pulse", "鑴夊"),
    "速度": ("速度", "speed", "閫熷害"),
    "DPI": ("DPI", "dpi"),
}


def canonicalize_parameter_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept both legacy Chinese and canonical English label headers."""
    output = frame.copy()
    for canonical, aliases in PARAM_COLUMN_ALIASES.items():
        source = next(
            (column for column in aliases if column in output.columns),
            None,
        )
        if source is None:
            raise ValueError(
                f"Missing parameter column {canonical!r}; tried {aliases}"
            )
        values = pd.to_numeric(output[source], errors="raise")
        if canonical in output.columns and source != canonical:
            canonical_values = pd.to_numeric(
                output[canonical], errors="raise"
            )
            if not np.allclose(
                canonical_values.to_numpy(float),
                values.to_numpy(float),
                equal_nan=True,
            ):
                raise ValueError(
                    f"Conflicting aliases for parameter {canonical!r}"
                )
        output[canonical] = values
    return output


def normalize_params(params: np.ndarray) -> np.ndarray:
    normalized = []
    for value, (_, _, lower, upper) in zip(params, PARAM_SPECS):
        normalized.append(np.clip((float(value) - lower) / (upper - lower), 0.0, 1.0))
    return np.asarray(normalized, dtype=np.float32)


def denormalize_params(params: np.ndarray) -> np.ndarray:
    physical = []
    for value, (_, _, lower, upper) in zip(params, PARAM_SPECS):
        physical.append(float(value) * (upper - lower) + lower)
    return np.asarray(physical, dtype=np.float32)


BACKBONE_CHOICES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
    "mobilenet_v3_large",
]


def _build_torchvision_model(factory_name: str, weights_name: str, pretrained: bool):
    factory = getattr(models, factory_name)
    if not pretrained:
        try:
            return factory(weights=None)
        except TypeError:
            return factory(pretrained=False)

    try:
        weights_enum = getattr(models, weights_name)
        return factory(weights=weights_enum.DEFAULT)
    except Exception:
        return factory(pretrained=True)


def get_backbone(backbone_name: str = "resnet18", pretrained: bool = True):
    if backbone_name == "resnet18":
        backbone = models.resnet18(pretrained=pretrained)
        feat_dim = 512
        backbone = nn.Sequential(*list(backbone.children())[:-1])
    elif backbone_name == "resnet34":
        backbone = models.resnet34(pretrained=pretrained)
        feat_dim = 512
        backbone = nn.Sequential(*list(backbone.children())[:-1])
    elif backbone_name == "resnet50":
        backbone = models.resnet50(pretrained=pretrained)
        feat_dim = 2048
        backbone = nn.Sequential(*list(backbone.children())[:-1])
    elif backbone_name == "efficientnet_b0":
        net = _build_torchvision_model("efficientnet_b0", "EfficientNet_B0_Weights", pretrained)
        feat_dim = int(net.classifier[1].in_features)
        backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d((1, 1)))
    elif backbone_name == "convnext_tiny":
        net = _build_torchvision_model("convnext_tiny", "ConvNeXt_Tiny_Weights", pretrained)
        feat_dim = int(net.classifier[2].in_features)
        backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d((1, 1)))
    elif backbone_name == "mobilenet_v3_large":
        net = _build_torchvision_model("mobilenet_v3_large", "MobileNet_V3_Large_Weights", pretrained)
        feat_dim = int(net.classifier[0].in_features)
        backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d((1, 1)))
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    return backbone, feat_dim


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3, num_params: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_params),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class BNRegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3, num_params: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_params),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def make_head(head_type: str, input_dim: int, hidden_dim: int, dropout: float, num_params: int) -> nn.Module:
    if head_type == "simple":
        return RegressionHead(input_dim, hidden_dim, dropout, num_params)
    if head_type == "bn":
        return BNRegressionHead(input_dim, hidden_dim, dropout, num_params)
    raise ValueError(f"Unsupported head_type: {head_type}")


class DualStreamInverseNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        fusion: str = "concat_diff",
        hidden_dim: int = 256,
        dropout: float = 0.3,
        head_type: str = "simple",
    ):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.fusion = fusion

        if fusion == "concat":
            head_in = feat_dim * 2
        elif fusion == "diff":
            head_in = feat_dim
        elif fusion == "concat_diff":
            head_in = feat_dim * 3
        elif fusion == "gated":
            head_in = feat_dim * 3
        elif fusion == "concat_absdiff":
            head_in = feat_dim * 4
        elif fusion == "cross_attention":
            head_in = feat_dim * 3
        elif fusion == "masked_concat_diff":
            head_in = feat_dim * 3
        elif fusion in {"input_signeddiff", "input_absdiff"}:
            head_in = feat_dim
        elif fusion == "input4_late":
            head_in = feat_dim * 4
        else:
            raise ValueError(f"Unsupported fusion: {fusion}")

        if fusion in {"gated", "cross_attention"}:
            self.before_gate = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),
                nn.Sigmoid(),
            )
            self.effect_gate = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),
                nn.Sigmoid(),
            )
        if fusion == "cross_attention":
            self.diff_gate = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),
                nn.Sigmoid(),
            )

        self.head = make_head(head_type, head_in, hidden_dim, dropout, num_params=len(PARAM_SPECS))

    def _feature(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image).squeeze(-1).squeeze(-1)

    def _pixel_delta(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        signed = batch["effect"] - batch["before"]
        abs_delta = torch.abs(signed)
        return signed, abs_delta

    def _change_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        _, abs_delta = self._pixel_delta(batch)
        mask = abs_delta.mean(dim=1, keepdim=True)
        flat = mask.flatten(1)
        scale = flat.amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        return (mask / scale).clamp(0.0, 1.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.fusion in {"input_signeddiff", "input_absdiff", "input4_late"}:
            signed_delta, abs_delta = self._pixel_delta(batch)
            if self.fusion == "input_signeddiff":
                fused = self._feature(signed_delta)
            elif self.fusion == "input_absdiff":
                fused = self._feature(abs_delta)
            else:
                before_feat = self._feature(batch["before"])
                effect_feat = self._feature(batch["effect"])
                signed_feat = self._feature(signed_delta)
                abs_feat = self._feature(abs_delta)
                fused = torch.cat([before_feat, effect_feat, signed_feat, abs_feat], dim=1)
            return self.head(fused)

        before_image = batch["before"]
        effect_image = batch["effect"]
        if self.fusion == "masked_concat_diff":
            mask = self._change_mask(batch)
            before_image = before_image * mask
            effect_image = effect_image * mask

        before_feat = self._feature(before_image)
        effect_feat = self._feature(effect_image)

        if self.fusion == "concat":
            fused = torch.cat([before_feat, effect_feat], dim=1)
        elif self.fusion == "diff":
            fused = effect_feat - before_feat
        elif self.fusion == "gated":
            joint = torch.cat([before_feat, effect_feat], dim=1)
            gated_before = before_feat * self.before_gate(joint)
            gated_effect = effect_feat * self.effect_gate(joint)
            diff = gated_effect - gated_before
            fused = torch.cat([gated_before, gated_effect, diff], dim=1)
        elif self.fusion == "concat_absdiff":
            diff = effect_feat - before_feat
            abs_diff = torch.abs(diff)
            fused = torch.cat([before_feat, effect_feat, diff, abs_diff], dim=1)
        elif self.fusion == "cross_attention":
            joint = torch.cat([before_feat, effect_feat], dim=1)
            attended_before = before_feat * self.before_gate(joint)
            attended_effect = effect_feat * self.effect_gate(joint)
            attended_diff = (effect_feat - before_feat) * self.diff_gate(joint)
            fused = torch.cat([attended_before, attended_effect, attended_diff], dim=1)
        else:
            diff = effect_feat - before_feat
            fused = torch.cat([before_feat, effect_feat, diff], dim=1)
        return self.head(fused)


class SingleImageInverseNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        image_key: str = "effect",
        head_type: str = "simple",
    ):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.head = make_head(head_type, feat_dim, hidden_dim, dropout, num_params=len(PARAM_SPECS))
        self.image_key = image_key

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        feat = self.backbone(batch[self.image_key]).squeeze(-1).squeeze(-1)
        return self.head(feat)


class LetterboxSquare:
    """Resize without changing aspect ratio, then pad to a square canvas.

    A per-image median fill avoids introducing a fixed black-border cue that
    could become correlated with a held-out acquisition or pattern group.
    """

    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        resampling = getattr(Image, "Resampling", Image)
        contained = ImageOps.contain(
            image,
            (self.size, self.size),
            method=resampling.BILINEAR,
        )
        median = tuple(int(round(value)) for value in ImageStat.Stat(contained).median[:3])
        canvas = Image.new("RGB", (self.size, self.size), color=median)
        left = (self.size - contained.width) // 2
        top = (self.size - contained.height) // 2
        canvas.paste(contained, (left, top))
        return canvas


class InverseExperimentDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        before_dir: str,
        after_dir: str,
        img_size: int = 224,
        is_train: bool = True,
        resize_mode: str = "stretch",
        augmentation_mode: str = "weak",
    ):
        self.df = canonicalize_parameter_columns(
            infer_metadata_columns(load_table(csv_path))
        )
        self.before_dir = Path(before_dir)
        self.after_dir = Path(after_dir)
        self.img_size = img_size
        self.is_train = is_train
        if resize_mode not in {"stretch", "letterbox"}:
            raise ValueError(f"Unsupported resize_mode: {resize_mode}")
        if augmentation_mode not in {"none", "weak", "strong"}:
            raise ValueError(f"Unsupported augmentation_mode: {augmentation_mode}")
        self.resize_mode = resize_mode
        self.augmentation_mode = augmentation_mode
        self.transform = self._build_transform()

    def _build_transform(self):
        resize = (
            transforms.Resize((self.img_size, self.img_size))
            if self.resize_mode == "stretch"
            else LetterboxSquare(self.img_size)
        )
        common = [
            resize,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if not self.is_train:
            return transforms.Compose(common)
        if self.augmentation_mode == "none":
            return transforms.Compose(common)
        if self.augmentation_mode == "strong":
            return transforms.Compose(
                [
                    resize,
                    transforms.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.15,
                        hue=0.03,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.25),
                    transforms.RandomAffine(
                        degrees=10,
                        translate=(0.05, 0.05),
                        scale=(0.90, 1.10),
                        fill=(124, 116, 104),
                    ),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                        p=0.20,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        return transforms.Compose(
            [
                resize,
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def _open_rgb(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.df.iloc[index]
        sample_id = str(row["sample_id"])
        before_id = str(row["before_id"]) if "before_id" in row and pd.notna(row["before_id"]) else sample_id

        before_path = self.before_dir / f"{before_id}.jpg"
        after_path = self.after_dir / f"{sample_id}.jpg"

        params_raw = np.asarray([row[column] for column, _, _, _ in PARAM_SPECS], dtype=np.float32)
        params_norm = normalize_params(params_raw)

        before_image = self._open_rgb(before_path)
        effect_image = self._open_rgb(after_path)
        if self.is_train and self.augmentation_mode != "none":
            # Paired inputs must receive exactly the same stochastic transform.
            # Independent crops/flips/jitter would create artificial differences
            # and unfairly penalize difference-only and dual-input conditions.
            torch_state = torch.get_rng_state()
            numpy_state = np.random.get_state()
            python_state = random.getstate()
            before_tensor = self.transform(before_image)
            torch.set_rng_state(torch_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            effect_tensor = self.transform(effect_image)
        else:
            before_tensor = self.transform(before_image)
            effect_tensor = self.transform(effect_image)

        sample = {
            "before": before_tensor,
            "effect": effect_tensor,
            "params": torch.tensor(params_norm, dtype=torch.float32),
            "params_raw": torch.tensor(params_raw, dtype=torch.float32),
            "sample_id": sample_id,
            "before_id": before_id,
            "split": row["split"] if "split" in row else ("train" if self.is_train else "val"),
            "pattern_id": str(row["pattern_id"]) if "pattern_id" in row and pd.notna(row["pattern_id"]) else "",
            "batch_id": str(row["batch_id"]) if "batch_id" in row and pd.notna(row["batch_id"]) else "",
        }
        return sample


@dataclass
class EvalResult:
    loss: float
    mae_norm: float
    mape_physical: float
    param_mae_norm: dict[str, float]
    param_mape_physical: dict[str, float]


def move_tensor_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def compute_physical_errors(pred_norm: np.ndarray, true_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_raw = np.stack([denormalize_params(item) for item in pred_norm], axis=0)
    abs_err = np.abs(pred_raw - true_raw)
    ape = abs_err / np.maximum(np.abs(true_raw), 1e-6) * 100.0
    return pred_raw, ape


def run_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_tensor_batch(batch, device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch["params"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch["before"].size(0)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> EvalResult:
    model.eval()
    total_loss = 0.0
    all_pred_norm = []
    all_true_norm = []
    all_true_raw = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_tensor_batch(batch, device)
        pred = model(batch)
        loss = criterion(pred, batch["params"])
        total_loss += loss.item() * batch["before"].size(0)

        all_pred_norm.append(pred.detach().cpu().numpy())
        all_true_norm.append(batch["params"].detach().cpu().numpy())
        all_true_raw.append(batch["params_raw"].detach().cpu().numpy())

    pred_norm = np.concatenate(all_pred_norm, axis=0)
    true_norm = np.concatenate(all_true_norm, axis=0)
    true_raw = np.concatenate(all_true_raw, axis=0)
    pred_raw, ape = compute_physical_errors(pred_norm, true_raw)

    mae_norm = float(np.mean(np.abs(pred_norm - true_norm)))
    mape_physical = float(np.mean(ape))
    param_mae_norm = {}
    param_mape_physical = {}
    for idx, (_, en_name, _, _) in enumerate(PARAM_SPECS):
        param_mae_norm[en_name] = float(np.mean(np.abs(pred_norm[:, idx] - true_norm[:, idx])))
        param_mape_physical[en_name] = float(np.mean(ape[:, idx]))

    return EvalResult(
        loss=float(total_loss / max(len(loader.dataset), 1)),
        mae_norm=mae_norm,
        mape_physical=mape_physical,
        param_mae_norm=param_mae_norm,
        param_mape_physical=param_mape_physical,
    )


@torch.no_grad()
def export_predictions(model, loader, device, output_csv: Path, run_name: str, split_name: str) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in tqdm(loader, desc=f"predict-{split_name}", leave=False):
        moved = move_tensor_batch(batch, device)
        pred_norm = model(moved).detach().cpu().numpy()
        true_raw = moved["params_raw"].detach().cpu().numpy()
        pred_raw, ape = compute_physical_errors(pred_norm, true_raw)

        for row_idx in range(pred_norm.shape[0]):
            item = {
                "run_name": run_name,
                "split": split_name,
                "sample_id": batch["sample_id"][row_idx],
                "before_id": batch["before_id"][row_idx],
                "pattern_id": batch["pattern_id"][row_idx],
                "batch_id": batch["batch_id"][row_idx],
            }
            for param_idx, (_, en_name, _, _) in enumerate(PARAM_SPECS):
                true_value = float(true_raw[row_idx, param_idx])
                pred_value = float(pred_raw[row_idx, param_idx])
                abs_err = abs(pred_value - true_value)
                item[f"true_{en_name}"] = true_value
                item[f"pred_{en_name}"] = pred_value
                item[f"abs_err_{en_name}"] = abs_err
                item[f"ape_{en_name}"] = float(ape[row_idx, param_idx])
            item["mean_ape"] = float(np.mean(ape[row_idx]))
            rows.append(item)

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def build_loader(
    csv_path: str,
    before_dir: str,
    after_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    is_train: bool,
    resize_mode: str = "stretch",
    augmentation_mode: str = "weak",
):
    dataset = InverseExperimentDataset(
        csv_path=csv_path,
        before_dir=before_dir,
        after_dir=after_dir,
        img_size=img_size,
        is_train=is_train,
        resize_mode=resize_mode,
        augmentation_mode=augmentation_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=is_train,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified inverse-model experiment runner.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", default="")
    parser.add_argument("--before-dir", required=True)
    parser.add_argument("--after-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone", choices=BACKBONE_CHOICES, default="resnet18")
    parser.add_argument(
        "--fusion",
        choices=[
            "concat",
            "diff",
            "concat_diff",
            "gated",
            "concat_absdiff",
            "cross_attention",
            "masked_concat_diff",
            "input_signeddiff",
            "input_absdiff",
            "input4_late",
        ],
        default="concat_diff",
    )
    parser.add_argument("--input-mode", choices=["dual", "after_only", "before_only"], default="dual")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--head-type", choices=["simple", "bn"], default="simple")
    parser.add_argument(
        "--resize-mode",
        choices=["stretch", "letterbox"],
        default="stretch",
    )
    parser.add_argument(
        "--augmentation-mode",
        choices=["none", "weak", "strong"],
        default="weak",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["val_loss", "val_mean_mape"],
        default="val_loss",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.input_mode == "dual":
        model = DualStreamInverseNet(
            backbone_name=args.backbone,
            pretrained=not args.no_pretrained,
            fusion=args.fusion,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            head_type=args.head_type,
        )
    else:
        image_key = "effect" if args.input_mode == "after_only" else "before"
        model = SingleImageInverseNet(
            backbone_name=args.backbone,
            pretrained=not args.no_pretrained,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            image_key=image_key,
            head_type=args.head_type,
        )
    return model.to(device)


def selection_score(result: EvalResult, selection_metric: str) -> float:
    if selection_metric == "val_loss":
        return float(result.loss)
    if selection_metric == "val_mean_mape":
        return float(result.mape_physical)
    raise ValueError(selection_metric)


def write_run_manifest(run_dir: Path, payload: dict) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    manifest.update(payload)
    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name or f"inverse_{Path(args.train_csv).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    checkpoints_dir = run_dir / "checkpoints"
    predictions_dir = run_dir / "predictions"
    logs_dir = run_dir / "logs"
    for path in (checkpoints_dir, predictions_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"run_dir: {run_dir}")

    train_loader = build_loader(
        csv_path=args.train_csv,
        before_dir=args.before_dir,
        after_dir=args.after_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=True,
        resize_mode=args.resize_mode,
        augmentation_mode=args.augmentation_mode,
    )
    val_loader = build_loader(
        csv_path=args.val_csv,
        before_dir=args.before_dir,
        after_dir=args.after_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
        resize_mode=args.resize_mode,
        augmentation_mode="none",
    )
    test_loader = None
    if args.test_csv:
        test_loader = build_loader(
            csv_path=args.test_csv,
            before_dir=args.before_dir,
            after_dir=args.after_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_train=False,
            resize_mode=args.resize_mode,
            augmentation_mode="none",
        )

    model = build_model(args, device)
    parameter_count = count_trainable_parameters(model)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    config_payload = vars(args).copy()
    config_payload["device"] = str(device)
    config_payload["run_name"] = run_name
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, ensure_ascii=False, indent=2)
    write_run_manifest(
        run_dir,
        {
            "run_name": run_name,
            "config": config_payload,
            "parameter_count": parameter_count,
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "test_csv": args.test_csv,
        },
    )

    history_rows = []
    best_state = None
    best_result = None
    best_val_loss = float("inf")
    best_selection_score = float("inf")
    log_file = logs_dir / f"{run_name}_history.csv"

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result.loss,
            "val_mae": val_result.mae_norm,
            "val_mape": val_result.mape_physical,
            "lr": current_lr,
        }
        for key, value in val_result.param_mae_norm.items():
            row[f"val_{key}_loss"] = value
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(log_file, index=False, encoding="utf-8-sig")

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_result.loss:.6f} | "
            f"val_mae={val_result.mae_norm:.6f} | val_mape={val_result.mape_physical:.4f}"
        )

        current_selection_score = selection_score(
            val_result,
            args.selection_metric,
        )
        if current_selection_score < best_selection_score:
            best_selection_score = current_selection_score
            best_val_loss = val_result.loss
            best_result = deepcopy(asdict(val_result))
            best_state = deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": best_state,
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_selection_score": best_selection_score,
                    "run_name": run_name,
                    "config": config_payload,
                },
                checkpoints_dir / "best_model.pth",
            )
            shutil.copy2(checkpoints_dir / "best_model.pth", run_dir / "best_checkpoint.pth")

    if best_state is None or best_result is None:
        raise RuntimeError("Training finished without a valid checkpoint")

    model.load_state_dict(best_state)

    val_start = time.perf_counter()
    val_predictions = export_predictions(
        model=model,
        loader=val_loader,
        device=device,
        output_csv=predictions_dir / "val_predictions.csv",
        run_name=run_name,
        split_name="val",
    )
    val_inference_seconds = time.perf_counter() - val_start
    test_predictions = None
    test_inference_seconds = None
    if test_loader is not None:
        test_start = time.perf_counter()
        test_predictions = export_predictions(
            model=model,
            loader=test_loader,
            device=device,
            output_csv=predictions_dir / "test_predictions.csv",
            run_name=run_name,
            split_name="test",
        )
        test_inference_seconds = time.perf_counter() - test_start

    summary = {
        "run_name": run_name,
        "parameter_count": parameter_count,
        "best_val_loss": best_val_loss,
        "best_selection_score": best_selection_score,
        "selection_metric": args.selection_metric,
        "best_eval": best_result,
        "num_train_samples": int(len(train_loader.dataset)),
        "num_val_samples": int(len(val_loader.dataset)),
        "num_test_samples": int(len(test_loader.dataset)) if test_loader is not None else 0,
        "evaluation_scope": (
            "train_val_plus_test" if test_loader is not None else "inner_validation_only"
        ),
        "val_prediction_file": str(predictions_dir / "val_predictions.csv"),
        "test_prediction_file": str(predictions_dir / "test_predictions.csv") if test_predictions is not None else "",
        "mean_val_prediction_mape": float(val_predictions["mean_ape"].mean()) if len(val_predictions) else float("nan"),
        "mean_test_prediction_mape": float(test_predictions["mean_ape"].mean()) if test_predictions is not None and len(test_predictions) else float("nan"),
        "val_inference_seconds": val_inference_seconds,
        "test_inference_seconds": test_inference_seconds,
        "val_inference_ms_per_sample": float(val_inference_seconds / max(len(val_predictions), 1) * 1000.0),
        "test_inference_ms_per_sample": float(test_inference_seconds / max(len(test_predictions), 1) * 1000.0) if test_predictions is not None else None,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    try:
        shutil.copy2(log_file, run_dir / "train_log.csv")
        figure_dir = logs_dir / f"{run_name}_figures"
        plot_inverse_training_log(log_file, figure_dir, model_name=run_name)
    except Exception as exc:
        print(f"[WARN] failed to plot training curves: {exc}")

    print(f"best checkpoint: {checkpoints_dir / 'best_model.pth'}")
    print(f"val predictions: {predictions_dir / 'val_predictions.csv'}")
    if test_predictions is not None:
        print(f"test predictions: {predictions_dir / 'test_predictions.csv'}")
    print(f"summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
