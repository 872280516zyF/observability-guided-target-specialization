#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_inverse_experiment import (  # noqa: E402
    BNRegressionHead,
    BACKBONE_CHOICES,
    InverseExperimentDataset,
    PARAM_SPECS,
    compute_physical_errors,
    get_backbone,
)
from utils.seed import set_seed  # noqa: E402


class Shared4BNNet(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool, hidden_dim: int, dropout: float):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.head = BNRegressionHead(feat_dim, hidden_dim, dropout, num_params=4)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        feat = self.backbone(batch["effect"]).squeeze(-1).squeeze(-1)
        return self.head(feat)


class ThreePlusOneBNNet(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool, hidden_dim: int, dropout: float, dpi_attention: bool = False):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.main_head = BNRegressionHead(feat_dim, hidden_dim, dropout, num_params=3)
        self.dpi_attention = dpi_attention
        if dpi_attention:
            self.attention = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim),
                nn.Sigmoid(),
            )
        self.dpi_head = BNRegressionHead(feat_dim, hidden_dim, dropout, num_params=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        feat = self.backbone(batch["effect"]).squeeze(-1).squeeze(-1)
        main = self.main_head(feat)
        dpi_feat = feat * self.attention(feat) if self.dpi_attention else feat
        dpi = self.dpi_head(dpi_feat)
        return torch.cat([main, dpi], dim=1)


DPI_INDEX = 3
PARAM_NAMES = [name for _, name, _, _ in PARAM_SPECS]


class DpiResidualAttentionBNNet(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        residual_scale: float = 0.25,
    ):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.base_head = BNRegressionHead(feat_dim, hidden_dim, dropout, num_params=4)
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        self.residual_scale = float(residual_scale)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        feat = self.backbone(batch["effect"]).squeeze(-1).squeeze(-1)
        base = self.base_head(feat)
        dpi_feat = feat * self.attention(feat)
        residual = self.residual_head(dpi_feat) * self.residual_scale
        pred = base.clone()
        pred[:, DPI_INDEX : DPI_INDEX + 1] = (pred[:, DPI_INDEX : DPI_INDEX + 1] + residual).clamp(0.0, 1.0)
        return pred


class DpiMoEBNNet(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        n_experts: int = 4,
    ):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.main_head = BNRegressionHead(feat_dim, hidden_dim, dropout, num_params=3)
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, n_experts),
        )
        self.experts = nn.ModuleList([BNRegressionHead(feat_dim, hidden_dim, dropout, num_params=1) for _ in range(n_experts)])

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        feat = self.backbone(batch["effect"]).squeeze(-1).squeeze(-1)
        main = self.main_head(feat)
        weights = torch.softmax(self.gate(feat), dim=1)
        expert_pred = torch.cat([expert(feat) for expert in self.experts], dim=1)
        dpi = (expert_pred * weights).sum(dim=1, keepdim=True)
        return torch.cat([main, dpi], dim=1)


class DpiOrdinalAuxBNNet(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        n_bins: int = 8,
        blend_alpha: float = 0.0,
    ):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.main_head = BNRegressionHead(feat_dim, hidden_dim, dropout, num_params=3)
        self.dpi_head = BNRegressionHead(feat_dim, hidden_dim, dropout, num_params=1)
        self.bin_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_bins),
        )
        centers = (torch.arange(n_bins, dtype=torch.float32) + 0.5) / float(n_bins)
        self.register_buffer("bin_centers", centers.view(1, n_bins))
        self.blend_alpha = float(blend_alpha)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feat = self.backbone(batch["effect"]).squeeze(-1).squeeze(-1)
        main = self.main_head(feat)
        dpi_reg = self.dpi_head(feat)
        logits = self.bin_head(feat)
        if self.blend_alpha > 0:
            dpi_cls = (torch.softmax(logits, dim=1) * self.bin_centers).sum(dim=1, keepdim=True)
            dpi = ((1.0 - self.blend_alpha) * dpi_reg + self.blend_alpha * dpi_cls).clamp(0.0, 1.0)
        else:
            dpi = dpi_reg
        return torch.cat([main, dpi], dim=1), {"dpi_bin_logits": logits}


class TextureDescriptor(nn.Module):
    """Small differentiable descriptor bank for DPI-sensitive local texture."""

    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("lap", lap)
        self.output_dim = 14

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = (image * self.std + self.mean).clamp(0.0, 1.0)
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).clamp(0.0, 1.0)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = torch.abs(F.conv2d(gray, self.lap, padding=1))
        blur3 = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        blur9 = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_var = F.avg_pool2d(gray.square(), kernel_size=9, stride=1, padding=4) - blur9.square()
        local_var = local_var.clamp_min(0.0)
        high = torch.abs(gray - blur9)
        dark = (gray < 0.12).float()
        foreground = (gray > 0.16).float()

        maps = [gray, grad, lap, local_var, high, dark, foreground]
        stats = []
        for item in maps:
            flat = item.flatten(1)
            stats.append(flat.mean(dim=1, keepdim=True))
            stats.append(flat.std(dim=1, keepdim=True))
        return torch.cat(stats, dim=1)


class TextureAwareDPIBNNet(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        texture_dim: int = 64,
    ):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.main_head = BNRegressionHead(feat_dim, hidden_dim, dropout, num_params=3)
        self.texture = TextureDescriptor()
        self.texture_mlp = nn.Sequential(
            nn.Linear(self.texture.output_dim, texture_dim),
            nn.BatchNorm1d(texture_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.25),
            nn.Linear(texture_dim, texture_dim),
            nn.ReLU(inplace=True),
        )
        self.dpi_attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )
        self.dpi_head = BNRegressionHead(feat_dim + texture_dim, hidden_dim, dropout, num_params=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        image = batch["effect"]
        feat = self.backbone(image).squeeze(-1).squeeze(-1)
        main = self.main_head(feat)
        tex = self.texture_mlp(self.texture(image))
        dpi_feat = torch.cat([feat * self.dpi_attention(feat), tex], dim=1)
        dpi = self.dpi_head(dpi_feat)
        return torch.cat([main, dpi], dim=1)


class TrainableTextureEncoder(nn.Module):
    """Small CNN encoder over explicit texture maps for DPI-sensitive details."""

    def __init__(self, output_dim: int = 64):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = output_dim

    def _texture_maps(self, image: torch.Tensor) -> torch.Tensor:
        x = (image * self.std + self.mean).clamp(0.0, 1.0)
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).clamp(0.0, 1.0)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        blur = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        high = torch.abs(gray - blur)
        return torch.cat([gray, grad, high], dim=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(self._texture_maps(image))


class TextureCNNOrdinalDPIBNNet(nn.Module):
    """Texture-aware DPI branch with trainable texture CNN and optional ordinal auxiliary loss."""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        texture_dim: int = 64,
        n_bins: int = 8,
        blend_alpha: float = 0.0,
    ):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.main_head = BNRegressionHead(feat_dim, hidden_dim, dropout, num_params=3)
        self.texture = TextureDescriptor()
        self.texture_mlp = nn.Sequential(
            nn.Linear(self.texture.output_dim, texture_dim),
            nn.BatchNorm1d(texture_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.25),
            nn.Linear(texture_dim, texture_dim),
            nn.ReLU(inplace=True),
        )
        self.texture_cnn = TrainableTextureEncoder(texture_dim)
        dpi_dim = feat_dim + texture_dim * 2
        self.dpi_attention = nn.Sequential(
            nn.Linear(dpi_dim, dpi_dim),
            nn.ReLU(inplace=True),
            nn.Linear(dpi_dim, dpi_dim),
            nn.Sigmoid(),
        )
        self.dpi_head = BNRegressionHead(dpi_dim, hidden_dim, dropout, num_params=1)
        self.bin_head = nn.Sequential(
            nn.Linear(dpi_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_bins),
        )
        centers = (torch.arange(n_bins, dtype=torch.float32) + 0.5) / float(n_bins)
        self.register_buffer("bin_centers", centers.view(1, n_bins))
        self.blend_alpha = float(blend_alpha)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = batch["effect"]
        feat = self.backbone(image).squeeze(-1).squeeze(-1)
        main = self.main_head(feat)
        tex_hand = self.texture_mlp(self.texture(image))
        tex_cnn = self.texture_cnn(image)
        dpi_feat = torch.cat([feat, tex_hand, tex_cnn], dim=1)
        dpi_feat = dpi_feat * self.dpi_attention(dpi_feat)
        dpi_reg = self.dpi_head(dpi_feat)
        logits = self.bin_head(dpi_feat)
        if self.blend_alpha > 0:
            dpi_cls = (torch.softmax(logits, dim=1) * self.bin_centers).sum(dim=1, keepdim=True)
            dpi = ((1.0 - self.blend_alpha) * dpi_reg + self.blend_alpha * dpi_cls).clamp(0.0, 1.0)
        else:
            dpi = dpi_reg
        return torch.cat([main, dpi], dim=1), {"dpi_bin_logits": logits}


class IntegratedTextureExpertBNNet(nn.Module):
    """Integrated parameter-wise experts inspired by the post-hoc ensemble.

    Frequency, pulse width and speed use a trainable texture-CNN expert, while
    DPI uses the previously best texture-descriptor DPI expert. When
    shared_backbone is False, the two experts keep separate visual backbones to
    reduce parameter competition; when True, they share one backbone.
    """

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        texture_dim: int = 64,
        shared_backbone: bool = False,
    ):
        super().__init__()
        self.shared_backbone = bool(shared_backbone)
        if self.shared_backbone:
            self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
            self.main_backbone = None
            self.dpi_backbone = None
        else:
            self.main_backbone, feat_dim = get_backbone(backbone_name, pretrained)
            self.dpi_backbone, dpi_feat_dim = get_backbone(backbone_name, pretrained)
            if dpi_feat_dim != feat_dim:
                raise ValueError(f"Backbone feature mismatch: main={feat_dim}, dpi={dpi_feat_dim}")

        self.main_texture_cnn = TrainableTextureEncoder(texture_dim)
        self.main_head = BNRegressionHead(feat_dim + texture_dim, hidden_dim, dropout, num_params=3)

        self.dpi_texture = TextureDescriptor()
        self.dpi_texture_mlp = nn.Sequential(
            nn.Linear(self.dpi_texture.output_dim, texture_dim),
            nn.BatchNorm1d(texture_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.25),
            nn.Linear(texture_dim, texture_dim),
            nn.ReLU(inplace=True),
        )
        self.dpi_attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )
        self.dpi_head = BNRegressionHead(feat_dim + texture_dim, hidden_dim, dropout, num_params=1)

    def _main_features(self, image: torch.Tensor) -> torch.Tensor:
        if self.shared_backbone:
            return self.backbone(image).squeeze(-1).squeeze(-1)
        return self.main_backbone(image).squeeze(-1).squeeze(-1)

    def _dpi_features(self, image: torch.Tensor) -> torch.Tensor:
        if self.shared_backbone:
            return self.backbone(image).squeeze(-1).squeeze(-1)
        return self.dpi_backbone(image).squeeze(-1).squeeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        image = batch["effect"]
        main_feat = self._main_features(image)
        main_tex = self.main_texture_cnn(image)
        main = self.main_head(torch.cat([main_feat, main_tex], dim=1))

        dpi_feat = self._dpi_features(image)
        dpi_tex = self.dpi_texture_mlp(self.dpi_texture(image))
        dpi = self.dpi_head(torch.cat([dpi_feat * self.dpi_attention(dpi_feat), dpi_tex], dim=1))
        return torch.cat([main, dpi], dim=1)


class GlobalStatisticDescriptor(nn.Module):
    """Fourteen non-texture image statistics for the equal-capacity control.

    The trainable projection has exactly the same dimensions as the texture
    descriptor projection. The inputs contain global colour/intensity
    summaries only: RGB mean/std/min/max plus grayscale mean/std.
    """

    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.output_dim = 14

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = (image * self.std + self.mean).clamp(0.0, 1.0)
        flat = x.flatten(2)
        rgb_mean = flat.mean(dim=2)
        rgb_std = flat.std(dim=2)
        rgb_min = flat.amin(dim=2)
        rgb_max = flat.amax(dim=2)
        gray = (
            0.299 * x[:, 0:1]
            + 0.587 * x[:, 1:2]
            + 0.114 * x[:, 2:3]
        ).flatten(1)
        gray_mean = gray.mean(dim=1, keepdim=True)
        gray_std = gray.std(dim=1, keepdim=True)
        return torch.cat(
            [rgb_mean, rgb_std, rgb_min, rgb_max, gray_mean, gray_std],
            dim=1,
        )


class TrainableRGBControlEncoder(TrainableTextureEncoder):
    """Parameter-matched control encoder operating on RGB rather than texture maps."""

    def _texture_maps(self, image: torch.Tensor) -> torch.Tensor:
        return (image * self.std + self.mean).clamp(0.0, 1.0)


class TargetedIntegratedExpertBNNet(nn.Module):
    """Two-backbone expert model with a configurable specialist target.

    This class supports the data-selected P^obs model and the three non-P^obs
    placement controls without changing trainable parameter count. With
    texture_guided=False, the texture branches are replaced by parameter-
    matched RGB/global-statistic controls.
    """

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        texture_dim: int = 64,
        expert_target: str = "dpi",
        texture_guided: bool = True,
    ):
        super().__init__()
        if expert_target not in PARAM_NAMES:
            raise ValueError(
                f"Unsupported expert target {expert_target!r}; choose from {PARAM_NAMES}"
            )
        self.expert_target = expert_target
        self.specialist_index = PARAM_NAMES.index(expert_target)
        self.remaining_indices = [
            index for index in range(len(PARAM_NAMES))
            if index != self.specialist_index
        ]
        self.texture_guided = bool(texture_guided)

        self.general_backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.specialist_backbone, specialist_feat_dim = get_backbone(
            backbone_name, pretrained
        )
        if specialist_feat_dim != feat_dim:
            raise ValueError(
                "Backbone feature mismatch: "
                f"general={feat_dim}, specialist={specialist_feat_dim}"
            )

        encoder_cls = (
            TrainableTextureEncoder
            if self.texture_guided
            else TrainableRGBControlEncoder
        )
        descriptor_cls = (
            TextureDescriptor
            if self.texture_guided
            else GlobalStatisticDescriptor
        )
        self.general_aux_encoder = encoder_cls(texture_dim)
        self.general_head = BNRegressionHead(
            feat_dim + texture_dim,
            hidden_dim,
            dropout,
            num_params=3,
        )
        self.specialist_descriptor = descriptor_cls()
        self.specialist_descriptor_mlp = nn.Sequential(
            nn.Linear(self.specialist_descriptor.output_dim, texture_dim),
            nn.BatchNorm1d(texture_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.25),
            nn.Linear(texture_dim, texture_dim),
            nn.ReLU(inplace=True),
        )
        self.specialist_attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )
        self.specialist_head = BNRegressionHead(
            feat_dim + texture_dim,
            hidden_dim,
            dropout,
            num_params=1,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        image = batch["effect"]
        general_feat = self.general_backbone(image).squeeze(-1).squeeze(-1)
        general_aux = self.general_aux_encoder(image)
        general = self.general_head(
            torch.cat([general_feat, general_aux], dim=1)
        )

        specialist_feat = self.specialist_backbone(image).squeeze(-1).squeeze(-1)
        specialist_aux = self.specialist_descriptor_mlp(
            self.specialist_descriptor(image)
        )
        specialist = self.specialist_head(
            torch.cat(
                [
                    specialist_feat
                    * self.specialist_attention(specialist_feat),
                    specialist_aux,
                ],
                dim=1,
            )
        )

        columns: list[torch.Tensor] = []
        general_column = 0
        for param_index in range(len(PARAM_NAMES)):
            if param_index == self.specialist_index:
                columns.append(specialist)
            else:
                columns.append(general[:, general_column : general_column + 1])
                general_column += 1
        return torch.cat(columns, dim=1)


class TargetedIntegratedOrdinalExpertBNNet(TargetedIntegratedExpertBNNet):
    """Targeted expert with a distance-aware discrete-level auxiliary head.

    The specialist parameter keeps its regression head. A second head predicts
    ordered physical levels in normalized space, and its expected level is
    blended with the regression output. This is suitable for discrete grids
    such as DPI 25--175 in steps of 5 (31 levels) or pulse width 25--100 in
    steps of 5 (16 levels).
    """

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        texture_dim: int = 64,
        expert_target: str = "dpi",
        texture_guided: bool = True,
        ordinal_levels: int = 31,
        ordinal_blend_alpha: float = 0.35,
    ):
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            dropout=dropout,
            texture_dim=texture_dim,
            expert_target=expert_target,
            texture_guided=texture_guided,
        )
        if ordinal_levels < 2:
            raise ValueError("ordinal_levels must be at least 2")
        self.ordinal_levels = int(ordinal_levels)
        self.ordinal_blend_alpha = float(ordinal_blend_alpha)
        specialist_input_dim = int(self.specialist_head.head[0].in_features)
        self.ordinal_head = nn.Linear(specialist_input_dim, self.ordinal_levels)
        centers = torch.linspace(0.0, 1.0, steps=self.ordinal_levels)
        self.register_buffer("ordinal_centers", centers.view(1, -1))

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, object]]:
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
        specialist_regression = self.specialist_head(specialist_context)
        ordinal_logits = self.ordinal_head(specialist_context)
        ordinal_expectation = (
            torch.softmax(ordinal_logits, dim=1) * self.ordinal_centers
        ).sum(dim=1, keepdim=True)
        specialist = (
            (1.0 - self.ordinal_blend_alpha) * specialist_regression
            + self.ordinal_blend_alpha * ordinal_expectation
        ).clamp(0.0, 1.0)

        columns: list[torch.Tensor] = []
        general_column = 0
        for param_index in range(len(PARAM_NAMES)):
            if param_index == self.specialist_index:
                columns.append(specialist)
            else:
                columns.append(general[:, general_column : general_column + 1])
                general_column += 1
        return torch.cat(columns, dim=1), {
            "ordinal_logits": ordinal_logits,
            "ordinal_target_index": self.specialist_index,
        }


class PStarConditionedIntegratedTextureExpertBNNet(nn.Module):
    """Integrated experts where non-dominant heads are conditioned on P* evidence.

    This tests whether the visually dominant parameter can organize prediction
    of the remaining parameters. The formal variants use only predicted P* or
    latent P* expert features. The oracle mode uses ground-truth normalized DPI
    and is therefore diagnostic only.
    """

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        texture_dim: int = 64,
        condition_mode: str = "latent",
        detach_condition: bool = True,
    ):
        super().__init__()
        if condition_mode not in {"scalar", "latent", "both", "oracle_scalar"}:
            raise ValueError(f"Unsupported P* condition mode: {condition_mode}")
        self.condition_mode = condition_mode
        self.detach_condition = bool(detach_condition)

        self.main_backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.dpi_backbone, dpi_feat_dim = get_backbone(backbone_name, pretrained)
        if dpi_feat_dim != feat_dim:
            raise ValueError(f"Backbone feature mismatch: main={feat_dim}, dpi={dpi_feat_dim}")

        self.main_texture_cnn = TrainableTextureEncoder(texture_dim)

        self.dpi_texture = TextureDescriptor()
        self.dpi_texture_mlp = nn.Sequential(
            nn.Linear(self.dpi_texture.output_dim, texture_dim),
            nn.BatchNorm1d(texture_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.25),
            nn.Linear(texture_dim, texture_dim),
            nn.ReLU(inplace=True),
        )
        self.dpi_attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )
        dpi_context_dim = feat_dim + texture_dim
        self.dpi_head = BNRegressionHead(dpi_context_dim, hidden_dim, dropout, num_params=1)

        self.pstar_context_mlp = nn.Sequential(
            nn.Linear(dpi_context_dim, texture_dim),
            nn.BatchNorm1d(texture_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.25),
            nn.Linear(texture_dim, texture_dim),
            nn.ReLU(inplace=True),
        )

        if condition_mode in {"scalar", "oracle_scalar"}:
            condition_dim = 1
        elif condition_mode == "latent":
            condition_dim = texture_dim
        else:
            condition_dim = texture_dim + 1
        self.main_head = BNRegressionHead(feat_dim + texture_dim + condition_dim, hidden_dim, dropout, num_params=3)

    def _condition_tensor(self, batch: dict[str, torch.Tensor], dpi_context: torch.Tensor, dpi: torch.Tensor) -> torch.Tensor:
        if self.condition_mode == "oracle_scalar":
            return batch["params"][:, DPI_INDEX : DPI_INDEX + 1]

        context_source = dpi_context.detach() if self.detach_condition else dpi_context
        dpi_source = dpi.detach() if self.detach_condition else dpi

        if self.condition_mode == "scalar":
            return dpi_source
        latent = self.pstar_context_mlp(context_source)
        if self.condition_mode == "latent":
            return latent
        return torch.cat([latent, dpi_source], dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        image = batch["effect"]

        dpi_feat = self.dpi_backbone(image).squeeze(-1).squeeze(-1)
        dpi_feat_att = dpi_feat * self.dpi_attention(dpi_feat)
        dpi_tex = self.dpi_texture_mlp(self.dpi_texture(image))
        dpi_context = torch.cat([dpi_feat_att, dpi_tex], dim=1)
        dpi = self.dpi_head(dpi_context)

        main_feat = self.main_backbone(image).squeeze(-1).squeeze(-1)
        main_tex = self.main_texture_cnn(image)
        condition = self._condition_tensor(batch, dpi_context, dpi)
        main = self.main_head(torch.cat([main_feat, main_tex, condition], dim=1))

        return torch.cat([main, dpi], dim=1)


class MultiScaleTextureCNNOrdinalDPIBNNet(nn.Module):
    """Full-image, local-crop and texture-CNN DPI branch with ordinal auxiliary supervision."""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        texture_dim: int = 64,
        crop_scale: float = 0.58,
        n_bins: int = 8,
        blend_alpha: float = 0.0,
    ):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.main_head = BNRegressionHead(feat_dim, hidden_dim, dropout, num_params=3)
        self.crop_scale = float(crop_scale)
        self.texture = TextureDescriptor()
        self.texture_mlp = nn.Sequential(
            nn.Linear(self.texture.output_dim, texture_dim),
            nn.BatchNorm1d(texture_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.25),
            nn.Linear(texture_dim, texture_dim),
            nn.ReLU(inplace=True),
        )
        self.texture_cnn = TrainableTextureEncoder(texture_dim)
        dpi_dim = feat_dim * 2 + texture_dim * 2
        self.dpi_attention = nn.Sequential(
            nn.Linear(dpi_dim, dpi_dim),
            nn.ReLU(inplace=True),
            nn.Linear(dpi_dim, dpi_dim),
            nn.Sigmoid(),
        )
        self.dpi_head = BNRegressionHead(dpi_dim, hidden_dim, dropout, num_params=1)
        self.bin_head = nn.Sequential(
            nn.Linear(dpi_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_bins),
        )
        centers = (torch.arange(n_bins, dtype=torch.float32) + 0.5) / float(n_bins)
        self.register_buffer("bin_centers", centers.view(1, n_bins))
        self.blend_alpha = float(blend_alpha)

    def _center_crop(self, image: torch.Tensor) -> torch.Tensor:
        _, _, h, w = image.shape
        ch = max(8, int(round(h * self.crop_scale)))
        cw = max(8, int(round(w * self.crop_scale)))
        top = (h - ch) // 2
        left = (w - cw) // 2
        crop = image[:, :, top : top + ch, left : left + cw]
        return F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)

    def _feature(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image).squeeze(-1).squeeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = batch["effect"]
        full = self._feature(image)
        local = self._feature(self._center_crop(image))
        main = self.main_head(full)
        tex_hand = self.texture_mlp(self.texture(image))
        tex_cnn = self.texture_cnn(image)
        dpi_feat = torch.cat([full, local, tex_hand, tex_cnn], dim=1)
        dpi_feat = dpi_feat * self.dpi_attention(dpi_feat)
        dpi_reg = self.dpi_head(dpi_feat)
        logits = self.bin_head(dpi_feat)
        if self.blend_alpha > 0:
            dpi_cls = (torch.softmax(logits, dim=1) * self.bin_centers).sum(dim=1, keepdim=True)
            dpi = ((1.0 - self.blend_alpha) * dpi_reg + self.blend_alpha * dpi_cls).clamp(0.0, 1.0)
        else:
            dpi = dpi_reg
        return torch.cat([main, dpi], dim=1), {"dpi_bin_logits": logits}


class LocalCropDPIBNNet(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        crop_scale: float = 0.58,
    ):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.main_head = BNRegressionHead(feat_dim, hidden_dim, dropout, num_params=3)
        self.crop_scale = float(crop_scale)
        self.crop_gate = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.25),
            nn.Linear(hidden_dim, 1),
        )
        self.dpi_attention = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim * 2),
            nn.Sigmoid(),
        )
        self.dpi_head = BNRegressionHead(feat_dim * 2, hidden_dim, dropout, num_params=1)

    def _crops(self, image: torch.Tensor) -> torch.Tensor:
        _, _, h, w = image.shape
        ch = max(8, int(round(h * self.crop_scale)))
        cw = max(8, int(round(w * self.crop_scale)))
        windows = [
            (0, 0),
            (0, w - cw),
            (h - ch, 0),
            (h - ch, w - cw),
            ((h - ch) // 2, (w - cw) // 2),
        ]
        crops = []
        for top, left in windows:
            crop = image[:, :, top : top + ch, left : left + cw]
            crops.append(F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False))
        return torch.stack(crops, dim=1)

    def _feature(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image).squeeze(-1).squeeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        image = batch["effect"]
        full = self._feature(image)
        main = self.main_head(full)
        crops = self._crops(image)
        b, n, c, h, w = crops.shape
        crop_feat = self._feature(crops.reshape(b * n, c, h, w)).reshape(b, n, -1)
        weights = torch.softmax(self.crop_gate(crop_feat).squeeze(-1), dim=1)
        local = (crop_feat * weights.unsqueeze(-1)).sum(dim=1)
        dpi_feat = torch.cat([full, local], dim=1)
        dpi_feat = dpi_feat * self.dpi_attention(dpi_feat)
        dpi = self.dpi_head(dpi_feat)
        return torch.cat([main, dpi], dim=1)


class TextureLocalDPIBNNet(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        texture_dim: int = 64,
        crop_scale: float = 0.58,
    ):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.main_head = BNRegressionHead(feat_dim, hidden_dim, dropout, num_params=3)
        self.crop_scale = float(crop_scale)
        self.texture = TextureDescriptor()
        self.texture_mlp = nn.Sequential(
            nn.Linear(self.texture.output_dim, texture_dim),
            nn.BatchNorm1d(texture_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.25),
            nn.Linear(texture_dim, texture_dim),
            nn.ReLU(inplace=True),
        )
        self.crop_gate = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.25),
            nn.Linear(hidden_dim, 1),
        )
        dpi_dim = feat_dim * 2 + texture_dim
        self.dpi_attention = nn.Sequential(
            nn.Linear(dpi_dim, dpi_dim),
            nn.ReLU(inplace=True),
            nn.Linear(dpi_dim, dpi_dim),
            nn.Sigmoid(),
        )
        self.dpi_head = BNRegressionHead(dpi_dim, hidden_dim, dropout, num_params=1)

    def _crops(self, image: torch.Tensor) -> torch.Tensor:
        _, _, h, w = image.shape
        ch = max(8, int(round(h * self.crop_scale)))
        cw = max(8, int(round(w * self.crop_scale)))
        windows = [
            (0, 0),
            (0, w - cw),
            (h - ch, 0),
            (h - ch, w - cw),
            ((h - ch) // 2, (w - cw) // 2),
        ]
        crops = []
        for top, left in windows:
            crop = image[:, :, top : top + ch, left : left + cw]
            crops.append(F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False))
        return torch.stack(crops, dim=1)

    def _feature(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image).squeeze(-1).squeeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        image = batch["effect"]
        full = self._feature(image)
        main = self.main_head(full)
        crops = self._crops(image)
        b, n, c, h, w = crops.shape
        crop_feat = self._feature(crops.reshape(b * n, c, h, w)).reshape(b, n, -1)
        weights = torch.softmax(self.crop_gate(crop_feat).squeeze(-1), dim=1)
        local = (crop_feat * weights.unsqueeze(-1)).sum(dim=1)
        tex = self.texture_mlp(self.texture(image))
        dpi_feat = torch.cat([full, local, tex], dim=1)
        dpi_feat = dpi_feat * self.dpi_attention(dpi_feat)
        dpi = self.dpi_head(dpi_feat)
        return torch.cat([main, dpi], dim=1)


class LegacyStyleMultiBranchHead3(nn.Module):
    """Legacy-inspired separate heads for frequency, pulse width and speed."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        def branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        self.freq_head = branch()
        self.pulse_head = branch()
        self.speed_head = branch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(x)
        return torch.cat([self.freq_head(feat), self.pulse_head(feat), self.speed_head(feat)], dim=1)


class DualMainTextureDPIBNNet(nn.Module):
    """Dual-stream legacy-style heads for first three params plus texture-aware DPI."""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        texture_dim: int = 64,
        dual_dpi_attention: bool = False,
    ):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.main_head = LegacyStyleMultiBranchHead3(feat_dim * 3, hidden_dim, dropout)
        self.texture = TextureDescriptor()
        self.texture_mlp = nn.Sequential(
            nn.Linear(self.texture.output_dim, texture_dim),
            nn.BatchNorm1d(texture_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.25),
            nn.Linear(texture_dim, texture_dim),
            nn.ReLU(inplace=True),
        )
        self.dual_dpi_attention = bool(dual_dpi_attention)
        if self.dual_dpi_attention:
            self.channel_attention = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim // 4, feat_dim),
                nn.Sigmoid(),
            )
            dpi_dim = feat_dim * 3 + texture_dim
        else:
            self.dpi_attention = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim),
                nn.Sigmoid(),
            )
            dpi_dim = feat_dim + texture_dim
        self.dpi_head = BNRegressionHead(dpi_dim, hidden_dim, dropout, num_params=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        before_feat = self.backbone(batch["before"]).squeeze(-1).squeeze(-1)
        effect_feat = self.backbone(batch["effect"]).squeeze(-1).squeeze(-1)
        diff_feat = effect_feat - before_feat
        main = self.main_head(torch.cat([before_feat, effect_feat, diff_feat], dim=1))
        tex = self.texture_mlp(self.texture(batch["effect"]))
        if self.dual_dpi_attention:
            before_att = before_feat * self.channel_attention(before_feat)
            effect_att = effect_feat * self.channel_attention(effect_feat)
            dpi_feat = torch.cat([before_att, effect_att, torch.abs(effect_att - before_att), tex], dim=1)
        else:
            dpi_feat = torch.cat([effect_feat * self.dpi_attention(effect_feat), tex], dim=1)
        dpi = self.dpi_head(dpi_feat)
        return torch.cat([main, dpi], dim=1)


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.variant == "target_shared4_bn":
        model = Shared4BNNet(args.backbone, not args.no_pretrained, args.hidden_dim, args.dropout)
    elif args.variant == "target_3plus1_bn":
        model = ThreePlusOneBNNet(args.backbone, not args.no_pretrained, args.hidden_dim, args.dropout, dpi_attention=False)
    elif args.variant == "target_3plus1_dpi_attention_bn":
        model = ThreePlusOneBNNet(args.backbone, not args.no_pretrained, args.hidden_dim, args.dropout, dpi_attention=True)
    elif args.variant == "target_3plus1_dpi_residual_attention_bn":
        model = DpiResidualAttentionBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            residual_scale=args.dpi_residual_scale,
        )
    elif args.variant == "target_3plus1_dpi_moe_bn":
        model = DpiMoEBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            n_experts=args.dpi_moe_experts,
        )
    elif args.variant == "target_3plus1_dpi_ordinal_bn":
        model = DpiOrdinalAuxBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            n_bins=args.dpi_ordinal_bins,
            blend_alpha=0.0,
        )
    elif args.variant == "target_3plus1_dpi_ordinal_blend_bn":
        model = DpiOrdinalAuxBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            n_bins=args.dpi_ordinal_bins,
            blend_alpha=args.dpi_ordinal_blend_alpha,
        )
    elif args.variant == "target_texture_dpi_branch_bn":
        model = TextureAwareDPIBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
        )
    elif args.variant == "target_texture_cnn_dpi_branch_bn":
        model = TextureCNNOrdinalDPIBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            n_bins=args.dpi_ordinal_bins,
            blend_alpha=0.0,
        )
    elif args.variant == "target_texture_cnn_ordinal_dpi_branch_bn":
        model = TextureCNNOrdinalDPIBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            n_bins=args.dpi_ordinal_bins,
            blend_alpha=args.dpi_ordinal_blend_alpha,
        )
    elif args.variant == "target_integrated_texture_expert_bn":
        model = IntegratedTextureExpertBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            shared_backbone=False,
        )
    elif args.variant == "target_integrated_shared_texture_expert_bn":
        model = IntegratedTextureExpertBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            shared_backbone=True,
        )
    elif args.variant == "target_integrated_texture_expert_any_bn":
        model = TargetedIntegratedExpertBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            expert_target=args.expert_target,
            texture_guided=True,
        )
    elif args.variant == "target_integrated_texture_expert_any_ordinal_bn":
        model = TargetedIntegratedOrdinalExpertBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            expert_target=args.expert_target,
            texture_guided=True,
            ordinal_levels=args.expert_ordinal_levels,
            ordinal_blend_alpha=args.expert_ordinal_blend_alpha,
        )
    elif args.variant == "target_integrated_nonguided_control_bn":
        model = TargetedIntegratedExpertBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            expert_target=args.expert_target,
            texture_guided=False,
        )
    elif args.variant == "target_pstar_scalar_conditioned_integrated_texture_expert_bn":
        model = PStarConditionedIntegratedTextureExpertBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            condition_mode="scalar",
            detach_condition=args.pstar_condition_detach,
        )
    elif args.variant == "target_pstar_latent_conditioned_integrated_texture_expert_bn":
        model = PStarConditionedIntegratedTextureExpertBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            condition_mode="latent",
            detach_condition=args.pstar_condition_detach,
        )
    elif args.variant == "target_pstar_both_conditioned_integrated_texture_expert_bn":
        model = PStarConditionedIntegratedTextureExpertBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            condition_mode="both",
            detach_condition=args.pstar_condition_detach,
        )
    elif args.variant == "target_oracle_pstar_conditioned_integrated_texture_expert_bn":
        model = PStarConditionedIntegratedTextureExpertBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            condition_mode="oracle_scalar",
            detach_condition=True,
        )
    elif args.variant == "target_multiscale_texture_cnn_ordinal_dpi_branch_bn":
        model = MultiScaleTextureCNNOrdinalDPIBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            crop_scale=args.local_crop_scale,
            n_bins=args.dpi_ordinal_bins,
            blend_alpha=args.dpi_ordinal_blend_alpha,
        )
    elif args.variant == "target_localcrop_dpi_branch_bn":
        model = LocalCropDPIBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            crop_scale=args.local_crop_scale,
        )
    elif args.variant == "target_texture_local_dpi_branch_bn":
        model = TextureLocalDPIBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            crop_scale=args.local_crop_scale,
        )
    elif args.variant == "target_dualmain_texture_dpi_branch_bn":
        model = DualMainTextureDPIBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            dual_dpi_attention=False,
        )
    elif args.variant == "target_dualmain_dualatt_texture_dpi_branch_bn":
        model = DualMainTextureDPIBNNet(
            args.backbone,
            not args.no_pretrained,
            args.hidden_dim,
            args.dropout,
            texture_dim=args.texture_dim,
            dual_dpi_attention=True,
        )
    else:
        raise ValueError(args.variant)
    return model.to(device)


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_output_weights(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            float(args.frequency_loss_weight),
            float(args.pulse_width_loss_weight),
            float(args.speed_loss_weight),
            float(args.dpi_loss_weight),
        ],
        dtype=torch.float32,
        device=device,
    )


def unpack_model_output(output) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(output, tuple):
        pred, aux = output
        return pred, aux
    return output, {}


def weighted_regression_loss(pred: torch.Tensor, target: torch.Tensor, output_weights: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(pred, target, reduction="none")
    elif loss_type == "l1":
        loss = F.l1_loss(pred, target, reduction="none")
    elif loss_type == "mse":
        loss = F.mse_loss(pred, target, reduction="none")
    else:
        raise ValueError(loss_type)
    return (loss * output_weights.view(1, -1)).mean()


def dpi_bin_targets(target: torch.Tensor, n_bins: int) -> torch.Tensor:
    boundaries = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=target.device)[1:-1]
    return torch.bucketize(target[:, DPI_INDEX].contiguous(), boundaries).long()


def soft_ordinal_level_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_index: int,
    sigma: float,
) -> torch.Tensor:
    """Distance-aware soft-label loss for an ordered normalized level grid."""
    n_levels = int(logits.shape[1])
    target_position = target[:, int(target_index)].clamp(0.0, 1.0) * float(
        n_levels - 1
    )
    level_positions = torch.arange(
        n_levels, dtype=logits.dtype, device=logits.device
    ).view(1, -1)
    sigma_value = max(float(sigma), 1e-3)
    soft_targets = torch.exp(
        -0.5 * ((level_positions - target_position.view(-1, 1)) / sigma_value) ** 2
    )
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp_min(
        1e-12
    )
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def combined_loss(
    output,
    target: torch.Tensor,
    output_weights: torch.Tensor,
    loss_type: str,
    dpi_aux_weight: float,
    dpi_ordinal_bins: int,
    expert_ordinal_sigma: float = 0.75,
) -> torch.Tensor:
    pred, aux = unpack_model_output(output)
    loss = weighted_regression_loss(pred, target, output_weights, loss_type)
    if "dpi_bin_logits" in aux and dpi_aux_weight > 0:
        loss = loss + float(dpi_aux_weight) * F.cross_entropy(aux["dpi_bin_logits"], dpi_bin_targets(target, dpi_ordinal_bins))
    if "ordinal_logits" in aux and dpi_aux_weight > 0:
        loss = loss + float(dpi_aux_weight) * soft_ordinal_level_loss(
            aux["ordinal_logits"],
            target,
            int(aux["ordinal_target_index"]),
            expert_ordinal_sigma,
        )
    return loss


def move_tensor_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


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
    group_balanced_sampler: bool = False,
):
    dataset = InverseExperimentDataset(
        csv_path,
        before_dir,
        after_dir,
        img_size=img_size,
        is_train=is_train,
        resize_mode=resize_mode,
        augmentation_mode=augmentation_mode,
    )
    sampler = None
    shuffle = bool(is_train)
    if is_train and group_balanced_sampler:
        if "before_id" not in dataset.df.columns:
            raise ValueError(
                "--group-balanced-sampler requires before_id in the training CSV"
            )
        group_counts = dataset.df["before_id"].value_counts()
        weights = dataset.df["before_id"].map(
            lambda value: 1.0 / float(group_counts.loc[value])
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(weights.to_numpy(), dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=is_train,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader, output_weights: torch.Tensor, device: torch.device, args: argparse.Namespace) -> dict:
    model.eval()
    total_loss = 0.0
    pred_norm_rows = []
    true_norm_rows = []
    true_raw_rows = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_tensor_batch(batch, device)
        output = model(batch)
        pred, _ = unpack_model_output(output)
        loss = combined_loss(
            output,
            batch["params"],
            output_weights,
            args.loss_type,
            args.dpi_aux_weight,
            args.dpi_ordinal_bins,
            args.expert_ordinal_sigma,
        )
        total_loss += loss.item() * batch["params"].size(0)
        pred_norm_rows.append(pred.detach().cpu().numpy())
        true_norm_rows.append(batch["params"].detach().cpu().numpy())
        true_raw_rows.append(batch["params_raw"].detach().cpu().numpy())
    pred_norm = np.concatenate(pred_norm_rows, axis=0)
    true_norm = np.concatenate(true_norm_rows, axis=0)
    true_raw = np.concatenate(true_raw_rows, axis=0)
    _, ape = compute_physical_errors(pred_norm, true_raw)
    return {
        "loss": float(total_loss / max(len(loader.dataset), 1)),
        "mae_norm": float(np.mean(np.abs(pred_norm - true_norm))),
        "mape_physical": float(np.mean(ape)),
        "param_mape_physical": {name: float(np.mean(ape[:, idx])) for idx, (_, name, _, _) in enumerate(PARAM_SPECS)},
    }


def train_one_epoch(model: nn.Module, loader, output_weights: torch.Tensor, optimizer, device: torch.device, args: argparse.Namespace) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_tensor_batch(batch, device)
        optimizer.zero_grad()
        output = model(batch)
        loss = combined_loss(
            output,
            batch["params"],
            output_weights,
            args.loss_type,
            args.dpi_aux_weight,
            args.dpi_ordinal_bins,
            args.expert_ordinal_sigma,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch["params"].size(0)
    return float(total_loss / max(len(loader.dataset), 1))


@torch.no_grad()
def export_predictions(model: nn.Module, loader, device: torch.device, output_csv: Path, run_name: str, split_name: str) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in tqdm(loader, desc=f"predict-{split_name}", leave=False):
        moved = move_tensor_batch(batch, device)
        output = model(moved)
        pred_tensor, _ = unpack_model_output(output)
        pred_norm = pred_tensor.detach().cpu().numpy()
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
            for param_idx, (_, name, _, _) in enumerate(PARAM_SPECS):
                item[f"true_{name}"] = float(true_raw[row_idx, param_idx])
                item[f"pred_{name}"] = float(pred_raw[row_idx, param_idx])
                item[f"abs_err_{name}"] = abs(item[f"pred_{name}"] - item[f"true_{name}"])
                item[f"ape_{name}"] = float(ape[row_idx, param_idx])
            item["mean_ape"] = float(np.mean(ape[row_idx]))
            rows.append(item)
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def selection_score(metrics: dict, args: argparse.Namespace) -> float:
    selection_metric = args.selection_metric
    if selection_metric == "val_loss":
        return float(metrics["loss"])
    if selection_metric == "val_mean_mape":
        return float(metrics["mape_physical"])
    if selection_metric == "val_dpi_mape":
        return float(metrics["param_mape_physical"]["dpi"])
    if selection_metric == "val_dpi_mean_combo":
        return float(metrics["param_mape_physical"]["dpi"]) + float(args.selection_mean_weight) * float(metrics["mape_physical"])
    if selection_metric == "val_dpi_guardrail_combo":
        param = metrics["param_mape_physical"]
        non_dpi_max = max(float(param["frequency"]), float(param["pulse_width"]), float(param["speed"]))
        return (
            float(param["dpi"])
            + float(args.selection_mean_weight) * float(metrics["mape_physical"])
            + float(args.selection_non_dpi_max_weight) * non_dpi_max
        )
    if selection_metric == "val_expert_mape":
        return float(metrics["param_mape_physical"][args.expert_target])
    if selection_metric == "val_expert_guardrail_combo":
        param = metrics["param_mape_physical"]
        non_expert_max = max(
            float(value)
            for name, value in param.items()
            if name != args.expert_target
        )
        return (
            float(param[args.expert_target])
            + float(args.selection_mean_weight) * float(metrics["mape_physical"])
            + float(args.selection_non_dpi_max_weight) * non_expert_max
        )
    raise ValueError(selection_metric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair target-only DPI-branch ablation runner.")
    parser.add_argument(
        "--variant",
        choices=[
            "target_shared4_bn",
            "target_3plus1_bn",
            "target_3plus1_dpi_attention_bn",
            "target_3plus1_dpi_residual_attention_bn",
            "target_3plus1_dpi_moe_bn",
            "target_3plus1_dpi_ordinal_bn",
            "target_3plus1_dpi_ordinal_blend_bn",
            "target_texture_dpi_branch_bn",
            "target_texture_cnn_dpi_branch_bn",
            "target_texture_cnn_ordinal_dpi_branch_bn",
            "target_integrated_texture_expert_bn",
            "target_integrated_shared_texture_expert_bn",
            "target_integrated_texture_expert_any_bn",
            "target_integrated_texture_expert_any_ordinal_bn",
            "target_integrated_nonguided_control_bn",
            "target_pstar_scalar_conditioned_integrated_texture_expert_bn",
            "target_pstar_latent_conditioned_integrated_texture_expert_bn",
            "target_pstar_both_conditioned_integrated_texture_expert_bn",
            "target_oracle_pstar_conditioned_integrated_texture_expert_bn",
            "target_multiscale_texture_cnn_ordinal_dpi_branch_bn",
            "target_localcrop_dpi_branch_bn",
            "target_texture_local_dpi_branch_bn",
            "target_dualmain_texture_dpi_branch_bn",
            "target_dualmain_dualatt_texture_dpi_branch_bn",
        ],
        required=True,
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument(
        "--test-csv",
        default="",
        help=(
            "Optional outer-test CSV. Omit during inner-validation-only model "
            "selection so candidate configurations cannot inspect outer-test data."
        ),
    )
    parser.add_argument("--before-dir", required=True)
    parser.add_argument("--after-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=224)
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
        "--group-balanced-sampler",
        action="store_true",
        help="Sample before_id groups uniformly during training.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone", choices=BACKBONE_CHOICES, default="resnet18")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--loss-type", choices=["smooth_l1", "l1", "mse"], default="smooth_l1")
    parser.add_argument("--frequency-loss-weight", type=float, default=1.0)
    parser.add_argument("--pulse-width-loss-weight", type=float, default=1.0)
    parser.add_argument("--speed-loss-weight", type=float, default=1.0)
    parser.add_argument("--dpi-loss-weight", type=float, default=1.0)
    parser.add_argument("--dpi-aux-weight", type=float, default=0.25)
    parser.add_argument("--dpi-ordinal-bins", type=int, default=8)
    parser.add_argument("--dpi-ordinal-blend-alpha", type=float, default=0.35)
    parser.add_argument("--expert-ordinal-levels", type=int, default=31)
    parser.add_argument("--expert-ordinal-blend-alpha", type=float, default=0.35)
    parser.add_argument("--expert-ordinal-sigma", type=float, default=0.75)
    parser.add_argument("--dpi-moe-experts", type=int, default=4)
    parser.add_argument("--dpi-residual-scale", type=float, default=0.25)
    parser.add_argument("--texture-dim", type=int, default=64)
    parser.add_argument(
        "--expert-target",
        choices=PARAM_NAMES,
        default="dpi",
        help=(
            "Target assigned to the specialist branch for the grouped outer-CV "
            "P^obs and placement-control experiments."
        ),
    )
    parser.add_argument("--local-crop-scale", type=float, default=0.58)
    parser.add_argument("--pstar-condition-detach", dest="pstar_condition_detach", action="store_true", default=True)
    parser.add_argument("--no-pstar-condition-detach", dest="pstar_condition_detach", action="store_false")
    parser.add_argument(
        "--selection-metric",
        choices=[
            "val_loss",
            "val_mean_mape",
            "val_dpi_mape",
            "val_dpi_mean_combo",
            "val_dpi_guardrail_combo",
            "val_expert_mape",
            "val_expert_guardrail_combo",
        ],
        default="val_dpi_mape",
    )
    parser.add_argument("--selection-mean-weight", type=float, default=0.25)
    parser.add_argument("--selection-non-dpi-max-weight", type=float, default=0.10)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_name = args.run_name or f"{args.variant}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    checkpoints_dir = run_dir / "checkpoints"
    predictions_dir = run_dir / "predictions"
    logs_dir = run_dir / "logs"
    for path in (checkpoints_dir, predictions_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_loader(
        args.train_csv,
        args.before_dir,
        args.after_dir,
        args.img_size,
        args.batch_size,
        args.num_workers,
        True,
        resize_mode=args.resize_mode,
        augmentation_mode=args.augmentation_mode,
        group_balanced_sampler=args.group_balanced_sampler,
    )
    val_loader = build_loader(
        args.val_csv,
        args.before_dir,
        args.after_dir,
        args.img_size,
        args.batch_size,
        args.num_workers,
        False,
        resize_mode=args.resize_mode,
        augmentation_mode="weak",
        group_balanced_sampler=False,
    )
    test_loader = None
    if args.test_csv:
        test_loader = build_loader(
            args.test_csv,
            args.before_dir,
            args.after_dir,
            args.img_size,
            args.batch_size,
            args.num_workers,
            False,
            resize_mode=args.resize_mode,
            augmentation_mode="weak",
            group_balanced_sampler=False,
        )
    model = build_model(args, device)
    parameter_count = count_trainable_parameters(model)
    output_weights = build_output_weights(args, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    config_payload = vars(args).copy()
    config_payload.update({"run_name": run_name, "device": str(device), "parameter_count": parameter_count, "input_mode": "after_only"})
    write_json(run_dir / "config.json", config_payload)
    write_json(run_dir / "run_manifest.json", {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "config": config_payload,
    })

    best_score = float("inf")
    best_state = None
    best_eval = None
    history = []
    log_file = logs_dir / f"{run_name}_history.csv"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, output_weights, optimizer, device, args)
        val_metrics = evaluate(model, val_loader, output_weights, device, args)
        scheduler.step()
        current_score = selection_score(val_metrics, args)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mape": val_metrics["mape_physical"],
            "selection_metric": args.selection_metric,
            "selection_score": current_score,
            "lr": scheduler.get_last_lr()[0],
        }
        for name, value in val_metrics["param_mape_physical"].items():
            row[f"val_{name}_mape"] = value
        history.append(row)
        pd.DataFrame(history).to_csv(log_file, index=False, encoding="utf-8-sig")
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | val_mape={val_metrics['mape_physical']:.4f} | "
            f"{args.selection_metric}={current_score:.4f}"
        )
        if current_score < best_score:
            best_score = current_score
            best_state = deepcopy(model.state_dict())
            best_eval = val_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": best_state,
                    "optimizer": optimizer.state_dict(),
                    "best_selection_score": best_score,
                    "best_eval": best_eval,
                    "config": config_payload,
                },
                checkpoints_dir / "best_model.pth",
            )
            shutil.copy2(checkpoints_dir / "best_model.pth", run_dir / "best_checkpoint.pth")

    if best_state is None:
        raise RuntimeError("Training finished without a checkpoint")
    model.load_state_dict(best_state)
    shutil.copy2(log_file, run_dir / "train_log.csv")
    val_start = time.perf_counter()
    val_predictions = export_predictions(model, val_loader, device, predictions_dir / "val_predictions.csv", run_name, "val")
    val_seconds = time.perf_counter() - val_start
    test_predictions = None
    test_seconds = None
    test_metrics = None
    if test_loader is not None:
        test_start = time.perf_counter()
        test_predictions = export_predictions(
            model,
            test_loader,
            device,
            predictions_dir / "test_predictions.csv",
            run_name,
            "test",
        )
        test_seconds = time.perf_counter() - test_start
        test_metrics = evaluate(model, test_loader, output_weights, device, args)
    summary = {
        "run_name": run_name,
        "variant": args.variant,
        "parameter_count": parameter_count,
        "selection_metric": args.selection_metric,
        "loss_type": args.loss_type,
        "frequency_loss_weight": args.frequency_loss_weight,
        "pulse_width_loss_weight": args.pulse_width_loss_weight,
        "speed_loss_weight": args.speed_loss_weight,
        "dpi_loss_weight": args.dpi_loss_weight,
        "selection_mean_weight": args.selection_mean_weight,
        "selection_non_dpi_max_weight": args.selection_non_dpi_max_weight,
        "dpi_aux_weight": args.dpi_aux_weight,
        "dpi_ordinal_bins": args.dpi_ordinal_bins,
        "dpi_ordinal_blend_alpha": args.dpi_ordinal_blend_alpha,
        "expert_ordinal_levels": args.expert_ordinal_levels,
        "expert_ordinal_blend_alpha": args.expert_ordinal_blend_alpha,
        "expert_ordinal_sigma": args.expert_ordinal_sigma,
        "dpi_moe_experts": args.dpi_moe_experts,
        "dpi_residual_scale": args.dpi_residual_scale,
        "texture_dim": args.texture_dim,
        "expert_target": args.expert_target,
        "local_crop_scale": args.local_crop_scale,
        "pstar_condition_detach": args.pstar_condition_detach,
        "resize_mode": args.resize_mode,
        "augmentation_mode": args.augmentation_mode,
        "group_balanced_sampler": args.group_balanced_sampler,
        "evaluation_scope": (
            "inner_validation_and_outer_test"
            if test_loader is not None
            else "inner_validation_only"
        ),
        "best_selection_score": best_score,
        "best_val_loss": best_eval["loss"] if best_eval else np.nan,
        "best_eval": best_eval,
        "test_metrics": test_metrics,
        "num_train_samples": int(len(train_loader.dataset)),
        "num_val_samples": int(len(val_loader.dataset)),
        "num_test_samples": int(len(test_loader.dataset)) if test_loader is not None else 0,
        "mean_val_prediction_mape": float(val_predictions["mean_ape"].mean()),
        "mean_test_prediction_mape": (
            float(test_predictions["mean_ape"].mean())
            if test_predictions is not None
            else None
        ),
        "val_inference_ms_per_sample": float(val_seconds / max(len(val_predictions), 1) * 1000.0),
        "test_inference_ms_per_sample": (
            float(test_seconds / max(len(test_predictions), 1) * 1000.0)
            if test_predictions is not None and test_seconds is not None
            else None
        ),
    }
    write_json(run_dir / "summary.json", summary)
    print(f"best checkpoint: {run_dir / 'best_checkpoint.pth'}")
    if test_predictions is not None:
        print(f"test predictions: {predictions_dir / 'test_predictions.csv'}")
    else:
        print("outer-test evaluation: skipped")
    print(f"summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
