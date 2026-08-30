"""ROI-guided inverse predictor for laser parameters.

Design intent:
- f / p / v use target-region features through ROI-guided spatial pooling.
- d / DPI keeps an independent attention branch because DPI tends to encode
  scan-density / texture-coverage cues that should not be forced into the same
  ROI-only representation.

The model outputs normalized parameters in [0, 1]:
    [frequency, pulse_width, speed, DPI]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _build_resnet_spatial(backbone_name: str = "resnet18", pretrained: bool = False):
    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        feat_dim = 512
    elif backbone_name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)
        feat_dim = 512
    elif backbone_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        feat_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    encoder = nn.Sequential(*list(resnet.children())[:-2])
    return encoder, feat_dim


def _build_resnet_vector(backbone_name: str = "resnet18", pretrained: bool = False):
    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        feat_dim = 512
    elif backbone_name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)
        feat_dim = 512
    else:
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        feat_dim = 2048
    resnet.fc = nn.Identity()
    return resnet, feat_dim


class ROIWeightedPool(nn.Module):
    """Mask-guided spatial pooling over a feature map.

    If the mask is empty or nearly empty, the small epsilon keeps the operation
    stable. A soft mask is allowed; values should be in [0, 1].
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, feat: torch.Tensor, roi_mask: torch.Tensor) -> torch.Tensor:
        if roi_mask.dim() == 3:
            roi_mask = roi_mask.unsqueeze(1)
        roi = F.interpolate(roi_mask.float(), size=feat.shape[-2:], mode="bilinear", align_corners=False)
        roi = roi.clamp(0.0, 1.0)
        denom = roi.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        pooled = (feat * roi).sum(dim=(2, 3), keepdim=True) / denom
        return pooled.flatten(1)


class FPVHead(nn.Module):
    """Shared trunk plus separate f/p/v heads."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.35):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        def branch():
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

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        x = self.shared(fused)
        return torch.cat(
            [
                self.freq_head(x),
                self.pulse_head(x),
                self.speed_head(x),
            ],
            dim=1,
        )


class DPIAttentionBranch(nn.Module):
    """Independent full-image DPI predictor with channel attention."""

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = False, dropout: float = 0.5):
        super().__init__()
        self.backbone, feat_dim = _build_resnet_vector(backbone_name, pretrained)
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 4, feat_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, before: torch.Tensor, effect: torch.Tensor) -> torch.Tensor:
        f_before = self.backbone(before)
        f_effect = self.backbone(effect)
        a_before = self.attention(f_before)
        a_effect = self.attention(f_effect)
        f_before = f_before * a_before
        f_effect = f_effect * a_effect
        diff = torch.abs(f_effect - f_before)
        return self.head(torch.cat([f_before, f_effect, diff], dim=1))


class ROIGuidedHybridInversePredictor(nn.Module):
    """ROI-guided f/p/v predictor + independent DPI attention branch."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = False,
        hidden_dim: int = 256,
        dropout: float = 0.35,
        use_global_context: bool = False,
    ):
        super().__init__()
        self.encoder, feat_dim = _build_resnet_spatial(backbone_name, pretrained)
        self.roi_pool = ROIWeightedPool()
        self.use_global_context = use_global_context
        fpv_in = feat_dim * 3
        if use_global_context:
            # Optional safeguard: concatenate ROI features and global features.
            fpv_in += feat_dim * 3
        self.fpv_head = FPVHead(fpv_in, hidden_dim=hidden_dim, dropout=dropout)
        self.dpi_branch = DPIAttentionBranch(backbone_name=backbone_name, pretrained=pretrained)

    def _extract_roi_features(
        self,
        before: torch.Tensor,
        effect: torch.Tensor,
        roi_mask: torch.Tensor,
    ) -> torch.Tensor:
        before_map = self.encoder(before)
        effect_map = self.encoder(effect)
        f_before_roi = self.roi_pool(before_map, roi_mask)
        f_effect_roi = self.roi_pool(effect_map, roi_mask)
        diff_roi = f_effect_roi - f_before_roi
        fused = [f_before_roi, f_effect_roi, diff_roi]
        if self.use_global_context:
            f_before_global = F.adaptive_avg_pool2d(before_map, (1, 1)).flatten(1)
            f_effect_global = F.adaptive_avg_pool2d(effect_map, (1, 1)).flatten(1)
            diff_global = f_effect_global - f_before_global
            fused.extend([f_before_global, f_effect_global, diff_global])
        return torch.cat(fused, dim=1)

    def forward(self, batch):
        before = batch["before"]
        effect = batch["effect"]
        if "roi_mask" not in batch:
            raise KeyError("ROIGuidedHybridInversePredictor requires batch['roi_mask']")
        roi_mask = batch["roi_mask"]
        fpv = self.fpv_head(self._extract_roi_features(before, effect, roi_mask))
        dpi = self.dpi_branch(before, effect)
        return torch.cat([fpv, dpi], dim=1)
