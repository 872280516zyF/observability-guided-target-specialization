from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

from .attention_fusion import AttentionFusion


LOCAL_WEIGHT_FILENAMES = {
    "resnet50": "resnet50-11ad3fa6.pth",
    "resnet101": "resnet101-63fe2227.pth",
}


class DualResNetRegressor(nn.Module):
    """
    双分支 ResNet 回归模型：
    - 分支1：洗水前图片
    - 分支2：样板图片
    - 特征层融合（可选注意力融合）
    - 输出：连续激光参数（回归）
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_params: int = 4,
        use_attention: bool = True,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.pretrained_path = self._resolve_pretrained_path(backbone, pretrained_path)
        self._cached_state_dict = None

        self.before_backbone, feat_dim = self._build_backbone()
        self.after_backbone, _ = self._build_backbone()

        # 去掉分类头，只保留全局特征输出 (B, 2048)
        self.before_backbone.fc = nn.Identity()
        self.after_backbone.fc = nn.Identity()

        self.use_attention = use_attention
        if use_attention:
            self.fusion = AttentionFusion(feat_dim)
            fused_dim = feat_dim
        else:
            self.fusion = None
            fused_dim = feat_dim * 2

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_params),
        )

    def _resolve_pretrained_path(self, backbone: str, override: Optional[str]) -> Optional[Path]:
        candidates = []
        if override:
            candidates.append(Path(override))

        project_root = Path(__file__).resolve().parents[2]
        default_dir = project_root / "pretrained"
        filename = LOCAL_WEIGHT_FILENAMES.get(backbone)
        if filename:
            candidates.append(default_dir / filename)

        for path in candidates:
            if path.exists():
                return path
        return None

    def _build_backbone(self):
        if self.backbone_name == "resnet50":
            backbone = models.resnet50(weights=None)
            feat_dim = 2048
            weights_enum = models.ResNet50_Weights.IMAGENET1K_V2
        elif self.backbone_name == "resnet101":
            backbone = models.resnet101(weights=None)
            feat_dim = 2048
            weights_enum = models.ResNet101_Weights.IMAGENET1K_V2
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        if self.pretrained:
            state_dict = self._load_pretrained_state(weights_enum)
            backbone.load_state_dict(state_dict)

        return backbone, feat_dim

    def _load_pretrained_state(self, weights_enum):
        if self.pretrained_path is not None:
            if self._cached_state_dict is None:
                print(f"[DualResNet] Loading local pretrained weights: {self.pretrained_path}")
                self._cached_state_dict = torch.load(self.pretrained_path, map_location="cpu")
            return self._cached_state_dict

        print("[DualResNet] Local pretrained文件未找到，将尝试默认权重（可能触发网络下载）")
        if self._cached_state_dict is None:
            self._cached_state_dict = weights_enum.get_state_dict(progress=True)
        return self._cached_state_dict

    def forward(self, before_img: torch.Tensor, pattern_img: torch.Tensor):
        # 输入形状: (B, 3, H, W)
        feat_before = self.before_backbone(before_img)  # (B, 2048)
        feat_pattern = self.after_backbone(pattern_img)  # (B, 2048)

        if self.use_attention:
            fused, attn = self.fusion(feat_before, feat_pattern)
        else:
            fused = torch.cat([feat_before, feat_pattern], dim=1)
            attn = None

        preds = self.regressor(fused)
        return preds, attn


