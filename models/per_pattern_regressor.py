import torch
import torch.nn as nn
import torchvision.models as models


class PerPatternRegressor(nn.Module):
    """
    针对单个pattern的参数回归器：
    - 输入：洗前图 + 目标图案（同一个pattern）
    - 输出：该pattern对应的参数（可以是全部4个，也可以只预测部分关键参数）
    
    因为每个pattern单独训练，任务更简单，R²会高很多
    """
    
    def __init__(
        self,
        num_params: int = 4,
        backbone: str = "resnet50",
        pretrained: bool = True,
        hidden_dims: list = [512, 256],
    ):
        super().__init__()
        self.num_params = num_params
        
        # 双分支特征提取
        if backbone == "resnet50":
            self.before_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.pattern_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 2048
        elif backbone == "resnet18":
            self.before_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.pattern_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.before_backbone.fc = nn.Identity()
        self.pattern_backbone.fc = nn.Identity()
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # 回归头（针对单个pattern，可以更简单）
        layers = []
        input_dim = feat_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, num_params))
        self.regressor = nn.Sequential(*layers)
    
    def forward(self, before_img: torch.Tensor, pattern_img: torch.Tensor):
        """
        Args:
            before_img: (B, 3, H, W)
            pattern_img: (B, 3, H, W)
        Returns:
            preds: (B, num_params) - 预测的参数（归一化后）
        """
        feat_before = self.before_backbone(before_img)
        feat_pattern = self.pattern_backbone(pattern_img)
        
        fused = torch.cat([feat_before, feat_pattern], dim=1)
        fused = self.fusion(fused)
        
        preds = self.regressor(fused)
        return preds

