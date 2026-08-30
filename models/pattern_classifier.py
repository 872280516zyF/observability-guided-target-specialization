import torch
import torch.nn as nn
import torchvision.models as models


class PatternClassifier(nn.Module):
    """
    样板图分类器：输入洗前图+目标图案，输出pattern_id（6分类）
    这个任务很简单，因为只有6个pattern，准确率会很高
    """
    
    def __init__(self, num_patterns: int = 6, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.num_patterns = num_patterns
        
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
        
        # 去掉分类头
        self.before_backbone.fc = nn.Identity()
        self.pattern_backbone.fc = nn.Identity()
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_patterns),
        )
    
    def forward(self, before_img: torch.Tensor, pattern_img: torch.Tensor):
        """
        Args:
            before_img: (B, 3, H, W)
            pattern_img: (B, 3, H, W)
        Returns:
            logits: (B, num_patterns) - 分类logits
            probs: (B, num_patterns) - 分类概率
        """
        feat_before = self.before_backbone(before_img)
        feat_pattern = self.pattern_backbone(pattern_img)
        
        fused = torch.cat([feat_before, feat_pattern], dim=1)
        fused = self.fusion(fused)
        
        logits = self.classifier(fused)
        probs = torch.softmax(logits, dim=1)
        
        return logits, probs

