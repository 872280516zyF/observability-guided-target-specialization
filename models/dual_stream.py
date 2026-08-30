"""
双流网络模型定义
用于反向模型：图像 → 参数预测
注意：此文件需要与训练时完全一致！
"""
import torch
import torch.nn as nn
from torchvision import models


def get_backbone(backbone_name='resnet18', pretrained=True):
    """获取Backbone网络"""
    if backbone_name == 'resnet18':
        backbone = models.resnet18(pretrained=pretrained)
        feat_dim = 512
    elif backbone_name == 'resnet34':
        backbone = models.resnet34(pretrained=pretrained)
        feat_dim = 512
    elif backbone_name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
        feat_dim = 2048
    else:
        raise ValueError(f"不支持的Backbone: {backbone_name}")

    # 移除最后的全连接层
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    return backbone, feat_dim


class RegressionHead(nn.Module):
    """回归头：从融合特征预测参数
    注意：结构必须与训练时完全一致！
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3, num_params=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 2, num_params),
            nn.Sigmoid()  # 输出 [0,1]
        )

    def forward(self, x):
        return self.head(x)


class DualStreamNet(nn.Module):
    """双流网络：原始布料图 + 效果图 → 参数"""

    def __init__(self, backbone_name='resnet18', pretrained=True, fusion='concat_diff',
                 hidden_dim=256, dropout=0.3):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.fusion = fusion

        if fusion == 'concat':
            head_in = feat_dim * 2  # before + effect
        elif fusion == 'diff':
            head_in = feat_dim      # effect - before
        elif fusion == 'concat_diff':
            head_in = feat_dim * 3  # before + effect + diff
        else:
            raise ValueError(f"不支持的融合方式: {fusion}")

        self.head = RegressionHead(head_in, hidden_dim, dropout, num_params=4)

    def forward(self, batch):
        # 支持两种输入格式：字典或张量元组
        if isinstance(batch, dict):
            f_before = self.backbone(batch['before']).squeeze(-1).squeeze(-1)
            f_effect = self.backbone(batch['effect']).squeeze(-1).squeeze(-1)
        else:
            # 假设输入是 (before, effect) 张量元组
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                before, effect = batch
                f_before = self.backbone(before).squeeze(-1).squeeze(-1)
                f_effect = self.backbone(effect).squeeze(-1).squeeze(-1)
            else:
                # 支持create_batch_for_inverse返回的格式
                f_before = self.backbone(batch['before']).squeeze(-1).squeeze(-1) if 'before' in batch else None
                f_effect = self.backbone(batch['effect']).squeeze(-1).squeeze(-1) if 'effect' in batch else None
                if f_before is None:
                    # 尝试直接用张量
                    f_before = self.backbone(batch[:, :3, :, :]).squeeze(-1).squeeze(-1)
                    f_effect = self.backbone(batch[:, 3:, :, :]).squeeze(-1).squeeze(-1)

        if self.fusion == 'concat':
            fused = torch.cat([f_before, f_effect], dim=1)
        elif self.fusion == 'diff':
            fused = f_effect - f_before
        elif self.fusion == 'concat_diff':
            diff = f_effect - f_before
            fused = torch.cat([f_before, f_effect, diff], dim=1)

        return self.head(fused)