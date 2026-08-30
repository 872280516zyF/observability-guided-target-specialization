import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    """
    特征注意力融合模块：
    输入两个全局特征向量 feat_before, feat_after (B, C)
    通过 MLP 计算两个分支的注意力权重，并进行加权求和。
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, feat_before: torch.Tensor, feat_after: torch.Tensor):
        # feat_before / feat_after: (B, C)
        concat = torch.cat([feat_before, feat_after], dim=1)  # (B, 2C)
        attn = self.mlp(concat)  # (B, 2)
        w_before = attn[:, 0:1]
        w_after = attn[:, 1:2]

        fused = w_before * feat_before + w_after * feat_after  # (B, C)
        return fused, attn


