"""
预测头模块
包含纯回归头和多任务（回归+分类）头。
"""

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """纯回归头：输出4个连续参数"""

    def __init__(self, in_dim, hidden_dim=256, dropout=0.3, num_params=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_params),
            nn.Sigmoid(),  # 输出[0,1]，因为参数做了min-max归一化
        )

    def forward(self, x):
        return self.head(x)


class MultiTaskHead(nn.Module):
    """
    多任务预测头：
    - 回归分支：输出频率、脉宽、DPI（3个连续值）
    - 分类分支：输出速度类别（5类）
    """

    def __init__(self, in_dim, hidden_dim=256, dropout=0.3, num_speed_classes=5):
        super().__init__()
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 回归分支：频率、脉宽、DPI
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

        # 分类分支：速度
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_speed_classes),
        )

    def forward(self, x):
        shared_feat = self.shared(x)
        reg_out = self.reg_head(shared_feat)    # (B, 3): freq, pulse, dpi (归一化)
        cls_out = self.cls_head(shared_feat)    # (B, 5): speed logits
        return reg_out, cls_out
