"""
单流网络：仅输入效果图 → 预测4个参数
"""

import torch.nn as nn
from models.backbone import get_backbone
from models.heads import RegressionHead, MultiTaskHead


class SingleStreamNet(nn.Module):

    def __init__(self, backbone_name='resnet18', pretrained=True,
                 hidden_dim=256, dropout=0.3, multitask=False,
                 num_speed_classes=5):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.multitask = multitask

        if multitask:
            self.head = MultiTaskHead(feat_dim, hidden_dim, dropout, num_speed_classes)
        else:
            self.head = RegressionHead(feat_dim, hidden_dim, dropout, num_params=4)

    def forward(self, batch):
        feat = self.backbone(batch['image'])
        return self.head(feat)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
