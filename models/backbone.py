"""
骨干网络模块
提供ResNet和EfficientNet系列backbone。
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(name='resnet18', pretrained=True):
    """
    获取骨干网络（去掉分类头）。

    Args:
        name: 'resnet18' / 'resnet34' / 'resnet50' / 'efficientnet_b0'
        pretrained: 是否使用ImageNet预训练权重
    Returns:
        backbone: nn.Module, 输出特征向量
        feat_dim: int, 特征维度
    """
    weights = "IMAGENET1K_V1" if pretrained else None

    if name == 'resnet18':
        model = models.resnet18(weights=weights)
        feat_dim = 512
    elif name == 'resnet34':
        model = models.resnet34(weights=weights)
        feat_dim = 512
    elif name == 'resnet50':
        model = models.resnet50(weights=weights)
        feat_dim = 2048
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        # EfficientNet结构不同，单独处理
        feat_dim = 1280
        backbone = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        return backbone, feat_dim
    else:
        raise ValueError(f"不支持的backbone: {name}")

    # 对ResNet系列，去掉最后的全连接层
    backbone = nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3, model.layer4,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )

    return backbone, feat_dim


def get_backbone_6ch(name='resnet18', pretrained=True):
    """
    获取6通道输入的骨干网络（用于拼接原始图+效果图的通道维度）。
    修改第一层卷积为6通道输入，保留预训练权重（复制3通道权重到6通道）。
    """
    weights = "IMAGENET1K_V1" if pretrained else None

    if name == 'resnet18':
        model = models.resnet18(weights=weights)
        feat_dim = 512
    elif name == 'resnet34':
        model = models.resnet34(weights=weights)
        feat_dim = 512
    else:
        raise ValueError(f"6通道模式暂只支持resnet18/resnet34，收到: {name}")

    # 修改第一层卷积: 3ch → 6ch
    old_conv = model.conv1
    new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if pretrained:
        # 将3通道权重复制两次到6通道
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight

    model.conv1 = new_conv

    backbone = nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3, model.layer4,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )

    return backbone, feat_dim
