import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """通道注意力模块 (Squeeze-and-Excitation)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_attention = torch.cat([avg_out, max_out], dim=1)
        x_attention = self.conv1(x_attention)
        attention_weights = self.sigmoid(x_attention)
        return x * attention_weights


class CBAMBlock(nn.Module):
    """CBAM注意力模块 (通道注意力 + 空间注意力)"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = SEBlock(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块"""
    def __init__(self, channels_list):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels_list[0], 1) 
            for channels in channels_list
        ])
        self.fusion_conv = nn.Conv2d(channels_list[0] * len(channels_list), channels_list[0], 1)
        self.attention = CBAMBlock(channels_list[0])

    def forward(self, features):
        # 统一特征尺寸
        target_size = features[0].shape[2:]
        resized_features = []
        
        for i, feat in enumerate(features):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            feat = self.convs[i](feat)
            resized_features.append(feat)
        
        # 特征融合
        fused = torch.cat(resized_features, dim=1)
        fused = self.fusion_conv(fused)
        fused = self.attention(fused)
        
        return fused


class EnhancedConvBlock(nn.Module):
    """增强的卷积块，包含注意力机制"""
    def __init__(self, in_channels, out_channels, downsample=False, use_attention=True):
        super().__init__()
        stride = 2 if downsample else 1
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAMBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_attention:
            x = self.attention(x)
        return x


class EnhancedUpBlock(nn.Module):
    """增强的上采样块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.attention = CBAMBlock(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape != skip.shape:
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x


class ForwardEffectUNetAttention(nn.Module):
    """
    增强版 UNet，添加多种注意力机制来减少损失：
    1. CBAM注意力 (通道+空间)
    2. 多尺度特征融合
    3. 残差连接优化
    """

    def __init__(self, num_params: int = 4, base_channels: int = 64, use_attention=True):
        super().__init__()
        self.num_params = num_params
        self.use_attention = use_attention
        in_channels = 6 + num_params  # before + pattern + params map

        # 编码器
        self.enc1 = EnhancedConvBlock(in_channels, base_channels, downsample=False, use_attention=use_attention)
        self.enc2 = EnhancedConvBlock(base_channels, base_channels * 2, downsample=True, use_attention=use_attention)
        self.enc3 = EnhancedConvBlock(base_channels * 2, base_channels * 4, downsample=True, use_attention=use_attention)
        self.enc4 = EnhancedConvBlock(base_channels * 4, base_channels * 8, downsample=True, use_attention=use_attention)

        self.bottleneck = EnhancedConvBlock(base_channels * 8, base_channels * 16, downsample=True, use_attention=use_attention)

        # 解码器
        self.up4 = EnhancedUpBlock(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up3 = EnhancedUpBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up2 = EnhancedUpBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up1 = EnhancedUpBlock(base_channels * 2 + base_channels, base_channels)

        # 多尺度特征融合
        self.multi_scale_fusion = MultiScaleFusion([base_channels, base_channels * 2, base_channels * 4])

        # 输出头
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=1),
            nn.Sigmoid(),
        )

        # 残差连接
        self.residual_conv = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, before_img, pattern_img, params_map):
        x = torch.cat([before_img, pattern_img, params_map], dim=1)
        
        # 编码路径
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 瓶颈层
        b = self.bottleneck(e4)

        # 解码路径
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # 多尺度特征融合
        multi_scale_features = [e1, e2, e3]
        fused_features = self.multi_scale_fusion(multi_scale_features)
        
        # 与解码器输出融合
        final_features = d1 + fused_features

        # 输出 + 残差连接
        out = self.head(final_features)
        residual = self.residual_conv(x)
        out = out + residual
        
        # Match the [0, 1] image range used by the forward training/evaluation code.
        out = torch.sigmoid(out)
        
        return out


if __name__ == "__main__":
    # 测试模型
    model = ForwardEffectUNetAttention(num_params=4, base_channels=64)
    
    # 创建测试输入
    batch_size = 2
    height, width = 164, 374
    before_img = torch.randn(batch_size, 3, height, width)
    pattern_img = torch.randn(batch_size, 3, height, width)
    params_map = torch.randn(batch_size, 4, height, width)
    
    # 前向传播
    output = model(before_img, pattern_img, params_map)
    print(f"输入尺寸: before_img {before_img.shape}, pattern_img {pattern_img.shape}, params_map {params_map.shape}")
    print(f"输出尺寸: {output.shape}")
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
