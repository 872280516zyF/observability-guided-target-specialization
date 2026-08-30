import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
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

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
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

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape != skip.shape:
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ForwardEffectUNet(nn.Module):
    """
    简化版 UNet，用于模拟 “洗前图 + 目标图案 + 参数 → 洗后效果”。
    输入：before_img(3通道) + pattern_img(3通道) + params_map(num_params 通道)。
    输出：预测的洗后图像（与输入尺寸一致）。
    """

    def __init__(self, num_params: int = 4, base_channels: int = 64):
        super().__init__()
        self.num_params = num_params
        in_channels = 6 + num_params  # before + pattern + params map

        self.enc1 = ConvBlock(in_channels, base_channels, downsample=False)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, downsample=True)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, downsample=True)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, downsample=True)

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16, downsample=True)

        self.up4 = UpBlock(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2 + base_channels, base_channels)

        self.head = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=1),
            nn.Sigmoid(),  # 输出范围 [0, 1]，与图像处理更匹配
        )

        # 添加参数预测分支
        self.param_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels * 16, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_params)
        )

    def forward(self, before_img, pattern_img, params_map):
        x = torch.cat([before_img, pattern_img, params_map], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = self.head(d1)
        # 预测参数
        pred_params = self.param_predictor(b)
        return out, pred_params

