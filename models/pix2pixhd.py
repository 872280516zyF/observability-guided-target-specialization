import torch
import torch.nn as nn
import torch.nn.functional as F


def imagenet_denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    return (img * std + mean).clamp(0, 1)


def pattern_soft_mask(pattern_img):
    pattern_01 = imagenet_denormalize(pattern_img)
    gray = pattern_01.mean(dim=1, keepdim=True)
    flat = gray.flatten(1)
    min_v = flat.min(dim=1).values.view(-1, 1, 1, 1)
    max_v = flat.max(dim=1).values.view(-1, 1, 1, 1)
    mask = (gray - min_v) / (max_v - min_v + 1e-6)
    mask = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
    return mask.clamp(0, 1)


def apply_output_mode(raw_output, before_img, pattern_img, output_mode):
    if output_mode == "direct":
        return raw_output
    if output_mode == "pattern_blend":
        mask = pattern_soft_mask(pattern_img)
        before_01 = imagenet_denormalize(before_img)
        return (before_01 * (1.0 - mask) + raw_output * mask).clamp(0, 1)
    if output_mode == "residual_delta":
        before_01 = imagenet_denormalize(before_img)
        delta = (raw_output - 0.5) * 2.0
        return (before_01 + delta).clamp(0, 1)
    if output_mode == "masked_residual_delta":
        mask = pattern_soft_mask(pattern_img)
        before_01 = imagenet_denormalize(before_img)
        delta = (raw_output - 0.5) * 2.0
        return (before_01 + mask * delta).clamp(0, 1)
    raise ValueError(f"Unsupported output_mode: {output_mode}")


class GlobalGenerator(nn.Module):
    """A U-Net style global generator used as the Pix2PixHD backbone."""

    def __init__(self, in_channels=6, num_params=4, base_channels=64, output_channels=3, output_mode="direct"):
        super().__init__()
        total_in_channels = in_channels + num_params
        self.output_mode = output_mode

        self.down1 = nn.Conv2d(total_in_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.down4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1)
        self.down5 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1)

        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        self.bn4 = nn.BatchNorm2d(base_channels * 8)
        self.bn5 = nn.BatchNorm2d(base_channels * 8)

        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(base_channels * 16, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(base_channels * 4, base_channels, kernel_size=4, stride=2, padding=1)
        self.up5 = nn.ConvTranspose2d(base_channels * 2, output_channels, kernel_size=4, stride=2, padding=1)

        self.bn_up1 = nn.BatchNorm2d(base_channels * 8)
        self.bn_up2 = nn.BatchNorm2d(base_channels * 4)
        self.bn_up3 = nn.BatchNorm2d(base_channels * 2)
        self.bn_up4 = nn.BatchNorm2d(base_channels)

        self.param_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_params),
        )

    def forward(self, before_img, pattern_img, params_map):
        x = torch.cat([before_img, pattern_img, params_map], dim=1)

        d1 = F.leaky_relu(self.down1(x), 0.2)
        d2 = F.leaky_relu(self.bn2(self.down2(d1)), 0.2)
        d3 = F.leaky_relu(self.bn3(self.down3(d2)), 0.2)
        d4 = F.leaky_relu(self.bn4(self.down4(d3)), 0.2)
        d5 = F.leaky_relu(self.bn5(self.down5(d4)), 0.2)

        u1 = F.relu(self.bn_up1(self.up1(d5)))
        if u1.shape[2:] != d4.shape[2:]:
            u1 = F.interpolate(u1, size=d4.shape[2:], mode="bilinear", align_corners=False)
        u1 = torch.cat([u1, d4], dim=1)

        u2 = F.relu(self.bn_up2(self.up2(u1)))
        if u2.shape[2:] != d3.shape[2:]:
            u2 = F.interpolate(u2, size=d3.shape[2:], mode="bilinear", align_corners=False)
        u2 = torch.cat([u2, d3], dim=1)

        u3 = F.relu(self.bn_up3(self.up3(u2)))
        if u3.shape[2:] != d2.shape[2:]:
            u3 = F.interpolate(u3, size=d2.shape[2:], mode="bilinear", align_corners=False)
        u3 = torch.cat([u3, d2], dim=1)

        u4 = F.relu(self.bn_up4(self.up4(u3)))
        if u4.shape[2:] != d1.shape[2:]:
            u4 = F.interpolate(u4, size=d1.shape[2:], mode="bilinear", align_corners=False)
        u4 = torch.cat([u4, d1], dim=1)

        raw_output = torch.sigmoid(self.up5(u4))
        output = apply_output_mode(raw_output, before_img, pattern_img, self.output_mode)
        pred_params = self.param_predictor(d5)
        return output, pred_params


class LocalEnhancer(nn.Module):
    """A shallow local refinement stage on top of the global generator."""

    def __init__(self, global_generator, num_params=4, base_channels=32, output_mode="direct", use_pattern_mask_channel=False):
        super().__init__()
        self.global_generator = global_generator
        self.output_mode = output_mode
        self.use_pattern_mask_channel = use_pattern_mask_channel

        extra_channels = 1 if use_pattern_mask_channel else 0
        self.local_conv1 = nn.Conv2d(6 + num_params + extra_channels, base_channels, kernel_size=3, padding=1)
        self.local_conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.fusion_conv = nn.Conv2d(base_channels * 2 + 3, 3, kernel_size=3, padding=1)

    def forward(self, before_img, pattern_img, params_map):
        global_output, pred_params = self.global_generator(before_img, pattern_img, params_map)
        global_output = F.interpolate(global_output, size=before_img.shape[2:], mode="bilinear", align_corners=False)

        local_inputs = [before_img, pattern_img, params_map]
        if self.use_pattern_mask_channel:
            local_inputs.append(pattern_soft_mask(pattern_img))
        local_input = torch.cat(local_inputs, dim=1)
        local_features = F.relu(self.bn1(self.local_conv1(local_input)))
        local_features = F.relu(self.bn2(self.local_conv2(local_features)))

        fused = torch.cat([local_features, global_output], dim=1)
        raw_enhanced = torch.sigmoid(self.fusion_conv(fused))
        enhanced_output = apply_output_mode(raw_enhanced, before_img, pattern_img, self.output_mode)
        return enhanced_output, pred_params


class Pix2PixHD(nn.Module):
    """Forward generator used for the second-system cGAN experiments."""

    def __init__(
        self,
        num_params=4,
        use_local_enhancer=True,
        base_channels=64,
        local_channels=32,
        output_mode="direct",
        use_pattern_mask_channel=False,
    ):
        super().__init__()
        self.use_local_enhancer = use_local_enhancer
        self.output_mode = output_mode
        self.use_pattern_mask_channel = use_pattern_mask_channel
        self.global_generator = GlobalGenerator(num_params=num_params, base_channels=base_channels, output_mode=output_mode)
        if use_local_enhancer:
            self.local_enhancer = LocalEnhancer(
                self.global_generator,
                num_params=num_params,
                base_channels=local_channels,
                output_mode=output_mode,
                use_pattern_mask_channel=use_pattern_mask_channel,
            )

    def forward(self, before_img, pattern_img, params_map):
        if self.use_local_enhancer:
            return self.local_enhancer(before_img, pattern_img, params_map)
        return self.global_generator(before_img, pattern_img, params_map)


class MultiscaleDiscriminator(nn.Module):
    """A multi-scale PatchGAN discriminator that returns raw logits."""

    def __init__(self, in_channels=3, base_channels=64, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()

        for _ in range(num_scales):
            disc = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(base_channels * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1),
                nn.InstanceNorm2d(base_channels * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
            )
            self.discriminators.append(disc)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x, before_img, pattern_img, params_map):
        inputs = torch.cat([x, before_img, pattern_img, params_map], dim=1)

        outputs = []
        current = inputs
        for idx, disc in enumerate(self.discriminators):
            outputs.append(disc(current))
            if idx < self.num_scales - 1:
                current = self.downsample(current)
        return outputs
