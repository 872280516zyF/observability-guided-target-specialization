import os

# Force headless plotting/backends before importing cv2 or matplotlib.
# This avoids Qt/xcb crashes when evaluation figures are generated on Linux/WSL servers.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import torchvision
import cv2
cv2.setNumThreads(0)
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json

from datasets.laser_dataset_v2 import LaserParamDatasetV2, adaptive_collate_fn
from models.forward_unet import ForwardEffectUNet
from models.pix2pixhd import Pix2PixHD, MultiscaleDiscriminator
from utils.train_utils import save_checkpoint
from utils.seed import set_seed
from utils.experiment_plots import plot_forward_training_log


def _format_metric_value(value, precision=4):
    if value is None:
        return "None"
    try:
        if np.isnan(value):
            return "nan"
    except TypeError:
        pass
    return f"{float(value):.{precision}f}"


def _format_eval_metrics(metrics):
    if not metrics:
        return [
            "MSE: None",
            "MAE: None",
            "PSNR: None",
            "SSIM: None",
            "R²: None",
            "LPIPS: None",
            "FID: None",
        ]
    return [
        f"MSE: {_format_metric_value(metrics.get('mse'), 6)}",
        f"MAE: {_format_metric_value(metrics.get('mae'), 6)}",
        f"PSNR: {_format_metric_value(metrics.get('psnr'), 2)}",
        f"SSIM: {_format_metric_value(metrics.get('ssim'), 4)}",
        f"R²: {_format_metric_value(metrics.get('r2'), 4)}",
        f"Global R²: {_format_metric_value(metrics.get('r2_global'), 4)}",
        f"LPIPS: {_format_metric_value(metrics.get('lpips'), 4)}",
        f"FID: {_format_metric_value(metrics.get('fid'), 2)}",
    ]


def print_and_save_run_summary(
    *,
    config_path,
    log_file,
    summary_path,
    summary_json_path,
    best_epoch,
    best_val_loss,
    best_eval_metrics,
    best_eval_source,
    final_eval_epoch,
    final_eval_metrics,
    mode_name,
    saved_figures=None,
):
    """Print a copy-paste friendly experiment summary and persist it beside logs."""
    summary_lines = [
        "",
        "=" * 80,
        "可复制实验结果摘要",
        "=" * 80,
        f"配置文件名: {config_path}",
        f"训练模式: {mode_name}",
        f"训练日志路径: {log_file}",
        f"最佳模型 epoch: {best_epoch}",
        f"最佳验证损失: {_format_metric_value(best_val_loss, 6)}",
        f"最佳指标来源: {best_eval_source}",
        "最佳/对应评估指标:",
        *_format_eval_metrics(best_eval_metrics),
        f"最后一次全面评估 epoch: {final_eval_epoch}",
        "最终评估指标:",
        *_format_eval_metrics(final_eval_metrics),
    ]
    if saved_figures:
        summary_lines.append("训练曲线图:")
        summary_lines.extend([f"  {path}" for path in saved_figures])
    summary_lines.append("=" * 80)

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    payload = {
        "config_path": config_path,
        "mode": mode_name,
        "training_log": log_file,
        "best_epoch": best_epoch,
        "best_val_loss": None if best_val_loss == float("inf") else float(best_val_loss),
        "best_eval_source": best_eval_source,
        "best_eval_metrics": best_eval_metrics,
        "final_eval_epoch": final_eval_epoch,
        "final_eval_metrics": final_eval_metrics,
        "saved_figures": saved_figures or [],
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"实验摘要已保存: {summary_path}")
    print(f"实验摘要JSON已保存: {summary_json_path}")

# 尝试导入LPIPS和FID评估指标
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("警告: lpips库未安装，LPIPS评估将不可用。使用 'pip install lpips' 安装。")

try:
    from scipy import linalg
    from torchvision.models import inception_v3
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("警告: scipy库未安装，FID评估将不可用。使用 'pip install scipy' 安装。")


def create_param_map(params: torch.Tensor, height: int, width: int):
    """
    将 (B, num_params) 的参数向量拓展为 (B, num_params, H, W) 常数图，用于与图像拼接。
    """
    return params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)


class ForwardInputAblationDataset(torch.utils.data.Dataset):
    """Replace selected forward conditions with neutral normalized values."""

    VALID_MODES = {
        "full",
        "no_before",
        "no_pattern",
        "no_params",
        "pattern_params",
        "before_params",
        "images_only",
        "pattern_only",
        "before_only",
    }

    def __init__(self, dataset, mode: str = "full"):
        self.dataset = dataset
        self.mode = (mode or "full").strip()
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported forward input ablation mode: {self.mode}")

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.mode == "full":
            return sample
        sample = dict(sample)

        zero_before = self.mode in {"no_before", "pattern_params", "pattern_only"}
        zero_pattern = self.mode in {"no_pattern", "before_params", "before_only"}
        zero_params = self.mode in {"no_params", "images_only", "pattern_only", "before_only"}

        if zero_before and "before_img" in sample:
            sample["before_img"] = torch.zeros_like(sample["before_img"])
        if zero_pattern and "pattern_img" in sample:
            sample["pattern_img"] = torch.zeros_like(sample["pattern_img"])
        if zero_params and "targets" in sample:
            sample["targets"] = torch.zeros_like(sample["targets"])
        return sample


def denormalize_image(img, device=None):
    """将ImageNet归一化的图像反归一化到[0, 1]范围"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    if device is not None:
        mean = mean.to(device)
        std = std.to(device)
    return img * std + mean


def prepare_discriminator_inputs(before_img, pattern_img, after_img=None):
    """Keep discriminator inputs in the same image range for real and fake pairs."""
    before_img_01 = denormalize_image(before_img, before_img.device)
    pattern_img_01 = denormalize_image(pattern_img, pattern_img.device)
    if after_img is None:
        return before_img_01, pattern_img_01, None
    after_img_01 = denormalize_image(after_img, after_img.device)
    return before_img_01, pattern_img_01, after_img_01


def pattern_to_soft_mask(pattern_img):
    """Build a per-sample soft foreground mask from the pattern condition."""
    pattern_01 = denormalize_image(pattern_img, pattern_img.device).clamp(0, 1)
    gray = pattern_01.mean(dim=1, keepdim=True)
    flat = gray.flatten(1)
    min_v = flat.min(dim=1).values.view(-1, 1, 1, 1)
    max_v = flat.max(dim=1).values.view(-1, 1, 1, 1)
    mask = (gray - min_v) / (max_v - min_v + 1e-6)
    mask = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
    return mask.clamp(0, 1)


def mask_guided_shape_loss(pred, target, before_img, pattern_img, lambda_fg=0.0, lambda_bg=0.0):
    """Use the pattern as a spatial guide: match target inside, preserve before outside."""
    if lambda_fg <= 0 and lambda_bg <= 0:
        return pred.new_tensor(0.0)
    mask = pattern_to_soft_mask(pattern_img)
    before_01 = denormalize_image(before_img, before_img.device).clamp(0, 1)
    loss = pred.new_tensor(0.0)
    if lambda_fg > 0:
        fg_norm = mask.mean().clamp_min(1e-4)
        fg_loss = (torch.abs(pred - target) * mask).mean() / fg_norm
        loss = loss + lambda_fg * fg_loss
    if lambda_bg > 0:
        bg = 1.0 - mask
        bg_norm = bg.mean().clamp_min(1e-4)
        bg_loss = (torch.abs(pred - before_01) * bg).mean() / bg_norm
        loss = loss + lambda_bg * bg_loss
    return loss


def load_generator_warmstart(generator, checkpoint_path, device):
    """Warm-start the cGAN generator from a supervised checkpoint when available."""
    if not checkpoint_path:
        return False

    if not os.path.exists(checkpoint_path):
        print(f"警告：未找到生成器预训练权重，跳过热启动: {checkpoint_path}")
        return False

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model_state = generator.state_dict()
    adapted_state = {}
    skipped_keys = []
    adapted_keys = []
    for key, value in state_dict.items():
        if key not in model_state:
            adapted_state[key] = value
            continue
        target = model_state[key]
        if value.shape == target.shape:
            adapted_state[key] = value
            continue
        if (
            key == "local_enhancer.local_conv1.weight"
            and value.ndim == 4
            and target.ndim == 4
            and value.shape[0] == target.shape[0]
            and value.shape[2:] == target.shape[2:]
            and value.shape[1] < target.shape[1]
        ):
            patched = target.clone()
            patched[:, : value.shape[1], :, :] = value
            patched[:, value.shape[1] :, :, :] = 0.0
            adapted_state[key] = patched
            adapted_keys.append(key)
            continue
        skipped_keys.append((key, tuple(value.shape), tuple(target.shape)))

    missing_keys, unexpected_keys = generator.load_state_dict(adapted_state, strict=False)

    print(f"已加载生成器预训练权重: {checkpoint_path}")
    if adapted_keys:
        print(f"  已适配尺寸变化的权重: {adapted_keys}")
    if skipped_keys:
        print(f"  跳过尺寸不匹配权重数量: {len(skipped_keys)}")
        print(f"  跳过示例: {skipped_keys[:3]}")
    if missing_keys:
        print(f"  缺失键数量: {len(missing_keys)}")
        print(f"  缺失键示例: {missing_keys[:5]}")
    if unexpected_keys:
        print(f"  额外键数量: {len(unexpected_keys)}")
        print(f"  额外键示例: {unexpected_keys[:5]}")
    if len(missing_keys) > 20 or len(unexpected_keys) > 20:
        print("  警告：当前热启动匹配度较低，说明预训练权重与当前生成器结构可能不一致。")
    return True


class SSIMLoss(nn.Module):
    """SSIM损失 - 提高结构相似性"""
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)
    
    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1, img2):
        channel = img1.size(1)
        
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def forward(self, pred, target):
        return 1 - self._ssim(pred, target)


class PerceptualLoss(nn.Module):
    """感知损失 - 使用VGG特征提高视觉质量"""
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4']):
        super().__init__()
        self.layers = layers
        self.layer_weights = [1.0, 1.0, 1.0, 1.0]
        
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18]) # relu3_4
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27]) # relu4_4
        
        for param in self.parameters():
            param.requires_grad = False
        
        # 将VGG模型转换为float32，确保与输入类型一致
        self.slice1 = self.slice1.float()
        self.slice2 = self.slice2.float()
        self.slice3 = self.slice3.float()
        self.slice4 = self.slice4.float()
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred, target):
        # 确保mean和std与输入张量在同一设备上
        if self.mean.device != pred.device:
            self.mean = self.mean.to(pred.device)
            self.std = self.std.to(pred.device)
        
        # 确保VGG层与输入在同一设备上
        if self.slice1[0].weight.device != pred.device:
            self.slice1 = self.slice1.to(pred.device)
            self.slice2 = self.slice2.to(pred.device)
            self.slice3 = self.slice3.to(pred.device)
            self.slice4 = self.slice4.to(pred.device)
        
        # 将输入转换为float32，避免与VGG的float32权重不匹配
        pred_float = pred.float()
        target_float = target.float()
        
        # 归一化到ImageNet标准
        pred_norm = (pred_float - self.mean) / self.std
        target_norm = (target_float - self.mean) / self.std
        
        # 确保VGG层在float32模式下运行
        with torch.amp.autocast('cuda', enabled=False):
            loss = 0.0
            
            if 'relu1_2' in self.layers:
                pred_feat1 = self.slice1(pred_norm)
                target_feat1 = self.slice1(target_norm)
                loss += self.layer_weights[0] * F.l1_loss(pred_feat1, target_feat1)
            
            if 'relu2_2' in self.layers:
                pred_feat2 = self.slice2(pred_feat1)
                target_feat2 = self.slice2(target_feat1)
                loss += self.layer_weights[1] * F.l1_loss(pred_feat2, target_feat2)
            
            if 'relu3_4' in self.layers:
                pred_feat3 = self.slice3(pred_feat2)
                target_feat3 = self.slice3(target_feat2)
                loss += self.layer_weights[2] * F.l1_loss(pred_feat3, target_feat3)
            
            if 'relu4_4' in self.layers:
                pred_feat4 = self.slice4(pred_feat3)
                target_feat4 = self.slice4(target_feat3)
                loss += self.layer_weights[3] * F.l1_loss(pred_feat4, target_feat4)
        
        return loss


def sobel_edges(img):
    """Return Sobel edge magnitude for images in [0, 1]."""
    channels = img.shape[1]
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=img.device,
        dtype=img.dtype,
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=img.device,
        dtype=img.dtype,
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    gx = F.conv2d(img, kernel_x, padding=1, groups=channels)
    gy = F.conv2d(img, kernel_y, padding=1, groups=channels)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def laplacian_highpass(img):
    """Return Laplacian high-frequency response for images in [0, 1]."""
    channels = img.shape[1]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=img.device,
        dtype=img.dtype,
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    return F.conv2d(img, kernel, padding=1, groups=channels)


class EdgeTextureLoss(nn.Module):
    """Edge and high-frequency texture loss for visual forward simulation."""

    def __init__(self, lambda_edge=0.0, lambda_texture=0.0):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.lambda_texture = lambda_texture

    def forward(self, pred, target):
        loss = pred.new_tensor(0.0)
        edge = pred.new_tensor(0.0)
        texture = pred.new_tensor(0.0)
        if self.lambda_edge > 0:
            edge = F.l1_loss(sobel_edges(pred), sobel_edges(target))
            loss = loss + self.lambda_edge * edge
        if self.lambda_texture > 0:
            texture = F.l1_loss(laplacian_highpass(pred), laplacian_highpass(target))
            loss = loss + self.lambda_texture * texture
        return loss, edge, texture


class CombinedLoss(nn.Module):
    """组合损失：L1 + SSIM + 感知损失"""
    def __init__(self, lambda_l1=1.0, lambda_ssim=1.0, lambda_perceptual=0.5, lambda_edge=0.0, lambda_texture=0.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
        self.lambda_edge = lambda_edge
        self.lambda_texture = lambda_texture

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = PerceptualLoss()
        self.edge_texture_loss = EdgeTextureLoss(lambda_edge, lambda_texture)

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        edge_texture, _, _ = self.edge_texture_loss(pred, target)

        total = self.lambda_l1 * l1 + self.lambda_ssim * ssim + self.lambda_perceptual * perceptual + edge_texture

        return total, l1, ssim, perceptual


class EnhancedL1Loss(nn.Module):
    """增强的L1损失，结合结构相似性，更好地利用洗水后图片作为监督"""
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred, target):
        l1_loss = self.l1_loss(pred, target)
        
        # 结构相似性计算 - 更好地利用洗水后图片的视觉信息
        pred_mean = pred.mean(dim=[1, 2, 3], keepdim=True)
        target_mean = target.mean(dim=[1, 2, 3], keepdim=True)
        pred_std = pred.std(dim=[1, 2, 3], keepdim=True)
        target_std = target.std(dim=[1, 2, 3], keepdim=True)
        
        # 对比度相似性
        contrast_sim = (2 * pred_std * target_std) / (pred_std**2 + target_std**2 + 1e-8)
        contrast_sim = contrast_sim.mean()
        
        # 亮度相似性
        brightness_sim = (2 * pred_mean * target_mean) / (pred_mean**2 + target_mean**2 + 1e-8)
        brightness_sim = brightness_sim.mean()
        
        # 结合L1损失和结构相似性
        return l1_loss + self.alpha * (2 - contrast_sim - brightness_sim)


class Discriminator(nn.Module):
    """判别器网络，输入为洗水后图片+条件信息（洗前图+样板图+参数图），输出真伪概率"""
    def __init__(self, num_params, base_channels=64):
        super().__init__()
        
        # 输入通道数：洗后图(3) + 洗前图(3) + 样板图(3) + 参数图(num_params)
        in_channels = 3 + 3 + 3 + num_params
        
        self.conv_layers = nn.Sequential(
            # 第1层: 输入 -> 64通道
            nn.Conv2d(in_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第2层: 64 -> 128通道
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第3层: 128 -> 256通道
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第4层: 256 -> 512通道
            nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_channels*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第5层: 512 -> 1通道（输出真伪概率）
            nn.Conv2d(base_channels*8, 1, 4, 1, 0, bias=False)
        )
    
    def forward(self, after_img, before_img, pattern_img, param_map):
        # 拼接所有条件信息
        x = torch.cat([after_img, before_img, pattern_img, param_map], dim=1)
        return self.conv_layers(x)


class cGANLoss(nn.Module):
    """cGAN损失函数，结合对抗损失和L1损失"""
    def __init__(self, lambda_l1=100.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.l1_criterion = nn.L1Loss()
    
    def forward(self, discriminator_output, is_real, generated_img=None, target_img=None):
        # 对抗损失
        if isinstance(discriminator_output, list):
            # 多尺度判别器输出
            adv_loss = 0.0
            for i, output in enumerate(discriminator_output):
                # 确保输入在[0, 1]范围内
                output_clamped = output
                # 为每个尺度的输出使用对应的目标值
                target = is_real[i] if isinstance(is_real, list) else is_real
                adv_loss += self.adv_criterion(output_clamped, target)
            adv_loss /= len(discriminator_output)
        else:
            # 单尺度判别器输出
            # 确保输入在[0, 1]范围内
            output_clamped = discriminator_output
            adv_loss = self.adv_criterion(output_clamped, is_real)
        
        # L1损失（如果提供了生成图像和目标图像）
        l1_loss = 0.0
        if generated_img is not None and target_img is not None:
            l1_loss = self.l1_criterion(generated_img, target_img)
        
        # 总损失 = 对抗损失 + λ * L1损失
        total_loss = adv_loss + self.lambda_l1 * l1_loss
        
        return total_loss, adv_loss, l1_loss


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, use_combined_loss=True):
    """训练循环，充分利用洗水后图片作为监督信号"""
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    total_perceptual = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        before_img = batch["before_img"].to(device)
        pattern_img = batch["pattern_img"].to(device)
        after_img = batch["after_img"].to(device)  # 真实洗水后图片作为监督
        params = batch["targets"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            # 使用参数图预测洗水效果
            param_map = create_param_map(params, before_img.size(2), before_img.size(3))
            pred_after, pred_params = model(before_img, pattern_img, param_map)
            
            # 关键改进：使用真实洗水后图片作为监督信号
            # 模型输出经过Sigmoid，范围是[0, 1]，所以目标值也需要在[0, 1]范围内
            # 由于数据集中的图像已经经过ImageNet归一化，需要反归一化到[0, 1]范围
            after_img_normalized = denormalize_image(after_img, after_img.device)
            
            # 图像损失
            if use_combined_loss and isinstance(criterion, CombinedLoss):
                img_loss, l1_loss, ssim_loss, perceptual_loss = criterion(pred_after, after_img_normalized)
                total_l1 += l1_loss.item() * before_img.size(0)
                total_ssim += ssim_loss.item() * before_img.size(0)
                total_perceptual += perceptual_loss.item() * before_img.size(0)
            else:
                img_loss = criterion(pred_after, after_img_normalized)
            
            # 参数预测损失
            param_loss = nn.MSELoss()(pred_params, params)
            # 总损失
            loss = img_loss + 0.1 * param_loss  # 参数损失权重

        if scaler is not None:
            scaler.scale(loss).backward()
            # 梯度裁剪稳定训练
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # 梯度裁剪稳定训练
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * before_img.size(0)

    avg_loss = total_loss / len(loader.dataset)
    if use_combined_loss and isinstance(criterion, CombinedLoss):
        avg_l1 = total_l1 / len(loader.dataset)
        avg_ssim = total_ssim / len(loader.dataset)
        avg_perceptual = total_perceptual / len(loader.dataset)
        return avg_loss, avg_l1, avg_ssim, avg_perceptual
    return avg_loss, 0, 0, 0


@torch.no_grad()
def evaluate(model, loader, criterion, device, record_per_sample=False, use_combined_loss=True):
    """
    评估函数，同样使用洗水后图片作为监督
    
    Args:
        record_per_sample: 是否记录每个样本的损失（用于困难样本分析）
    """
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    total_perceptual = 0.0
    
    # 记录每个样本的损失（如果启用）
    per_sample_losses = [] if record_per_sample else None

    for batch in tqdm(loader, desc="Val", leave=False):
        before_img = batch["before_img"].to(device)
        pattern_img = batch["pattern_img"].to(device)
        after_img = batch["after_img"].to(device)
        params = batch["targets"].to(device)

        param_map = create_param_map(params, before_img.size(2), before_img.size(3))
        pred_after, pred_params = model(before_img, pattern_img, param_map)
        # 模型输出经过Sigmoid，范围是[0, 1]，所以目标值也需要在[0, 1]范围内
        # 由于数据集中的图像已经经过ImageNet归一化，需要反归一化到[0, 1]范围
        after_img_normalized = denormalize_image(after_img, after_img.device)
        # 图像损失
        if use_combined_loss and isinstance(criterion, CombinedLoss):
            img_loss, l1_loss, ssim_loss, perceptual_loss = criterion(pred_after, after_img_normalized)
            total_l1 += l1_loss.item() * before_img.size(0)
            total_ssim += ssim_loss.item() * before_img.size(0)
            total_perceptual += perceptual_loss.item() * before_img.size(0)
        else:
            img_loss = criterion(pred_after, after_img_normalized)
        # 参数预测损失
        param_loss = nn.MSELoss()(pred_params, params)
        # 总损失 - img_loss已经是标量（当使用CombinedLoss时，返回的是total）
        loss = img_loss + 0.1 * param_loss  # 参数损失权重
        total_loss += loss.item() * before_img.size(0)
        
        # 记录每个样本的损失（如果启用）
        if record_per_sample and per_sample_losses is not None:
            batch_size = before_img.size(0)
            for i in range(batch_size):
                sample_info = {}
                # 获取样本ID
                if 'sample_id' in batch:
                    sample_id_val = batch['sample_id'][i]
                    if torch.is_tensor(sample_id_val):
                        sample_info['sample_id'] = str(sample_id_val.item())
                    else:
                        sample_info['sample_id'] = str(sample_id_val)
                elif 'sample_idx' in batch:
                    idx_val = batch['sample_idx'][i]
                    if torch.is_tensor(idx_val):
                        sample_info['sample_idx'] = idx_val.item()
                        try:
                            dataset_sample = loader.dataset[idx_val.item()]
                            if 'sample_id' in dataset_sample:
                                sample_info['sample_id'] = str(dataset_sample['sample_id'])
                        except:
                            pass
                    else:
                        sample_info['sample_idx'] = idx_val
                        try:
                            dataset_sample = loader.dataset[idx_val]
                            if 'sample_id' in dataset_sample:
                                sample_info['sample_id'] = str(dataset_sample['sample_id'])
                        except:
                            pass
                
                if 'pattern_id' in batch:
                    pattern_id_val = batch['pattern_id'][i]
                    if torch.is_tensor(pattern_id_val):
                        sample_info['pattern_id'] = str(pattern_id_val.item())
                    else:
                        sample_info['pattern_id'] = str(pattern_id_val)
                
                # 计算单个样本的损失
                sample_pred = pred_after[i:i+1]
                sample_target = after_img[i:i+1]
                # 确保目标值在[0, 1]范围内，与Sigmoid输出匹配
                mean_sample = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(sample_target.device)
                std_sample = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(sample_target.device)
                sample_target_normalized = sample_target * std_sample + mean_sample  # 反归一化到[0, 1]
                
                # 处理CombinedLoss返回元组的情况
                if use_combined_loss and isinstance(criterion, CombinedLoss):
                    sample_loss_tuple = criterion(sample_pred, sample_target_normalized)
                    sample_loss = sample_loss_tuple[0]  # 取total损失
                else:
                    sample_loss = criterion(sample_pred, sample_target_normalized)
                sample_info['loss'] = sample_loss.item()
                
                per_sample_losses.append(sample_info)

    avg_loss = total_loss / len(loader.dataset)
    if use_combined_loss and isinstance(criterion, CombinedLoss):
        avg_l1 = total_l1 / len(loader.dataset)
        avg_ssim = total_ssim / len(loader.dataset)
        avg_perceptual = total_perceptual / len(loader.dataset)
        return avg_loss, avg_l1, avg_ssim, avg_perceptual, per_sample_losses
    return avg_loss, 0, 0, 0, per_sample_losses


def train_generator(generator, discriminator, batch, g_optimizer, criterion, device, scaler=None):
    """训练生成器：生成假图像，让判别器误判为真"""
    before_img = batch["before_img"].to(device)
    pattern_img = batch["pattern_img"].to(device)
    after_img = batch["after_img"].to(device)
    params = batch["targets"].to(device)
    
    g_optimizer.zero_grad()
    
    with torch.amp.autocast('cuda', enabled=scaler is not None):
            # 生成器生成假洗后图
            param_map = create_param_map(params, before_img.size(2), before_img.size(3))
            fake_after, pred_params = generator(before_img, pattern_img, param_map)
            
            # 判别器判断假图像
            before_img_01, pattern_img_01, _ = prepare_discriminator_inputs(before_img, pattern_img)
            fake_output = discriminator(fake_after, before_img_01, pattern_img_01, param_map)
            
            # 模型输出经过Sigmoid，范围是[0, 1]，所以目标值也需要在[0, 1]范围内
            # 由于数据集中的图像已经经过ImageNet归一化，需要反归一化到[0, 1]范围
            after_img_normalized = denormalize_image(after_img, after_img.device)
            
            # 处理多尺度判别器的输出
            if isinstance(fake_output, list):
                # 为每个尺度的输出创建对应的全1张量
                fake_target = [torch.ones_like(output) for output in fake_output]
            else:
                # 单尺度判别器
                fake_target = torch.ones_like(fake_output)
            
            # 生成器希望判别器将假图像判断为真（标签为1）
            g_loss_adv, g_adv_loss, g_l1_loss = criterion(
                fake_output, 
                fake_target,  # 希望判别器判断为真
                fake_after, 
                after_img_normalized
            )
            
            # 参数预测损失
            param_loss = nn.MSELoss()(pred_params, params)
            # 总生成器损失
            g_loss = g_loss_adv + 0.1 * param_loss  # 参数损失权重
    
    if scaler is not None:
        scaler.scale(g_loss).backward()
        # 梯度裁剪稳定训练
        scaler.unscale_(g_optimizer)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        scaler.step(g_optimizer)
        scaler.update()
    else:
        g_loss.backward()
        # 梯度裁剪稳定训练
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        g_optimizer.step()
    
    return g_loss.item(), g_adv_loss.item(), g_l1_loss.item()


def train_generator_joint(generator, discriminator, batch, g_optimizer, cgan_criterion, supervised_criterion,
                         device, scaler=None, joint_alpha=0.5, use_combined_supervised=True,
                         lambda_mask_fg=0.0, lambda_mask_bg=0.0):
    """联合训练生成器：同时使用对抗损失和监督损失（支持组合损失）
    
    Args:
        use_combined_supervised: 是否使用组合损失（L1 + SSIM + 感知损失）作为监督损失
    """
    before_img = batch["before_img"].to(device)
    pattern_img = batch["pattern_img"].to(device)
    after_img = batch["after_img"].to(device)
    params = batch["targets"].to(device)
    
    g_optimizer.zero_grad()
    
    with torch.amp.autocast('cuda', enabled=scaler is not None):
            # 生成器生成假洗后图
            param_map = create_param_map(params, before_img.size(2), before_img.size(3))
            fake_after, pred_params = generator(before_img, pattern_img, param_map)
            
            # 模型输出经过Sigmoid，范围是[0, 1]，所以目标值也需要在[0, 1]范围内
            # 由于数据集中的图像已经经过ImageNet归一化，需要反归一化到[0, 1]范围
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(after_img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(after_img.device)
            after_img_normalized = after_img * std + mean  # 反归一化到[0, 1]范围
            
            # 对抗损失：判别器判断
            before_img_01, pattern_img_01, after_img_normalized = prepare_discriminator_inputs(before_img, pattern_img, after_img)
            fake_output = discriminator(fake_after, before_img_01, pattern_img_01, param_map)
            # 处理多尺度判别器的输出
            if isinstance(fake_output, list):
                # 为每个尺度的输出创建对应的全1张量
                fake_target = [torch.ones_like(output) for output in fake_output]
            else:
                # 单尺度判别器
                fake_target = torch.ones_like(fake_output)
            g_loss_adv, g_adv_loss, g_l1_loss = cgan_criterion(
                fake_output, 
                fake_target,  # 希望判别器判断为真
                fake_after, 
                after_img_normalized
            )
            
            # 监督损失：使用组合损失（L1 + SSIM + 感知损失）
            if use_combined_supervised and isinstance(supervised_criterion, CombinedLoss):
                supervised_loss, sup_l1, sup_ssim, sup_perceptual = supervised_criterion(fake_after, after_img_normalized)
            else:
                # 简单的L1损失
                supervised_loss = supervised_criterion(fake_after, after_img_normalized)
                sup_l1 = supervised_loss
                sup_ssim = torch.tensor(0.0)
                sup_perceptual = torch.tensor(0.0)
            supervised_loss = supervised_loss + mask_guided_shape_loss(
                fake_after,
                after_img_normalized,
                before_img,
                pattern_img,
                lambda_fg=lambda_mask_fg,
                lambda_bg=lambda_mask_bg,
            )
            
            # 参数预测损失
            param_loss = nn.MSELoss()(pred_params, params)
        
            # 联合损失：对抗损失 + α × 监督损失 + 参数损失
            g_loss = g_loss_adv + joint_alpha * supervised_loss + 0.1 * param_loss  # 参数损失权重
    
    if scaler is not None:
        scaler.scale(g_loss).backward()
        # 梯度裁剪稳定训练
        scaler.unscale_(g_optimizer)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        scaler.step(g_optimizer)
        scaler.update()
    else:
        g_loss.backward()
        # 梯度裁剪稳定训练
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        g_optimizer.step()
    
    if use_combined_supervised and isinstance(supervised_criterion, CombinedLoss):
        return g_loss.item(), g_adv_loss.item(), g_l1_loss.item(), supervised_loss.item(), sup_l1.item(), sup_ssim.item(), sup_perceptual.item()
    else:
        return g_loss.item(), g_adv_loss.item(), g_l1_loss.item(), supervised_loss.item(), 0.0, 0.0, 0.0


def train_discriminator(generator, discriminator, batch, d_optimizer, criterion, device, scaler=None, label_smoothing=0.1):
    """训练判别器：区分真实图像和生成图像
    
    Args:
        label_smoothing: 标签平滑系数，0.0-0.3之间，用于稳定训练，防止判别器过强
    """
    before_img = batch["before_img"].to(device)
    pattern_img = batch["pattern_img"].to(device)
    after_img = batch["after_img"].to(device)
    params = batch["targets"].to(device)
    
    d_optimizer.zero_grad()
    
    with torch.amp.autocast('cuda', enabled=scaler is not None):
        param_map = create_param_map(params, before_img.size(2), before_img.size(3))
        
        # 真实图像判别（使用标签平滑：1.0 -> 0.9~1.0）
        # 由于数据集中的图像已经经过ImageNet归一化，需要反归一化到[0, 1]范围
        before_img_01, pattern_img_01, after_img_01 = prepare_discriminator_inputs(before_img, pattern_img, after_img)
        real_output = discriminator(after_img_01, before_img_01, pattern_img_01, param_map)
        
        # 处理多尺度判别器的输出
        if isinstance(real_output, list):
            # 多尺度判别器
            real_loss = 0.0
            for output in real_output:
                real_target = torch.ones_like(output) * (1.0 - label_smoothing)
                loss, _, _ = criterion(output, real_target)
                real_loss += loss
            real_loss /= len(real_output)
        else:
            # 单尺度判别器
            real_target = torch.ones_like(real_output) * (1.0 - label_smoothing)
            real_loss, _, _ = criterion(real_output, real_target)
        
        # 生成图像判别（使用标签平滑：0.0 -> 0.0~0.1）
        fake_after, _ = generator(before_img, pattern_img, param_map)
        fake_after = fake_after.detach()
        fake_output = discriminator(fake_after, before_img_01, pattern_img_01, param_map)
        
        # 处理多尺度判别器的输出
        if isinstance(fake_output, list):
            # 多尺度判别器
            fake_loss = 0.0
            for output in fake_output:
                fake_target = torch.zeros_like(output)
                loss, _, _ = criterion(output, fake_target)
                fake_loss += loss
            fake_loss /= len(fake_output)
        else:
            # 单尺度判别器
            fake_target = torch.zeros_like(fake_output)
            fake_loss, _, _ = criterion(fake_output, fake_target)
        
        # 判别器总损失
        d_loss = (real_loss + fake_loss) / 2
    
    if scaler is not None:
        scaler.scale(d_loss).backward()
        # 梯度裁剪稳定训练
        scaler.unscale_(d_optimizer)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        scaler.step(d_optimizer)
        scaler.update()
    else:
        d_loss.backward()
        # 梯度裁剪稳定训练
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        d_optimizer.step()
    
    return d_loss.item()


def train_one_epoch_cgan(generator, discriminator, loader, g_optimizer, d_optimizer, criterion, device, scaler=None, 
                          use_joint_training=True, joint_alpha=0.5, d_train_freq=3, label_smoothing=0.2, 
                          adaptive_d_freq=True, epoch=None, total_epochs=None, cfg=None):
    """改进的cGANs训练循环：解决视觉质量下降和模糊问题
    
    Args:
        d_train_freq: 判别器训练频率，每N个batch训练一次判别器
        label_smoothing: 标签平滑系数，用于稳定训练
        adaptive_d_freq: 是否自适应调整判别器训练频率
        epoch: 当前epoch，用于自适应调整
        total_epochs: 总epoch数，用于自适应调整
        cfg: 配置文件，用于读取联合训练损失设置
    """
    generator.train()
    discriminator.train()
    
    g_total_loss = 0.0
    d_total_loss = 0.0
    g_total_adv_loss = 0.0
    g_total_l1_loss = 0.0
    g_total_supervised_loss = 0.0

    # 自适应调整判别器训练频率
    if adaptive_d_freq and epoch is not None and total_epochs is not None:
        # 训练早期：减少判别器训练频率，让生成器先学习基础
        if epoch < total_epochs * 0.3:  # 前30%的epoch
            current_d_freq = max(d_train_freq, 5)  # 至少每5个batch训练一次
        # 训练中期：正常频率
        elif epoch < total_epochs * 0.7:  # 30%-70%的epoch
            current_d_freq = d_train_freq
        # 训练后期：增加判别器训练频率，提升细节质量
        else:
            current_d_freq = max(1, d_train_freq // 2)  # 增加训练频率
    else:
        current_d_freq = d_train_freq

    # 创建监督损失函数（用于联合训练）
    if use_joint_training:
        # 从配置文件读取联合训练损失设置
        if cfg and "trainer" in cfg:
            joint_losses = cfg["trainer"].get("joint_losses", ["l1", "ssim", "perceptual"])
            lambda_l1 = cfg["trainer"].get("joint_lambda_l1", 1.0)
            lambda_ssim = cfg["trainer"].get("joint_lambda_ssim", 2.0)
            lambda_perceptual = cfg["trainer"].get("joint_lambda_perceptual", 1.0)
            lambda_edge = cfg["trainer"].get("joint_lambda_edge", 0.0)
            lambda_texture = cfg["trainer"].get("joint_lambda_texture", 0.0)
            lambda_mask_fg = cfg["trainer"].get("joint_lambda_mask_fg", 0.0)
            lambda_mask_bg = cfg["trainer"].get("joint_lambda_mask_bg", 0.0)
        else:
            # 默认值
            joint_losses = ["l1", "ssim", "perceptual"]
            lambda_l1 = 1.0
            lambda_ssim = 2.0
            lambda_perceptual = 1.0
            lambda_edge = 0.0
            lambda_texture = 0.0
            lambda_mask_fg = 0.0
            lambda_mask_bg = 0.0

        # 创建组合损失
        supervised_criterion = CombinedLoss(
            lambda_l1 if "l1" in joint_losses else 0.0,
            lambda_ssim if "ssim" in joint_losses else 0.0,
            lambda_perceptual if "perceptual" in joint_losses else 0.0,
            lambda_edge if "edge" in joint_losses else 0.0,
            lambda_texture if "texture" in joint_losses else 0.0,
        )
        
        # 打印使用的损失组合
        used_losses = []
        if "l1" in joint_losses:
            used_losses.append(f"L1({lambda_l1})")
        if "ssim" in joint_losses:
            used_losses.append(f"SSIM({lambda_ssim})")
        if "perceptual" in joint_losses:
            used_losses.append(f"Perceptual({lambda_perceptual})")
        if "edge" in joint_losses:
            used_losses.append(f"Edge({lambda_edge})")
        if "texture" in joint_losses:
            used_losses.append(f"Texture({lambda_texture})")
        if "mask" in joint_losses:
            used_losses.append(f"MaskFG({lambda_mask_fg})")
            used_losses.append(f"MaskBG({lambda_mask_bg})")
        
        if used_losses:
            print(f"使用组合监督损失: {' + '.join(used_losses)}")
        else:
            print("警告：未启用任何联合训练损失！")
    else:
        supervised_criterion = None
        lambda_mask_fg = 0.0
        lambda_mask_bg = 0.0
    
    # 追踪判别器和生成器的损失平衡
    d_loss_history = []
    g_loss_history = []
    balance_window = 20  # 窗口大小，用于计算平均损失

    for batch_idx, batch in enumerate(tqdm(loader, desc="Train cGANs", leave=False)):
        # 动态调整判别器训练频率（基于损失平衡）
        train_d = True
        if len(d_loss_history) > balance_window and len(g_loss_history) > balance_window:
            avg_d_loss = sum(d_loss_history[-balance_window:]) / balance_window
            avg_g_loss = sum(g_loss_history[-balance_window:]) / balance_window
            
            # 如果判别器太强（D_loss太低），跳过一些判别器训练
            if avg_d_loss < 0.1:  # D太自信，减少训练
                train_d = batch_idx % max(current_d_freq, 5) == 0
            # 如果判别器太弱（D_loss太高），增加训练
            elif avg_d_loss > 0.6:  # D太弱，增加训练
                train_d = True
            # 正常平衡状态，按原频率训练
            else:
                train_d = batch_idx % current_d_freq == 0
        
        # 控制判别器训练频率
        if train_d:
            # 训练判别器（使用标签平滑）
            d_loss = train_discriminator(generator, discriminator, batch, d_optimizer, criterion, device, scaler, label_smoothing)
            d_total_loss += d_loss * batch["before_img"].size(0)
            d_loss_history.append(d_loss)
        else:
            # 跳过判别器训练，只训练生成器
            d_loss = 0.0
        
        # 训练生成器（支持联合训练）
        if use_joint_training:
            g_loss, g_adv_loss, g_l1_loss, g_supervised_loss, sup_l1, sup_ssim, sup_perceptual = train_generator_joint(
                generator, discriminator, batch, g_optimizer, criterion, supervised_criterion, 
                device, scaler, joint_alpha, use_combined_supervised=True,
                lambda_mask_fg=lambda_mask_fg, lambda_mask_bg=lambda_mask_bg,
            )
            g_total_supervised_loss += g_supervised_loss * batch["before_img"].size(0)
        else:
            g_loss, g_adv_loss, g_l1_loss = train_generator(
                generator, discriminator, batch, g_optimizer, criterion, device, scaler
            )
            g_supervised_loss = 0.0
        
        g_loss_history.append(g_loss)
        g_total_loss += g_loss * batch["before_img"].size(0)
        g_total_adv_loss += g_adv_loss * batch["before_img"].size(0)
        g_total_l1_loss += g_l1_loss * batch["before_img"].size(0)

    dataset_size = len(loader.dataset)
    return (
        g_total_loss / dataset_size,
        d_total_loss / dataset_size,
        g_total_adv_loss / dataset_size,
        g_total_l1_loss / dataset_size,
        g_total_supervised_loss / dataset_size if use_joint_training else 0.0
    )


@torch.no_grad()
def evaluate_cgan(generator, discriminator, loader, criterion, device, use_joint_training=True, joint_alpha=0.5, cfg=None):
    """cGANs评估函数：支持联合监督学习和对抗训练评估"""
    generator.eval()
    discriminator.eval()
    
    g_total_loss = 0.0
    d_total_loss = 0.0
    g_total_adv_loss = 0.0
    g_total_l1_loss = 0.0
    g_total_supervised_loss = 0.0

    # 创建监督损失函数（用于联合训练评估）
    # 使用组合损失（L1 + SSIM + 感知损失）作为监督损失，提高图像质量
    if use_joint_training:
        # 从配置文件读取联合训练损失设置
        if cfg and "trainer" in cfg:
            joint_losses = cfg["trainer"].get("joint_losses", ["l1", "ssim", "perceptual"])
            lambda_l1 = cfg["trainer"].get("joint_lambda_l1", 1.0)
            lambda_ssim = cfg["trainer"].get("joint_lambda_ssim", 2.0)
            lambda_perceptual = cfg["trainer"].get("joint_lambda_perceptual", 1.0)
            lambda_edge = cfg["trainer"].get("joint_lambda_edge", 0.0)
            lambda_texture = cfg["trainer"].get("joint_lambda_texture", 0.0)
            lambda_mask_fg = cfg["trainer"].get("joint_lambda_mask_fg", 0.0)
            lambda_mask_bg = cfg["trainer"].get("joint_lambda_mask_bg", 0.0)
        else:
            # 默认值
            joint_losses = ["l1", "ssim", "perceptual"]
            lambda_l1 = 1.0
            lambda_ssim = 2.0
            lambda_perceptual = 1.0
            lambda_edge = 0.0
            lambda_texture = 0.0
            lambda_mask_fg = 0.0
            lambda_mask_bg = 0.0

        # 创建组合损失
        supervised_criterion = CombinedLoss(
            lambda_l1 if "l1" in joint_losses else 0.0,
            lambda_ssim if "ssim" in joint_losses else 0.0,
            lambda_perceptual if "perceptual" in joint_losses else 0.0,
            lambda_edge if "edge" in joint_losses else 0.0,
            lambda_texture if "texture" in joint_losses else 0.0,
        )
    else:
        supervised_criterion = None

    for batch in tqdm(loader, desc="Val cGANs", leave=False):
        before_img = batch["before_img"].to(device)
        pattern_img = batch["pattern_img"].to(device)
        after_img = batch["after_img"].to(device)
        params = batch["targets"].to(device)
        
        param_map = create_param_map(params, before_img.size(2), before_img.size(3))
        
        # 生成器评估
        fake_after, pred_params = generator(before_img, pattern_img, param_map)
        before_img_01, pattern_img_01, after_img_01 = prepare_discriminator_inputs(before_img, pattern_img, after_img)
        fake_output = discriminator(fake_after, before_img_01, pattern_img_01, param_map)
        
        # 对抗损失评估
        # 处理多尺度判别器的输出
        if isinstance(fake_output, list):
            # 为每个尺度的输出创建对应的全1张量
            fake_target = [torch.ones_like(output) for output in fake_output]
        else:
            # 单尺度判别器
            fake_target = torch.ones_like(fake_output)
        g_loss_adv, g_adv_loss, g_l1_loss = criterion(
            fake_output, 
            fake_target,
            fake_after, 
            after_img_01
        )
        
        # 监督损失评估（如果使用联合训练）
        if use_joint_training and supervised_criterion is not None:
            supervised_loss, _, _, _ = supervised_criterion(fake_after, after_img_01)
            supervised_loss = supervised_loss + mask_guided_shape_loss(
                fake_after,
                after_img_01,
                before_img,
                pattern_img,
                lambda_fg=lambda_mask_fg,
                lambda_bg=lambda_mask_bg,
            )
        else:
            supervised_loss = 0.0
        
        # 参数预测损失评估
        param_loss = nn.MSELoss()(pred_params, params) if use_joint_training else 0.0
        
        # 总损失计算
        g_loss = g_loss_adv + (joint_alpha * supervised_loss if use_joint_training else 0.0) + 0.1 * param_loss  # 参数损失权重
        
        # 判别器评估
        # 由于数据集中的图像已经经过ImageNet归一化，需要反归一化到[0, 1]范围
        real_output = discriminator(after_img_01, before_img_01, pattern_img_01, param_map)
        # 处理多尺度判别器的输出
        if isinstance(real_output, list):
            # 为每个尺度的输出创建对应的全1张量
            real_target = [torch.ones_like(output) for output in real_output]
        else:
            # 单尺度判别器
            real_target = torch.ones_like(real_output)
        real_loss, _, _ = criterion(real_output, real_target)
        
        fake_output_val = discriminator(fake_after, before_img_01, pattern_img_01, param_map)
        # 处理多尺度判别器的输出
        if isinstance(fake_output_val, list):
            # 为每个尺度的输出创建对应的全0张量
            fake_target = [torch.zeros_like(output) for output in fake_output_val]
        else:
            # 单尺度判别器
            fake_target = torch.zeros_like(fake_output_val)
        fake_loss, _, _ = criterion(fake_output_val, fake_target)
        
        d_loss = (real_loss + fake_loss) / 2
        
        g_total_loss += g_loss.item() * before_img.size(0)
        d_total_loss += d_loss.item() * before_img.size(0)
        g_total_adv_loss += g_adv_loss.item() * before_img.size(0)
        g_total_l1_loss += g_l1_loss.item() * before_img.size(0)
        if use_joint_training:
            g_total_supervised_loss += supervised_loss.item() * before_img.size(0)

    dataset_size = len(loader.dataset)
    return (
        g_total_loss / dataset_size,
        d_total_loss / dataset_size,
        g_total_adv_loss / dataset_size,
        g_total_l1_loss / dataset_size,
        g_total_supervised_loss / dataset_size if use_joint_training else 0.0
    )


def analyze_hard_samples_forward(per_sample_losses, save_dir, top_k=20):
    """
    分析困难样本（损失最高的样本）
    
    Args:
        per_sample_losses: 每个样本的损失信息列表
        save_dir: 保存目录
        top_k: 显示前k个困难样本
    """
    if not per_sample_losses:
        print("没有样本损失数据可分析")
        return
    
    # 按损失排序
    sorted_samples = sorted(per_sample_losses, key=lambda x: x['loss'], reverse=True)
    
    print(f"\n前 {top_k} 个困难样本（损失最高）:")
    print("-" * 80)
    for i, sample in enumerate(sorted_samples[:top_k], 1):
        sample_id = sample.get('sample_id', sample.get('sample_idx', 'N/A'))
        pattern_id = sample.get('pattern_id', 'N/A')
        loss = sample['loss']
        print(f"{i:2d}. 样本ID: {sample_id:15s} | 样板ID: {pattern_id:15s} | 损失: {loss:.4f}")
    
    # 保存到文件
    import json
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "hard_samples_forward.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_samples[:top_k], f, ensure_ascii=False, indent=2)
    print(f"\n困难样本分析结果已保存至: {output_path}")


def compute_image_metrics(pred, target):
    """计算图像质量评估指标"""
    # 确保输入在[0, 1]范围内
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    
    # MSE (均方误差)
    mse = F.mse_loss(pred, target)
    
    # MAE (平均绝对误差)
    mae = F.l1_loss(pred, target)
    
    # PSNR (峰值信噪比)
    psnr = 20 * torch.log10(torch.tensor(1.0, device=pred.device) / torch.sqrt(mse.clamp_min(1e-12)))
    
    # SSIM (结构相似性)
    ssim = compute_ssim(pred, target)
    
    # R² (决定系数) - 优化计算方式
    # 按样本和通道分别计算，然后取平均，更准确反映模型拟合度
    r2 = compute_r2_score(pred, target)
    r2_global = compute_r2_score(pred, target, per_channel=False)
    edge_mae = F.l1_loss(sobel_edges(pred), sobel_edges(target))
    texture_mae = F.l1_loss(laplacian_highpass(pred), laplacian_highpass(target))

    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'psnr': psnr.item(),
        'ssim': ssim.item(),
        'r2': r2,
        'r2_global': r2_global,
        'edge_mae': edge_mae.item(),
        'texture_mae': texture_mae.item(),
    }


def compute_r2_score(pred, target, per_channel=True):
    """
    计算R²决定系数 - 优化版本
    
    Args:
        pred: 预测图像 (B, C, H, W)
        target: 目标图像 (B, C, H, W)
        per_channel: 是否按通道分别计算
    
    Returns:
        R²分数 (范围：-∞ 到 1，越接近1越好)
    """
    # 确保输入在[0, 1]范围内
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    
    if per_channel:
        # 按通道分别计算R²，然后取平均
        # 这样可以避免不同通道间的干扰
        batch_size, num_channels = pred.shape[0], pred.shape[1]
        r2_per_channel = []
        
        for c in range(num_channels):
            pred_c = pred[:, c, :, :]
            target_c = target[:, c, :, :]
            
            # 按样本计算R²
            r2_per_sample = []
            for b in range(batch_size):
                pred_sample = pred_c[b].flatten()
                target_sample = target_c[b].flatten()
                
                # 计算R²
                target_mean = target_sample.mean()
                ss_res = ((target_sample - pred_sample) ** 2).sum()
                ss_tot = ((target_sample - target_mean) ** 2).sum()
                
                # 处理特殊情况
                if ss_tot < 1e-8:
                    # 如果目标值几乎不变，R²定义为1（完美预测）
                    r2_sample = 1.0 if ss_res < 1e-8 else 0.0
                else:
                    r2_sample = 1 - ss_res / ss_tot
                
                r2_per_sample.append(r2_sample.item())
            
            # 该通道的平均R²
            r2_channel = np.mean(r2_per_sample)
            r2_per_channel.append(r2_channel)
        
        # 所有通道的平均R²
        r2 = np.mean(r2_per_channel)
    else:
        # 原始计算方式（整个批次）
        target_mean = target.mean()
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target_mean) ** 2).sum()
        
        if ss_tot < 1e-8:
            r2 = 1.0 if ss_res < 1e-8 else 0.0
        else:
            r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
    
    return r2


def select_forward_checkpoint_score(metrics, cfg):
    """Lower is better. MAE remains the default; visual mode favors edges/textures."""
    trainer_cfg = cfg.get("trainer", {})
    mode = trainer_cfg.get("best_model_metric", "mae")
    if mode == "mae":
        return float(metrics["mae"])
    if mode == "visual":
        weights = trainer_cfg.get("visual_score_weights", {})
        mae_w = float(weights.get("mae", 0.25))
        edge_w = float(weights.get("edge_mae", 0.35))
        texture_w = float(weights.get("texture_mae", 0.25))
        ssim_w = float(weights.get("ssim_loss", 0.15))
        return (
            mae_w * float(metrics["mae"])
            + edge_w * float(metrics.get("edge_mae", 0.0))
            + texture_w * float(metrics.get("texture_mae", 0.0))
            + ssim_w * (1.0 - float(metrics["ssim"]))
        )
    if mode not in metrics:
        raise ValueError(f"Unsupported best_model_metric: {mode}. Available metrics: {sorted(metrics.keys())}")
    value = metrics[mode]
    if value is None:
        raise ValueError(f"Metric {mode} is None and cannot be used for checkpoint selection.")
    return float(value)


@torch.no_grad()
def evaluate_image_metrics_fast(model, loader, device):
    """Compute real validation image metrics every epoch without LPIPS/FID/sample saving."""
    model.eval()
    all_metrics = []
    all_batch_sizes = []

    for batch in tqdm(loader, desc="Image metrics", leave=False):
        before_img = batch["before_img"].to(device)
        pattern_img = batch["pattern_img"].to(device)
        after_img = batch["after_img"].to(device)
        params = batch["targets"].to(device)

        param_map = create_param_map(params, before_img.size(2), before_img.size(3))
        pred_after, _ = model(before_img, pattern_img, param_map)
        target_after = denormalize_image(after_img, device)

        all_metrics.append(compute_image_metrics(pred_after, target_after))
        all_batch_sizes.append(before_img.size(0))

    weights = np.asarray(all_batch_sizes, dtype=np.float64)
    avg_mse = float(np.average([m["mse"] for m in all_metrics], weights=weights))
    return {
        "mse": avg_mse,
        "mae": float(np.average([m["mae"] for m in all_metrics], weights=weights)),
        "psnr": float(20 * np.log10(1.0 / np.sqrt(max(avg_mse, 1e-12)))),
        "ssim": float(np.average([m["ssim"] for m in all_metrics], weights=weights)),
        "r2": float(np.average([m["r2"] for m in all_metrics], weights=weights)),
        "r2_global": float(np.average([m["r2_global"] for m in all_metrics], weights=weights)),
        "edge_mae": float(np.average([m["edge_mae"] for m in all_metrics], weights=weights)),
        "texture_mae": float(np.average([m["texture_mae"] for m in all_metrics], weights=weights)),
        "lpips": None,
        "fid": None,
    }


def analyze_r2_score(pred, target, save_dir=None):
    """
    详细分析R²分数，帮助诊断模型问题
    
    Args:
        pred: 预测图像 (B, C, H, W)
        target: 目标图像 (B, C, H, W)
        save_dir: 保存分析结果的目录
    
    Returns:
        包含详细R²分析结果的字典
    """
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    
    batch_size, num_channels, height, width = pred.shape
    
    analysis = {
        'overall_r2': 0.0,
        'r2_per_channel': [],
        'r2_per_sample': [],
        'variance_per_channel': [],
        'mse_per_channel': [],
        'recommendations': []
    }
    
    # 计算每个通道的统计信息
    for c in range(num_channels):
        pred_c = pred[:, c, :, :]
        target_c = target[:, c, :, :]
        
        # 计算该通道的方差（反映信息量）
        target_variance = target_c.var().item()
        analysis['variance_per_channel'].append(target_variance)
        
        # 计算该通道的MSE
        mse_c = F.mse_loss(pred_c, target_c).item()
        analysis['mse_per_channel'].append(mse_c)
        
        # 计算该通道的R²
        r2_samples = []
        for b in range(batch_size):
            pred_flat = pred_c[b].flatten()
            target_flat = target_c[b].flatten()
            
            target_mean = target_flat.mean()
            ss_res = ((target_flat - pred_flat) ** 2).sum()
            ss_tot = ((target_flat - target_mean) ** 2).sum()
            
            if ss_tot < 1e-8:
                r2_s = 1.0 if ss_res < 1e-8 else 0.0
            else:
                r2_s = (1 - ss_res / ss_tot).item()
            
            r2_samples.append(r2_s)
        
        r2_channel = np.mean(r2_samples)
        analysis['r2_per_channel'].append(r2_channel)
        analysis['r2_per_sample'].extend(r2_samples)
    
    analysis['overall_r2'] = np.mean(analysis['r2_per_channel'])
    
    # 生成改进建议
    recommendations = []
    
    # 检查方差
    avg_variance = np.mean(analysis['variance_per_channel'])
    if avg_variance < 0.01:
        recommendations.append("目标图像方差过低，可能存在大量空白或纯色区域，建议检查数据质量")
    
    # 检查各通道R²差异
    r2_std = np.std(analysis['r2_per_channel'])
    if r2_std > 0.2:
        recommendations.append(f"各通道R²差异较大(std={r2_std:.3f})，建议检查颜色通道平衡")
    
    # 检查整体R²
    if analysis['overall_r2'] < 0.5:
        recommendations.append(f"R²过低({analysis['overall_r2']:.3f})，建议：")
        recommendations.append("  1. 增加训练轮数")
        recommendations.append("  2. 提高监督损失权重(joint_alpha)")
        recommendations.append("  3. 启用SSIM和感知损失")
        recommendations.append("  4. 检查数据预处理是否正确")
    elif analysis['overall_r2'] < 0.8:
        recommendations.append(f"R²中等({analysis['overall_r2']:.3f})，建议：")
        recommendations.append("  1. 继续训练更多轮数")
        recommendations.append("  2. 微调学习率")
    
    analysis['recommendations'] = recommendations
    
    # 打印分析结果
    print(f"\n{'='*60}")
    print("R²详细分析")
    print(f"{'='*60}")
    print(f"整体R²: {analysis['overall_r2']:.4f}")
    print(f"\n各通道R²:")
    channel_names = ['R', 'G', 'B']
    for i, (r2, var, mse) in enumerate(zip(analysis['r2_per_channel'], 
                                            analysis['variance_per_channel'],
                                            analysis['mse_per_channel'])):
        name = channel_names[i] if i < 3 else f'Ch{i}'
        print(f"  {name}通道: R²={r2:.4f}, 方差={var:.4f}, MSE={mse:.6f}")
    
    print(f"\n改进建议:")
    for rec in recommendations:
        print(f"  • {rec}")
    
    # 保存分析结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import json
        output_path = os.path.join(save_dir, "r2_analysis.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        print(f"\nR²分析结果已保存至: {output_path}")
    
    return analysis


class LPIPSMetric:
    """LPIPS感知相似度评估"""
    def __init__(self, device='cuda'):
        self.device = device
        self.loss_fn = None
        if LPIPS_AVAILABLE:
            try:
                self.loss_fn = lpips.LPIPS(net='alex').to(device)
                self.loss_fn.eval()
            except Exception as e:
                print(f"LPIPS初始化失败: {e}")
    
    def __call__(self, pred, target):
        """
        计算LPIPS距离
        Args:
            pred: 预测图像 (B, 3, H, W)，范围[0, 1]
            target: 目标图像 (B, 3, H, W)，范围[0, 1]
        Returns:
            LPIPS距离，范围[0, ∞)，越低越好
        """
        if self.loss_fn is None:
            return None
        
        # LPIPS期望输入范围[-1, 1]
        pred = pred * 2 - 1
        target = target * 2 - 1
        
        with torch.no_grad():
            distance = self.loss_fn(pred, target)
        
        return distance.mean().item()


class FIDMetric:
    """FID (Frechet Inception Distance) 评估"""
    def __init__(self, device='cuda'):
        self.device = device
        self.inception_model = None
        if FID_AVAILABLE:
            try:
                self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
                self.inception_model.eval()
                # 移除最后的分类层，获取2048维特征
                self.inception_model.fc = nn.Identity()
            except Exception as e:
                print(f"FID初始化失败: {e}")
    
    def get_features(self, images):
        """
        使用Inception网络提取特征
        Args:
            images: 图像张量 (B, 3, H, W)，范围[0, 1]
        Returns:
            特征向量 (B, 2048)
        """
        if self.inception_model is None:
            return None
        
        # Inception网络期望输入299x299
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 归一化到Inception网络的输入范围
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images = (images - mean) / std
        
        with torch.no_grad():
            features = self.inception_model(images)
        
        return features.cpu().numpy()
    
    def calculate_fid(self, real_features, fake_features):
        """
        计算FID分数
        Args:
            real_features: 真实图像特征 (N, 2048)
            fake_features: 生成图像特征 (N, 2048)
        Returns:
            FID分数，越低越好
        """
        if real_features is None or fake_features is None:
            return None
        
        # 计算均值和协方差
        mu_real = np.mean(real_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)
        
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # 计算FID
        diff = mu_real - mu_fake
        
        # 计算协方差矩阵的平方根
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
        
        return float(fid)


def compute_ssim(img1, img2, window_size=11, channel=3):
    """计算SSIM"""
    # 创建高斯窗口
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window.to(img1.device)
    
    window = create_window(window_size, channel)
    
    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    # SSIM计算
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


@torch.no_grad()
def comprehensive_evaluate(model, loader, device, save_dir="outputs", num_samples=10):
    """全面评估模型性能"""
    model.eval()
    
    all_metrics = []
    all_batch_sizes = []
    sample_count = 0
    
    # 初始化LPIPS和FID评估器
    lpips_metric = LPIPSMetric(device=device)
    fid_metric = FIDMetric(device=device)
    
    # 收集所有预测和真实图像用于FID计算
    all_real_features = []
    all_fake_features = []
    all_lpips_scores = []
    
    print(f"\n{'='*80}")
    print("全面评估模型性能")
    print(f"{'='*80}")
    
    for batch in tqdm(loader, desc="Comprehensive Eval", leave=False):
        before_img = batch["before_img"].to(device)
        pattern_img = batch["pattern_img"].to(device)
        after_img = batch["after_img"].to(device)
        params = batch["targets"].to(device)
        
        # 预测
        param_map = create_param_map(params, before_img.size(2), before_img.size(3))
        pred_after, pred_params = model(before_img, pattern_img, param_map)
        
        # 反归一化目标图像
        after_img_normalized = denormalize_image(after_img, device)
        
        # 计算图像质量指标
        batch_metrics = compute_image_metrics(pred_after, after_img_normalized)
        all_metrics.append(batch_metrics)
        all_batch_sizes.append(before_img.size(0))
        
        # 计算LPIPS
        if lpips_metric.loss_fn is not None:
            lpips_score = lpips_metric(pred_after, after_img_normalized)
            if lpips_score is not None:
                all_lpips_scores.append(lpips_score)
        
        # 提取FID特征
        if fid_metric.inception_model is not None:
            real_features = fid_metric.get_features(after_img_normalized)
            fake_features = fid_metric.get_features(pred_after)
            if real_features is not None and fake_features is not None:
                all_real_features.append(real_features)
                all_fake_features.append(fake_features)
        
        # 保存一些样本用于可视化
        if sample_count < num_samples:
            save_comparison_samples(
                before_img, pattern_img, pred_after, after_img_normalized,
                params, pred_params, save_dir, sample_count
            )
            sample_count += before_img.size(0)
    
    # 计算平均指标
    weights = np.asarray(all_batch_sizes, dtype=np.float64)
    avg_mse = float(np.average([m['mse'] for m in all_metrics], weights=weights))
    avg_metrics = {
        'mse': avg_mse,
        'mae': float(np.average([m['mae'] for m in all_metrics], weights=weights)),
        'psnr': float(20 * np.log10(1.0 / np.sqrt(max(avg_mse, 1e-12)))),
        'ssim': float(np.average([m['ssim'] for m in all_metrics], weights=weights)),
        'r2': float(np.average([m['r2'] for m in all_metrics], weights=weights)),
        'r2_global': float(np.average([m['r2_global'] for m in all_metrics], weights=weights)),
        'edge_mae': float(np.average([m['edge_mae'] for m in all_metrics], weights=weights)),
        'texture_mae': float(np.average([m['texture_mae'] for m in all_metrics], weights=weights)),
    }
    
    # 添加LPIPS指标
    if all_lpips_scores:
        avg_metrics['lpips'] = np.mean(all_lpips_scores)
    else:
        avg_metrics['lpips'] = None
    
    # 计算FID指标
    if all_real_features and all_fake_features:
        real_features_all = np.concatenate(all_real_features, axis=0)
        fake_features_all = np.concatenate(all_fake_features, axis=0)
        fid_score = fid_metric.calculate_fid(real_features_all, fake_features_all)
        avg_metrics['fid'] = fid_score
    else:
        avg_metrics['fid'] = None
    
    # 收集所有批次的预测和目标用于R²详细分析
    all_preds = []
    all_targets = []
    for batch in loader:
        before_img = batch["before_img"].to(device)
        pattern_img = batch["pattern_img"].to(device)
        after_img = batch["after_img"].to(device)
        params = batch["targets"].to(device)
        
        param_map = create_param_map(params, before_img.size(2), before_img.size(3))
        pred_after, _ = model(before_img, pattern_img, param_map)
        
        after_img_normalized = denormalize_image(after_img, device)
        
        all_preds.append(pred_after.cpu())
        all_targets.append(after_img_normalized.cpu())
        
        if len(all_preds) >= 5:  # 限制样本数量避免内存问题
            break
    
    if all_preds:
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        # 诊断用R²分析只抽取少量样本，不能覆盖主评估中的全验证集R²。
        r2_analysis = analyze_r2_score(all_preds_tensor, all_targets_tensor, save_dir)
        avg_metrics['r2_diagnostic_subset'] = r2_analysis['overall_r2']
    
    # 打印评估结果
    print(f"\n{'='*80}")
    print("图像质量评估结果")
    print(f"{'='*80}")
    print(f"MSE (均方误差):        {avg_metrics['mse']:.6f}  (越低越好，优秀<0.01)")
    print(f"MAE (平均绝对误差):    {avg_metrics['mae']:.6f}  (越低越好，优秀<0.05)")
    print(f"PSNR (峰值信噪比):     {avg_metrics['psnr']:.2f} dB  (越高越好，优秀>30dB)")
    print(f"SSIM (结构相似性):     {avg_metrics['ssim']:.4f}  (越高越好，优秀>0.9)")
    print(f"R² (决定系数):         {avg_metrics['r2']:.4f}  (越接近1越好，优秀>0.95)")
    
    # 打印LPIPS和FID指标
    if avg_metrics['lpips'] is not None:
        print(f"LPIPS (感知距离):      {avg_metrics['lpips']:.4f}  (越低越好，优秀<0.1)")
    else:
        print(f"LPIPS (感知距离):      未安装lpips库，使用 'pip install lpips' 安装")
    
    if avg_metrics['fid'] is not None:
        print(f"FID (分布距离):        {avg_metrics['fid']:.2f}   (越低越好，优秀<50)")
    else:
        print(f"FID (分布距离):        未安装scipy库，使用 'pip install scipy' 安装")
    
    # 评估等级
    print(f"\n{'='*80}")
    print("模型性能评估")
    print(f"{'='*80}")
    
    # 综合评估：考虑SSIM、PSNR、R²、LPIPS和FID
    ssim_score = avg_metrics['ssim']
    psnr_score = avg_metrics['psnr']
    r2_score = avg_metrics['r2']
    lpips_score = avg_metrics['lpips']
    fid_score = avg_metrics['fid']
    
    # 基础评估（基于SSIM、PSNR、R²）
    if ssim_score > 0.9 and psnr_score > 30 and r2_score > 0.95:
        base_grade = "优秀"
    elif ssim_score > 0.8 and psnr_score > 25 and r2_score > 0.90:
        base_grade = "良好"
    elif ssim_score > 0.7 and psnr_score > 20 and r2_score > 0.80:
        base_grade = "中等"
    else:
        base_grade = "较差"
    
    # 考虑LPIPS和FID的评估
    if lpips_score is not None and fid_score is not None:
        if lpips_score < 0.1 and fid_score < 50:
            grade = "优秀"
        elif lpips_score < 0.2 and fid_score < 100:
            grade = "良好"
        elif lpips_score < 0.3 and fid_score < 200:
            grade = "中等"
        else:
            grade = "较差"
        print(f"✓ 模型性能（基于感知质量）：{grade}")
    else:
        print(f"✓ 模型性能（基于像素级指标）：{base_grade}")
    
    # 各指标专项评估
    print(f"\n各指标评估：")
    
    # R²专项评估
    if r2_score > 0.95:
        print(f"  R²指标：优秀 ({r2_score:.4f}) - 模型拟合度很高")
    elif r2_score > 0.90:
        print(f"  R²指标：良好 ({r2_score:.4f}) - 模型拟合度较好")
    elif r2_score > 0.80:
        print(f"  R²指标：中等 ({r2_score:.4f}) - 模型拟合度一般")
    else:
        print(f"  R²指标：较差 ({r2_score:.4f}) - 模型拟合度不足，建议改进")
    
    # LPIPS专项评估
    if lpips_score is not None:
        if lpips_score < 0.1:
            print(f"  LPIPS指标：优秀 ({lpips_score:.4f}) - 感知质量很高")
        elif lpips_score < 0.2:
            print(f"  LPIPS指标：良好 ({lpips_score:.4f}) - 感知质量较好")
        elif lpips_score < 0.3:
            print(f"  LPIPS指标：中等 ({lpips_score:.4f}) - 感知质量一般")
        else:
            print(f"  LPIPS指标：较差 ({lpips_score:.4f}) - 感知质量不足，建议改进")
    
    # FID专项评估
    if fid_score is not None:
        if fid_score < 50:
            print(f"  FID指标：优秀 ({fid_score:.2f}) - 分布匹配度很高")
        elif fid_score < 100:
            print(f"  FID指标：良好 ({fid_score:.2f}) - 分布匹配度较好")
        elif fid_score < 200:
            print(f"  FID指标：中等 ({fid_score:.2f}) - 分布匹配度一般")
        else:
            print(f"  FID指标：较差 ({fid_score:.2f}) - 分布匹配度不足，建议改进")
    
    # 保存评估结果
    os.makedirs(save_dir, exist_ok=True)
    import json
    output_path = os.path.join(save_dir, "evaluation_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(avg_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存至: {output_path}")
    
    return avg_metrics


def save_comparison_samples(before_img, pattern_img, pred_after, real_after, params, pred_params, save_dir, idx):
    """保存对比样本"""
    # 反归一化用于显示
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(before_img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(before_img.device)
    
    before_img_denorm = before_img * std + mean
    pattern_img_denorm = pattern_img * std + mean
    
    # 转换为numpy
    before_np = before_img_denorm[0].cpu().numpy().transpose(1, 2, 0)
    pattern_np = pattern_img_denorm[0].cpu().numpy().transpose(1, 2, 0)
    pred_np = pred_after[0].cpu().numpy().transpose(1, 2, 0)
    real_np = real_after[0].cpu().numpy().transpose(1, 2, 0)
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(np.clip(before_np, 0, 1))
    axes[0, 0].set_title('洗前图', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.clip(pattern_np, 0, 1))
    axes[0, 1].set_title('样板图', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(np.clip(pred_np, 0, 1))
    axes[1, 0].set_title('预测图', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.clip(real_np, 0, 1))
    axes[1, 1].set_title('真实图', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"comparison_{idx:04d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存单独的预测图像
    pred_img_path = os.path.join(save_dir, f"pred_{idx:04d}.png")
    pred_img_uint8 = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
    pred_img_bgr = cv2.cvtColor(pred_img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(pred_img_path, pred_img_bgr)


def main(config_path: str = "configs/forward_model.yaml"):
    print(f"当前使用配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=== 前向模型训练（使用洗水后图片作为监督） ===")

    # 检查是否使用cGANs模式
    use_cgan = cfg.get("use_cgan", False)
    use_joint_training = cfg.get("use_joint_training", True)  # 默认启用联合训练
    joint_alpha = cfg.get("joint_alpha", 0.5)  # 监督损失权重
    
    print(f"训练模式: {'cGANs' if use_cgan else '标准'}")
    if use_cgan:
        print(f"联合训练: {'启用' if use_joint_training else '禁用'}")
        if use_joint_training:
            print(f"监督损失权重: {joint_alpha}")

    forward_transforms_cfg = dict(cfg.get("transforms") or {})
    # For supervised image generation, unsynchronized jitter/flip of only the
    # conditional images breaks alignment with the after-image target.
    forward_transforms_cfg.setdefault("augment", False)

    train_ds = LaserParamDatasetV2(
        annotation_path=cfg["data"]["train_manifest"],
        before_dir=cfg["data"]["before_dir"],
        after_dir=cfg["data"]["after_dir"],
        pattern_dir=cfg["data"]["pattern_dir"],
        pattern_manifest=cfg["data"].get("pattern_manifest"),
        label_stats_path=cfg["data"]["label_stats"],
        has_after=True,
        transforms_cfg=forward_transforms_cfg,
    )
    val_ds = LaserParamDatasetV2(
        annotation_path=cfg["data"]["val_manifest"],
        before_dir=cfg["data"]["before_dir"],
        after_dir=cfg["data"]["after_dir"],
        pattern_dir=cfg["data"]["pattern_dir"],
        pattern_manifest=cfg["data"].get("pattern_manifest"),
        label_stats_path=cfg["data"]["label_stats"],
        has_after=True,
        transforms_cfg=forward_transforms_cfg,
    )

    # 详细的数据统计信息
    print(f"训练集样本数: {len(train_ds)}")
    print(f"验证集样本数: {len(val_ds)}")
    print(f"参数列: {train_ds.param_cols}")
    
    # 样板图统计
    input_ablation_cfg = cfg.get("input_ablation", {}) or {}
    input_ablation_mode = str(input_ablation_cfg.get("mode", "full")).strip()
    if input_ablation_mode != "full":
        print(f"Forward input ablation mode: {input_ablation_mode}")
        train_ds = ForwardInputAblationDataset(train_ds, input_ablation_mode)
        val_ds = ForwardInputAblationDataset(val_ds, input_ablation_mode)
    else:
        print("Forward input ablation mode: full")

    if "pattern_id" in train_ds.df.columns:
        pattern_counts = train_ds.df["pattern_id"].value_counts()
        print(f"样板图种类数: {len(pattern_counts)}")
        print("样板图分布:")
        for pattern_id, count in pattern_counts.items():
            print(f"  样板图 {pattern_id}: {count} 个样本")
    
    # 测试数据形状以确保正确
    sample = train_ds[0]
    print(f"\n样本形状检查:")
    print(f"  before_img: {sample['before_img'].shape}")
    print(f"  pattern_img: {sample['pattern_img'].shape}")
    print(f"  after_img: {sample['after_img'].shape}" if "after_img" in sample else "  after_img: 未找到")
    print(f"  targets: {sample['targets'].shape}")
    print(f"  参数列数量: {len(train_ds.param_cols)}")

    # 检查是否使用自适应尺寸
    transforms_cfg = forward_transforms_cfg
    resize = transforms_cfg.get("resize", [164, 374])
    adaptive_size = resize is None or resize == "adaptive" or resize == "none"
    
    # 根据是否使用自适应尺寸选择collate_fn
    if adaptive_size:
        collate_fn = adaptive_collate_fn
        print("使用自适应尺寸模式，启用自适应批处理函数")
    else:
        collate_fn = None
        print(f"使用固定尺寸模式: {resize}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=True,
        num_workers=cfg["trainer"].get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,  # 避免训练最后一个batch size为1，导致BatchNorm错误
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=False,
        num_workers=cfg["trainer"].get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 根据模式选择模型和训练方式
    if use_cgan:
        # cGANs模式：生成器 + 判别器
        model_type = cfg["model"].get("type", "unet").lower()
        
        if model_type == "pix2pixhd":
            # 使用Pix2PixHD生成器
            generator = Pix2PixHD(
                num_params=len(train_ds.param_cols),
                use_local_enhancer=cfg["model"].get("use_local_enhancer", True),
                base_channels=cfg["model"].get("base_channels", 64),
                local_channels=cfg["model"].get("local_channels", 32),
                output_mode=cfg["model"].get("output_mode", "direct"),
                use_pattern_mask_channel=cfg["model"].get("use_pattern_mask_channel", False),
            ).to(device)
            
            # 使用多尺度判别器
            discriminator = MultiscaleDiscriminator(
                in_channels=3 + 3 + 3 + len(train_ds.param_cols),  # 洗后图(3) + 洗前图(3) + 样板图(3) + 参数图(num_params)
                base_channels=cfg["model"].get("base_channels", 64),
                num_scales=cfg["model"].get("num_scales", 3),
            ).to(device)
        else:
            # 默认使用UNet生成器
            generator = ForwardEffectUNet(
                num_params=len(train_ds.param_cols),
                base_channels=cfg["model"].get("base_channels", 64),
            ).to(device)
            
            # 默认使用标准判别器
            discriminator = Discriminator(
                num_params=len(train_ds.param_cols),
                base_channels=cfg["model"].get("base_channels", 64),
            ).to(device)

        # cGANs损失函数和优化器
        pretrained_generator_path = cfg["trainer"].get("pretrained_generator_path")
        if pretrained_generator_path:
            load_generator_warmstart(generator, pretrained_generator_path, device)

        cgan_lambda_l1 = 0.0 if use_joint_training else cfg["trainer"].get("lambda_l1", 100.0)
        criterion = cGANLoss(lambda_l1=cgan_lambda_l1)
        
        g_betas = (
            float(cfg["optimizer"].get("g_beta1", cfg["optimizer"].get("beta1", 0.5))),
            float(cfg["optimizer"].get("g_beta2", cfg["optimizer"].get("beta2", 0.999))),
        )
        d_betas = (
            float(cfg["optimizer"].get("d_beta1", cfg["optimizer"].get("beta1", 0.5))),
            float(cfg["optimizer"].get("d_beta2", cfg["optimizer"].get("beta2", 0.999))),
        )

        g_optimizer = torch.optim.AdamW(
            generator.parameters(),
            lr=float(cfg["optimizer"].get("g_lr", cfg["optimizer"]["lr"])),
            weight_decay=float(cfg["optimizer"]["weight_decay"]),
            betas=g_betas,
        )
        d_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(cfg["optimizer"].get("d_lr", cfg["optimizer"]["lr"])),
            weight_decay=float(cfg["optimizer"]["weight_decay"]),
            betas=d_betas,
        )
        
        # 使用ReduceLROnPlateau调度器，当验证损失停滞时降低学习率
        g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            g_optimizer, mode='min', factor=0.5, patience=5
        )
        d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            d_optimizer, mode='min', factor=0.5, patience=5
        )
        
        print(f"\n使用cGANs损失函数: {criterion.__class__.__name__}")
        print(f"生成器学习率: {cfg['optimizer'].get('g_lr', cfg['optimizer']['lr'])}")
        print(f"判别器学习率: {cfg['optimizer'].get('d_lr', cfg['optimizer']['lr'])}")
        print(f"L1损失权重: {cgan_lambda_l1}")
        
        # 在cGANs模式下，初始化标准模式的变量以避免错误
        model = generator  # 生成器可以用于标准推理
        optimizer = g_optimizer
        scheduler = g_scheduler
    else:
        # 标准模式：单一模型
        model_type = cfg["model"].get("type", "unet").lower()
        
        if model_type == "pix2pixhd":
            # 使用Pix2PixHD模型
            model = Pix2PixHD(
                num_params=len(train_ds.param_cols),
                use_local_enhancer=cfg["model"].get("use_local_enhancer", True),
                base_channels=cfg["model"].get("base_channels", 64),
                local_channels=cfg["model"].get("local_channels", 32),
                output_mode=cfg["model"].get("output_mode", "direct"),
                use_pattern_mask_channel=cfg["model"].get("use_pattern_mask_channel", False),
            ).to(device)
        else:
            # 默认使用UNet模型
            model = ForwardEffectUNet(
                num_params=len(train_ds.param_cols),
                base_channels=cfg["model"].get("base_channels", 64),
            ).to(device)

        # 使用增强的损失函数，更好地利用洗水后图片
        use_combined_loss = cfg["trainer"].get("use_combined_loss", True)
        if use_combined_loss:
            # 使用组合损失：L1 + SSIM + 感知损失
            lambda_l1 = cfg["trainer"].get("lambda_l1", 1.0)
            lambda_ssim = cfg["trainer"].get("lambda_ssim", 1.0)
            lambda_perceptual = cfg["trainer"].get("lambda_perceptual", 0.5)
            lambda_edge = cfg["trainer"].get("lambda_edge", 0.0)
            lambda_texture = cfg["trainer"].get("lambda_texture", 0.0)
            criterion = CombinedLoss(lambda_l1, lambda_ssim, lambda_perceptual, lambda_edge, lambda_texture)
            print(
                f"\n使用组合损失函数: L1({lambda_l1}) + SSIM({lambda_ssim}) + "
                f"Perceptual({lambda_perceptual}) + Edge({lambda_edge}) + Texture({lambda_texture})"
            )
        else:
            criterion = EnhancedL1Loss(alpha=0.1)
            print(f"\n使用增强损失函数: {criterion.__class__.__name__}")
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["optimizer"]["lr"]),
            weight_decay=float(cfg["optimizer"]["weight_decay"]),
            betas=(
                float(cfg["optimizer"].get("beta1", 0.9)),
                float(cfg["optimizer"].get("beta2", 0.999)),
            ),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["trainer"]["epochs"],
        )
        
        # 在标准模式下，初始化cGANs模式的变量以避免错误
        generator = model
        discriminator = None
        g_optimizer = optimizer
        d_optimizer = None
        g_scheduler = scheduler
        d_scheduler = None

    scaler = torch.amp.GradScaler('cuda', enabled=cfg["trainer"].get("amp", True) and device.type == "cuda")
    best_val_loss = float("inf")
    best_visual_quality_loss = float("inf")  # 新增：视觉质量评估
    best_eval_metrics = None
    best_eval_source = "None"
    last_full_eval_metrics = None
    last_full_eval_epoch = None
    
    # 早停机制
    early_stopping_patience = cfg["trainer"].get("early_stopping_patience", 10)
    early_stopping_counter = 0
    best_epoch = 0
    
    os.makedirs(cfg["trainer"]["save_dir"], exist_ok=True)
    
    # 训练日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(cfg["trainer"]["save_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.csv")
    summary_file = os.path.join(log_dir, f"run_summary_{timestamp}.txt")
    summary_json_file = os.path.join(log_dir, f"run_summary_{timestamp}.json")
    
    # 初始化日志文件
    with open(log_file, 'w', encoding='utf-8') as f:
        if use_cgan:
            f.write("epoch,train_g_loss,train_d_loss,train_adv_loss,train_l1_loss,train_supervised_loss,val_g_loss,val_d_loss,val_adv_loss,val_l1_loss,val_supervised_loss,psnr,ssim,mse,mae,r2,lr_g,lr_d\n")
        else:
            f.write("epoch,train_loss,train_l1,train_ssim,train_perceptual,val_loss,val_l1,val_ssim,val_perceptual,psnr,ssim,mse,mae,r2,lr\n")
    
    # 评估配置
    eval_freq = cfg["trainer"].get("eval_freq", 5)  # 每N个epoch评估一次
    save_samples = cfg["trainer"].get("save_samples", True)  # 是否保存样本图像
    num_eval_samples = cfg["trainer"].get("num_eval_samples", 10)  # 每次评估保存的样本数

    print(f"\n开始训练，共 {cfg['trainer']['epochs']} 个epoch...")
    print(f"评估频率: 每 {eval_freq} 个epoch评估一次")
    print(f"保存样本: {'启用' if save_samples else '禁用'}")
    print(f"早停耐心值: {early_stopping_patience} 个epoch")
    print(f"训练日志: {log_file}")
    
    for epoch in range(1, cfg["trainer"]["epochs"] + 1):
        print(f"\nEpoch [{epoch}/{cfg['trainer']['epochs']}]")

        if use_cgan:
            # cGANs训练（支持联合训练）
            # 从配置读取训练参数，改进训练稳定性
            d_train_freq = cfg["trainer"].get("d_train_freq", 3)  # 调整为更频繁的判别器训练
            label_smoothing = cfg["trainer"].get("label_smoothing", 0.2)  # 增加标签平滑以稳定训练
            adaptive_d_freq = cfg["trainer"].get("adaptive_d_freq", True)  # 启用自适应判别器训练频率
            
            g_train_loss, d_train_loss, g_adv_loss, g_l1_loss, g_supervised_loss = train_one_epoch_cgan(
                generator, discriminator, train_loader, g_optimizer, d_optimizer, criterion, device,
                use_joint_training=use_joint_training, joint_alpha=joint_alpha, 
                d_train_freq=d_train_freq, label_smoothing=label_smoothing,
                adaptive_d_freq=adaptive_d_freq, epoch=epoch, total_epochs=cfg["trainer"]["epochs"],
                cfg=cfg
            )
            
            g_val_loss, d_val_loss, g_val_adv_loss, g_val_l1_loss, g_val_supervised_loss = evaluate_cgan(
                generator, discriminator, val_loader, criterion, device,
                use_joint_training=use_joint_training, joint_alpha=joint_alpha, cfg=cfg
            )
            
            # 使用验证损失更新学习率调度器
            g_scheduler.step(g_val_loss)
            d_scheduler.step(d_val_loss)

            if use_joint_training:
                print(f"生成器 - 训练损失: {g_train_loss:.4f} (对抗: {g_adv_loss:.4f}, L1: {g_l1_loss:.4f}, 监督: {g_supervised_loss:.4f})")
                print(f"判别器 - 训练损失: {d_train_loss:.4f}")
                print(f"生成器 - 验证损失: {g_val_loss:.4f} (对抗: {g_val_adv_loss:.4f}, L1: {g_val_l1_loss:.4f}, 监督: {g_val_supervised_loss:.4f})")
                print(f"判别器 - 验证损失: {d_val_loss:.4f}")
            else:
                print(f"生成器 - 训练损失: {g_train_loss:.4f} (对抗: {g_adv_loss:.4f}, L1: {g_l1_loss:.4f})")
                print(f"判别器 - 训练损失: {d_train_loss:.4f}")
                print(f"生成器 - 验证损失: {g_val_loss:.4f} (对抗: {g_val_adv_loss:.4f}, L1: {g_val_l1_loss:.4f})")
                print(f"判别器 - 验证损失: {d_val_loss:.4f}")

            # 计算视觉质量损失（结合L1损失和对抗损失）
            avg_metrics = evaluate_image_metrics_fast(generator, val_loader, device)
            visual_quality_loss = select_forward_checkpoint_score(avg_metrics, cfg)
            
            # 定期进行全面评估
            did_full_eval = False
            if epoch % eval_freq == 0 or epoch == cfg["trainer"]["epochs"]:
                did_full_eval = True
                print(f"\n{'='*80}")
                print(f"全面评估 cGANs 模型 (Epoch {epoch})")
                print(f"{'='*80}")
                
                eval_save_dir = os.path.join(cfg["trainer"]["save_dir"], f"eval_epoch_{epoch}")
                full_eval_metrics = comprehensive_evaluate(
                    model=generator,
                    loader=val_loader,
                    device=device,
                    save_dir=eval_save_dir,
                    num_samples=num_eval_samples if save_samples else 0
                )
                avg_metrics.update(full_eval_metrics)
                last_full_eval_metrics = dict(avg_metrics)
                last_full_eval_epoch = epoch
                
                # 打印详细指标对比
                print(f"\n{'='*80}")
                print("cGANs 训练 vs 验证指标对比")
                print(f"{'='*80}")
                print(f"训练 L1: {g_l1_loss:.4f} | 验证 L1: {g_val_l1_loss:.4f} | 评估 MAE: {avg_metrics['mae']:.4f}")
                print(f"训练对抗损失: {g_adv_loss:.4f} | 验证对抗损失: {g_val_adv_loss:.4f}")
                print(
                    f"评估 SSIM: {avg_metrics['ssim']:.4f} | 评估 PSNR: {avg_metrics['psnr']:.2f} dB | "
                    f"Edge MAE: {avg_metrics.get('edge_mae', 0.0):.4f} | Texture MAE: {avg_metrics.get('texture_mae', 0.0):.4f}"
                )
                visual_quality_loss = select_forward_checkpoint_score(avg_metrics, cfg)
            # 记录训练日志
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{epoch},{g_train_loss:.6f},{d_train_loss:.6f},{g_adv_loss:.6f},{g_l1_loss:.6f},{g_supervised_loss:.6f},{g_val_loss:.6f},{d_val_loss:.6f},{g_val_adv_loss:.6f},{g_val_l1_loss:.6f},{g_val_supervised_loss:.6f},{avg_metrics['psnr']:.6f},{avg_metrics['ssim']:.6f},{avg_metrics['mse']:.6f},{avg_metrics['mae']:.6f},{avg_metrics['r2']:.6f},{g_optimizer.param_groups[0]['lr']:.6f},{d_optimizer.param_groups[0]['lr']:.6f}\n")
            
            # 双重保存策略：基于总损失和视觉质量
            if visual_quality_loss < best_val_loss:
                best_val_loss = visual_quality_loss
                best_epoch = epoch
                best_eval_metrics = dict(avg_metrics)
                best_eval_source = "full_eval" if did_full_eval else "fast_image_metrics"
                early_stopping_counter = 0  # 重置早停计数器
                save_checkpoint(
                    generator,
                    g_optimizer,
                    epoch,
                    cfg["trainer"]["save_dir"],
                    "best_forward_model_cgan.pth",
                )
                print(f">>> 保存最佳cGANs生成器模型（选择指标: {cfg['trainer'].get('best_model_metric', 'mae')}={visual_quality_loss:.6f}）")
            else:
                early_stopping_counter += 1
                print(f">>> 验证损失未改善，早停计数: {early_stopping_counter}/{early_stopping_patience}")
            
            # 额外保存视觉质量最好的模型
            if visual_quality_loss < best_visual_quality_loss:
                best_visual_quality_loss = visual_quality_loss
                save_checkpoint(
                    generator,
                    g_optimizer,
                    epoch,
                    cfg["trainer"]["save_dir"],
                    "best_visual_quality_cgan.pth",
                )
                print(">>> 保存视觉质量最佳cGANs生成器模型")
            
            # 早停检查
            if early_stopping_counter >= early_stopping_patience:
                print(f"\n{'='*80}")
                print(f"早停触发！验证损失连续 {early_stopping_patience} 个epoch未改善")
                print(f"最佳模型在第 {best_epoch} 个epoch")
                print(f"最佳验证损失: {best_val_loss:.4f}")
                print(f"{'='*80}")
                break
        else:
            # 标准训练
            train_loss, train_l1, train_ssim, train_perceptual = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler, 
                use_combined_loss=use_combined_loss
            )
            record_samples = (epoch == cfg["trainer"]["epochs"]) and cfg["trainer"].get("analyze_hard_samples", True)
            val_loss, val_l1, val_ssim, val_perceptual, per_sample_losses = evaluate(
                model, val_loader, criterion, device, record_per_sample=record_samples,
                use_combined_loss=use_combined_loss
            )
            scheduler.step()

            if use_combined_loss and isinstance(criterion, CombinedLoss):
                print(f"Train Loss: {train_loss:.4f} (L1: {train_l1:.4f}, SSIM: {train_ssim:.4f}, Perceptual: {train_perceptual:.4f})")
                print(f"Val Loss: {val_loss:.4f} (L1: {val_l1:.4f}, SSIM: {val_ssim:.4f}, Perceptual: {val_perceptual:.4f})")
            else:
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            avg_metrics = evaluate_image_metrics_fast(model, val_loader, device)
            visual_quality_loss = select_forward_checkpoint_score(avg_metrics, cfg)

            # 定期进行全面评估
            did_full_eval = False
            if epoch % eval_freq == 0 or epoch == cfg["trainer"]["epochs"]:
                did_full_eval = True
                print(f"\n{'='*80}")
                print(f"全面评估 (Epoch {epoch})")
                print(f"{'='*80}")
                
                eval_save_dir = os.path.join(cfg["trainer"]["save_dir"], f"eval_epoch_{epoch}")
                full_eval_metrics = comprehensive_evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    save_dir=eval_save_dir,
                    num_samples=num_eval_samples if save_samples else 0
                )
                avg_metrics.update(full_eval_metrics)
                last_full_eval_metrics = dict(avg_metrics)
                last_full_eval_epoch = epoch
                
                # 打印详细指标对比
                print(f"\n{'='*80}")
                print("训练 vs 验证指标对比")
                print(f"{'='*80}")
                print(f"训练 L1: {train_l1:.4f} | 验证 L1: {val_l1:.4f} | 评估 MAE: {avg_metrics['mae']:.4f}")
                print(f"训练 SSIM: {1-train_ssim:.4f} | 验证 SSIM: {1-val_ssim:.4f} | 评估 SSIM: {avg_metrics['ssim']:.4f}")
                print(f"训练 PSNR: {20*np.log10(1/np.sqrt(train_l1)):.2f} dB | 验证 PSNR: {20*np.log10(1/np.sqrt(val_l1)):.2f} dB | 评估 PSNR: {avg_metrics['psnr']:.2f} dB")
                print(f"Edge MAE: {avg_metrics.get('edge_mae', 0.0):.4f} | Texture MAE: {avg_metrics.get('texture_mae', 0.0):.4f}")
                visual_quality_loss = select_forward_checkpoint_score(avg_metrics, cfg)
            # 记录训练日志
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{epoch},{train_loss:.6f},{train_l1:.6f},{train_ssim:.6f},{train_perceptual:.6f},{val_loss:.6f},{val_l1:.6f},{val_ssim:.6f},{val_perceptual:.6f},{avg_metrics['psnr']:.6f},{avg_metrics['ssim']:.6f},{avg_metrics['mse']:.6f},{avg_metrics['mae']:.6f},{avg_metrics['r2']:.6f},{optimizer.param_groups[0]['lr']:.6f}\n")

            if visual_quality_loss < best_val_loss:
                best_val_loss = visual_quality_loss
                best_epoch = epoch
                best_eval_metrics = dict(avg_metrics)
                best_eval_source = "full_eval" if did_full_eval else "fast_image_metrics"
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    cfg["trainer"]["save_dir"],
                    "best_forward_model.pth",
                )
                print(">>> 保存最佳前向模型（使用洗水后图片监督）")

    # 困难样本分析（默认启用）
    if cfg["trainer"].get("analyze_hard_samples", True):
        print("\n开始分析困难样本...")
        # 在最后一个epoch重新评估以获取样本级损失
        _, _, _, _, per_sample_losses = evaluate(
            model, val_loader, criterion, device, record_per_sample=True,
            use_combined_loss=use_combined_loss if not use_cgan else False
        )
        if per_sample_losses:
            analyze_hard_samples_forward(
                per_sample_losses, 
                cfg["trainer"]["save_dir"], 
                top_k=cfg["trainer"].get("hard_samples_topk", 20)
            )
    
    print("\n训练完成！")
    if use_cgan:
        print("cGANs模型通过对抗训练学习到了更真实的洗后效果生成能力。")
        print("如需验证仿真效果，可加载 best_forward_model_cgan.pth 进行推理。")
    else:
        print("模型已学习到参数对洗水效果的影响，可以利用洗水后图片进行更准确的预测。")
        print("如需验证仿真效果，可加载 best_forward_model.pth 进行推理。")


    saved_figures = []
    try:
        model_name = "%s_%s" % ("cgan" if use_cgan else "supervised", cfg["model"].get("type", "unet"))
        figure_dir = os.path.join(log_dir, os.path.splitext(os.path.basename(log_file))[0] + "_figures")
        saved_figures = plot_forward_training_log(log_file, figure_dir, model_name=model_name)
        if saved_figures:
            print("训练曲线图已保存:")
            for path in saved_figures:
                print(f"  {path}")
    except Exception as exc:
        print(f"警告: 训练曲线绘制失败，不影响模型结果: {exc}")

    print_and_save_run_summary(
        config_path=config_path,
        log_file=log_file,
        summary_path=summary_file,
        summary_json_path=summary_json_file,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_eval_metrics=best_eval_metrics,
        best_eval_source=best_eval_source,
        final_eval_epoch=last_full_eval_epoch,
        final_eval_metrics=last_full_eval_metrics,
        mode_name="cGANs" if use_cgan else "supervised",
        saved_figures=saved_figures,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train second-system forward image-generation model.")
    parser.add_argument("--config", default="configs/forward_model.yaml", help="Path to forward-model YAML config.")
    args = parser.parse_args()
    main(args.config)

