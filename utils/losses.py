import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    """
    Huber Loss - 对异常值更鲁棒
    """
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = pred - target
        is_small = torch.abs(error) < self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        return torch.where(is_small, squared_loss, linear_loss).mean()


class CombinedLoss(nn.Module):
    """
    组合损失函数：
    - MSE Loss (主要)
    - MAE Loss (辅助，对异常值更鲁棒)
    - 可选的参数权重
    """
    def __init__(self, mse_weight=0.7, mae_weight=0.3, param_weights=None):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.param_weights = param_weights
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')
    
    def forward(self, pred, target):
        # 计算每个样本每个参数的损失
        mse = self.mse_loss(pred, target)  # (B, num_params)
        mae = self.mae_loss(pred, target)  # (B, num_params)
        
        # 应用参数权重（如果提供）
        if self.param_weights is not None:
            weights = torch.tensor(self.param_weights, device=pred.device, dtype=pred.dtype)
            mse = mse * weights
            mae = mae * weights
        
        # 对参数维度求平均，然后加权组合
        mse_mean = mse.mean(dim=1).mean()
        mae_mean = mae.mean(dim=1).mean()
        
        total_loss = self.mse_weight * mse_mean + self.mae_weight * mae_mean
        return total_loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss - 结合MSE和MAE的优点
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()


class FocalRegressionLoss(nn.Module):
    """
    Focal Loss for Regression - 关注难样本
    """
    def __init__(self, alpha=2.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target):
        mse = self.mse(pred, target)
        # 计算权重：误差越大，权重越大
        weights = (mse / (mse.mean() + 1e-8)) ** self.gamma
        focal_loss = self.alpha * weights * mse
        return focal_loss.mean()

