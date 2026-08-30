import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_metrics(y_true, y_pred):
    """
    y_true, y_pred: torch.Tensor 或 numpy.ndarray，形状 (N, num_params)
    返回: MSE / MAE / R2
    """
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "R2": float(r2),
    }


def calculate_ssim(img1, img2):
    """计算两幅图像的结构相似性指数"""
    if hasattr(img1, "detach"):
        img1 = img1.detach().cpu().numpy()
    if hasattr(img2, "detach"):
        img2 = img2.detach().cpu().numpy()
    
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    
    # 确保图像是二维或三维的
    if img1.ndim == 3:
        # 如果是彩色图像，转换为灰度图或计算多通道SSIM
        if img1.shape[0] == 3:  # CHW格式
            img1 = img1.transpose(1, 2, 0)
        if img2.shape[0] == 3:
            img2 = img2.transpose(1, 2, 0)
        
        # 计算多通道SSIM
        ssim_score = ssim(img1, img2, channel_axis=-1, data_range=1.0)
    else:
        # 单通道图像
        ssim_score = ssim(img1, img2, data_range=1.0)
    
    return float(ssim_score)


def calculate_psnr(img1, img2):
    """计算两幅图像的峰值信噪比"""
    if hasattr(img1, "detach"):
        img1 = img1.detach().cpu().numpy()
    if hasattr(img2, "detach"):
        img2 = img2.detach().cpu().numpy()
    
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    
    # 确保图像范围在0-1之间
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    # 计算PSNR
    psnr_score = psnr(img1, img2, data_range=1.0)
    
    return float(psnr_score)


def calculate_mse(img1, img2):
    """计算两幅图像的均方误差"""
    if hasattr(img1, "detach"):
        img1 = img1.detach().cpu().numpy()
    if hasattr(img2, "detach"):
        img2 = img2.detach().cpu().numpy()
    
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    
    # 计算MSE
    mse_score = np.mean((img1 - img2) ** 2)
    
    return float(mse_score)


