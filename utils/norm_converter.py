"""
前向模型与反向模型之间的归一化转换工具
解决：前向用Z-Score，反向用Min-Max [0,1]
"""
import torch
from typing import Dict, List, Tuple

# 反向模型 (laser_predict) 的参数范围（从其配置文件读取）
INVERSE_MODEL_RANGES = {
    'freq':   {'min': 20,    'max': 95},
    'pulse':  {'min': 25,    'max': 100},
    'speed':  {'min': 30000, 'max': 50000},
    'dpi':    {'min': 30,    'max': 175},
}

# 参数顺序：[频率, 脉宽, 速度, DPI]
PARAM_ORDER = ['freq', 'pulse', 'speed', 'dpi']


class NormalizationConverter:
    """归一化转换器：在Z-Score(前向模型)和Min-Max(反向模型)之间转换"""
    
    def __init__(self, forward_normalizer):
        """
        Args:
            forward_normalizer: 前向模型的 LabelNormalizer 实例
        """
        self.forward_normalizer = forward_normalizer
        self.device = torch.device('cpu')
        
        # 缓存Min-Max参数（转换为tensor，加速计算）
        self._update_device()
    
    def to(self, device):
        """切换设备"""
        self.device = device
        self._update_device()
        return self
    
    def _update_device(self):
        """更新设备相关的tensor缓存"""
        self.minmax_mins = torch.tensor(
            [INVERSE_MODEL_RANGES[p]['min'] for p in PARAM_ORDER],
            dtype=torch.float32, device=self.device
        )
        self.minmax_maxs = torch.tensor(
            [INVERSE_MODEL_RANGES[p]['max'] for p in PARAM_ORDER],
            dtype=torch.float32, device=self.device
        )
        self.minmax_ranges = self.minmax_maxs - self.minmax_mins
    
    def zscore_to_minmax(self, zscore_params: torch.Tensor) -> torch.Tensor:
        """
        前向模型参数 → 反向模型参数
        Z-Score → Min-Max [0, 1]
        
        Args:
            zscore_params: (B, 4) 前向模型输出/输入的Z-Score参数
            
        Returns:
            minmax_params: (B, 4) 反向模型期望的Min-Max参数
        """
        if zscore_params.device != self.device:
            self.to(zscore_params.device)
        
        # 1. Z-Score → 物理值
        physical = self.forward_normalizer.denormalize(zscore_params)
        
        # 2. 物理值 → Min-Max [0, 1]
        minmax = (physical - self.minmax_mins) / self.minmax_ranges
        minmax = torch.clamp(minmax, 0.0, 1.0)  # 确保在有效范围内
        
        return minmax
    
    def minmax_to_zscore(self, minmax_params: torch.Tensor) -> torch.Tensor:
        """
        反向模型参数 → 前向模型参数
        Min-Max [0, 1] → Z-Score
        
        Args:
            minmax_params: (B, 4) 反向模型输出/输入的Min-Max参数
            
        Returns:
            zscore_params: (B, 4) 前向模型期望的Z-Score参数
        """
        if minmax_params.device != self.device:
            self.to(minmax_params.device)
        
        # 1. Min-Max → 物理值
        physical = minmax_params * self.minmax_ranges + self.minmax_mins
        
        # 2. 物理值 → Z-Score
        zscore = self.forward_normalizer.normalize(physical)
        
        return zscore


class ImageFormatConverter:
    """图像格式转换器：前向模型输出 → 反向模型输入"""
    
    def __init__(self, inverse_model_size: int = 224):
        self.inverse_model_size = inverse_model_size
        
        # ImageNet归一化参数
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def register_buffer(self, name, tensor):
        """模拟nn.Module的buffer行为"""
        setattr(self, name, tensor)
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self
    
    def forward_to_inverse(self, forward_output: torch.Tensor) -> torch.Tensor:
        """
        前向模型输出 → 反向模型输入
        [0, 1]范围 + (B,3,164,374) → ImageNet归一化 + (B,3,224,224)
        
        Args:
            forward_output: (B, 3, H, W) 前向模型输出，范围 [0, 1] (Sigmoid)
            
        Returns:
            inverse_input: (B, 3, 224, 224) 反向模型期望的输入格式
        """
        import torch.nn.functional as F
        
        # 确保在正确设备上
        if forward_output.device != self.mean.device:
            self.to(forward_output.device)
        
        # 1. Resize到反向模型期望尺寸
        resized = F.interpolate(
            forward_output,
            size=(self.inverse_model_size, self.inverse_model_size),
            mode='bilinear',
            align_corners=False
        )
        
        # 2. [0, 1] → ImageNet归一化
        normalized = (resized - self.mean) / self.std
        
        return normalized
    
    def inverse_to_forward(self, inverse_input: torch.Tensor, 
                          target_size: Tuple[int, int] = (164, 374)) -> torch.Tensor:
        """
        反向模型输入 → 前向模型输入格式（双向闭环需要）
        """
        import torch.nn.functional as F
        
        if inverse_input.device != self.mean.device:
            self.to(inverse_input.device)
        
        # 1. ImageNet归一化 → [0, 1]
        denormalized = inverse_input * self.std + self.mean
        denormalized = torch.clamp(denormalized, 0.0, 1.0)
        
        # 2. Resize到前向模型期望尺寸
        resized = F.interpolate(
            denormalized,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        
        return resized


def create_batch_for_inverse(before_img: torch.Tensor, effect_img: torch.Tensor,
                             img_converter: ImageFormatConverter = None) -> Dict[str, torch.Tensor]:
    """
    为反向模型创建batch字典
    
    Args:
        before_img: (B, 3, H, W) 洗前图
        effect_img: (B, 3, H, W) 洗后图（可以是真实的或前向模型预测的）
        img_converter: 可选的图像格式转换器
    """
    if img_converter is not None:
        before_converted = img_converter.forward_to_inverse(before_img)
        effect_converted = img_converter.forward_to_inverse(effect_img)
    else:
        before_converted = before_img
        effect_converted = effect_img
        
    return {
        'before': before_converted,
        'effect': effect_converted
    }