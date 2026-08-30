"""
多任务激光加工效果学习模型
整合：前向预测、参数预测、效果评估三个任务
改进版：使用预训练ViT作为特征提取器，改进池化策略，增强参数预测头
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .forward_unet import ForwardEffectUNet
from .dual_vit import DualViTRegressor, CrossAttentionFusion


class EnhancedL1Loss(nn.Module):
    """增强版L1：在像素L1基础上增加亮度/对比度相似性，与单任务前向模型保持一致"""
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = self.l1_loss(pred, target)

        # 亮度/对比度相似性（全局）
        pred_mean = pred.mean(dim=[1, 2, 3], keepdim=True)
        target_mean = target.mean(dim=[1, 2, 3], keepdim=True)
        pred_std = pred.std(dim=[1, 2, 3], keepdim=True)
        target_std = target.std(dim=[1, 2, 3], keepdim=True)

        contrast_sim = (2 * pred_std * target_std) / (pred_std ** 2 + target_std ** 2 + 1e-8)
        brightness_sim = (2 * pred_mean * target_mean) / (pred_mean ** 2 + target_mean ** 2 + 1e-8)

        contrast_sim = contrast_sim.mean()
        brightness_sim = brightness_sim.mean()

        # 总损失：L1 + α * (2 - 对比度 - 亮度)
        return l1 + self.alpha * (2 - contrast_sim - brightness_sim)

# 尝试导入timm
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class SharedFeatureExtractor(nn.Module):
    """共享特征提取器 - 使用预训练ResNet作为骨干网络提取洗前图和样板图的联合特征"""
    
    def __init__(self, base_channels: int = 64, backbone: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        
        self.base_channels = base_channels
        self.backbone_name = backbone
        
        # 使用预训练ResNet作为骨干网络
        try:
            from torchvision import models
            
            if backbone == 'resnet50':
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
                feat_dim = 2048
            elif backbone == 'resnet34':
                resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
                feat_dim = 512
            elif backbone == 'resnet18':
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
                feat_dim = 512
            else:
                raise ValueError(f"不支持的backbone: {backbone}，支持: resnet18, resnet34, resnet50")
            
            # 移除最后的全连接层和平均池化，保留到layer4（保留空间信息）
            # ResNet结构: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
            self.before_backbone = nn.Sequential(*list(resnet.children())[:-2])  # 保留到layer4
            self.pattern_backbone = nn.Sequential(*list(resnet.children())[:-2])  # 保留到layer4
            
            print(f"使用预训练{backbone}作为特征提取器，特征维度: {feat_dim}")
            
        except ImportError:
            print("警告: torchvision未安装，回退到简单卷积网络")
            # 回退到原始简单网络
            self.before_backbone = nn.Sequential(
                nn.Conv2d(3, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.GELU(),
                nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
                nn.BatchNorm2d(base_channels*2),
                nn.GELU(),
            )
            self.pattern_backbone = nn.Sequential(
                nn.Conv2d(3, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.GELU(),
                nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
                nn.BatchNorm2d(base_channels*2),
                nn.GELU(),
            )
            feat_dim = base_channels * 2
        
        # 特征融合层（降维并融合两个分支的特征）
        # ResNet50输出是2048维，两个分支拼接后是4096维
        fusion_dim = feat_dim * 2
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(fusion_dim, base_channels * 4, kernel_size=1),  # 1x1卷积降维
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            nn.Dropout2d(0.1),  # 添加空间Dropout，增强正则化
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
        )
        
        # 用于参数预测和质量评估的全局特征（池化）
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.shared_feature_dim = base_channels * 4  # 全局特征维度
        
    def forward(self, before_img: torch.Tensor, pattern_img: torch.Tensor) -> tuple:
        """
        Args:
            before_img: (B, 3, H, W) 洗前图
            pattern_img: (B, 3, H, W) 样板图
        Returns:
            shared_features_spatial: (B, base_channels*4, H', W') 空间特征（用于前向预测）
            shared_features_global: (B, base_channels*4) 全局特征（用于参数预测和质量评估）
        """
        # 使用ResNet提取特征（保留空间信息）
        # ResNet输出尺寸约为输入尺寸的1/32（经过5次下采样）
        before_feat = self.before_backbone(before_img)  # (B, feat_dim, H/32, W/32)
        pattern_feat = self.pattern_backbone(pattern_img)  # (B, feat_dim, H/32, W/32)
        
        # 拼接两个分支的特征
        fused_feat = torch.cat([before_feat, pattern_feat], dim=1)  # (B, feat_dim*2, H/32, W/32)
        
        # 融合特征并降维（保留空间信息）
        shared_feat_spatial = self.feature_fusion(fused_feat)  # (B, base_channels*4, H/32, W/32)
        
        # 全局特征（用于参数预测和质量评估）
        shared_feat_global = self.global_pool(shared_feat_spatial)  # (B, base_channels*4, 1, 1)
        shared_feat_global = shared_feat_global.view(shared_feat_global.size(0), -1)  # (B, base_channels*4)
        
        return shared_feat_spatial, shared_feat_global


class ForwardPredictionHead(nn.Module):
    """前向预测任务头 - 预测洗后效果图（使用UNet解码器保留空间信息）"""
    
    def __init__(self, num_params: int = 4, base_channels: int = 64):
        super().__init__()
        
        # 参数编码为空间特征图（通过广播）
        self.param_encoder = nn.Sequential(
            nn.Linear(num_params, base_channels),
            nn.GELU(),
            nn.Linear(base_channels, base_channels),
            nn.GELU(),
        )
        
        # UNet解码器（从共享空间特征生成洗后图）
        # 输入：共享空间特征 (base_channels*4) + 参数特征图 (base_channels)
        # ResNet输出尺寸约为输入尺寸的1/32，需要上采样到原始尺寸
        self.decoder = nn.Sequential(
            # 第一层：融合共享特征和参数特征
            nn.Conv2d(base_channels * 4 + base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            # 第一次上采样：H/32 -> H/16 (2倍)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            # 第二次上采样：H/16 -> H/8 (2倍)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            # 第三次上采样：H/8 -> H/4 (2倍)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.GELU(),
            # 第四次上采样：H/4 -> H/2 (2倍)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels // 2, base_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.GELU(),
            # 第五次上采样：H/2 -> H (2倍)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels // 4, base_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 8),
            nn.GELU(),
            # 输出层：生成RGB图像
            nn.Conv2d(base_channels // 8, 3, kernel_size=1),
        )
        
    def forward(self, shared_features_spatial: torch.Tensor, params: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """
        Args:
            shared_features_spatial: (B, base_channels*4, H', W') 共享空间特征（ResNet输出，尺寸约为输入的1/32）
            params: (B, num_params) 激光参数
            target_size: (H, W) 目标图像尺寸，默认None（使用decoder输出尺寸）
        Returns:
            pred_after: (B, 3, H, W) 预测洗后图
        """
        B, C, H, W = shared_features_spatial.shape
        
        # 编码参数为特征向量
        param_feat = self.param_encoder(params)  # (B, base_channels)
        
        # 将参数特征广播为空间特征图
        param_map = param_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # (B, base_channels, H', W')
        
        # 拼接共享特征和参数特征图
        combined_feat = torch.cat([shared_features_spatial, param_map], dim=1)  # (B, base_channels*4 + base_channels, H', W')
        
        # 解码为图像（经过5次上采样，从H/32恢复到接近H）
        pred_after = self.decoder(combined_feat)  # (B, 3, H_decoded, W_decoded)
        
        # 如果指定了目标尺寸且与解码后尺寸不一致，进行最终调整
        if target_size is not None and pred_after.shape[2:] != target_size:
            pred_after = F.interpolate(
                pred_after, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        return pred_after


class ParameterPredictionHead(nn.Module):
    """参数预测任务头 - 预测激光参数（增强正则化版本）"""
    
    def __init__(self, shared_feature_dim: int, num_params: int = 4, dropout_rate: float = 0.5):
        super().__init__()
        
        # 增强正则化：更深的网络但更强的Dropout
        self.regressor = nn.Sequential(
            # 第一层：输入层，更强的Dropout
            nn.Linear(shared_feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),  # 提高到0.5
            
            # 第二层：中间层
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.8),  # 0.4
            
            # 第三层：中间层（新增）
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.6),  # 0.3
            
            # 输出层：不使用Dropout
            nn.Linear(128, num_params),
        )
        
    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_features: (B, shared_feature_dim) 共享特征
        Returns:
            pred_params: (B, num_params) 预测激光参数
        """
        return self.regressor(shared_features)


class QualityAssessmentHead(nn.Module):
    """质量评估任务头 - 评估预测效果质量"""
    
    def __init__(self, shared_feature_dim: int, num_quality_metrics: int = 3):
        super().__init__()
        
        # 质量评估：SSIM、PSNR、MSE等指标
        self.quality_predictor = nn.Sequential(
            nn.Linear(shared_feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_quality_metrics),
            nn.Sigmoid(),  # 输出0-1之间的质量分数
        )
        
    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_features: (B, shared_feature_dim) 共享特征
        Returns:
            quality_scores: (B, num_quality_metrics) 质量评估分数
        """
        return self.quality_predictor(shared_features)


class MultiTaskLaserModel(nn.Module):
    """
    多任务激光加工效果学习模型
    整合三个核心任务：
    1. 前向预测：输入洗前图+样板图+参数 → 输出预测洗后图
    2. 参数预测：输入洗前图+样板图 → 输出激光参数
    3. 质量评估：评估预测效果的质量
    """
    
    def __init__(self, num_params: int = 4, base_channels: int = 64, 
                 backbone: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        
        self.num_params = num_params
        
        # 共享特征提取器（使用预训练ResNet）
        self.shared_extractor = SharedFeatureExtractor(
            base_channels=base_channels,
            backbone=backbone,
            pretrained=pretrained
        )
        
        # 任务特定头
        self.forward_head = ForwardPredictionHead(num_params, base_channels)
        self.param_head = ParameterPredictionHead(
            self.shared_extractor.shared_feature_dim, 
            num_params,
            dropout_rate=0.5  # 增强Dropout以缓解过拟合
        )
        self.quality_head = QualityAssessmentHead(
            self.shared_extractor.shared_feature_dim, num_quality_metrics=3
        )
        
        # 任务权重（可训练）
        self.task_weights = nn.Parameter(torch.ones(3))  # 三个任务的权重
        
    def forward(
        self, 
        before_img: torch.Tensor, 
        pattern_img: torch.Tensor,
        params: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            before_img: (B, 3, H, W) 洗前图
            pattern_img: (B, 3, H, W) 样板图
            params: (B, num_params) 激光参数（可选，用于前向预测）
            task_mask: (B, 3) 任务掩码，控制哪些任务需要计算
        Returns:
            outputs: 包含各任务输出的字典
        """
        # 提取共享特征（返回空间特征和全局特征）
        shared_features_spatial, shared_features_global = self.shared_extractor(before_img, pattern_img)
        
        # 获取目标图像尺寸（与输入图像相同）
        target_h, target_w = before_img.shape[2], before_img.shape[3]
        
        outputs = {}
        
        # 前向预测任务（使用空间特征）
        if params is not None:
            pred_after = self.forward_head(shared_features_spatial, params, target_size=(target_h, target_w))
            outputs['pred_after'] = pred_after
        
        # 参数预测任务（使用全局特征）
        pred_params = self.param_head(shared_features_global)
        outputs['pred_params'] = pred_params
        
        # 质量评估任务（使用全局特征）
        quality_scores = self.quality_head(shared_features_global)
        outputs['quality_scores'] = quality_scores
        
        # 任务权重
        outputs['task_weights'] = F.softmax(self.task_weights, dim=0)
        
        return outputs
    
    def predict_forward(self, before_img: torch.Tensor, pattern_img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """仅进行前向预测"""
        return self.forward(before_img, pattern_img, params)['pred_after']
    
    def predict_params(self, before_img: torch.Tensor, pattern_img: torch.Tensor) -> torch.Tensor:
        """仅进行参数预测"""
        return self.forward(before_img, pattern_img)['pred_params']
    
    def assess_quality(self, before_img: torch.Tensor, pattern_img: torch.Tensor) -> torch.Tensor:
        """仅进行质量评估"""
        return self.forward(before_img, pattern_img)['quality_scores']


class MultiTaskLoss(nn.Module):
    """多任务损失函数（加入物理先验知识约束）"""
    
    def __init__(self, task_weights: Optional[Dict[str, float]] = None, 
                 use_physics_constraint: bool = True,
                 physics_weight: float = 0.1):
        super().__init__()
        
        # 默认任务权重
        self.default_weights = {
            'forward': 1.0,    # 前向预测损失权重
            'param': 0.8,      # 参数预测损失权重  
            'quality': 0.5,    # 质量评估损失权重
        }
        
        if task_weights is not None:
            self.default_weights.update(task_weights)
        
        # 是否使用物理约束
        self.use_physics_constraint = use_physics_constraint
        self.physics_weight = physics_weight
        
        # 参数重要性权重（基于先验知识：脉宽和DPI影响最大）
        # 参数顺序：[频率, 脉宽, 速度, DPI]
        # 脉宽和DPI权重更高（2.0），频率和速度权重较低（1.0）
        self.param_importance_weights = torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float32)
        
        # 损失函数
        # 前向预测：使用增强版L1（亮度/对比度相似性），与单任务前向模型对齐
        self.forward_loss = EnhancedL1Loss(alpha=0.1)
        # 参数预测使用Huber Loss（对异常值更稳健，减少波动）
        self.param_loss = nn.SmoothL1Loss(reduction='none')  # 改为none以便加权
        self.quality_loss = nn.MSELoss() # 质量评估使用MSE损失
        
    def compute_physics_constraint_loss(
        self, 
        pred_params: torch.Tensor, 
        target_params: torch.Tensor,
        pred_after: torch.Tensor,
        target_after: torch.Tensor
    ) -> torch.Tensor:
        """
        计算物理约束损失，确保参数与图像亮度的一致性
        
        先验知识：
        1. 脉宽和DPI值越大，洗水效果越大（布颜色更浅，图像更亮）
        2. 频率和速度值越小，布颜色更浅（图像更亮）
        
        Args:
            pred_params: (B, 4) 预测参数 [频率, 脉宽, 速度, DPI]
            target_params: (B, 4) 真实参数 [频率, 脉宽, 速度, DPI]
            pred_after: (B, 3, H, W) 预测洗后图像
            target_after: (B, 3, H, W) 真实洗后图像
        Returns:
            physics_loss: 物理约束损失
        """
        B = pred_params.shape[0]
        
        # 计算图像平均亮度（转换为灰度后取平均）
        # 注意：图像可能已经归一化（ImageNet标准：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）
        # 使用RGB转灰度的标准公式：0.299*R + 0.587*G + 0.114*B
        pred_brightness = (0.299 * pred_after[:, 0] + 0.587 * pred_after[:, 1] + 0.114 * pred_after[:, 2]).mean(dim=[1, 2])  # (B,)
        target_brightness = (0.299 * target_after[:, 0] + 0.587 * target_after[:, 1] + 0.114 * target_after[:, 2]).mean(dim=[1, 2])  # (B,)
        
        # 归一化亮度到[0, 1]范围
        # 如果图像已归一化，亮度值可能在负值范围，需要先反归一化或使用相对亮度
        # 使用min-max归一化到[0, 1]范围（基于批次内的最小值和最大值）
        batch_min = torch.min(torch.cat([pred_brightness, target_brightness]))
        batch_max = torch.max(torch.cat([pred_brightness, target_brightness]))
        if batch_max > batch_min + 1e-6:
            pred_brightness = (pred_brightness - batch_min) / (batch_max - batch_min + 1e-6)
            target_brightness = (target_brightness - batch_min) / (batch_max - batch_min + 1e-6)
        else:
            # 如果所有亮度值相同，设为0.5
            pred_brightness = torch.ones_like(pred_brightness) * 0.5
            target_brightness = torch.ones_like(target_brightness) * 0.5
        
        # 根据先验知识计算期望的亮度影响
        # 参数顺序：[频率, 脉宽, 速度, DPI]
        # 频率和速度：值越小，亮度越大（负相关）
        # 脉宽和DPI：值越大，亮度越大（正相关）
        
        # 归一化参数到[0, 1]范围（假设参数已经归一化）
        # 如果参数未归一化，需要先归一化
        pred_freq = pred_params[:, 0]  # 频率
        pred_pulse = pred_params[:, 1]  # 脉宽
        pred_speed = pred_params[:, 2]  # 速度
        pred_dpi = pred_params[:, 3]  # DPI
        
        target_freq = target_params[:, 0]
        target_pulse = target_params[:, 1]
        target_speed = target_params[:, 2]
        target_dpi = target_params[:, 3]
        
        # 计算期望亮度（基于参数）
        # 使用线性组合：亮度 = -α*频率 - β*速度 + γ*脉宽 + δ*DPI
        # 这里使用简单的线性关系，权重可以调整
        # 注意：参数应该已经归一化到[0, 1]或接近的范围
        alpha, beta, gamma, delta = 0.3, 0.2, 0.4, 0.4  # 脉宽和DPI权重更高
        
        # 预测参数对应的期望亮度（线性组合）
        pred_expected_brightness_raw = (-alpha * pred_freq - beta * pred_speed + 
                                        gamma * pred_pulse + delta * pred_dpi)
        
        # 真实参数对应的期望亮度
        target_expected_brightness_raw = (-alpha * target_freq - beta * target_speed + 
                                         gamma * target_pulse + delta * target_dpi)
        
        # 归一化到[0, 1]范围（使用批次内的min-max归一化）
        all_expected = torch.cat([pred_expected_brightness_raw, target_expected_brightness_raw])
        exp_min = all_expected.min()
        exp_max = all_expected.max()
        if exp_max > exp_min + 1e-6:
            pred_expected_brightness = (pred_expected_brightness_raw - exp_min) / (exp_max - exp_min + 1e-6)
            target_expected_brightness = (target_expected_brightness_raw - exp_min) / (exp_max - exp_min + 1e-6)
        else:
            pred_expected_brightness = torch.ones_like(pred_expected_brightness_raw) * 0.5
            target_expected_brightness = torch.ones_like(target_expected_brightness_raw) * 0.5
        
        # 计算一致性损失：
        # 1. 预测亮度应该与预测参数一致
        # 2. 真实亮度应该与真实参数一致
        # 3. 预测参数和真实参数的差异应该反映在亮度差异上
        
        # 损失1：预测亮度与预测参数的一致性
        consistency_loss_1 = F.mse_loss(pred_brightness, pred_expected_brightness)
        
        # 损失2：真实亮度与真实参数的一致性（作为正则化项）
        consistency_loss_2 = F.mse_loss(target_brightness, target_expected_brightness)
        
        # 损失3：参数差异应该与亮度差异成正比
        param_diff = torch.abs(pred_params - target_params)
        brightness_diff = torch.abs(pred_brightness - target_brightness)
        
        # 计算参数差异的加权和（脉宽和DPI权重更高）
        weighted_param_diff = (param_diff * self.param_importance_weights.to(param_diff.device)).sum(dim=1)
        
        # 期望：参数差异越大，亮度差异应该越大
        # 使用相关性损失：如果参数差异大但亮度差异小，或反之，则惩罚
        correlation_loss = F.mse_loss(
            brightness_diff,
            weighted_param_diff * 0.5  # 缩放因子，可调整
        )
        
        # 总物理约束损失
        physics_loss = consistency_loss_1 + 0.5 * consistency_loss_2 + 0.3 * correlation_loss
        
        return physics_loss
        
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: 模型输出字典
            targets: 目标值字典
        Returns:
            loss_dict: 各任务损失和总损失
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 前向预测损失
        if 'pred_after' in outputs and 'target_after' in targets:
            forward_loss = self.forward_loss(outputs['pred_after'], targets['target_after'])
            loss_dict['forward_loss'] = forward_loss
            total_loss += self.default_weights['forward'] * forward_loss
        
        # 参数预测损失（加权版本）
        if 'pred_params' in outputs and 'target_params' in targets:
            pred_params = outputs['pred_params']
            target_params = targets['target_params']
            
            # 计算逐元素的损失（reduction='none'）
            param_loss_per_element = self.param_loss(pred_params, target_params)  # (B, 4)
            
            # 应用参数重要性权重
            param_weights = self.param_importance_weights.to(param_loss_per_element.device)
            weighted_param_loss = (param_loss_per_element * param_weights.unsqueeze(0)).mean()
            
            loss_dict['param_loss'] = weighted_param_loss
            total_loss += self.default_weights['param'] * weighted_param_loss
        
        # 质量评估损失
        if 'quality_scores' in outputs and 'target_quality' in targets:
            quality_loss = self.quality_loss(outputs['quality_scores'], targets['target_quality'])
            loss_dict['quality_loss'] = quality_loss
            total_loss += self.default_weights['quality'] * quality_loss
        
        # 物理约束损失（如果启用且同时有前向预测和参数预测）
        if self.use_physics_constraint and \
           'pred_after' in outputs and 'target_after' in targets and \
           'pred_params' in outputs and 'target_params' in targets:
            physics_loss = self.compute_physics_constraint_loss(
                outputs['pred_params'],
                targets['target_params'],
                outputs['pred_after'],
                targets['target_after']
            )
            loss_dict['physics_loss'] = physics_loss
            total_loss += self.physics_weight * physics_loss
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict