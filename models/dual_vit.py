from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("警告: timm库未安装，将使用torchvision的ViT实现。建议安装timm: pip install timm")


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块：
    - Query来自洗前图片特征
    - Key-Value来自洗后图片特征
    - 实现跨模态信息交互
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query_feat: torch.Tensor, kv_feat: torch.Tensor):
        """
        Args:
            query_feat: (B, embed_dim) - 洗前图片特征
            kv_feat: (B, embed_dim) - 洗后图片特征
        Returns:
            fused: (B, embed_dim) - 融合后的特征
            attention_weights: (B,) - 平均注意力权重（用于可视化）
        """
        B = query_feat.size(0)
        
        # 投影到Q, K, V
        Q = self.q_proj(query_feat).view(B, self.num_heads, self.head_dim)  # (B, num_heads, head_dim)
        K = self.k_proj(kv_feat).view(B, self.num_heads, self.head_dim)       # (B, num_heads, head_dim)
        V = self.v_proj(kv_feat).view(B, self.num_heads, self.head_dim)       # (B, num_heads, head_dim)
        
        # 计算注意力分数: 对于全局特征，计算每个特征维度的相似度
        # Q: (B, num_heads, head_dim), K: (B, num_heads, head_dim)
        # 计算点积并缩放
        scores = (Q * K).sum(dim=-1) / (self.head_dim ** 0.5)  # (B, num_heads)
        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads)
        attn_weights = self.dropout(attn_weights)
        
        # 对每个头应用注意力权重
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # (B, num_heads, 1)
        weighted_V = V * attn_weights_expanded  # (B, num_heads, head_dim)
        
        # 将所有头的输出拼接起来，而不是求和
        weighted_V = weighted_V.contiguous().view(B, self.embed_dim)  # (B, embed_dim)
        
        # 投影并残差连接
        output = self.out_proj(weighted_V)  # (B, embed_dim)
        output = self.norm(output + query_feat)  # 残差连接
        
        return output, attn_weights.mean(dim=-1)  # 返回平均注意力权重用于可视化


class DualViTRegressor(nn.Module):
    """
    双分支 Vision Transformer 回归模型：
    - 分支1：洗水前图片 (ViT特征提取)
    - 分支2：洗水后图片 (ViT特征提取)
    - 交叉注意力融合：Query来自洗前，Key-Value来自洗后
    - 输出：连续激光参数（回归）
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_params: int = 4,
        use_cross_attention: bool = True,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        img_size: int = 224,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.use_cross_attention = use_cross_attention
        
        # 构建ViT骨干网络
        # 优先尝试使用timm，如果失败则降级到torchvision
        use_timm = TIMM_AVAILABLE
        if use_timm:
            try:
                self.before_backbone = self._build_timm_vit(model_name, pretrained, pretrained_path, img_size)
                self.after_backbone = self._build_timm_vit(model_name, pretrained, pretrained_path, img_size)
                embed_dim = self.before_backbone.embed_dim
                print(f"[DualViT] 使用timm库构建ViT模型，embed_dim={embed_dim}")
            except Exception as e:
                print(f"[DualViT] timm构建失败: {e}")
                print(f"[DualViT] 尝试使用torchvision的ViT作为备选方案")
                use_timm = False
        
        if not use_timm:
            # 使用torchvision的ViT
            self.before_backbone, embed_dim = self._build_torchvision_vit(pretrained, img_size)
            self.after_backbone, _ = self._build_torchvision_vit(pretrained, img_size)
            print(f"[DualViT] 使用torchvision构建ViT模型，embed_dim={embed_dim}")
        
        # 特征融合模块
        if use_cross_attention:
            self.fusion = CrossAttentionFusion(embed_dim, num_heads=8, dropout=0.1)
            fused_dim = embed_dim
        else:
            self.fusion = None
            fused_dim = embed_dim * 2
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_params),
        )
    
    def _build_timm_vit(self, model_name: str, pretrained: bool, pretrained_path: Optional[str], img_size: int):
        """使用timm库构建ViT模型"""
        if pretrained_path:
            # 从本地路径加载预训练权重
            print(f"[DualViT] 从本地路径加载预训练权重: {pretrained_path}")
            model = timm.create_model(
                model_name,
                pretrained=False,  # 设置为False，避免网络下载
                img_size=img_size,
                num_classes=0,  # 不使用分类头
            )
            state_dict = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"[DualViT] 已从本地路径加载预训练权重: {pretrained_path}")
        elif pretrained:
            try:
                print(f"[DualViT] 尝试从网络下载预训练权重: {model_name}")
                model = timm.create_model(
                    model_name,
                    pretrained=True,
                    img_size=img_size,
                    num_classes=0,  # 不使用分类头
                )
                print(f"[DualViT] 成功加载预训练权重")
            except Exception as e:
                print(f"[DualViT] 警告: 无法从网络下载预训练权重: {e}")
                print(f"[DualViT] 将使用随机初始化的权重继续训练")
                model = timm.create_model(
                    model_name,
                    pretrained=False,
                    img_size=img_size,
                    num_classes=0,
                )
        else:
            model = timm.create_model(
                model_name,
                pretrained=False,
                img_size=img_size,
                num_classes=0,
            )
            print(f"[DualViT] 使用随机初始化的权重")
        return model
    
    def _build_torchvision_vit(self, pretrained: bool, img_size: int):
        """使用torchvision构建ViT模型（备用方案）"""
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            
            if pretrained:
                try:
                    print(f"[DualViT] 尝试从torchvision加载预训练权重")
                    weights = ViT_B_16_Weights.IMAGENET1K_V1
                    model = vit_b_16(weights=weights)
                    print(f"[DualViT] 成功加载torchvision预训练权重")
                except Exception as e:
                    print(f"[DualViT] 警告: 无法加载torchvision预训练权重: {e}")
                    print(f"[DualViT] 将使用随机初始化的权重")
                    model = vit_b_16(weights=None)
            else:
                model = vit_b_16(weights=None)
                print(f"[DualViT] 使用随机初始化的torchvision ViT权重")
            
            # 移除分类头，只保留特征提取部分
            model.heads = nn.Identity()
            
            # torchvision ViT的embed_dim是768
            embed_dim = 768
            
            return model, embed_dim
        except ImportError:
            raise ImportError("需要torchvision >= 0.13.0 或安装timm库")
    
    def forward(self, before_img: torch.Tensor, after_img: torch.Tensor):
        """
        Args:
            before_img: (B, 3, H, W) - 洗前图片
            after_img: (B, 3, H, W) - 洗后图片
        Returns:
            preds: (B, num_params) - 预测的激光参数
            attn: 注意力权重（用于可视化）
        """
        # 提取特征
        # 判断是否使用timm（通过检查是否有forward_features方法）
        if hasattr(self.before_backbone, 'forward_features'):
            # timm模型
            feat_before = self.before_backbone.forward_features(before_img)
            feat_after = self.after_backbone.forward_features(after_img)
            
            # timm返回的可能是(B, num_patches+1, embed_dim)或(B, embed_dim)
            if len(feat_before.shape) == 3:
                # 如果是3D，取CLS token (第一个token)
                feat_before = feat_before[:, 0]  # (B, embed_dim)
                feat_after = feat_after[:, 0]    # (B, embed_dim)
            # 如果已经是2D，直接使用
        else:
            # torchvision ViT
            feat_before = self.before_backbone(before_img)  # (B, embed_dim)
            feat_after = self.after_backbone(after_img)     # (B, embed_dim)
        
        # 特征融合
        if self.use_cross_attention:
            fused, attn = self.fusion(feat_before, feat_after)
        else:
            fused = torch.cat([feat_before, feat_after], dim=1)
            attn = None
        
        # 回归预测
        preds = self.regressor(fused)
        
        return preds, attn

