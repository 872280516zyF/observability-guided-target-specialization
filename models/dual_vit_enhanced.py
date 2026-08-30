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


class EnhancedCrossAttentionFusion(nn.Module):
    """
    增强版交叉注意力融合模块：
    - 多层注意力机制
    - 残差连接
    - 门控机制
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_layers = num_layers
        
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        # 多层交叉注意力
        self.attention_layers = nn.ModuleList([
            self._build_attention_layer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # 最终融合
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def _build_attention_layer(self, embed_dim, num_heads, dropout):
        return nn.ModuleDict({
            'q_proj': nn.Linear(embed_dim, embed_dim),
            'k_proj': nn.Linear(embed_dim, embed_dim),
            'v_proj': nn.Linear(embed_dim, embed_dim),
            'out_proj': nn.Linear(embed_dim, embed_dim),
            'norm1': nn.LayerNorm(embed_dim),
            'norm2': nn.LayerNorm(embed_dim),
            'ffn': nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            )
        })
    
    def forward(self, query_feat: torch.Tensor, kv_feat: torch.Tensor):
        """
        Args:
            query_feat: (B, embed_dim) - 洗前图片特征
            kv_feat: (B, embed_dim) - 洗后图片特征
        Returns:
            fused: (B, embed_dim) - 融合后的特征
            attention_weights: (B,) - 平均注意力权重
        """
        B = query_feat.size(0)
        x = query_feat
        kv = kv_feat
        
        attn_weights_list = []
        
        # 多层交叉注意力
        for layer in self.attention_layers:
            # 交叉注意力
            Q = layer['q_proj'](x).view(B, self.num_heads, self.head_dim)
            K = layer['k_proj'](kv).view(B, self.num_heads, self.head_dim)
            V = layer['v_proj'](kv).view(B, self.num_heads, self.head_dim)
            
            scores = (Q * K).sum(dim=-1) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights_list.append(attn_weights.mean(dim=-1))
            
            attn_weights_expanded = attn_weights.unsqueeze(-1)
            weighted_V = (V * attn_weights_expanded).view(B, self.embed_dim)
            
            attn_out = layer['out_proj'](weighted_V)
            x = layer['norm1'](x + attn_out)  # 残差连接
            
            # 前馈网络
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)  # 残差连接
        
        # 门控融合
        concat_feat = torch.cat([x, kv_feat], dim=1)
        gate_weights = self.gate(concat_feat)
        gated_x = x * gate_weights
        gated_kv = kv_feat * (1 - gate_weights)
        
        # 最终融合
        final_concat = torch.cat([gated_x, gated_kv], dim=1)
        fused = self.fusion_proj(final_concat)
        
        avg_attn = torch.stack(attn_weights_list).mean(dim=0)
        return fused, avg_attn


class DualViTEnhanced(nn.Module):
    """
    增强版双分支 Vision Transformer 回归模型：
    - 多层交叉注意力融合
    - 更深的回归头
    - 残差连接和批归一化
    - Dropout正则化
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_params: int = 4,
        use_cross_attention: bool = True,
        pretrained: bool = False,
        pretrained_path: Optional[str] = None,
        img_size: int = 224,
        hidden_dims: list = [1024, 512, 256],
        dropout: float = 0.3,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.use_cross_attention = use_cross_attention
        
        # 构建ViT骨干网络
        use_timm = TIMM_AVAILABLE
        if use_timm:
            try:
                self.before_backbone = self._build_timm_vit(model_name, pretrained, pretrained_path, img_size)
                self.after_backbone = self._build_timm_vit(model_name, pretrained, pretrained_path, img_size)
                embed_dim = self.before_backbone.embed_dim
                print(f"[DualViTEnhanced] 使用timm库构建ViT模型，embed_dim={embed_dim}")
            except Exception as e:
                print(f"[DualViTEnhanced] timm构建失败: {e}")
                print(f"[DualViTEnhanced] 尝试使用torchvision的ViT作为备选方案")
                use_timm = False
        
        if not use_timm:
            self.before_backbone, embed_dim = self._build_torchvision_vit(pretrained, img_size)
            self.after_backbone, _ = self._build_torchvision_vit(pretrained, img_size)
            print(f"[DualViTEnhanced] 使用torchvision构建ViT模型，embed_dim={embed_dim}")
        
        # 特征融合模块
        if use_cross_attention:
            self.fusion = EnhancedCrossAttentionFusion(
                embed_dim, 
                num_heads=8, 
                dropout=0.1, 
                num_layers=2
            )
            fused_dim = embed_dim
        else:
            self.fusion = None
            fused_dim = embed_dim * 2
        
        # 增强的回归头 - 更深的网络
        layers = []
        input_dim = fused_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.5),
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, num_params))
        
        self.regressor = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def _build_timm_vit(self, model_name: str, pretrained: bool, pretrained_path: Optional[str], img_size: int):
        """使用timm库构建ViT模型"""
        if pretrained_path:
            model = timm.create_model(
                model_name,
                pretrained=False,
                img_size=img_size,
                num_classes=0,
            )
            state_dict = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"[DualViTEnhanced] 已从本地路径加载预训练权重: {pretrained_path}")
        elif pretrained:
            try:
                print(f"[DualViTEnhanced] 尝试从网络下载预训练权重: {model_name}")
                model = timm.create_model(
                    model_name,
                    pretrained=True,
                    img_size=img_size,
                    num_classes=0,
                )
                print(f"[DualViTEnhanced] 成功加载预训练权重")
            except Exception as e:
                print(f"[DualViTEnhanced] 警告: 无法从网络下载预训练权重: {e}")
                print(f"[DualViTEnhanced] 将使用随机初始化的权重继续训练")
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
            print(f"[DualViTEnhanced] 使用随机初始化的权重")
        return model
    
    def _build_torchvision_vit(self, pretrained: bool, img_size: int):
        """使用torchvision构建ViT模型（备用方案）"""
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            
            if pretrained:
                try:
                    print(f"[DualViTEnhanced] 尝试从torchvision加载预训练权重")
                    weights = ViT_B_16_Weights.IMAGENET1K_V1
                    model = vit_b_16(weights=weights)
                    print(f"[DualViTEnhanced] 成功加载torchvision预训练权重")
                except Exception as e:
                    print(f"[DualViTEnhanced] 警告: 无法加载torchvision预训练权重: {e}")
                    print(f"[DualViTEnhanced] 将使用随机初始化的权重")
                    model = vit_b_16(weights=None)
            else:
                model = vit_b_16(weights=None)
                print(f"[DualViTEnhanced] 使用随机初始化的torchvision ViT权重")
            
            model.heads = nn.Identity()
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
        if hasattr(self.before_backbone, 'forward_features'):
            feat_before = self.before_backbone.forward_features(before_img)
            feat_after = self.after_backbone.forward_features(after_img)
            
            if len(feat_before.shape) == 3:
                feat_before = feat_before[:, 0]
                feat_after = feat_after[:, 0]
        else:
            feat_before = self.before_backbone(before_img)
            feat_after = self.after_backbone(after_img)
        
        # 特征融合
        if self.use_cross_attention:
            fused, attn = self.fusion(feat_before, feat_after)
        else:
            fused = torch.cat([feat_before, feat_after], dim=1)
            attn = None
        
        # 回归预测
        preds = self.regressor(fused)
        
        return preds, attn

