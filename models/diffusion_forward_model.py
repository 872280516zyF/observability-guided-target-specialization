import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """获取时间步嵌入"""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:
        emb = torch.pad(emb, (0, 1), mode='constant')
    
    return emb


class TimeEmbedding(nn.Module):
    """时间嵌入模块"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 第一个线性变换
        self.time_proj1 = nn.Linear(input_dim, output_dim)
        self.time_proj2 = nn.Linear(output_dim, output_dim)
        
        # 激活函数
        self.activation = nn.SiLU()
    
    def forward(self, timestep_embed: torch.Tensor) -> torch.Tensor:
        # timestep_embed: [batch_size, input_dim]
        x = self.time_proj1(timestep_embed)
        x = self.activation(x)
        x = self.time_proj2(x)
        return x


class ParamEmbedding(nn.Module):
    """参数嵌入模块"""
    
    def __init__(self, num_params: int, embed_dim: int):
        super().__init__()
        self.num_params = num_params
        self.embed_dim = embed_dim
        
        # 参数嵌入
        self.param_proj = nn.Linear(num_params, embed_dim)
        self.activation = nn.SiLU()
        
        # 残差连接
        self.residual_proj = nn.Linear(num_params, embed_dim) if num_params != embed_dim else nn.Identity()
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        # params: [batch_size, num_params]
        x = self.param_proj(params)
        x = self.activation(x)
        
        # 残差连接
        residual = self.residual_proj(params)
        x = x + residual
        
        return x


class CrossAttention(nn.Module):
    """交叉注意力机制"""
    
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = query_dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 查询、键、值的投影
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        
        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
    
    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = query.shape
        
        # 计算查询、键、值
        q = self.to_q(query)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, -1).transpose(1, 2)
        
        # 计算注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = torch.matmul(attn, v)
        
        # 还原回原始形状
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        # 输出投影
        output = self.to_out(attn)
        
        return output


class ResnetBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.activation = nn.SiLU()
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
        # 下采样/上采样
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None, use_skip_connection: bool = True) -> torch.Tensor:
        if skip is not None and use_skip_connection:
            # 如果提供了跳跃连接，将其作为残差连接
            # 确保跳跃连接的通道数与输出通道数匹配
            if skip.shape[1] != self.out_channels:
                skip = skip[:, :self.out_channels, :, :]  # 截断到匹配的通道数
            
            # 如果skip的通道数与in_channels不匹配，使用简单的1x1卷积调整
            if skip.shape[1] != self.in_channels:
                # 创建临时卷积层，使用与输入相同的数据类型
                temp_conv = nn.Conv2d(skip.shape[1], self.out_channels, 1, dtype=x.dtype).to(x.device)
                skip = temp_conv(skip)
            else:
                skip = self.skip_conv(skip)
            
            h = self.conv1(x)
            h = self.norm1(h)
            h = self.activation(h)
            h = self.dropout(h)
            h = self.conv2(h)
            h = self.norm2(h)
            return h + skip
        elif not use_skip_connection:
            # 当use_skip_connection=False时，直接处理输入，不使用残差连接
            # 这是为了处理解码器中已拼接的特征图
            h = self.conv1(x)
            h = self.norm1(h)
            h = self.activation(h)
            h = self.dropout(h)
            h = self.conv2(h)
            h = self.norm2(h)
            return h
        else:
            # 使用原始输入进行残差连接
            skip = self.skip_conv(x)
            h = self.conv1(x)
            h = self.norm1(h)
            h = self.activation(h)
            h = self.dropout(h)
            h = self.conv2(h)
            h = self.norm2(h)
            return h + skip


class AttentionBlock(nn.Module):
    """注意力块"""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        
        # 层归一化
        self.norm = nn.LayerNorm(channels)
        
        # 多头注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 输出投影
        self.to_out = nn.Linear(channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        
        # 重塑为序列格式
        x_flat = x.view(batch_size, channels, -1).transpose(1, 2)  # [B, N, C]
        
        # 应用注意力
        x_flat = self.norm(x_flat)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = self.to_out(attn_out)
        
        # 还原为原始格式
        x = attn_out.transpose(1, 2).contiguous().view(batch_size, channels, height, width)
        
        return x


class DiffusionUNet(nn.Module):
    """扩散UNet网络"""
    
    def __init__(
        self,
        in_channels: int = 6,  # RGB图像 + 参数图
        base_channels: int = 128,
        channel_multiplier: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        dropout: float = 0.0,
        use_geometric: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_multiplier = channel_multiplier
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.use_geometric = use_geometric
        
        # 计算各层的通道数
        self.channels = [base_channels * m for m in channel_multiplier]
        
        # 输入投影层
        self.input_proj = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 编码器（下采样）
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for i, ch in enumerate(self.channels):
            out_ch = ch
            for j in range(num_res_blocks):
                block = ResnetBlock(in_ch, out_ch, dropout)
                self.down_blocks.append(block)
                
                if i in attention_resolutions:
                    attn_block = AttentionBlock(out_ch)
                    self.down_blocks.append(attn_block)
                
                in_ch = out_ch
            
            # 下采样
            if i < len(channel_multiplier) - 1:
                self.down_blocks.append(nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1))
        
        # 中间层
        self.middle_blocks = nn.ModuleList()
        # 残差块
        for _ in range(num_res_blocks):
            self.middle_blocks.append(ResnetBlock(in_ch, in_ch, dropout))
        
        # 注意力
        self.middle_attn = AttentionBlock(in_ch)
        
        # 另一个残差块
        self.middle_blocks.append(ResnetBlock(in_ch, in_ch, dropout))
        
        # 解码器（上采样）
        self.up_blocks = nn.ModuleList()
        in_ch = self.channels[-1]
        for i, ch in enumerate(reversed(self.channels)):
            out_ch = self.channels[-(i+1)] if i > 0 else base_channels
            
            for j in range(num_res_blocks + 1):
                # 在解码器中，ResnetBlock应该接收拼接后的通道数
                # 但skip_conv应该处理原始输入通道数
                block = ResnetBlock(in_ch, out_ch, dropout)
                self.up_blocks.append(block)
                
                if i < len(channel_multiplier) - 1 and j == num_res_blocks:
                    self.up_blocks.append(nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1))
                
                in_ch = out_ch
        
        # 输出层
        self.out_norm = nn.GroupNorm(32, base_channels)
        self.out_activation = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, 3, 3, padding=1)  # 输出RGB图像
        
        # 几何注意力（可选）
        if self.use_geometric:
            self.geometric_attn = nn.MultiheadAttention(embed_dim=base_channels, num_heads=8)
        
        # 参数条件嵌入
        self.param_embed_dim = base_channels
        self.param_embed = ParamEmbedding(num_params=4, embed_dim=self.param_embed_dim)
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # 参数嵌入
        param_embed = self.param_embed(params)  # [batch_size, embed_dim]
        
        # 时间嵌入
        timestep_embed = get_timestep_embedding(timestep, self.base_channels)  # [batch_size, base_channels]
        
        # 输入投影
        h = self.input_proj(x)
        
        # 存储跳跃连接
        skips = []
        
        # 编码器
        for block in self.down_blocks:
            if isinstance(block, ResnetBlock):
                h = block(h)
                skips.append(h)
            elif isinstance(block, AttentionBlock):
                h = block(h)
            elif isinstance(block, nn.Conv2d):
                h = block(h)
        
        # 中间层
        for block in self.middle_blocks:
            if isinstance(block, ResnetBlock):
                h = block(h)
            elif isinstance(block, AttentionBlock):
                h = block(h)
        
        # 解码器
        for i, block in enumerate(self.up_blocks):
            if isinstance(block, ResnetBlock):
                # 添加跳跃连接 - 作为skip参数传递
                if i // (self.num_res_blocks + 1) < len(skips):
                    skip_index = min(i // (self.num_res_blocks + 1), len(skips) - 1)
                    skip = skips[skip_index]
                    
                    # 确保跳跃连接和当前特征图的空间尺寸匹配
                    if h.shape[2:] != skip.shape[2:]:
                        skip = F.interpolate(skip, size=h.shape[2:], mode='bilinear', align_corners=False)
                    
                    # 传入主路径特征图和跳跃连接（启用skip连接）
                    h = block(h, skip=skip, use_skip_connection=True)
                else:
                    # 没有跳跃连接时，正常处理
                    h = block(h)
            elif isinstance(block, nn.ConvTranspose2d):
                h = block(h)
        
        # 输出
        h = self.out_norm(h)
        h = self.out_activation(h)
        output = self.out_conv(h)
        
        # 确保输出尺寸与输入尺寸匹配
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return output


class GaussianDDPM(nn.Module):
    """高斯扩散模型"""
    
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        model_mean_type: str = 'epsilon',
        model_var_type: str = 'fixed_small',
        loss_type: str = 'mse'
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        # 噪声调度
        if beta_schedule == 'cosine':
            betas = self.cosine_beta_schedule(timesteps)
        elif beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'sqrt':
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # 前向过程参数
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # 采样相关参数
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # 后验方差
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', 
                           torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', 
                           torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        
        # 损失类型
        if loss_type == 'mse':
            self.loss = F.mse_loss
        elif loss_type == 'l1':
            self.loss = F.l1_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """余弦Beta调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向扩散过程采样"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """从噪声预测初始图像"""
        return (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1) * x_t -
            self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1) * noise
        )
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """后验均值和方差"""
        posterior_mean = (
            self.posterior_mean_coef1[t].view(-1, 1, 1, 1) * x_start +
            self.posterior_mean_coef2[t].view(-1, 1, 1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1)
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, params: torch.Tensor, clip_denoised: bool = True):
        """模型预测的均值和方差"""
        # 预测噪声
        pred_noise = self.model(x, t, params)
        
        # 预测初始图像
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        
        if clip_denoised:
            x_start = torch.clamp(x_start, -1, 1)
        
        # 计算后验
        posterior_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_start, x, t)
        
        return posterior_mean, posterior_variance, posterior_log_variance, x_start
    
    def training_losses(self, x_start: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, params: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """训练损失"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 对目标图像进行扩散
        x_t = self.q_sample(x_start, t, noise=noise)
        
        # 将条件信息与扩散后的图像合并
        x_cond = torch.cat([x_t, condition], dim=1)  # [B, C_target + C_condition, H, W]
        
        # 处理参数：如果params是4D张量，取平均值池化
        if params.dim() == 4:  # [B, C, H, W]
            params_vec = params.mean(dim=[2, 3])  # [B, C]
        else:  # [B, C]
            params_vec = params
            
        # 预测噪声
        pred_noise = self.model(x_cond, t, params_vec)
        
        loss = self.loss(noise, pred_noise, reduction='none')
        loss = loss.mean(dim=[1, 2, 3])
        
        return loss
    
    def forward(self, before_img: torch.Tensor, pattern_img: torch.Tensor, params_map: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """前向传播 - 计算训练损失"""
        batch_size = before_img.size(0)
        device = before_img.device
        
        # 准备目标图像（洗后图像）
        x_start = target_img
        
        # 准备条件信息（洗前图像 + 样板图案）
        condition = torch.cat([before_img, pattern_img], dim=1)  # [B, 6, H, W]
        
        # 随机采样时间步
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        
        # 处理参数映射图：如果params_map是4D张量，取平均值池化
        if params_map.dim() == 4:  # [B, C, H, W]
            params_vec = params_map.mean(dim=[2, 3])  # [B, C]
        else:  # [B, C]
            params_vec = params_map
        
        # 计算损失
        loss = self.training_losses(x_start, t, condition, params_vec, noise=None)
        
        return loss.mean()
    
    @torch.no_grad()
    def sample(self, before_img: torch.Tensor, pattern_img: torch.Tensor, params_map: torch.Tensor, 
               shape, device: str, use_ddim: bool = False, ddim_steps: int = 20, **kwargs):
        """采样生成图像"""
        batch_size = shape[0]
        
        # 准备输入
        condition = torch.cat([pattern_img, params_map], dim=1)
        x = torch.cat([before_img, condition], dim=1)
        
        # 处理参数映射图：如果params_map是4D张量，取平均值池化
        if params_map.dim() == 4:  # [B, C, H, W]
            params_vec = params_map.mean(dim=[2, 3])  # [B, C]
        else:  # [B, C]
            params_vec = params_map
        
        if use_ddim:
            return self.ddim_sample(x, params_vec, shape, device, ddim_steps)
        else:
            return self.ddpm_sample(x, params_vec, shape, device)
    
    @torch.no_grad()
    def ddpm_sample(self, x: torch.Tensor, params: torch.Tensor, shape: Tuple, device: str):
        """DDPM采样"""
        x = torch.randn(shape, device=device)
        
        # 逐步去噪
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            # 预测均值和方差
            posterior_mean, posterior_variance, posterior_log_variance, x_start = self.p_mean_variance(x, t, params)
            
            if i == 0:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn_like(x)
            
            # 采样
            x = posterior_mean + torch.sqrt(posterior_variance) * noise
        
        return x
    
    @torch.no_grad()
    def ddim_sample(self, x: torch.Tensor, params: torch.Tensor, shape: Tuple, device: str, ddim_steps: int = 20):
        """DDIM采样（加速采样）"""
        device = x.device
        c = self.timesteps // ddim_steps
        
        x = torch.randn(shape, device=device)
        
        # 时间步序列
        seq = range(0, self.timesteps, c)
        
        x_start = None
        for i in reversed(seq):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            # 预测噪声
            pred_noise = self.model(x, t, params)
            
            # 预测初始图像
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            
            # 如果是最后一步，直接返回
            if i == 0:
                break
            
            # 下一个时间步
            next_t = torch.full((shape[0],), max(0, i - c), device=device, dtype=torch.long)
            
            # 计算下一步的图像
            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_next_t = self.alphas_cumprod[next_t]
            
            x = torch.sqrt(alpha_cumprod_next_t) * x_start + \
                torch.sqrt(1 - alpha_cumprod_next_t) * pred_noise
        
        return x_start


class DiffusionForwardModel(nn.Module):
    """扩散模型前向预测系统"""
    
    def __init__(
        self,
        num_params: int = 4,
        base_channels: int = 128,
        timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        model_mean_type: str = 'epsilon',
        model_var_type: str = 'fixed_small',
        loss_type: str = 'mse',
        use_geometric: bool = True
    ):
        super().__init__()
        
        # 计算UNet输入通道数 (目标图像3 + 条件信息6 = 9)
        # 条件信息 = 洗前图像3 + 样板图案3
        in_channels = 3 + 6  # target_image + (before_img + pattern_img)
        
        # 创建UNet模型
        unet_model = DiffusionUNet(
            in_channels=in_channels,  # 9个通道
            base_channels=base_channels,
            use_geometric=use_geometric
        )
        
        # 创建扩散模型
        self.diffusion = GaussianDDPM(
            model=unet_model,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=loss_type
        )
        
        # 后处理层
        self.postprocess = nn.Sequential(
            nn.Tanh()  # 归一化到[-1, 1]
        )
    
    def forward(self, before_img: torch.Tensor, pattern_img: torch.Tensor, params_map: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.diffusion.forward(before_img, pattern_img, params_map, target_img)
    
    @torch.no_grad()
    def sample(self, before_img: torch.Tensor, pattern_img: torch.Tensor, params_map: torch.Tensor, 
               shape, device: str, use_ddim: bool = False, ddim_steps: int = 20):
        """采样生成"""
        generated = self.diffusion.sample(
            before_img, pattern_img, params_map, shape, device, use_ddim, ddim_steps
        )
        
        # 后处理
        generated = self.postprocess(generated)
        
        return generated