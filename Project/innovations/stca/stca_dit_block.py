"""
STCADiTBlock - 时空因果注意力DiT块
===================================

架构级创新：将STCA集成到DiT模型架构中，推理时也使用

与原DiTBlock的区别：
1. 将单一Self-Attention拆分为Temporal + Spatial两阶段
2. 时间注意力使用因果掩码（t只能attend到<=t）
3. 空间注意力使用窗口机制（利用3D局部相关性）
4. 推理时同样生效，确保生成的时序一致性

论文贡献：
- 模型架构级创新，而非仅训练技巧
- 推理时强制因果约束，生成更符合物理规律的序列
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from timm.models.vision_transformer import Mlp


def modulate(x, shift, scale):
    """adaLN调制函数"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TemporalCausalAttentionDiT(nn.Module):
    """
    时间因果注意力（DiT版本）

    关键创新：
    - 因果掩码确保t时刻只能看到<=t的信息
    - 支持adaLN-Zero条件化
    - 推理时同样强制因果约束
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 缓存因果掩码
        self._causal_mask_cache = {}

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成因果掩码（下三角矩阵），带缓存"""
        cache_key = (seq_len, device)
        if cache_key not in self._causal_mask_cache:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            mask = mask.masked_fill(mask == 0, float('-inf'))
            mask = mask.masked_fill(mask == 1, 0.0)
            self._causal_mask_cache[cache_key] = mask
        return self._causal_mask_cache[cache_key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) 时序特征
        Returns:
            (B, T, C) 因果注意力输出
        """
        B, T, C = x.shape

        # QKV投影
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算（优先使用SDPA/Flash/Efficient，避免显式分配attn矩阵）
        if hasattr(F, "scaled_dot_product_attention"):
            dropout_p = float(self.attn_drop.p) if self.training else 0.0
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            # 兼容旧版本PyTorch
            attn = (q @ k.transpose(-2, -1)) * self.scale
            causal_mask = self._get_causal_mask(T, x.device)
            attn = attn + causal_mask.unsqueeze(0).unsqueeze(0)
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        # 输出
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SpatialWindowAttentionDiT(nn.Module):
    """
    空间窗口注意力（DiT版本）

    关键创新：
    - 利用3D占用的空间局部相关性
    - 窗口内注意力减少计算复杂度
    - 推理时同样使用，保持空间一致性
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: Tuple[int, int, int] = (2, 5, 5),  # 适配(4, 25, 25)的特征图
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 相对位置编码
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) 空间特征，N = D * H * W
            spatial_shape: (D, H, W) 空间维度
        Returns:
            (B, N, C) 窗口注意力输出
        """
        B, N, C = x.shape
        D, H, W = spatial_shape

        # 简化实现：对于小特征图直接使用全局注意力
        # 对于大特征图使用窗口注意力
        if N <= 100:  # 小特征图
            return self._global_attention(x)
        else:
            return self._window_attention(x, D, H, W)

    def _global_attention(self, x: torch.Tensor) -> torch.Tensor:
        """全局注意力（小特征图）"""
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def _window_attention(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        """窗口注意力（大特征图）"""
        B, N, C = x.shape

        # reshape到3D
        x = x.view(B, D, H, W, C)

        # 窗口划分
        wd, wh, ww = self.window_size
        pad_d = (wd - D % wd) % wd
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))

        Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

        # 划分窗口
        x = x.view(B, Dp // wd, wd, Hp // wh, wh, Wp // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x = x.view(-1, wd * wh * ww, C)  # (B*num_windows, window_size, C)

        # 窗口内注意力
        x = self._global_attention(x)

        # 恢复形状
        x = x.view(B, Dp // wd, Hp // wh, Wp // ww, wd, wh, ww, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, Dp, Hp, Wp, C)

        # 移除padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :D, :H, :W, :]

        x = x.view(B, N, C)
        return x


class STCADiTBlock(nn.Module):
    """
    时空因果注意力DiT块 - 架构级创新

    与原DiTBlock的核心区别：
    1. 时间维度：因果注意力（t只能看到<=t）
    2. 空间维度：窗口注意力（利用局部相关性）
    3. 推理时同样生效，不仅是训练技巧

    输入输出与原DiTBlock兼容，可直接替换
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        num_frames: int = 4,  # 时间帧数
        spatial_shape: Tuple[int, int, int] = (4, 25, 25),  # D, H, W
        **block_kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_frames = num_frames
        self.spatial_shape = spatial_shape
        self.spatial_size = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]

        # 时间因果注意力
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.temporal_attn = TemporalCausalAttentionDiT(
            hidden_size, num_heads, **block_kwargs
        )

        # 空间窗口注意力
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.spatial_attn = SpatialWindowAttentionDiT(
            hidden_size, num_heads, **block_kwargs
        )

        # MLP
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0
        )

        # adaLN-Zero调制（9个参数：3组 x 3个）
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) 输入特征，N = num_frames * spatial_size
            c: (B, C) 条件向量（时间步 + 标签）
        Returns:
            (B, N, C) 输出特征
        """
        B, N, C = x.shape

        # 获取adaLN调制参数
        modulation = self.adaLN_modulation(c)
        (shift_t, scale_t, gate_t,
         shift_s, scale_s, gate_s,
         shift_m, scale_m, gate_m) = modulation.chunk(9, dim=1)

        # 1. 时间因果注意力
        x_temporal = self._temporal_attention(
            x, shift_t, scale_t, gate_t
        )
        x = x + x_temporal

        # 2. 空间窗口注意力
        x_spatial = self._spatial_attention(
            x, shift_s, scale_s, gate_s
        )
        x = x + x_spatial

        # 3. MLP
        x_mlp = gate_m.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_m, scale_m)
        )
        x = x + x_mlp

        return x

    def _temporal_attention(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor
    ) -> torch.Tensor:
        """时间维度因果注意力"""
        B, N, C = x.shape
        T = self.num_frames
        S = self.spatial_size

        # 检查是否可以reshape
        if N != T * S:
            # 如果维度不匹配，使用简化的时间注意力
            return gate.unsqueeze(1) * self.temporal_attn(
                modulate(self.norm1(x), shift, scale)
            )

        # reshape: (B, T*S, C) -> (B*S, T, C)
        x_norm = modulate(self.norm1(x), shift, scale)
        x_t = x_norm.view(B, T, S, C).permute(0, 2, 1, 3)
        x_t = x_t.reshape(B * S, T, C)

        # 时间因果注意力
        x_t = self.temporal_attn(x_t)

        # reshape回来
        x_t = x_t.view(B, S, T, C).permute(0, 2, 1, 3)
        x_t = x_t.reshape(B, N, C)

        return gate.unsqueeze(1) * x_t

    def _spatial_attention(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor
    ) -> torch.Tensor:
        """空间维度窗口注意力"""
        B, N, C = x.shape
        T = self.num_frames
        S = self.spatial_size

        # 检查是否可以reshape
        if N != T * S:
            return gate.unsqueeze(1) * self.spatial_attn(
                modulate(self.norm2(x), shift, scale),
                self.spatial_shape
            )

        # reshape: (B, T*S, C) -> (B*T, S, C)
        x_norm = modulate(self.norm2(x), shift, scale)
        x_s = x_norm.view(B, T, S, C).reshape(B * T, S, C)

        # 空间窗口注意力
        x_s = self.spatial_attn(x_s, self.spatial_shape)

        # reshape回来
        x_s = x_s.view(B, T, S, C).reshape(B, N, C)

        return gate.unsqueeze(1) * x_s
