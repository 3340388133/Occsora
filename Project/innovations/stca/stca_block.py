"""
STCA Block - 时空因果注意力块
组合时间因果注意力和空间窗口注意力
"""

import torch
import torch.nn as nn
from .causal_attention import TemporalCausalAttention
from .window_attention import SpatialWindowAttention


class STCABlock(nn.Module):
    """时空因果注意力块"""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.temporal_attn = TemporalCausalAttention(dim, num_heads)
        self.spatial_attn = SpatialWindowAttention(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, keyframe_mask=None):
        """x: (B, T, H, W, D, C)"""
        B, T, H, W, D, C = x.shape

        # 时间注意力
        x_t = x.permute(0, 2, 3, 4, 1, 5).reshape(B*H*W*D, T, C)
        x_t = x_t + self.temporal_attn(self.norm1(x_t), keyframe_mask)
        x = x_t.view(B, H, W, D, T, C).permute(0, 4, 1, 2, 3, 5)

        # 空间注意力 + MLP
        x_s = x.reshape(B*T, H, W, D, C)
        x_s = x_s + self.spatial_attn(self.norm2(x_s))
        x_s = x_s + self.mlp(x_s)
        x = x_s.view(B, T, H, W, D, C)

        return x
