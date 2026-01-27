"""
Spatial Window Attention
空间窗口注意力 - 利用3D空间局部相关性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpatialWindowAttention(nn.Module):
    """
    空间窗口注意力

    核心思想：3D占用中相邻体素相关性更强
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        num_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # 相对位置编码
        self.relative_pos = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) *
                (2 * window_size[1] - 1) *
                (2 * window_size[2] - 1),
                num_heads
            )
        )
        nn.init.trunc_normal_(self.relative_pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, D, C) 3D空间特征
        Returns:
            (B, H, W, D, C) 窗口注意力输出
        """
        B, H, W, D, C = x.shape
        wh, ww, wd = self.window_size

        # 划分窗口
        x = x.view(B, H // wh, wh, W // ww, ww, D // wd, wd, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x = x.view(-1, wh * ww * wd, C)

        # 窗口内注意力
        qkv = self.qkv(x).reshape(-1, wh * ww * wd, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(-1, wh * ww * wd, C)
        x = self.proj(x)

        # 恢复形状
        num_windows = (H // wh) * (W // ww) * (D // wd)
        x = x.view(B, H // wh, W // ww, D // wd, wh, ww, wd, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, H, W, D, C)

        return x
