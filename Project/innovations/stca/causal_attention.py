"""
Temporal Causal Attention
时间因果注意力 - 核心创新组件

关键思想：t时刻只能attend到<=t的帧，符合物理因果律
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TemporalCausalAttention(nn.Module):
    """
    时间因果注意力机制

    与标准注意力的区别：
    - 使用下三角因果掩码
    - 支持关键帧增强
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

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成因果掩码（下三角矩阵）"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        keyframe_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) 时序特征
            keyframe_mask: (T,) 关键帧掩码，关键帧位置为1
        Returns:
            (B, T, C) 因果注意力输出
        """
        B, T, C = x.shape

        # QKV投影
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 应用因果掩码
        causal_mask = self._get_causal_mask(T, x.device)
        attn = attn + causal_mask.unsqueeze(0).unsqueeze(0)

        # 关键帧增强
        if keyframe_mask is not None:
            keyframe_boost = keyframe_mask.float() * 0.5
            attn = attn + keyframe_boost.view(1, 1, 1, T)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 输出
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
