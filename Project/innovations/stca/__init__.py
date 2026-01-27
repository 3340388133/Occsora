"""
Spatio-Temporal Causal Attention (STCA)
时空因果注意力模块

核心创新：
1. 时间因果掩码 - 符合物理因果律
2. 空间窗口注意力 - 利用局部相关性
3. 关键帧传播 - 长程依赖建模

架构级创新（v2.0）：
4. STCADiTBlock - 可直接替换DiTBlock
5. 推理时也使用因果约束
"""

from .causal_attention import TemporalCausalAttention
from .window_attention import SpatialWindowAttention
from .stca_block import STCABlock
from .stca_dit_block import (
    STCADiTBlock,
    TemporalCausalAttentionDiT,
    SpatialWindowAttentionDiT,
)

__all__ = [
    # 原始组件
    'TemporalCausalAttention',
    'SpatialWindowAttention',
    'STCABlock',
    # 架构级创新（v2.0）
    'STCADiTBlock',
    'TemporalCausalAttentionDiT',
    'SpatialWindowAttentionDiT',
]
