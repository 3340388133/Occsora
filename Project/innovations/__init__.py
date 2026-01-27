"""
OccSora-Plus Innovations
========================

SCI 2区创新点模块：
1. STCA - 时空因果注意力
2. SADS - 场景感知扩散调度
3. PCL - 物理一致性损失
"""

from .stca import STCABlock, TemporalCausalAttention
from .sads import SceneComplexityEncoder, AdaptiveNoiseScheduler
from .pcl import PhysicsConsistentLoss

__all__ = [
    'STCABlock',
    'TemporalCausalAttention',
    'SceneComplexityEncoder',
    'AdaptiveNoiseScheduler',
    'PhysicsConsistentLoss',
]
