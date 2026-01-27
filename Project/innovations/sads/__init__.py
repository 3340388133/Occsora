"""
Scene-Aware Diffusion Scheduling (SADS)
场景感知扩散调度模块

核心创新：
1. 场景复杂度编码
2. 自适应噪声调度

架构级创新（v2.0）：
3. SADSSampler - 推理时自适应采样
4. 自适应步数采样 - 复杂场景多步，简单场景少步
"""

from .complexity_encoder import SceneComplexityEncoder
from .adaptive_scheduler import AdaptiveNoiseScheduler
from .sads_sampler import (
    SADSSampler,
    SceneComplexityEstimator,
    AdaptiveBetaScheduler,
)

__all__ = [
    # 原始组件
    'SceneComplexityEncoder',
    'AdaptiveNoiseScheduler',
    # 架构级创新（v2.0）
    'SADSSampler',
    'SceneComplexityEstimator',
    'AdaptiveBetaScheduler',
]
