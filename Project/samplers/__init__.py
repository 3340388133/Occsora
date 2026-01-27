"""
混合多策略扩散采样器
====================

具有自动策略选择和质量-速度权衡优化的综合采样器集合。

模块说明:
    - base_sampler: 所有采样器的抽象基类
    - ddim_sampler: DDIM（去噪扩散隐式模型）采样器
    - dpm_solver: DPM-Solver++快速采样器
    - hybrid_sampler: 结合多种策略的智能采样器
    - scheduler: 噪声调度管理

核心功能:
    - 多种采样策略（DDIM、DPM-Solver++、Euler等）
    - 自动质量-速度优化
    - 可自定义的噪声调度
    - 进度跟踪和可视化

参考文献:
    - DDIM: https://arxiv.org/abs/2010.02502
    - DPM-Solver: https://arxiv.org/abs/2206.00927
    - DPM-Solver++: https://arxiv.org/abs/2211.01095
"""

from .base_sampler import BaseSampler, SamplerConfig
from .ddim_sampler import DDIMSampler
from .dpm_solver import DPMSolverSampler
from .euler_sampler import EulerSampler, EulerAncestralSampler
from .hybrid_sampler import HybridSampler
from .scheduler import NoiseScheduler, get_named_beta_schedule
from .factory import SamplerFactory

__all__ = [
    "BaseSampler",
    "SamplerConfig",
    "DDIMSampler",
    "DPMSolverSampler",
    "EulerSampler",
    "EulerAncestralSampler",
    "HybridSampler",
    "NoiseScheduler",
    "get_named_beta_schedule",
    "SamplerFactory",
]
