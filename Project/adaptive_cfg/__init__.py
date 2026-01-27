"""
自适应分类器无关引导模块
========================

本模块实现了用于4D占用生成的时序感知动态CFG调度。

核心功能:
    - 跨扩散时间步的动态引导尺度调度
    - 针对视频生成一致性的时序感知调整
    - 复杂场景控制的多条件引导融合

类说明:
    - AdaptiveCFGScheduler: 主调度器，支持多种调度策略
    - TemporalAwareCFG: 4D生成的时序感知引导
    - MultiConditionCFG: 多条件引导融合

参考文献:
    - Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)
    - Progressive Distillation for Fast Sampling (Salimans & Ho, 2022)
"""

from .scheduler import AdaptiveCFGScheduler
from .temporal_cfg import TemporalAwareCFG
from .multi_condition import MultiConditionCFG
from .strategies import (
    LinearSchedule,
    CosineSchedule,
    StepSchedule,
    AdaptiveSchedule,
    TemporalPyramidSchedule
)

__all__ = [
    "AdaptiveCFGScheduler",
    "TemporalAwareCFG",
    "MultiConditionCFG",
    "LinearSchedule",
    "CosineSchedule",
    "StepSchedule",
    "AdaptiveSchedule",
    "TemporalPyramidSchedule",
]
