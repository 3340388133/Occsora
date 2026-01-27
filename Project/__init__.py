"""
OccSora 增强项目
================

用于OccSora 4D占用生成模型的综合增强工具包。

模块说明:
    - adaptive_cfg: 自适应分类器无关引导，具有时序感知的动态调度
    - samplers: 混合多策略扩散采样器，支持质量-速度权衡
    - consistency: 时空一致性增强，支持多尺度细化

作者: OccSora Enhancement Team
版本: 1.0.0
"""

from .adaptive_cfg import AdaptiveCFGScheduler, TemporalAwareCFG, MultiConditionCFG
from .samplers import (
    BaseSampler,
    DDIMSampler,
    DPMSolverSampler,
    HybridSampler,
    SamplerFactory
)
from .consistency import (
    TemporalConsistencyFilter,
    SpatialSmoothingModule,
    ObjectTrajectoryRefiner,
    MultiScaleRefinementPipeline,
    SpatioTemporalEnhancer
)

__version__ = "1.0.0"
__all__ = [
    # 自适应CFG
    "AdaptiveCFGScheduler",
    "TemporalAwareCFG",
    "MultiConditionCFG",
    # 采样器
    "BaseSampler",
    "DDIMSampler",
    "DPMSolverSampler",
    "HybridSampler",
    "SamplerFactory",
    # 一致性增强
    "TemporalConsistencyFilter",
    "SpatialSmoothingModule",
    "ObjectTrajectoryRefiner",
    "MultiScaleRefinementPipeline",
    "SpatioTemporalEnhancer",
]
