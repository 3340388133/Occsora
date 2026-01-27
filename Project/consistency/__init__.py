"""
时空一致性增强模块
==================

用于提高生成的4D占用序列质量和一致性的综合后处理和增强模块。

组件说明:
    - TemporalConsistencyFilter: 帧一致性的时序滤波
    - SpatialSmoothingModule: 空间平滑和细化
    - ObjectTrajectoryRefiner: 物体级轨迹一致性
    - MultiScaleRefinementPipeline: 多尺度细化
    - SpatioTemporalEnhancer: 主增强管道

核心功能:
    - 时序滤波减少闪烁
    - 空间平滑使输出更清晰
    - 物体追踪保持运动一致性
    - 多尺度处理保留细节
    - 全面的质量指标
"""

from .temporal_filter import TemporalConsistencyFilter
from .spatial_smoothing import SpatialSmoothingModule
from .object_tracker import ObjectTrajectoryRefiner
from .multi_scale_refine import MultiScaleRefinementPipeline
from .enhancer import SpatioTemporalEnhancer
from .metrics import (
    TemporalConsistencyMetric,
    SpatialSmoothnessMetric,
    ObjectConsistencyMetric,
    QualityAssessment
)

__all__ = [
    "TemporalConsistencyFilter",
    "SpatialSmoothingModule",
    "ObjectTrajectoryRefiner",
    "MultiScaleRefinementPipeline",
    "SpatioTemporalEnhancer",
    "TemporalConsistencyMetric",
    "SpatialSmoothnessMetric",
    "ObjectConsistencyMetric",
    "QualityAssessment",
]
