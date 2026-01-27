"""
Spatio-Temporal Enhancer
========================

Main enhancement pipeline that combines all consistency modules.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging
import time

from .temporal_filter import TemporalConsistencyFilter
from .spatial_smoothing import SpatialSmoothingModule
from .object_tracker import ObjectTrajectoryRefiner
from .multi_scale_refine import MultiScaleRefinementPipeline


@dataclass
class EnhancementConfig:
    """Configuration for the enhancement pipeline."""
    enable_temporal_filter: bool = True
    enable_spatial_smoothing: bool = True
    enable_object_tracking: bool = True
    enable_multi_scale: bool = True

    temporal_filter_type: str = "ema"
    temporal_alpha: float = 0.3

    spatial_smoothing_type: str = "bilateral"
    spatial_kernel_size: int = 3

    multi_scale_levels: int = 3

    device: str = "cuda"


@dataclass
class EnhancementResult:
    """Result of enhancement pipeline."""
    enhanced_data: torch.Tensor
    metrics: Dict[str, float]
    elapsed_time: float
    applied_modules: List[str]


class SpatioTemporalEnhancer:
    """
    Spatio-Temporal Enhancement Pipeline for 4D Occupancy.

    Combines multiple enhancement modules into a unified pipeline:
        1. Multi-scale processing for detail preservation
        2. Temporal filtering for frame consistency
        3. Spatial smoothing for noise reduction
        4. Object tracking for trajectory consistency

    The pipeline is fully configurable and can selectively enable/disable
    individual components based on requirements.

    Example:
        >>> enhancer = SpatioTemporalEnhancer(
        ...     enable_temporal_filter=True,
        ...     enable_spatial_smoothing=True
        ... )
        >>> result = enhancer.enhance(generated_occupancy)
        >>> print(f"Enhanced in {result.elapsed_time:.2f}s")
    """

    def __init__(
        self,
        config: Optional[EnhancementConfig] = None,
        enable_temporal_filter: bool = True,
        enable_spatial_smoothing: bool = True,
        enable_object_tracking: bool = False,  # Disabled by default (slower)
        enable_multi_scale: bool = True,
        temporal_filter_type: str = "ema",
        temporal_alpha: float = 0.3,
        spatial_smoothing_type: str = "bilateral",
        spatial_kernel_size: int = 3,
        multi_scale_levels: int = 3,
        device: str = "cuda"
    ):
        """
        Initialize the enhancement pipeline.

        Args:
            config: Optional configuration object
            enable_temporal_filter: Enable temporal filtering
            enable_spatial_smoothing: Enable spatial smoothing
            enable_object_tracking: Enable object tracking
            enable_multi_scale: Enable multi-scale processing
            temporal_filter_type: Type of temporal filter
            temporal_alpha: Alpha for temporal EMA
            spatial_smoothing_type: Type of spatial smoothing
            spatial_kernel_size: Kernel size for spatial smoothing
            multi_scale_levels: Number of scale levels
            device: Computation device
        """
        if config is not None:
            self.config = config
        else:
            self.config = EnhancementConfig(
                enable_temporal_filter=enable_temporal_filter,
                enable_spatial_smoothing=enable_spatial_smoothing,
                enable_object_tracking=enable_object_tracking,
                enable_multi_scale=enable_multi_scale,
                temporal_filter_type=temporal_filter_type,
                temporal_alpha=temporal_alpha,
                spatial_smoothing_type=spatial_smoothing_type,
                spatial_kernel_size=spatial_kernel_size,
                multi_scale_levels=multi_scale_levels,
                device=device
            )

        self._init_modules()
        self._logger = logging.getLogger(self.__class__.__name__)

    def _init_modules(self):
        """Initialize enhancement modules."""
        # Temporal filter
        if self.config.enable_temporal_filter:
            self.temporal_filter = TemporalConsistencyFilter(
                filter_type=self.config.temporal_filter_type,
                alpha=self.config.temporal_alpha,
                device=self.config.device
            )
        else:
            self.temporal_filter = None

        # Spatial smoother
        if self.config.enable_spatial_smoothing:
            self.spatial_smoother = SpatialSmoothingModule(
                smoothing_type=self.config.spatial_smoothing_type,
                kernel_size=self.config.spatial_kernel_size,
                device=self.config.device
            )
        else:
            self.spatial_smoother = None

        # Object tracker
        if self.config.enable_object_tracking:
            self.object_tracker = ObjectTrajectoryRefiner(
                device=self.config.device
            )
        else:
            self.object_tracker = None

        # Multi-scale pipeline
        if self.config.enable_multi_scale:
            self.multi_scale = MultiScaleRefinementPipeline(
                num_scales=self.config.multi_scale_levels,
                device=self.config.device
            )
        else:
            self.multi_scale = None

    def enhance(
        self,
        data: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> EnhancementResult:
        """
        Apply full enhancement pipeline.

        Args:
            data: Input data (B, T, C, H, W) or (B, T, H, W, D)
            class_labels: Optional class labels for object tracking
            return_intermediates: Whether to return intermediate results

        Returns:
            EnhancementResult with enhanced data and metrics
        """
        start_time = time.time()
        applied_modules = []
        intermediates = {} if return_intermediates else None

        enhanced = data.clone()
        original = data.clone()

        # Step 1: Multi-scale processing
        if self.multi_scale is not None:
            self._logger.info("Applying multi-scale refinement...")
            enhanced = self.multi_scale.refine(
                enhanced,
                temporal_filter=self.temporal_filter,
                spatial_smoother=self.spatial_smoother
            )
            applied_modules.append("multi_scale")
            if return_intermediates:
                intermediates['after_multi_scale'] = enhanced.clone()

        else:
            # Apply temporal and spatial separately if no multi-scale
            # Step 2: Temporal filtering
            if self.temporal_filter is not None:
                self._logger.info("Applying temporal filtering...")
                enhanced = self.temporal_filter.filter(enhanced)
                applied_modules.append("temporal_filter")
                if return_intermediates:
                    intermediates['after_temporal'] = enhanced.clone()

            # Step 3: Spatial smoothing
            if self.spatial_smoother is not None:
                self._logger.info("Applying spatial smoothing...")
                enhanced = self.spatial_smoother.smooth(enhanced)
                applied_modules.append("spatial_smoother")
                if return_intermediates:
                    intermediates['after_spatial'] = enhanced.clone()

        # Step 4: Object tracking (applied last)
        if self.object_tracker is not None:
            self._logger.info("Applying object trajectory refinement...")
            enhanced = self.object_tracker.refine(enhanced, class_labels)
            applied_modules.append("object_tracker")
            if return_intermediates:
                intermediates['after_tracking'] = enhanced.clone()

        elapsed_time = time.time() - start_time

        # Compute metrics
        metrics = self._compute_metrics(original, enhanced)

        result = EnhancementResult(
            enhanced_data=enhanced,
            metrics=metrics,
            elapsed_time=elapsed_time,
            applied_modules=applied_modules
        )

        self._logger.info(
            f"Enhancement complete in {elapsed_time:.2f}s. "
            f"Applied: {applied_modules}"
        )

        return result

    def _compute_metrics(
        self,
        original: torch.Tensor,
        enhanced: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute quality metrics comparing original and enhanced.

        Args:
            original: Original data
            enhanced: Enhanced data

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Temporal consistency (lower is better)
        if len(original.shape) >= 3:
            original_diff = (original[:, 1:] - original[:, :-1]).abs().mean()
            enhanced_diff = (enhanced[:, 1:] - enhanced[:, :-1]).abs().mean()
            metrics['temporal_consistency_improvement'] = (
                (original_diff - enhanced_diff) / (original_diff + 1e-8)
            ).item()

        # Spatial smoothness
        metrics['spatial_smoothness_original'] = self._compute_smoothness(original).item()
        metrics['spatial_smoothness_enhanced'] = self._compute_smoothness(enhanced).item()

        # Preservation (how much original content is preserved)
        metrics['content_preservation'] = (
            1 - (original - enhanced).abs().mean() / (original.abs().mean() + 1e-8)
        ).item()

        # PSNR-like metric
        mse = ((original - enhanced) ** 2).mean()
        max_val = max(original.max().item(), enhanced.max().item())
        if mse > 0:
            metrics['psnr'] = (10 * torch.log10(max_val ** 2 / mse)).item()
        else:
            metrics['psnr'] = float('inf')

        return metrics

    def _compute_smoothness(self, data: torch.Tensor) -> torch.Tensor:
        """Compute spatial smoothness metric."""
        if len(data.shape) == 5:  # (B, T, C, H, W)
            # Compute gradient magnitude
            dx = (data[:, :, :, :, 1:] - data[:, :, :, :, :-1]).abs()
            dy = (data[:, :, :, 1:, :] - data[:, :, :, :-1, :]).abs()
            return (dx.mean() + dy.mean()) / 2
        return torch.tensor(0.0)

    def get_module_info(self) -> Dict[str, Any]:
        """Get information about enabled modules."""
        return {
            'temporal_filter': {
                'enabled': self.temporal_filter is not None,
                'type': self.config.temporal_filter_type if self.temporal_filter else None
            },
            'spatial_smoother': {
                'enabled': self.spatial_smoother is not None,
                'type': self.config.spatial_smoothing_type if self.spatial_smoother else None
            },
            'object_tracker': {
                'enabled': self.object_tracker is not None
            },
            'multi_scale': {
                'enabled': self.multi_scale is not None,
                'levels': self.config.multi_scale_levels if self.multi_scale else None
            }
        }

    def __repr__(self) -> str:
        enabled = []
        if self.temporal_filter:
            enabled.append("temporal")
        if self.spatial_smoother:
            enabled.append("spatial")
        if self.object_tracker:
            enabled.append("tracking")
        if self.multi_scale:
            enabled.append("multi_scale")

        return f"SpatioTemporalEnhancer(enabled=[{', '.join(enabled)}])"
