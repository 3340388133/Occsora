"""
Multi-Scale Refinement Pipeline
===============================

Multi-scale processing for improved detail preservation and
consistency across different spatial scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class ScaleConfig:
    """Configuration for a single scale level."""
    scale_factor: float
    kernel_size: int
    sigma: float
    weight: float


class MultiScaleRefinementPipeline:
    """
    Multi-Scale Refinement Pipeline for 4D Occupancy.

    Processes occupancy at multiple scales to:
        - Preserve fine details
        - Ensure global consistency
        - Reduce scale-dependent artifacts

    Processing Stages:
        1. Build scale pyramid
        2. Process each scale independently
        3. Merge scales with learned/fixed weights
        4. Apply residual refinement

    Example:
        >>> pipeline = MultiScaleRefinementPipeline(num_scales=3)
        >>> refined = pipeline.refine(occupancy_sequence)
    """

    def __init__(
        self,
        num_scales: int = 3,
        scale_factor: float = 0.5,
        merge_mode: str = "weighted",
        refinement_iterations: int = 2,
        device: str = "cuda"
    ):
        """
        Initialize multi-scale pipeline.

        Args:
            num_scales: Number of scale levels
            scale_factor: Downsampling factor between scales
            merge_mode: How to merge scales ("weighted", "attention", "residual")
            refinement_iterations: Number of refinement passes
            device: Computation device
        """
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        self.merge_mode = merge_mode
        self.refinement_iterations = refinement_iterations
        self.device = device

        # Initialize scale configurations
        self._init_scales()

    def _init_scales(self):
        """Initialize scale configurations."""
        self.scales = []

        for i in range(self.num_scales):
            factor = self.scale_factor ** i
            self.scales.append(ScaleConfig(
                scale_factor=factor,
                kernel_size=3 + 2 * i,  # Larger kernel at coarser scales
                sigma=1.0 + 0.5 * i,
                weight=1.0 / (i + 1)  # Higher weight for finer scales
            ))

        # Normalize weights
        total_weight = sum(s.weight for s in self.scales)
        for s in self.scales:
            s.weight /= total_weight

    def refine(
        self,
        data: torch.Tensor,
        temporal_filter: Optional[Any] = None,
        spatial_smoother: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Apply multi-scale refinement.

        Args:
            data: Input data (B, T, C, H, W) or (B, T, H, W, D)
            temporal_filter: Optional temporal filter to apply
            spatial_smoother: Optional spatial smoother to apply

        Returns:
            Refined data
        """
        for iteration in range(self.refinement_iterations):
            # Build scale pyramid
            pyramid = self._build_pyramid(data)

            # Process each scale
            processed_pyramid = []
            for scale_idx, (scale_data, scale_config) in enumerate(zip(pyramid, self.scales)):
                processed = self._process_scale(
                    scale_data,
                    scale_config,
                    temporal_filter,
                    spatial_smoother
                )
                processed_pyramid.append(processed)

            # Merge scales
            data = self._merge_scales(processed_pyramid, data.shape)

        return data

    def _build_pyramid(self, data: torch.Tensor) -> List[torch.Tensor]:
        """
        Build scale pyramid.

        Args:
            data: Input data

        Returns:
            List of tensors at different scales
        """
        pyramid = [data]

        current = data
        for i in range(1, self.num_scales):
            # Downsample
            scale = self.scales[i].scale_factor / self.scales[i-1].scale_factor

            # Handle different input shapes
            if len(data.shape) == 5:  # (B, T, C, H, W)
                B, T, C, H, W = current.shape
                current_flat = current.reshape(B * T, C, H, W)
                downsampled = F.interpolate(
                    current_flat,
                    scale_factor=scale,
                    mode='bilinear',
                    align_corners=False
                )
                new_H, new_W = downsampled.shape[2:]
                current = downsampled.reshape(B, T, C, new_H, new_W)
            else:
                current = F.interpolate(
                    current,
                    scale_factor=scale,
                    mode='trilinear' if len(data.shape) == 5 else 'bilinear',
                    align_corners=False
                )

            pyramid.append(current)

        return pyramid

    def _process_scale(
        self,
        data: torch.Tensor,
        config: ScaleConfig,
        temporal_filter: Optional[Any],
        spatial_smoother: Optional[Any]
    ) -> torch.Tensor:
        """
        Process data at a single scale.

        Args:
            data: Scale-specific data
            config: Scale configuration
            temporal_filter: Optional temporal filter
            spatial_smoother: Optional spatial smoother

        Returns:
            Processed data
        """
        processed = data

        # Apply temporal filtering
        if temporal_filter is not None:
            processed = temporal_filter.filter(processed)

        # Apply spatial smoothing
        if spatial_smoother is not None:
            processed = spatial_smoother.smooth(processed)
        else:
            # Default Gaussian smoothing
            processed = self._gaussian_smooth(processed, config.kernel_size, config.sigma)

        return processed

    def _gaussian_smooth(
        self,
        data: torch.Tensor,
        kernel_size: int,
        sigma: float
    ) -> torch.Tensor:
        """Apply Gaussian smoothing."""
        # Create Gaussian kernel
        x = torch.arange(kernel_size, device=data.device) - kernel_size // 2
        kernel_1d = torch.exp(-x.float()**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Handle different input shapes
        if len(data.shape) == 5:  # (B, T, C, H, W)
            B, T, C, H, W = data.shape
            data_flat = data.reshape(B * T, C, H, W)

            # Apply separable Gaussian
            kernel_h = kernel_1d.view(1, 1, kernel_size, 1).repeat(C, 1, 1, 1)
            kernel_w = kernel_1d.view(1, 1, 1, kernel_size).repeat(C, 1, 1, 1)

            padding = kernel_size // 2
            smoothed = F.conv2d(data_flat, kernel_h, padding=(padding, 0), groups=C)
            smoothed = F.conv2d(smoothed, kernel_w, padding=(0, padding), groups=C)

            return smoothed.reshape(B, T, C, H, W)

        return data

    def _merge_scales(
        self,
        pyramid: List[torch.Tensor],
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Merge processed scales back together.

        Args:
            pyramid: List of processed scale tensors
            target_shape: Target output shape

        Returns:
            Merged tensor
        """
        if self.merge_mode == "weighted":
            return self._weighted_merge(pyramid, target_shape)
        elif self.merge_mode == "residual":
            return self._residual_merge(pyramid, target_shape)
        elif self.merge_mode == "attention":
            return self._attention_merge(pyramid, target_shape)
        else:
            raise ValueError(f"Unknown merge mode: {self.merge_mode}")

    def _weighted_merge(
        self,
        pyramid: List[torch.Tensor],
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Weighted average of scales."""
        merged = torch.zeros(target_shape, device=pyramid[0].device, dtype=pyramid[0].dtype)

        for scale_data, scale_config in zip(pyramid, self.scales):
            # Upsample to target size
            if scale_data.shape != target_shape:
                if len(target_shape) == 5:
                    B, T, C, H, W = target_shape
                    B_s, T_s, C_s, H_s, W_s = scale_data.shape
                    scale_flat = scale_data.reshape(B_s * T_s, C_s, H_s, W_s)
                    upsampled = F.interpolate(
                        scale_flat,
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    )
                    scale_data = upsampled.reshape(B, T, C, H, W)
                else:
                    scale_data = F.interpolate(
                        scale_data,
                        size=target_shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )

            merged = merged + scale_config.weight * scale_data

        return merged

    def _residual_merge(
        self,
        pyramid: List[torch.Tensor],
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Residual-based merging."""
        # Start from coarsest scale
        current = pyramid[-1]

        # Add residuals from finer scales
        for i in range(len(pyramid) - 2, -1, -1):
            # Upsample current
            if current.shape != pyramid[i].shape:
                if len(target_shape) == 5:
                    B, T, C, H, W = pyramid[i].shape
                    B_c, T_c, C_c, H_c, W_c = current.shape
                    current_flat = current.reshape(B_c * T_c, C_c, H_c, W_c)
                    upsampled = F.interpolate(
                        current_flat,
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    )
                    current = upsampled.reshape(B, T, C, H, W)

            # Add residual
            residual = pyramid[i] - current
            current = current + 0.5 * residual

        return current

    def _attention_merge(
        self,
        pyramid: List[torch.Tensor],
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Attention-based merging."""
        # Upsample all scales to target
        upsampled_pyramid = []
        for scale_data in pyramid:
            if scale_data.shape != target_shape:
                if len(target_shape) == 5:
                    B, T, C, H, W = target_shape
                    B_s, T_s, C_s, H_s, W_s = scale_data.shape
                    scale_flat = scale_data.reshape(B_s * T_s, C_s, H_s, W_s)
                    upsampled = F.interpolate(
                        scale_flat,
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    )
                    scale_data = upsampled.reshape(B, T, C, H, W)
            upsampled_pyramid.append(scale_data)

        # Stack scales
        stacked = torch.stack(upsampled_pyramid, dim=0)  # (S, B, T, C, H, W)

        # Compute attention weights based on local variance
        variances = stacked.var(dim=[3, 4, 5], keepdim=True)  # (S, B, T, 1, 1, 1)
        attention = F.softmax(variances, dim=0)

        # Weighted sum
        merged = (stacked * attention).sum(dim=0)

        return merged

    def __repr__(self) -> str:
        return (
            f"MultiScaleRefinementPipeline("
            f"scales={self.num_scales}, "
            f"merge={self.merge_mode})"
        )
