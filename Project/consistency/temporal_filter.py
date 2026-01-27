"""
Temporal Consistency Filter
===========================

Filters for improving temporal consistency in generated sequences.

Implements various filtering strategies to reduce temporal artifacts
like flickering, jittering, and inconsistent motion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass
import logging


class FilterType(Enum):
    """Types of temporal filters."""
    EXPONENTIAL_MOVING_AVERAGE = "ema"
    GAUSSIAN = "gaussian"
    BILATERAL = "bilateral"
    KALMAN = "kalman"
    MOTION_COMPENSATED = "motion_compensated"
    ADAPTIVE = "adaptive"


@dataclass
class TemporalFilterConfig:
    """Configuration for temporal filtering."""
    filter_type: str = "ema"
    window_size: int = 5
    alpha: float = 0.3  # For EMA
    sigma_temporal: float = 1.0  # For Gaussian
    sigma_spatial: float = 2.0  # For bilateral
    motion_threshold: float = 0.1  # For motion-compensated


class TemporalConsistencyFilter:
    """
    Temporal Consistency Filter for 4D Occupancy Generation.

    Applies temporal filtering to reduce artifacts and improve
    consistency across generated frames.

    Filter Types:
        - EMA: Exponential Moving Average (fast, simple)
        - Gaussian: Gaussian temporal smoothing
        - Bilateral: Edge-preserving bilateral filter
        - Kalman: Kalman filter for motion-aware smoothing
        - Motion-Compensated: Uses motion estimation
        - Adaptive: Adaptively selects filter based on content

    Example:
        >>> filter = TemporalConsistencyFilter(filter_type="ema", alpha=0.3)
        >>> smoothed = filter.filter(generated_sequence)
    """

    def __init__(
        self,
        filter_type: str = "ema",
        window_size: int = 5,
        alpha: float = 0.3,
        sigma_temporal: float = 1.0,
        sigma_spatial: float = 2.0,
        motion_threshold: float = 0.1,
        preserve_edges: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize the temporal filter.

        Args:
            filter_type: Type of temporal filter
            window_size: Size of the temporal window
            alpha: EMA smoothing factor (0-1, higher = more smoothing)
            sigma_temporal: Temporal sigma for Gaussian filter
            sigma_spatial: Spatial sigma for bilateral filter
            motion_threshold: Threshold for motion detection
            preserve_edges: Whether to preserve temporal edges
            device: Computation device
        """
        self.filter_type = FilterType(filter_type)
        self.window_size = window_size
        self.alpha = alpha
        self.sigma_temporal = sigma_temporal
        self.sigma_spatial = sigma_spatial
        self.motion_threshold = motion_threshold
        self.preserve_edges = preserve_edges
        self.device = device

        # Pre-compute kernels
        self._init_kernels()

        # State for stateful filters
        self._state = None

        self._logger = logging.getLogger(self.__class__.__name__)

    def _init_kernels(self):
        """Initialize filter kernels."""
        # Gaussian temporal kernel
        if self.window_size > 0:
            t = torch.arange(self.window_size) - self.window_size // 2
            self.gaussian_kernel = torch.exp(-t.float()**2 / (2 * self.sigma_temporal**2))
            self.gaussian_kernel = self.gaussian_kernel / self.gaussian_kernel.sum()
            self.gaussian_kernel = self.gaussian_kernel.to(self.device)

    def filter(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply temporal filtering to a sequence.

        Args:
            sequence: Input sequence (B, T, C, H, W) or (B, T, H, W, D)
            mask: Optional mask for selective filtering

        Returns:
            Filtered sequence
        """
        if self.filter_type == FilterType.EXPONENTIAL_MOVING_AVERAGE:
            return self._ema_filter(sequence, mask)
        elif self.filter_type == FilterType.GAUSSIAN:
            return self._gaussian_filter(sequence, mask)
        elif self.filter_type == FilterType.BILATERAL:
            return self._bilateral_filter(sequence, mask)
        elif self.filter_type == FilterType.KALMAN:
            return self._kalman_filter(sequence, mask)
        elif self.filter_type == FilterType.MOTION_COMPENSATED:
            return self._motion_compensated_filter(sequence, mask)
        elif self.filter_type == FilterType.ADAPTIVE:
            return self._adaptive_filter(sequence, mask)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

    def _ema_filter(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Exponential Moving Average filter.

        Fast and simple, good for real-time applications.
        """
        B, T = sequence.shape[:2]
        filtered = torch.zeros_like(sequence)

        # Forward pass
        filtered[:, 0] = sequence[:, 0]
        for t in range(1, T):
            filtered[:, t] = self.alpha * sequence[:, t] + (1 - self.alpha) * filtered[:, t-1]

        # Optional backward pass for symmetric filtering
        if self.preserve_edges:
            backward = torch.zeros_like(sequence)
            backward[:, -1] = sequence[:, -1]
            for t in range(T - 2, -1, -1):
                backward[:, t] = self.alpha * sequence[:, t] + (1 - self.alpha) * backward[:, t+1]

            # Average forward and backward
            filtered = 0.5 * (filtered + backward)

        if mask is not None:
            filtered = filtered * mask + sequence * (1 - mask)

        return filtered

    def _gaussian_filter(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Gaussian temporal filter.

        Provides smooth blending across frames with controllable spread.
        """
        B, T = sequence.shape[:2]
        orig_shape = sequence.shape

        # Reshape for 1D convolution
        # (B, T, C, H, W) -> (B*C*H*W, 1, T)
        sequence_flat = sequence.permute(0, 2, 3, 4, 1)  # (B, C, H, W, T)
        flat_shape = sequence_flat.shape
        sequence_flat = sequence_flat.reshape(-1, 1, T)

        # Apply 1D Gaussian convolution
        kernel = self.gaussian_kernel.view(1, 1, -1)
        padding = self.window_size // 2

        filtered_flat = F.conv1d(sequence_flat, kernel, padding=padding)

        # Reshape back
        filtered = filtered_flat.reshape(flat_shape)
        filtered = filtered.permute(0, 4, 1, 2, 3)  # (B, T, C, H, W)

        if mask is not None:
            filtered = filtered * mask + sequence * (1 - mask)

        return filtered

    def _bilateral_filter(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Bilateral temporal filter.

        Edge-preserving filter that considers both temporal distance
        and intensity similarity.
        """
        B, T = sequence.shape[:2]
        filtered = torch.zeros_like(sequence)

        half_window = self.window_size // 2

        for t in range(T):
            weights_sum = torch.zeros_like(sequence[:, t])
            weighted_sum = torch.zeros_like(sequence[:, t])

            for dt in range(-half_window, half_window + 1):
                neighbor_t = t + dt
                if 0 <= neighbor_t < T:
                    # Temporal weight (Gaussian)
                    temporal_weight = np.exp(-dt**2 / (2 * self.sigma_temporal**2))

                    # Intensity weight (Gaussian on difference)
                    intensity_diff = (sequence[:, t] - sequence[:, neighbor_t]).abs()
                    intensity_weight = torch.exp(-intensity_diff**2 / (2 * self.sigma_spatial**2))

                    # Combined weight
                    weight = temporal_weight * intensity_weight

                    weighted_sum += weight * sequence[:, neighbor_t]
                    weights_sum += weight

            filtered[:, t] = weighted_sum / (weights_sum + 1e-8)

        if mask is not None:
            filtered = filtered * mask + sequence * (1 - mask)

        return filtered

    def _kalman_filter(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Kalman filter for temporal consistency.

        Motion-aware filtering that predicts and corrects based on
        a simple motion model.
        """
        B, T = sequence.shape[:2]
        filtered = torch.zeros_like(sequence)

        # Initialize state
        state = sequence[:, 0].clone()
        velocity = torch.zeros_like(state)

        # Kalman parameters
        process_noise = 0.1
        measurement_noise = 0.3
        kalman_gain = process_noise / (process_noise + measurement_noise)

        filtered[:, 0] = state

        for t in range(1, T):
            # Predict
            predicted_state = state + velocity

            # Update
            innovation = sequence[:, t] - predicted_state
            state = predicted_state + kalman_gain * innovation
            velocity = kalman_gain * innovation

            filtered[:, t] = state

        if mask is not None:
            filtered = filtered * mask + sequence * (1 - mask)

        return filtered

    def _motion_compensated_filter(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Motion-compensated temporal filter.

        Aligns frames based on motion before filtering.
        """
        B, T = sequence.shape[:2]

        # Estimate motion between frames
        motion_maps = self._estimate_motion(sequence)

        # Apply motion compensation
        aligned_sequence = self._warp_sequence(sequence, motion_maps)

        # Filter aligned sequence
        filtered_aligned = self._ema_filter(aligned_sequence, mask)

        # Inverse warp to original coordinates
        filtered = self._inverse_warp_sequence(filtered_aligned, motion_maps)

        return filtered

    def _estimate_motion(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Estimate motion between consecutive frames.

        Simple block-based motion estimation.
        """
        B, T = sequence.shape[:2]

        # Compute frame differences as proxy for motion
        motion_maps = torch.zeros(B, T - 1, *sequence.shape[2:], device=sequence.device)

        for t in range(T - 1):
            motion_maps[:, t] = (sequence[:, t + 1] - sequence[:, t]).abs()

        return motion_maps

    def _warp_sequence(
        self,
        sequence: torch.Tensor,
        motion_maps: torch.Tensor
    ) -> torch.Tensor:
        """Warp sequence based on motion (simplified)."""
        # For simplicity, return unchanged (full implementation would use optical flow)
        return sequence

    def _inverse_warp_sequence(
        self,
        sequence: torch.Tensor,
        motion_maps: torch.Tensor
    ) -> torch.Tensor:
        """Inverse warp sequence (simplified)."""
        return sequence

    def _adaptive_filter(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Adaptive temporal filter.

        Selects filtering strength based on local content characteristics.
        """
        B, T = sequence.shape[:2]

        # Compute local variance as indicator of detail
        local_variance = self._compute_local_variance(sequence)

        # Normalize variance
        variance_normalized = local_variance / (local_variance.max() + 1e-8)

        # Compute motion magnitude
        motion = torch.zeros_like(sequence)
        motion[:, 1:] = (sequence[:, 1:] - sequence[:, :-1]).abs()
        motion_normalized = motion / (motion.max() + 1e-8)

        # Adaptive alpha: less smoothing for high variance/motion areas
        adaptive_alpha = self.alpha * (1 - 0.5 * variance_normalized - 0.3 * motion_normalized)
        adaptive_alpha = adaptive_alpha.clamp(0.1, 0.9)

        # Apply adaptive EMA
        filtered = torch.zeros_like(sequence)
        filtered[:, 0] = sequence[:, 0]

        for t in range(1, T):
            alpha_t = adaptive_alpha[:, t]
            filtered[:, t] = alpha_t * sequence[:, t] + (1 - alpha_t) * filtered[:, t-1]

        if mask is not None:
            filtered = filtered * mask + sequence * (1 - mask)

        return filtered

    def _compute_local_variance(self, sequence: torch.Tensor) -> torch.Tensor:
        """Compute local variance for adaptive filtering."""
        # Simple spatial variance using convolution
        kernel_size = 3
        padding = kernel_size // 2

        # Compute mean
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=sequence.device) / (kernel_size ** 2)

        B, T = sequence.shape[:2]
        variance = torch.zeros_like(sequence)

        for t in range(T):
            for c in range(sequence.shape[2]):
                frame = sequence[:, t, c:c+1]
                mean = F.conv2d(frame, kernel, padding=padding)
                sq_mean = F.conv2d(frame ** 2, kernel, padding=padding)
                variance[:, t, c] = (sq_mean - mean ** 2).squeeze(1)

        return variance.abs()

    def reset_state(self):
        """Reset filter state (for stateful filters)."""
        self._state = None

    def get_statistics(self, sequence: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics about temporal consistency.

        Args:
            sequence: Input sequence

        Returns:
            Dictionary of statistics
        """
        B, T = sequence.shape[:2]

        # Temporal difference
        temporal_diff = (sequence[:, 1:] - sequence[:, :-1]).abs()

        return {
            'mean_temporal_diff': temporal_diff.mean().item(),
            'max_temporal_diff': temporal_diff.max().item(),
            'std_temporal_diff': temporal_diff.std().item(),
            'num_frames': T
        }

    def __repr__(self) -> str:
        return (
            f"TemporalConsistencyFilter("
            f"type={self.filter_type.value}, "
            f"window={self.window_size}, "
            f"alpha={self.alpha})"
        )
