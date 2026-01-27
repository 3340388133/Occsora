"""
Quality Metrics for Consistency Evaluation
==========================================

Comprehensive metrics for evaluating spatio-temporal consistency
and quality of generated 4D occupancy.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging


@dataclass
class MetricResult:
    """Result of a metric computation."""
    value: float
    per_frame: Optional[np.ndarray] = None
    details: Optional[Dict[str, Any]] = None


class TemporalConsistencyMetric:
    """
    Metric for evaluating temporal consistency.

    Measures how consistent the generated content is across frames.
    Lower values indicate better consistency (less flickering/jitter).

    Metrics computed:
        - Mean Absolute Difference (MAD)
        - Temporal Gradient Magnitude
        - Motion Smoothness
        - Flicker Index
    """

    def __init__(self, device: str = "cuda"):
        """Initialize the metric."""
        self.device = device

    def compute(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> MetricResult:
        """
        Compute temporal consistency metrics.

        Args:
            sequence: Input sequence (B, T, C, H, W) or (B, T, H, W, D)
            mask: Optional mask for selective computation

        Returns:
            MetricResult with consistency scores
        """
        B, T = sequence.shape[:2]

        # Compute frame differences
        frame_diffs = (sequence[:, 1:] - sequence[:, :-1]).abs()

        if mask is not None:
            frame_diffs = frame_diffs * mask[:, 1:]

        # Mean Absolute Difference
        mad = frame_diffs.mean().item()

        # Per-frame differences
        per_frame = frame_diffs.mean(dim=list(range(2, len(frame_diffs.shape)))).cpu().numpy()

        # Temporal gradient (second derivative)
        if T >= 3:
            temporal_grad = (sequence[:, 2:] - 2 * sequence[:, 1:-1] + sequence[:, :-2]).abs()
            temporal_grad_mean = temporal_grad.mean().item()
        else:
            temporal_grad_mean = 0.0

        # Flicker index (variance of differences)
        flicker_index = frame_diffs.var().item()

        # Motion smoothness (gradient of motion)
        if T >= 3:
            motion = sequence[:, 1:] - sequence[:, :-1]
            motion_change = (motion[:, 1:] - motion[:, :-1]).abs()
            motion_smoothness = motion_change.mean().item()
        else:
            motion_smoothness = 0.0

        return MetricResult(
            value=mad,
            per_frame=per_frame.mean(axis=0) if len(per_frame.shape) > 1 else per_frame,
            details={
                'mean_absolute_difference': mad,
                'temporal_gradient': temporal_grad_mean,
                'flicker_index': flicker_index,
                'motion_smoothness': motion_smoothness
            }
        )


class SpatialSmoothnessMetric:
    """
    Metric for evaluating spatial smoothness.

    Measures the smoothness of spatial content, which indicates
    noise level and artifact presence.

    Metrics computed:
        - Total Variation
        - Gradient Magnitude
        - Edge Density
        - Noise Estimate
    """

    def __init__(self, device: str = "cuda"):
        """Initialize the metric."""
        self.device = device

    def compute(
        self,
        data: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> MetricResult:
        """
        Compute spatial smoothness metrics.

        Args:
            data: Input data (B, C, H, W) or (B, T, C, H, W)
            mask: Optional mask

        Returns:
            MetricResult with smoothness scores
        """
        # Flatten temporal dimension if present
        if len(data.shape) == 5:
            B, T, C, H, W = data.shape
            data = data.reshape(B * T, C, H, W)

        # Compute gradients
        grad_x = (data[:, :, :, 1:] - data[:, :, :, :-1]).abs()
        grad_y = (data[:, :, 1:, :] - data[:, :, :-1, :]).abs()

        # Total Variation
        tv = grad_x.mean() + grad_y.mean()

        # Gradient magnitude
        # Align dimensions for computation
        grad_x_aligned = grad_x[:, :, :grad_y.shape[2], :]
        grad_y_aligned = grad_y[:, :, :, :grad_x.shape[3]]

        min_h = min(grad_x_aligned.shape[2], grad_y_aligned.shape[2])
        min_w = min(grad_x_aligned.shape[3], grad_y_aligned.shape[3])

        grad_x_crop = grad_x_aligned[:, :, :min_h, :min_w]
        grad_y_crop = grad_y_aligned[:, :, :min_h, :min_w]

        grad_mag = torch.sqrt(grad_x_crop**2 + grad_y_crop**2 + 1e-8)
        mean_grad_mag = grad_mag.mean().item()

        # Edge density (high gradient regions)
        edge_threshold = grad_mag.mean() + grad_mag.std()
        edge_density = (grad_mag > edge_threshold).float().mean().item()

        # Noise estimate (Laplacian-based)
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=data.dtype, device=data.device).view(1, 1, 3, 3)

        noise_estimates = []
        for c in range(data.shape[1]):
            laplacian = torch.nn.functional.conv2d(
                data[:, c:c+1], laplacian_kernel, padding=1
            )
            # Noise estimate based on median absolute deviation of Laplacian
            noise_est = laplacian.abs().median().item() * 1.4826
            noise_estimates.append(noise_est)

        noise_estimate = np.mean(noise_estimates)

        return MetricResult(
            value=tv.item(),
            details={
                'total_variation': tv.item(),
                'gradient_magnitude': mean_grad_mag,
                'edge_density': edge_density,
                'noise_estimate': noise_estimate
            }
        )


class ObjectConsistencyMetric:
    """
    Metric for evaluating object-level consistency.

    Measures how consistently objects are tracked and represented
    across frames.

    Metrics computed:
        - Object Count Stability
        - Centroid Drift
        - Size Consistency
        - Appearance Consistency
    """

    def __init__(self, device: str = "cuda"):
        """Initialize the metric."""
        self.device = device

    def compute(
        self,
        sequence: torch.Tensor,
        threshold: float = 0.5
    ) -> MetricResult:
        """
        Compute object consistency metrics.

        Args:
            sequence: Input sequence (B, T, H, W, D) or (B, T, C, H, W)
            threshold: Threshold for object detection

        Returns:
            MetricResult with object consistency scores
        """
        B, T = sequence.shape[:2]

        # Binary thresholding
        binary = (sequence > threshold).float()

        # Count objects per frame (simple connected component proxy)
        object_counts = []
        for t in range(T):
            frame = binary[:, t]
            # Approximate object count by counting local maxima
            count = (frame > 0).float().mean(dim=list(range(1, len(frame.shape)))).sum().item()
            object_counts.append(count)

        object_counts = np.array(object_counts)

        # Object count stability (variance of counts)
        count_stability = 1.0 / (1.0 + object_counts.std())

        # Centroid analysis
        centroids = []
        for t in range(T):
            frame = binary[:, t]
            # Compute center of mass
            coords = torch.where(frame > 0)
            if len(coords[0]) > 0:
                centroid = tuple(c.float().mean().item() for c in coords)
                centroids.append(centroid)
            else:
                centroids.append(None)

        # Centroid drift
        drifts = []
        for i in range(1, len(centroids)):
            if centroids[i] is not None and centroids[i-1] is not None:
                drift = sum((centroids[i][d] - centroids[i-1][d])**2 for d in range(len(centroids[i])))
                drifts.append(np.sqrt(drift))

        centroid_drift = np.mean(drifts) if drifts else 0.0

        # Size consistency (variance of object sizes)
        sizes = []
        for t in range(T):
            frame = binary[:, t]
            size = (frame > 0).float().sum().item()
            sizes.append(size)

        sizes = np.array(sizes)
        size_consistency = 1.0 / (1.0 + sizes.std() / (sizes.mean() + 1e-8))

        # Overall consistency score
        overall = (count_stability + size_consistency) / 2

        return MetricResult(
            value=overall,
            per_frame=object_counts,
            details={
                'object_count_stability': count_stability,
                'centroid_drift': centroid_drift,
                'size_consistency': size_consistency,
                'mean_object_count': object_counts.mean()
            }
        )


class QualityAssessment:
    """
    Comprehensive quality assessment combining all metrics.

    Provides a unified interface for evaluating 4D occupancy quality.
    """

    def __init__(self, device: str = "cuda"):
        """Initialize quality assessment."""
        self.device = device

        self.temporal_metric = TemporalConsistencyMetric(device)
        self.spatial_metric = SpatialSmoothnessMetric(device)
        self.object_metric = ObjectConsistencyMetric(device)

        self._logger = logging.getLogger(self.__class__.__name__)

    def assess(
        self,
        data: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment.

        Args:
            data: Input data to assess
            reference: Optional reference for comparison
            detailed: Whether to include detailed metrics

        Returns:
            Dictionary of quality metrics
        """
        results = {}

        # Temporal consistency
        temporal_result = self.temporal_metric.compute(data)
        results['temporal_consistency'] = temporal_result.value
        if detailed:
            results['temporal_details'] = temporal_result.details

        # Spatial smoothness
        spatial_result = self.spatial_metric.compute(data)
        results['spatial_smoothness'] = spatial_result.value
        if detailed:
            results['spatial_details'] = spatial_result.details

        # Object consistency
        object_result = self.object_metric.compute(data)
        results['object_consistency'] = object_result.value
        if detailed:
            results['object_details'] = object_result.details

        # Overall quality score (weighted combination)
        # Lower temporal = better, higher object = better, lower spatial = better
        temporal_score = 1.0 / (1.0 + temporal_result.value)
        spatial_score = 1.0 / (1.0 + spatial_result.value)
        object_score = object_result.value

        results['overall_quality'] = (
            0.4 * temporal_score +
            0.3 * spatial_score +
            0.3 * object_score
        )

        # Comparison with reference if provided
        if reference is not None:
            ref_temporal = self.temporal_metric.compute(reference)
            ref_spatial = self.spatial_metric.compute(reference)

            results['temporal_improvement'] = (
                ref_temporal.value - temporal_result.value
            ) / (ref_temporal.value + 1e-8)
            results['spatial_improvement'] = (
                ref_spatial.value - spatial_result.value
            ) / (ref_spatial.value + 1e-8)

        return results

    def generate_report(
        self,
        data: torch.Tensor,
        reference: Optional[torch.Tensor] = None
    ) -> str:
        """
        Generate a human-readable quality report.

        Args:
            data: Input data
            reference: Optional reference

        Returns:
            Formatted report string
        """
        results = self.assess(data, reference, detailed=True)

        report = []
        report.append("=" * 50)
        report.append("4D Occupancy Quality Assessment Report")
        report.append("=" * 50)
        report.append("")

        report.append(f"Overall Quality Score: {results['overall_quality']:.4f}")
        report.append("")

        report.append("Temporal Consistency:")
        report.append(f"  - Score: {results['temporal_consistency']:.4f}")
        for key, value in results.get('temporal_details', {}).items():
            report.append(f"  - {key}: {value:.4f}")
        report.append("")

        report.append("Spatial Smoothness:")
        report.append(f"  - Total Variation: {results['spatial_smoothness']:.4f}")
        for key, value in results.get('spatial_details', {}).items():
            report.append(f"  - {key}: {value:.4f}")
        report.append("")

        report.append("Object Consistency:")
        report.append(f"  - Score: {results['object_consistency']:.4f}")
        for key, value in results.get('object_details', {}).items():
            report.append(f"  - {key}: {value:.4f}")
        report.append("")

        if reference is not None:
            report.append("Improvement vs Reference:")
            report.append(f"  - Temporal: {results.get('temporal_improvement', 0):.2%}")
            report.append(f"  - Spatial: {results.get('spatial_improvement', 0):.2%}")

        report.append("=" * 50)

        return "\n".join(report)
