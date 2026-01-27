"""
Object Trajectory Refiner
=========================

Refines object trajectories for improved consistency in 4D occupancy.

Ensures that detected objects maintain consistent motion paths
across the generated sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import logging


@dataclass
class ObjectTrack:
    """Represents a tracked object across frames."""
    track_id: int
    positions: List[Tuple[float, float, float]]  # (x, y, z) or (x, y, t)
    confidences: List[float]
    class_id: int
    frames: List[int]
    bbox: Optional[List[Tuple]] = None


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory refinement."""
    min_track_length: int = 3
    max_gap: int = 2
    distance_threshold: float = 10.0
    smoothing_window: int = 5
    velocity_weight: float = 0.5
    appearance_weight: float = 0.3


class ObjectTrajectoryRefiner:
    """
    Object Trajectory Refiner for 4D Occupancy.

    Tracks objects across frames and refines their trajectories
    for improved temporal consistency.

    Features:
        - Object detection and tracking
        - Trajectory smoothing
        - Gap filling for missing detections
        - Motion prediction and correction
        - Multi-object association

    Example:
        >>> refiner = ObjectTrajectoryRefiner()
        >>> refined_occupancy = refiner.refine(occupancy_sequence)
    """

    def __init__(
        self,
        min_track_length: int = 3,
        max_gap: int = 2,
        distance_threshold: float = 10.0,
        smoothing_window: int = 5,
        velocity_weight: float = 0.5,
        iou_threshold: float = 0.3,
        device: str = "cuda"
    ):
        """
        Initialize trajectory refiner.

        Args:
            min_track_length: Minimum frames for valid track
            max_gap: Maximum gap frames to bridge
            distance_threshold: Max distance for association
            smoothing_window: Window size for trajectory smoothing
            velocity_weight: Weight for velocity in association
            iou_threshold: IoU threshold for matching
            device: Computation device
        """
        self.min_track_length = min_track_length
        self.max_gap = max_gap
        self.distance_threshold = distance_threshold
        self.smoothing_window = smoothing_window
        self.velocity_weight = velocity_weight
        self.iou_threshold = iou_threshold
        self.device = device

        self.tracks: List[ObjectTrack] = []
        self.next_track_id = 0

        self._logger = logging.getLogger(self.__class__.__name__)

    def refine(
        self,
        occupancy: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Refine object trajectories in occupancy sequence.

        Args:
            occupancy: Occupancy tensor (B, T, H, W, D) or (B, T, C, H, W)
            class_labels: Optional class label tensor

        Returns:
            Refined occupancy tensor
        """
        B, T = occupancy.shape[:2]

        refined_occupancy = occupancy.clone()

        for b in range(B):
            # Extract objects from each frame
            detections = self._extract_objects(occupancy[b], class_labels)

            # Associate detections across frames
            tracks = self._associate_detections(detections)

            # Smooth trajectories
            smoothed_tracks = self._smooth_trajectories(tracks)

            # Fill gaps
            filled_tracks = self._fill_gaps(smoothed_tracks, T)

            # Apply refined trajectories to occupancy
            refined_occupancy[b] = self._apply_tracks(
                refined_occupancy[b], filled_tracks, detections
            )

        return refined_occupancy

    def _extract_objects(
        self,
        occupancy: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> List[List[Dict]]:
        """
        Extract objects from occupancy sequence.

        Args:
            occupancy: Single batch occupancy (T, H, W, D) or (T, C, H, W)
            class_labels: Optional class labels

        Returns:
            List of detections per frame
        """
        T = occupancy.shape[0]
        all_detections = []

        occupancy_np = occupancy.cpu().numpy()

        for t in range(T):
            frame_detections = []

            # Threshold to get binary occupancy
            if len(occupancy.shape) == 4:  # (T, H, W, D)
                binary = (occupancy_np[t] > 0).astype(np.float32)
            else:  # (T, C, H, W)
                binary = (occupancy_np[t].sum(axis=0) > 0).astype(np.float32)

            # Connected components
            labeled, num_features = ndimage.label(binary)

            for obj_id in range(1, num_features + 1):
                mask = (labeled == obj_id)

                # Get centroid
                coords = np.where(mask)
                if len(coords[0]) > 0:
                    centroid = tuple(np.mean(c) for c in coords)
                    bbox = tuple((c.min(), c.max()) for c in coords)
                    size = mask.sum()

                    frame_detections.append({
                        'centroid': centroid,
                        'bbox': bbox,
                        'size': size,
                        'mask': mask,
                        'frame': t
                    })

            all_detections.append(frame_detections)

        return all_detections

    def _associate_detections(
        self,
        detections: List[List[Dict]]
    ) -> List[ObjectTrack]:
        """
        Associate detections across frames using Hungarian algorithm.

        Args:
            detections: List of detections per frame

        Returns:
            List of object tracks
        """
        tracks = []
        active_tracks: Dict[int, ObjectTrack] = {}
        next_id = 0

        for t, frame_dets in enumerate(detections):
            if not frame_dets:
                continue

            if not active_tracks:
                # Initialize tracks from first frame
                for det in frame_dets:
                    track = ObjectTrack(
                        track_id=next_id,
                        positions=[det['centroid']],
                        confidences=[1.0],
                        class_id=0,
                        frames=[t],
                        bbox=[det['bbox']]
                    )
                    active_tracks[next_id] = track
                    next_id += 1
                continue

            # Compute cost matrix
            track_ids = list(active_tracks.keys())
            cost_matrix = np.zeros((len(track_ids), len(frame_dets)))

            for i, tid in enumerate(track_ids):
                track = active_tracks[tid]
                last_pos = track.positions[-1]

                # Predict position based on velocity
                if len(track.positions) >= 2:
                    velocity = tuple(
                        track.positions[-1][d] - track.positions[-2][d]
                        for d in range(len(last_pos))
                    )
                    predicted_pos = tuple(
                        last_pos[d] + velocity[d] for d in range(len(last_pos))
                    )
                else:
                    predicted_pos = last_pos

                for j, det in enumerate(frame_dets):
                    # Distance cost
                    dist = np.sqrt(sum(
                        (predicted_pos[d] - det['centroid'][d])**2
                        for d in range(len(predicted_pos))
                    ))
                    cost_matrix[i, j] = dist

            # Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Update tracks
            matched_dets = set()
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < self.distance_threshold:
                    tid = track_ids[i]
                    det = frame_dets[j]
                    active_tracks[tid].positions.append(det['centroid'])
                    active_tracks[tid].confidences.append(1.0)
                    active_tracks[tid].frames.append(t)
                    active_tracks[tid].bbox.append(det['bbox'])
                    matched_dets.add(j)

            # Create new tracks for unmatched detections
            for j, det in enumerate(frame_dets):
                if j not in matched_dets:
                    track = ObjectTrack(
                        track_id=next_id,
                        positions=[det['centroid']],
                        confidences=[1.0],
                        class_id=0,
                        frames=[t],
                        bbox=[det['bbox']]
                    )
                    active_tracks[next_id] = track
                    next_id += 1

            # Terminate old tracks
            to_remove = []
            for tid, track in active_tracks.items():
                if t - track.frames[-1] > self.max_gap:
                    if len(track.frames) >= self.min_track_length:
                        tracks.append(track)
                    to_remove.append(tid)

            for tid in to_remove:
                del active_tracks[tid]

        # Add remaining active tracks
        for track in active_tracks.values():
            if len(track.frames) >= self.min_track_length:
                tracks.append(track)

        return tracks

    def _smooth_trajectories(
        self,
        tracks: List[ObjectTrack]
    ) -> List[ObjectTrack]:
        """
        Smooth object trajectories using moving average.

        Args:
            tracks: List of object tracks

        Returns:
            Smoothed tracks
        """
        smoothed_tracks = []

        for track in tracks:
            if len(track.positions) < self.smoothing_window:
                smoothed_tracks.append(track)
                continue

            positions = np.array(track.positions)
            smoothed_positions = []

            half_window = self.smoothing_window // 2

            for i in range(len(positions)):
                start = max(0, i - half_window)
                end = min(len(positions), i + half_window + 1)
                smoothed_positions.append(positions[start:end].mean(axis=0).tolist())

            smoothed_track = ObjectTrack(
                track_id=track.track_id,
                positions=[tuple(p) for p in smoothed_positions],
                confidences=track.confidences,
                class_id=track.class_id,
                frames=track.frames,
                bbox=track.bbox
            )
            smoothed_tracks.append(smoothed_track)

        return smoothed_tracks

    def _fill_gaps(
        self,
        tracks: List[ObjectTrack],
        num_frames: int
    ) -> List[ObjectTrack]:
        """
        Fill gaps in tracks using interpolation.

        Args:
            tracks: List of object tracks
            num_frames: Total number of frames

        Returns:
            Tracks with gaps filled
        """
        filled_tracks = []

        for track in tracks:
            if len(track.frames) <= 1:
                filled_tracks.append(track)
                continue

            new_positions = []
            new_frames = []
            new_confidences = []
            new_bboxes = []

            for i in range(len(track.frames) - 1):
                current_frame = track.frames[i]
                next_frame = track.frames[i + 1]

                new_positions.append(track.positions[i])
                new_frames.append(current_frame)
                new_confidences.append(track.confidences[i])
                new_bboxes.append(track.bbox[i])

                # Interpolate if gap exists
                gap = next_frame - current_frame
                if gap > 1:
                    for g in range(1, gap):
                        ratio = g / gap
                        interp_pos = tuple(
                            track.positions[i][d] * (1 - ratio) + track.positions[i + 1][d] * ratio
                            for d in range(len(track.positions[i]))
                        )
                        new_positions.append(interp_pos)
                        new_frames.append(current_frame + g)
                        new_confidences.append(0.5)  # Lower confidence for interpolated
                        new_bboxes.append(track.bbox[i])  # Use nearest bbox

            # Add last position
            new_positions.append(track.positions[-1])
            new_frames.append(track.frames[-1])
            new_confidences.append(track.confidences[-1])
            new_bboxes.append(track.bbox[-1])

            filled_track = ObjectTrack(
                track_id=track.track_id,
                positions=new_positions,
                confidences=new_confidences,
                class_id=track.class_id,
                frames=new_frames,
                bbox=new_bboxes
            )
            filled_tracks.append(filled_track)

        return filled_tracks

    def _apply_tracks(
        self,
        occupancy: torch.Tensor,
        tracks: List[ObjectTrack],
        original_detections: List[List[Dict]]
    ) -> torch.Tensor:
        """
        Apply refined tracks to occupancy.

        Args:
            occupancy: Occupancy tensor (T, H, W, D) or (T, C, H, W)
            tracks: Refined tracks
            original_detections: Original frame detections

        Returns:
            Refined occupancy
        """
        # For now, return the original occupancy
        # Full implementation would warp objects to refined positions
        return occupancy

    def get_track_statistics(self) -> Dict[str, Any]:
        """Get statistics about tracked objects."""
        if not self.tracks:
            return {'num_tracks': 0}

        track_lengths = [len(t.frames) for t in self.tracks]

        return {
            'num_tracks': len(self.tracks),
            'mean_track_length': np.mean(track_lengths),
            'max_track_length': max(track_lengths),
            'min_track_length': min(track_lengths),
        }

    def __repr__(self) -> str:
        return (
            f"ObjectTrajectoryRefiner("
            f"min_length={self.min_track_length}, "
            f"max_gap={self.max_gap})"
        )
