"""
时序感知CFG模块
===============

专门用于时序/4D生成任务的CFG处理。

本模块扩展了自适应CFG概念以处理时序序列，确保引导在帧之间一致应用，
同时允许时序感知的调整。

核心功能:
    - 帧级引导调整
    - 时序一致性正则化
    - 基于关键帧的引导调度
    - 运动感知引导缩放
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass

from .scheduler import AdaptiveCFGScheduler
from .strategies import TemporalPyramidSchedule


@dataclass
class TemporalCFGConfig:
    """时序感知CFG的配置。"""
    num_frames: int = 16
    cfg_min: float = 1.0
    cfg_max: float = 7.5
    temporal_consistency_weight: float = 0.5
    keyframe_boost: float = 1.2
    keyframe_indices: Optional[List[int]] = None
    motion_adaptive: bool = True
    inter_frame_smoothing: float = 0.3


class TemporalAwareCFG:
    """
    4D生成的时序感知分类器无关引导。

    该类为时序序列（视频、4D占用等）提供专门的CFG处理，
    其中保持帧间一致性至关重要。

    核心概念:
        - 关键帧引导: 重要帧（首帧、末帧、场景变化）获得更高的引导以确定结构
        - 帧间平滑: 引导尺度在帧之间平滑以防止时序伪影
        - 运动自适应缩放: 运动较多的帧可以获得调整后的引导

    使用示例:
        >>> temporal_cfg = TemporalAwareCFG(
        ...     num_frames=16,
        ...     cfg_max=7.5,
        ...     keyframe_indices=[0, 8, 15]
        ... )
        >>>
        >>> # 获取帧特定引导
        >>> for frame_idx in range(16):
        ...     cfg_scale = temporal_cfg.get_frame_guidance(timestep, frame_idx)
    """

    def __init__(
        self,
        num_frames: int = 16,
        cfg_min: float = 1.0,
        cfg_max: float = 7.5,
        num_timesteps: int = 1000,
        temporal_consistency_weight: float = 0.5,
        keyframe_boost: float = 1.2,
        keyframe_indices: Optional[List[int]] = None,
        motion_adaptive: bool = True,
        inter_frame_smoothing: float = 0.3,
        base_schedule: str = "temporal_pyramid"
    ):
        """
        初始化时序感知CFG。

        参数:
            num_frames: 序列中的帧数
            cfg_min: 最小引导尺度
            cfg_max: 最大引导尺度
            num_timesteps: 总扩散时间步
            temporal_consistency_weight: 时序一致性权重（0-1）
            keyframe_boost: 关键帧引导的乘数
            keyframe_indices: 关键帧索引（如果为None则自动检测）
            motion_adaptive: 启用运动自适应引导
            inter_frame_smoothing: 帧间平滑因子（0-1）
            base_schedule: 基础调度策略
        """
        self.num_frames = num_frames
        self.cfg_min = cfg_min
        self.cfg_max = cfg_max
        self.num_timesteps = num_timesteps
        self.temporal_consistency_weight = temporal_consistency_weight
        self.keyframe_boost = keyframe_boost
        self.inter_frame_smoothing = inter_frame_smoothing
        self.motion_adaptive = motion_adaptive

        # 如果未提供则自动检测关键帧
        if keyframe_indices is None:
            self.keyframe_indices = self._auto_detect_keyframes()
        else:
            self.keyframe_indices = keyframe_indices

        # 初始化基础调度器
        self.base_scheduler = AdaptiveCFGScheduler(
            cfg_min=cfg_min,
            cfg_max=cfg_max,
            schedule_type=base_schedule,
            num_timesteps=num_timesteps,
            num_frames=num_frames,
            temporal_weight=temporal_consistency_weight
        )

        # 预计算帧权重
        self._compute_frame_weights()

        # 运动估计缓存
        self._motion_cache = {}

    def _auto_detect_keyframes(self) -> List[int]:
        """
        自动检测关键帧索引。

        返回首帧、中间帧和末帧作为关键帧。
        """
        keyframes = [0]  # 首帧

        # 添加均匀间隔的关键帧
        if self.num_frames >= 4:
            keyframes.append(self.num_frames // 2)  # 中间帧

        keyframes.append(self.num_frames - 1)  # 末帧

        return sorted(set(keyframes))

    def _compute_frame_weights(self):
        """计算每帧的引导权重修正系数。"""
        self.frame_weights = np.ones(self.num_frames)

        # 增强关键帧
        for idx in self.keyframe_indices:
            self.frame_weights[idx] *= self.keyframe_boost

        # 应用高斯平滑以保持时序一致性
        if self.inter_frame_smoothing > 0:
            kernel_size = max(3, int(self.num_frames * self.inter_frame_smoothing))
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma = kernel_size / 4

            kernel = np.exp(-np.arange(-(kernel_size//2), kernel_size//2 + 1)**2 / (2*sigma**2))
            kernel = kernel / kernel.sum()

            # 带填充的卷积
            padded = np.pad(self.frame_weights, kernel_size//2, mode='edge')
            smoothed = np.convolve(padded, kernel, mode='valid')
            self.frame_weights = smoothed

        # 归一化以保持平均值
        self.frame_weights = self.frame_weights / self.frame_weights.mean()

    def get_frame_guidance(
        self,
        timestep: int,
        frame_idx: int,
        motion_magnitude: Optional[float] = None
    ) -> float:
        """
        获取给定时间步特定帧的引导尺度。

        参数:
            timestep: 当前扩散时间步
            frame_idx: 帧索引（0到num_frames-1）
            motion_magnitude: 可选的运动幅度用于自适应缩放

        返回:
            该帧的引导尺度
        """
        # 从调度器获取基础尺度
        base_scale = self.base_scheduler.get_guidance_scale(timestep, frame_idx)

        # 应用帧权重
        frame_weight = self.frame_weights[frame_idx]
        adjusted_scale = base_scale * frame_weight

        # 运动自适应调整
        if self.motion_adaptive and motion_magnitude is not None:
            # 更高运动 -> 略低引导以增加灵活性
            motion_factor = 1.0 - 0.2 * min(1.0, motion_magnitude)
            adjusted_scale *= motion_factor

        return np.clip(adjusted_scale, self.cfg_min, self.cfg_max)

    def get_all_frame_guidances(
        self,
        timestep: int,
        motion_magnitudes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        获取给定时间步所有帧的引导尺度。

        参数:
            timestep: 当前扩散时间步
            motion_magnitudes: 可选的每帧运动幅度数组

        返回:
            每帧的引导尺度数组
        """
        scales = np.zeros(self.num_frames)

        for i in range(self.num_frames):
            motion = motion_magnitudes[i] if motion_magnitudes is not None else None
            scales[i] = self.get_frame_guidance(timestep, i, motion)

        return scales

    def apply_temporal_guidance(
        self,
        noise_pred_uncond: torch.Tensor,
        noise_pred_cond: torch.Tensor,
        timestep: int,
        motion_magnitudes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        将时序感知引导应用于批量帧预测。

        期望输入形状为 (batch, frames, channels, ...) 或
        (batch * frames, channels, ...)。

        参数:
            noise_pred_uncond: 无条件预测
            noise_pred_cond: 条件预测
            timestep: 当前时间步
            motion_magnitudes: 可选的每帧运动幅度

        返回:
            具有时序一致性的引导预测
        """
        # 获取每帧引导尺度
        motion_np = motion_magnitudes.cpu().numpy() if motion_magnitudes is not None else None
        scales = self.get_all_frame_guidances(timestep, motion_np)
        scales_tensor = torch.tensor(scales, device=noise_pred_uncond.device, dtype=noise_pred_uncond.dtype)

        # 重塑用于广播
        orig_shape = noise_pred_uncond.shape

        if len(orig_shape) == 5:  # (batch, frames, channels, h, w)
            batch, frames = orig_shape[:2]
            scales_tensor = scales_tensor.view(1, frames, 1, 1, 1)
        elif len(orig_shape) == 4:  # (batch*frames, channels, h, w)
            # 假设batch维度包含帧
            scales_tensor = scales_tensor.view(-1, 1, 1, 1)

        # 使用每帧尺度应用CFG
        noise_pred = noise_pred_uncond + scales_tensor * (noise_pred_cond - noise_pred_uncond)

        return noise_pred

    def estimate_motion(
        self,
        frames: torch.Tensor
    ) -> torch.Tensor:
        """
        估计连续帧之间的运动幅度。

        参数:
            frames: 形状为 (batch, frames, channels, h, w) 的张量

        返回:
            每帧的运动幅度 (batch, frames)
        """
        batch, num_frames = frames.shape[:2]

        # 计算帧差
        frame_diffs = (frames[:, 1:] - frames[:, :-1]).abs()

        # 每帧平均运动
        motion = frame_diffs.mean(dim=[2, 3, 4])

        # 首帧填充零运动
        motion = F.pad(motion, (1, 0), value=0)

        # 归一化
        motion = motion / (motion.max() + 1e-6)

        return motion

    def get_temporal_consistency_loss(
        self,
        predictions: torch.Tensor,
        guidance_scales: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算时序一致性正则化损失。

        鼓励连续帧之间的平滑过渡。

        参数:
            predictions: 预测帧 (batch, frames, channels, h, w)
            guidance_scales: 可选的每帧引导尺度

        返回:
            时序一致性损失
        """
        # 计算时序差异
        temporal_diff = predictions[:, 1:] - predictions[:, :-1]

        # 按逆引导尺度加权（较低引导 = 需要更多一致性）
        if guidance_scales is not None:
            # 较高引导的帧应该更一致
            weights = guidance_scales[:, :-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            temporal_diff = temporal_diff * weights

        # 时序差异的L2损失
        consistency_loss = (temporal_diff ** 2).mean()

        return consistency_loss * self.temporal_consistency_weight

    def visualize_frame_guidance(
        self,
        timesteps: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ):
        """
        可视化跨帧和时间步的引导尺度。

        参数:
            timesteps: 要可视化的时间步列表
            save_path: 可选的图像保存路径
        """
        import matplotlib.pyplot as plt

        if timesteps is None:
            timesteps = [0, 250, 500, 750, 999]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 图1: 不同时间步下引导vs帧的关系
        ax1 = axes[0]
        frames = np.arange(self.num_frames)

        for t in timesteps:
            scales = self.get_all_frame_guidances(t)
            ax1.plot(frames, scales, label=f't={t}', linewidth=2)

        ax1.set_xlabel('Frame Index', fontsize=12)
        ax1.set_ylabel('Guidance Scale', fontsize=12)
        ax1.set_title('Per-Frame Guidance Scales', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 图2: 时间和帧的引导热力图
        ax2 = axes[1]
        all_timesteps = np.linspace(0, self.num_timesteps-1, 50).astype(int)
        heatmap = np.array([self.get_all_frame_guidances(t) for t in all_timesteps])

        im = ax2.imshow(heatmap, aspect='auto', cmap='viridis',
                       extent=[0, self.num_frames-1, self.num_timesteps, 0])
        ax2.set_xlabel('Frame Index', fontsize=12)
        ax2.set_ylabel('Timestep', fontsize=12)
        ax2.set_title('Temporal Guidance Heatmap', fontsize=14)
        plt.colorbar(im, ax=ax2, label='Guidance Scale')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.close()
        return fig

    def __repr__(self) -> str:
        return (
            f"TemporalAwareCFG("
            f"frames={self.num_frames}, "
            f"keyframes={self.keyframe_indices}, "
            f"cfg_range=[{self.cfg_min}, {self.cfg_max}])"
        )
