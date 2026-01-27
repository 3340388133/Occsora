"""
CFG调度策略
===========

用于动态分类器无关引导调整的各种调度策略。

每种策略定义了引导尺度如何随扩散时间步变化，
实现对生成质量-多样性权衡的精细控制。
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union
import math


class BaseSchedule(ABC):
    """
    CFG调度策略的抽象基类。

    所有调度策略都应继承此类并实现`get_scale`方法，
    以定义每个时间步的引导尺度。
    """

    def __init__(
        self,
        cfg_min: float = 1.0,
        cfg_max: float = 7.5,
        num_timesteps: int = 1000
    ):
        """
        初始化基础调度器。

        参数:
            cfg_min: 最小引导尺度（通常为1.0表示无引导）
            cfg_max: 最大引导尺度
            num_timesteps: 扩散总时间步数
        """
        self.cfg_min = cfg_min
        self.cfg_max = cfg_max
        self.num_timesteps = num_timesteps
        self._cache = {}

    @abstractmethod
    def get_scale(self, timestep: int) -> float:
        """
        获取特定时间步的引导尺度。

        参数:
            timestep: 当前扩散时间步（0到num_timesteps-1）

        返回:
            给定时间步的引导尺度
        """
        pass

    def get_scale_tensor(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        获取一批时间步的引导尺度。

        参数:
            timesteps: 时间步张量

        返回:
            引导尺度张量
        """
        scales = torch.tensor([self.get_scale(t.item()) for t in timesteps])
        return scales.to(timesteps.device)

    def get_all_scales(self) -> np.ndarray:
        """
        获取所有时间步的引导尺度。

        返回:
            每个时间步的引导尺度数组
        """
        return np.array([self.get_scale(t) for t in range(self.num_timesteps)])

    def visualize(self, save_path: Optional[str] = None):
        """
        可视化调度曲线。

        参数:
            save_path: 可选的图像保存路径
        """
        import matplotlib.pyplot as plt

        timesteps = np.arange(self.num_timesteps)
        scales = self.get_all_scales()

        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, scales, linewidth=2)
        plt.xlabel('时间步', fontsize=12)
        plt.ylabel('CFG尺度', fontsize=12)
        plt.title(f'{self.__class__.__name__} 调度曲线', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, self.num_timesteps)
        plt.ylim(self.cfg_min - 0.5, self.cfg_max + 0.5)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class LinearSchedule(BaseSchedule):
    """
    线性CFG调度策略。

    从t=T时的cfg_max线性插值到t=0时的cfg_min。
    提供从高引导（结构）到低引导（细节）的平滑过渡。

    公式:
        cfg(t) = cfg_min + (cfg_max - cfg_min) * (t / T)

    使用场景:
        需要均匀过渡的通用调度。
    """

    def __init__(
        self,
        cfg_min: float = 1.0,
        cfg_max: float = 7.5,
        num_timesteps: int = 1000,
        reverse: bool = False
    ):
        """
        初始化线性调度器。

        参数:
            cfg_min: 最小引导尺度
            cfg_max: 最大引导尺度
            num_timesteps: 总扩散时间步
            reverse: 如果为True，从cfg_min开始到cfg_max
        """
        super().__init__(cfg_min, cfg_max, num_timesteps)
        self.reverse = reverse

    def get_scale(self, timestep: int) -> float:
        ratio = timestep / max(self.num_timesteps - 1, 1)
        if self.reverse:
            return self.cfg_min + (self.cfg_max - self.cfg_min) * (1 - ratio)
        return self.cfg_min + (self.cfg_max - self.cfg_min) * ratio


class CosineSchedule(BaseSchedule):
    """
    余弦CFG调度策略。

    使用余弦退火实现边界处更平滑的过渡。
    这对于避免引导的突然变化特别有效。

    公式:
        cfg(t) = cfg_min + 0.5 * (cfg_max - cfg_min) * (1 + cos(pi * (1 - t/T)))

    使用场景:
        当平滑过渡很重要时，特别是对于视频生成，
        因为时序一致性很重要。
    """

    def __init__(
        self,
        cfg_min: float = 1.0,
        cfg_max: float = 7.5,
        num_timesteps: int = 1000,
        power: float = 1.0
    ):
        """
        初始化余弦调度器。

        参数:
            cfg_min: 最小引导尺度
            cfg_max: 最大引导尺度
            num_timesteps: 总扩散时间步
            power: 调整曲线陡峭程度的幂因子
        """
        super().__init__(cfg_min, cfg_max, num_timesteps)
        self.power = power

    def get_scale(self, timestep: int) -> float:
        ratio = timestep / max(self.num_timesteps - 1, 1)
        cosine_value = 0.5 * (1 + math.cos(math.pi * (1 - ratio)))
        cosine_value = cosine_value ** self.power
        return self.cfg_min + (self.cfg_max - self.cfg_min) * cosine_value


class StepSchedule(BaseSchedule):
    """
    阶梯式CFG调度策略。

    将扩散过程划分为具有不同引导尺度的独立阶段。
    这允许对不同生成阶段进行显式控制。

    阶段说明:
        - 早期（高t）: 结构形成 -> 高CFG
        - 中期: 细节细化 -> 中CFG
        - 晚期（低t）: 精细细节 -> 低CFG

    使用场景:
        当不同生成阶段需要明显不同的引导级别时。
    """

    def __init__(
        self,
        cfg_min: float = 1.0,
        cfg_max: float = 7.5,
        num_timesteps: int = 1000,
        boundaries: List[float] = [0.3, 0.7],
        scales: Optional[List[float]] = None
    ):
        """
        初始化阶梯调度器。

        参数:
            cfg_min: 最小引导尺度
            cfg_max: 最大引导尺度
            num_timesteps: 总扩散时间步
            boundaries: 阶段边界比例（如[0.3, 0.7]）
            scales: 每个阶段的引导尺度（如果为None则自动计算）
        """
        super().__init__(cfg_min, cfg_max, num_timesteps)
        self.boundaries = sorted(boundaries)

        if scales is None:
            # 自动计算每个阶段的尺度
            num_phases = len(boundaries) + 1
            self.scales = [
                cfg_max - i * (cfg_max - cfg_min) / (num_phases - 1)
                for i in range(num_phases)
            ]
        else:
            self.scales = scales

    def get_scale(self, timestep: int) -> float:
        ratio = timestep / max(self.num_timesteps - 1, 1)

        for i, boundary in enumerate(self.boundaries):
            if ratio < boundary:
                return self.scales[-(i + 1)]
        return self.scales[0]


class AdaptiveSchedule(BaseSchedule):
    """
    自适应CFG调度，基于生成质量估计。

    根据以下因素动态调整引导尺度:
        - 噪声水平估计
        - 梯度幅度
        - 预测置信度

    这实现了内容感知的引导，能够适应正在生成内容的特定特征。

    使用场景:
        复杂场景，其中不同区域可能需要不同的引导级别。
    """

    def __init__(
        self,
        cfg_min: float = 1.0,
        cfg_max: float = 7.5,
        num_timesteps: int = 1000,
        base_schedule: str = "cosine",
        adaptation_strength: float = 0.3
    ):
        """
        初始化自适应调度器。

        参数:
            cfg_min: 最小引导尺度
            cfg_max: 最大引导尺度
            num_timesteps: 总扩散时间步
            base_schedule: 基础调度类型（"linear", "cosine", "step"）
            adaptation_strength: 从基础的适应程度（0-1）
        """
        super().__init__(cfg_min, cfg_max, num_timesteps)
        self.adaptation_strength = adaptation_strength

        # 创建基础调度器
        if base_schedule == "linear":
            self.base = LinearSchedule(cfg_min, cfg_max, num_timesteps)
        elif base_schedule == "cosine":
            self.base = CosineSchedule(cfg_min, cfg_max, num_timesteps)
        elif base_schedule == "step":
            self.base = StepSchedule(cfg_min, cfg_max, num_timesteps)
        else:
            raise ValueError(f"未知的基础调度: {base_schedule}")

        self._noise_history = []

    def get_scale(self, timestep: int, noise_estimate: Optional[float] = None) -> float:
        """
        获取自适应引导尺度。

        参数:
            timestep: 当前时间步
            noise_estimate: 可选的噪声水平估计用于适应

        返回:
            适应后的引导尺度
        """
        base_scale = self.base.get_scale(timestep)

        if noise_estimate is not None:
            # 基于噪声水平进行适应
            # 高噪声 -> 增加引导，低噪声 -> 减少引导
            adaptation = (noise_estimate - 0.5) * 2 * self.adaptation_strength
            adapted_scale = base_scale * (1 + adaptation)
            return np.clip(adapted_scale, self.cfg_min, self.cfg_max)

        return base_scale

    def update_noise_history(self, noise_level: float):
        """更新噪声历史用于趋势分析。"""
        self._noise_history.append(noise_level)
        if len(self._noise_history) > 100:
            self._noise_history.pop(0)

    def get_noise_trend(self) -> float:
        """获取噪声降低趋势。"""
        if len(self._noise_history) < 2:
            return 0.0
        return self._noise_history[-1] - self._noise_history[-2]


class TemporalPyramidSchedule(BaseSchedule):
    """
    4D生成的时序金字塔CFG调度。

    专门为视频/4D占用生成设计，其中时序一致性至关重要。
    使用金字塔结构:
        - 早期时间步: 关注全局时序结构
        - 中期时间步: 细化每帧内容
        - 晚期时间步: 润色细节同时保持一致性

    金字塔方法确保时序关系在早期建立并在整个生成过程中保持。

    使用场景:
        4D占用生成、视频生成、任何时序序列生成。
    """

    def __init__(
        self,
        cfg_min: float = 1.0,
        cfg_max: float = 7.5,
        num_timesteps: int = 1000,
        num_frames: int = 16,
        temporal_weight: float = 0.5
    ):
        """
        初始化时序金字塔调度器。

        参数:
            cfg_min: 最小引导尺度
            cfg_max: 最大引导尺度
            num_timesteps: 总扩散时间步
            num_frames: 序列中的帧数
            temporal_weight: 时序一致性权重（0-1）
        """
        super().__init__(cfg_min, cfg_max, num_timesteps)
        self.num_frames = num_frames
        self.temporal_weight = temporal_weight

        # 预计算金字塔层级
        self._compute_pyramid_levels()

    def _compute_pyramid_levels(self):
        """计算时序调度的金字塔层级。"""
        self.pyramid_levels = []

        # 层级1: 全局结构（高CFG）
        self.pyramid_levels.append({
            'range': (0.7, 1.0),
            'cfg_scale': self.cfg_max,
            'temporal_focus': 'global'
        })

        # 层级2: 区域细化（中高CFG）
        self.pyramid_levels.append({
            'range': (0.4, 0.7),
            'cfg_scale': self.cfg_max * 0.75,
            'temporal_focus': 'regional'
        })

        # 层级3: 局部细节（中CFG）
        self.pyramid_levels.append({
            'range': (0.2, 0.4),
            'cfg_scale': self.cfg_max * 0.5,
            'temporal_focus': 'local'
        })

        # 层级4: 精细细节（低CFG）
        self.pyramid_levels.append({
            'range': (0.0, 0.2),
            'cfg_scale': self.cfg_min,
            'temporal_focus': 'fine'
        })

    def get_scale(self, timestep: int, frame_idx: Optional[int] = None) -> float:
        """
        获取带有可选帧特定调整的引导尺度。

        参数:
            timestep: 当前扩散时间步
            frame_idx: 可选的帧索引用于帧特定缩放

        返回:
            给定时间步和帧的引导尺度
        """
        ratio = timestep / max(self.num_timesteps - 1, 1)

        # 找到金字塔层级
        base_scale = self.cfg_min
        for level in self.pyramid_levels:
            if level['range'][0] <= ratio <= level['range'][1]:
                # 层级内的平滑插值
                level_ratio = (ratio - level['range'][0]) / (level['range'][1] - level['range'][0])
                base_scale = level['cfg_scale']
                break

        # 帧特定调整
        if frame_idx is not None:
            # 中心帧获得略高的引导以保持一致性
            frame_ratio = frame_idx / max(self.num_frames - 1, 1)
            center_distance = abs(frame_ratio - 0.5) * 2  # 中心为0，边缘为1
            frame_adjustment = 1 - center_distance * self.temporal_weight * 0.2
            base_scale *= frame_adjustment

        return np.clip(base_scale, self.cfg_min, self.cfg_max)

    def get_frame_scales(self, timestep: int) -> np.ndarray:
        """
        获取给定时间步所有帧的引导尺度。

        参数:
            timestep: 当前扩散时间步

        返回:
            每帧的引导尺度数组
        """
        return np.array([
            self.get_scale(timestep, frame_idx=i)
            for i in range(self.num_frames)
        ])


def create_schedule(
    schedule_type: str,
    cfg_min: float = 1.0,
    cfg_max: float = 7.5,
    num_timesteps: int = 1000,
    **kwargs
) -> BaseSchedule:
    """
    创建调度策略的工厂函数。

    参数:
        schedule_type: 调度类型（"linear", "cosine", "step", "adaptive", "temporal_pyramid"）
        cfg_min: 最小引导尺度
        cfg_max: 最大引导尺度
        num_timesteps: 总扩散时间步
        **kwargs: 额外的调度特定参数

    返回:
        配置好的调度器实例
    """
    schedules = {
        "linear": LinearSchedule,
        "cosine": CosineSchedule,
        "step": StepSchedule,
        "adaptive": AdaptiveSchedule,
        "temporal_pyramid": TemporalPyramidSchedule,
    }

    if schedule_type not in schedules:
        raise ValueError(f"未知的调度类型: {schedule_type}。"
                        f"可用类型: {list(schedules.keys())}")

    return schedules[schedule_type](
        cfg_min=cfg_min,
        cfg_max=cfg_max,
        num_timesteps=num_timesteps,
        **kwargs
    )
