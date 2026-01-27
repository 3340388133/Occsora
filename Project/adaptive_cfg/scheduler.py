"""
自适应CFG调度器
===============

在扩散采样过程中协调动态分类器无关引导调整的主调度器类。

本模块提供自适应CFG的核心功能，包括:
    - 多种调度策略支持
    - 运行时引导尺度计算
    - 与扩散采样循环的集成
    - 日志记录和可视化工具
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Union, Callable, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .strategies import (
    BaseSchedule,
    LinearSchedule,
    CosineSchedule,
    StepSchedule,
    AdaptiveSchedule,
    TemporalPyramidSchedule,
    create_schedule
)


class ScheduleType(Enum):
    """可用调度类型的枚举。"""
    LINEAR = "linear"
    COSINE = "cosine"
    STEP = "step"
    ADAPTIVE = "adaptive"
    TEMPORAL_PYRAMID = "temporal_pyramid"
    CUSTOM = "custom"


@dataclass
class CFGConfig:
    """自适应CFG调度器的配置。"""
    cfg_min: float = 1.0
    cfg_max: float = 7.5
    schedule_type: str = "cosine"
    num_timesteps: int = 1000
    warmup_steps: int = 0
    cooldown_steps: int = 0
    guidance_rescale: float = 0.0  # 来自Common Diffusion Noise Schedules
    dynamic_thresholding: bool = False
    dynamic_threshold_ratio: float = 0.995

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cfg_min': self.cfg_min,
            'cfg_max': self.cfg_max,
            'schedule_type': self.schedule_type,
            'num_timesteps': self.num_timesteps,
            'warmup_steps': self.warmup_steps,
            'cooldown_steps': self.cooldown_steps,
            'guidance_rescale': self.guidance_rescale,
            'dynamic_thresholding': self.dynamic_thresholding,
            'dynamic_threshold_ratio': self.dynamic_threshold_ratio,
        }


class AdaptiveCFGScheduler:
    """
    自适应分类器无关引导调度器。

    该调度器根据当前时间步使用各种调度策略，在扩散采样过程中动态调整引导尺度。

    核心思想是扩散过程的不同阶段受益于不同的引导强度:
        - 早期阶段（高噪声）: 更高的引导用于全局结构
        - 中期阶段: 平衡的引导用于内容细化
        - 晚期阶段（低噪声）: 较低的引导用于精细细节和多样性

    功能特性:
        - 多种内置调度策略
        - 支持自定义调度
        - 引导重缩放以提高样本质量
        - 动态阈值防止过饱和
        - 预热和冷却阶段
        - 全面的日志记录和可视化

    使用示例:
        >>> scheduler = AdaptiveCFGScheduler(
        ...     cfg_min=1.0,
        ...     cfg_max=7.5,
        ...     schedule_type="cosine",
        ...     num_timesteps=1000
        ... )
        >>>
        >>> # 在采样循环中
        >>> for t in timesteps:
        ...     cfg_scale = scheduler.get_guidance_scale(t)
        ...     # 在采样步骤中使用cfg_scale
        ...     noise_pred = scheduler.apply_guidance(
        ...         noise_pred_uncond, noise_pred_cond, cfg_scale
        ...     )
    """

    def __init__(
        self,
        cfg_min: float = 1.0,
        cfg_max: float = 7.5,
        schedule_type: str = "cosine",
        num_timesteps: int = 1000,
        warmup_steps: int = 0,
        cooldown_steps: int = 0,
        guidance_rescale: float = 0.0,
        dynamic_thresholding: bool = False,
        dynamic_threshold_ratio: float = 0.995,
        custom_schedule: Optional[BaseSchedule] = None,
        **schedule_kwargs
    ):
        """
        初始化自适应CFG调度器。

        参数:
            cfg_min: 最小引导尺度（默认: 1.0，无引导）
            cfg_max: 最大引导尺度（默认: 7.5）
            schedule_type: 调度策略类型
                          （"linear", "cosine", "step", "adaptive", "temporal_pyramid"）
            num_timesteps: 扩散总时间步数
            warmup_steps: 预热步数（从cfg_min逐渐增加）
            cooldown_steps: 冷却步数（逐渐降低到cfg_min）
            guidance_rescale: 引导重缩放因子（0.0禁用）
            dynamic_thresholding: 启用动态阈值
            dynamic_threshold_ratio: 动态阈值的百分位数
            custom_schedule: 可选的自定义调度实例
            **schedule_kwargs: 调度的额外参数
        """
        self.cfg_min = cfg_min
        self.cfg_max = cfg_max
        self.schedule_type = schedule_type
        self.num_timesteps = num_timesteps
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.guidance_rescale = guidance_rescale
        self.dynamic_thresholding = dynamic_thresholding
        self.dynamic_threshold_ratio = dynamic_threshold_ratio

        # 初始化调度器
        if custom_schedule is not None:
            self.schedule = custom_schedule
        else:
            self.schedule = create_schedule(
                schedule_type=schedule_type,
                cfg_min=cfg_min,
                cfg_max=cfg_max,
                num_timesteps=num_timesteps,
                **schedule_kwargs
            )

        # 跟踪
        self._step_count = 0
        self._history = []
        self._logger = logging.getLogger(self.__class__.__name__)

        # 缓存以提高效率
        self._scale_cache = {}

    def get_guidance_scale(
        self,
        timestep: Union[int, torch.Tensor],
        frame_idx: Optional[int] = None
    ) -> Union[float, torch.Tensor]:
        """
        获取当前时间步的引导尺度。

        参数:
            timestep: 当前扩散时间步（可以是int或tensor）
            frame_idx: 可选的帧索引用于时序感知调度

        返回:
            引导尺度（int输入返回float，tensor输入返回tensor）
        """
        # 处理tensor输入
        if isinstance(timestep, torch.Tensor):
            if timestep.dim() == 0:
                timestep = timestep.item()
            else:
                return torch.tensor([
                    self.get_guidance_scale(t.item(), frame_idx)
                    for t in timestep
                ], device=timestep.device)

        timestep = int(timestep)

        # 检查缓存
        cache_key = (timestep, frame_idx)
        if cache_key in self._scale_cache:
            return self._scale_cache[cache_key]

        # 从调度器获取基础尺度
        if hasattr(self.schedule, 'get_scale') and frame_idx is not None:
            try:
                base_scale = self.schedule.get_scale(timestep, frame_idx=frame_idx)
            except TypeError:
                base_scale = self.schedule.get_scale(timestep)
        else:
            base_scale = self.schedule.get_scale(timestep)

        # 应用预热
        if self.warmup_steps > 0:
            warmup_ratio = min(1.0, self._step_count / self.warmup_steps)
            base_scale = self.cfg_min + (base_scale - self.cfg_min) * warmup_ratio

        # 应用冷却
        if self.cooldown_steps > 0:
            remaining = self.num_timesteps - self._step_count
            if remaining < self.cooldown_steps:
                cooldown_ratio = remaining / self.cooldown_steps
                base_scale = self.cfg_min + (base_scale - self.cfg_min) * cooldown_ratio

        # 缓存并返回
        self._scale_cache[cache_key] = base_scale
        return base_scale

    def apply_guidance(
        self,
        noise_pred_uncond: torch.Tensor,
        noise_pred_cond: torch.Tensor,
        guidance_scale: Optional[float] = None,
        timestep: Optional[int] = None
    ) -> torch.Tensor:
        """
        将分类器无关引导应用于噪声预测。

        实现CFG公式:
            noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)

        可选引导重缩放和动态阈值。

        参数:
            noise_pred_uncond: 无条件噪声预测
            noise_pred_cond: 条件噪声预测
            guidance_scale: 可选的显式引导尺度（如果为None则自动计算）
            timestep: 当前时间步（如果guidance_scale为None则必需）

        返回:
            引导后的噪声预测
        """
        # 获取引导尺度
        if guidance_scale is None:
            if timestep is None:
                raise ValueError("必须提供guidance_scale或timestep")
            guidance_scale = self.get_guidance_scale(timestep)

        # 应用CFG
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # 应用引导重缩放（来自"Common Diffusion Noise Schedules"）
        if self.guidance_rescale > 0:
            std_cond = noise_pred_cond.std(dim=list(range(1, noise_pred_cond.ndim)), keepdim=True)
            std_cfg = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
            noise_pred = noise_pred * (std_cond / std_cfg) * self.guidance_rescale + \
                        noise_pred * (1 - self.guidance_rescale)

        # 应用动态阈值
        if self.dynamic_thresholding:
            noise_pred = self._apply_dynamic_thresholding(noise_pred)

        return noise_pred

    def _apply_dynamic_thresholding(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用动态阈值防止过饱和。

        该技术基于绝对值的百分位数裁剪超过动态计算阈值的值。

        参数:
            x: 输入张量

        返回:
            阈值处理后的张量
        """
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)

        # 计算动态阈值
        abs_x = x_flat.abs()
        threshold = torch.quantile(abs_x, self.dynamic_threshold_ratio, dim=-1, keepdim=True)
        threshold = torch.clamp(threshold, min=1.0)

        # 应用阈值
        x_flat = torch.clamp(x_flat, -threshold, threshold) / threshold

        return x_flat.reshape(x.shape)

    def step(self):
        """推进调度器一步。"""
        self._step_count += 1

    def reset(self):
        """重置调度器状态。"""
        self._step_count = 0
        self._history.clear()
        self._scale_cache.clear()

    def log_scale(self, timestep: int, scale: float):
        """记录引导尺度用于历史跟踪。"""
        self._history.append({
            'step': self._step_count,
            'timestep': timestep,
            'scale': scale
        })

    def get_config(self) -> CFGConfig:
        """获取调度器配置。"""
        return CFGConfig(
            cfg_min=self.cfg_min,
            cfg_max=self.cfg_max,
            schedule_type=self.schedule_type,
            num_timesteps=self.num_timesteps,
            warmup_steps=self.warmup_steps,
            cooldown_steps=self.cooldown_steps,
            guidance_rescale=self.guidance_rescale,
            dynamic_thresholding=self.dynamic_thresholding,
            dynamic_threshold_ratio=self.dynamic_threshold_ratio,
        )

    def visualize_schedule(self, save_path: Optional[str] = None):
        """
        可视化完整的调度曲线。

        参数:
            save_path: 可选的图像保存路径
        """
        import matplotlib.pyplot as plt

        timesteps = np.arange(self.num_timesteps)
        scales = [self.get_guidance_scale(t) for t in timesteps]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(timesteps, scales, linewidth=2, label='CFG尺度')
        ax.axhline(y=self.cfg_min, color='r', linestyle='--', alpha=0.5, label=f'最小值 ({self.cfg_min})')
        ax.axhline(y=self.cfg_max, color='g', linestyle='--', alpha=0.5, label=f'最大值 ({self.cfg_max})')

        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('引导尺度', fontsize=12)
        ax.set_title(f'自适应CFG调度 ({self.schedule_type})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self._logger.info(f"调度可视化已保存到 {save_path}")

        plt.close()
        return fig

    def __repr__(self) -> str:
        return (
            f"AdaptiveCFGScheduler("
            f"schedule={self.schedule_type}, "
            f"cfg_range=[{self.cfg_min}, {self.cfg_max}], "
            f"timesteps={self.num_timesteps})"
        )


class AdaptiveCFGWrapper(nn.Module):
    """
    自适应CFG的神经网络包装器。

    包装扩散模型以在前向传播中自动应用自适应CFG。
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: AdaptiveCFGScheduler,
        uncond_embedding: Optional[torch.Tensor] = None
    ):
        """
        初始化包装器。

        参数:
            model: 要包装的扩散模型
            scheduler: 自适应CFG调度器
            uncond_embedding: 可选的无条件嵌入
        """
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.register_buffer('uncond_embedding', uncond_embedding)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        带有自动CFG应用的前向传播。

        参数:
            x: 输入张量
            timestep: 当前时间步
            condition: 条件张量
            **kwargs: 模型的额外参数

        返回:
            引导后的模型输出
        """
        # 获取引导尺度
        cfg_scale = self.scheduler.get_guidance_scale(timestep)

        if cfg_scale == 1.0:
            # 不需要引导
            return self.model(x, timestep, condition, **kwargs)

        # 准备无条件输入
        if self.uncond_embedding is not None:
            uncond = self.uncond_embedding.expand(x.shape[0], -1, -1)
        else:
            uncond = torch.zeros_like(condition)

        # 批量前向以提高效率
        x_double = torch.cat([x, x], dim=0)
        t_double = torch.cat([timestep, timestep], dim=0)
        c_double = torch.cat([uncond, condition], dim=0)

        output = self.model(x_double, t_double, c_double, **kwargs)

        # 拆分并应用引导
        uncond_out, cond_out = output.chunk(2, dim=0)
        guided_output = self.scheduler.apply_guidance(
            uncond_out, cond_out, cfg_scale
        )

        self.scheduler.step()
        return guided_output
