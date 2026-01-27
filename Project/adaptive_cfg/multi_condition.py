"""
多条件CFG模块
=============

处理具有多个条件信号的分类器无关引导。

在复杂的生成任务（如4D占用生成）中，可能同时使用多个条件
（如轨迹、语义标签、场景上下文）。本模块提供跨多个条件组合CFG的策略。

核心功能:
    - 独立的每条件引导尺度
    - 条件重要性加权
    - 层次化条件组合
    - 自适应条件平衡
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class CompositionMode(Enum):
    """组合多个条件的模式。"""
    ADDITIVE = "additive"           # 引导预测的和
    MULTIPLICATIVE = "multiplicative"  # 乘积（几何平均）
    HIERARCHICAL = "hierarchical"    # 顺序应用
    WEIGHTED = "weighted"           # 加权组合
    ADAPTIVE = "adaptive"           # 学习/自适应权重


@dataclass
class ConditionConfig:
    """单个条件的配置。"""
    name: str
    cfg_scale: float = 7.5
    weight: float = 1.0
    dropout_prob: float = 0.0
    schedule_type: str = "constant"
    priority: int = 0  # 用于层次化组合


@dataclass
class MultiConditionConfig:
    """多条件CFG的配置。"""
    conditions: List[ConditionConfig] = field(default_factory=list)
    composition_mode: str = "weighted"
    normalize_weights: bool = True
    temperature: float = 1.0


class MultiConditionCFG:
    """
    多条件分类器无关引导。

    当存在多个条件信号时处理CFG，提供灵活的组合策略来合并它们的影响。

    组合模式:
        - ADDITIVE: noise = uncond + sum(scale_i * (cond_i - uncond))
        - MULTIPLICATIVE: 单独引导的几何平均
        - HIERARCHICAL: 按优先级顺序应用条件
        - WEIGHTED: 带归一化的加权组合
        - ADAPTIVE: 学习最优组合权重

    使用示例:
        >>> # 定义条件
        >>> trajectory_cond = ConditionConfig(
        ...     name="trajectory",
        ...     cfg_scale=7.5,
        ...     weight=1.0
        ... )
        >>> semantic_cond = ConditionConfig(
        ...     name="semantic",
        ...     cfg_scale=5.0,
        ...     weight=0.5
        ... )
        >>>
        >>> multi_cfg = MultiConditionCFG(
        ...     conditions=[trajectory_cond, semantic_cond],
        ...     composition_mode="weighted"
        ... )
        >>>
        >>> # 应用多条件引导
        >>> output = multi_cfg.apply_guidance(
        ...     uncond_pred,
        ...     {"trajectory": traj_pred, "semantic": sem_pred}
        ... )
    """

    def __init__(
        self,
        conditions: Optional[List[ConditionConfig]] = None,
        composition_mode: str = "weighted",
        normalize_weights: bool = True,
        temperature: float = 1.0,
        num_timesteps: int = 1000
    ):
        """
        初始化多条件CFG。

        参数:
            conditions: 条件配置列表
            composition_mode: 如何组合多个条件
            normalize_weights: 是否归一化条件权重
            temperature: softmax归一化的温度
            num_timesteps: 总扩散时间步
        """
        self.conditions = conditions or []
        self.composition_mode = CompositionMode(composition_mode)
        self.normalize_weights = normalize_weights
        self.temperature = temperature
        self.num_timesteps = num_timesteps

        # 构建条件查找表
        self._condition_map = {c.name: c for c in self.conditions}

        # 如果需要，为每个条件初始化调度器
        self._init_schedulers()

    def _init_schedulers(self):
        """初始化每个条件的调度器。"""
        from .scheduler import AdaptiveCFGScheduler

        self.schedulers = {}
        for cond in self.conditions:
            if cond.schedule_type != "constant":
                self.schedulers[cond.name] = AdaptiveCFGScheduler(
                    cfg_min=1.0,
                    cfg_max=cond.cfg_scale,
                    schedule_type=cond.schedule_type,
                    num_timesteps=self.num_timesteps
                )

    def add_condition(self, condition: ConditionConfig):
        """
        添加新条件。

        参数:
            condition: 要添加的条件配置
        """
        self.conditions.append(condition)
        self._condition_map[condition.name] = condition

    def remove_condition(self, name: str):
        """
        按名称移除条件。

        参数:
            name: 要移除的条件名称
        """
        self.conditions = [c for c in self.conditions if c.name != name]
        self._condition_map.pop(name, None)

    def get_condition_scale(
        self,
        condition_name: str,
        timestep: int
    ) -> float:
        """
        获取特定条件在特定时间步的引导尺度。

        参数:
            condition_name: 条件名称
            timestep: 当前时间步

        返回:
            该条件的引导尺度
        """
        if condition_name not in self._condition_map:
            raise ValueError(f"未知条件: {condition_name}")

        cond = self._condition_map[condition_name]

        # 检查是否使用调度器
        if condition_name in self.schedulers:
            return self.schedulers[condition_name].get_guidance_scale(timestep)

        return cond.cfg_scale

    def get_condition_weight(
        self,
        condition_name: str,
        timestep: Optional[int] = None
    ) -> float:
        """
        获取特定条件的权重。

        参数:
            condition_name: 条件名称
            timestep: 可选的时间步用于时变权重

        返回:
            该条件的权重
        """
        if condition_name not in self._condition_map:
            raise ValueError(f"未知条件: {condition_name}")

        weight = self._condition_map[condition_name].weight

        if self.normalize_weights:
            total_weight = sum(c.weight for c in self.conditions)
            weight = weight / total_weight

        return weight

    def apply_guidance(
        self,
        noise_pred_uncond: torch.Tensor,
        noise_pred_conds: Dict[str, torch.Tensor],
        timestep: int,
        condition_dropout: Optional[Dict[str, bool]] = None
    ) -> torch.Tensor:
        """
        应用多条件引导。

        参数:
            noise_pred_uncond: 无条件预测
            noise_pred_conds: 条件名称到条件预测的字典映射
            timestep: 当前时间步
            condition_dropout: 可选的字典指定哪些条件要丢弃

        返回:
            组合后的引导预测
        """
        if self.composition_mode == CompositionMode.ADDITIVE:
            return self._apply_additive(noise_pred_uncond, noise_pred_conds, timestep, condition_dropout)
        elif self.composition_mode == CompositionMode.MULTIPLICATIVE:
            return self._apply_multiplicative(noise_pred_uncond, noise_pred_conds, timestep, condition_dropout)
        elif self.composition_mode == CompositionMode.HIERARCHICAL:
            return self._apply_hierarchical(noise_pred_uncond, noise_pred_conds, timestep, condition_dropout)
        elif self.composition_mode == CompositionMode.WEIGHTED:
            return self._apply_weighted(noise_pred_uncond, noise_pred_conds, timestep, condition_dropout)
        else:
            raise ValueError(f"未知的组合模式: {self.composition_mode}")

    def _apply_additive(
        self,
        noise_pred_uncond: torch.Tensor,
        noise_pred_conds: Dict[str, torch.Tensor],
        timestep: int,
        condition_dropout: Optional[Dict[str, bool]] = None
    ) -> torch.Tensor:
        """
        应用加性组合。

        公式: pred = uncond + sum(scale_i * weight_i * (cond_i - uncond))
        """
        result = noise_pred_uncond.clone()

        for cond_name, cond_pred in noise_pred_conds.items():
            if cond_name not in self._condition_map:
                continue

            # 检查dropout
            if condition_dropout and condition_dropout.get(cond_name, False):
                continue

            scale = self.get_condition_scale(cond_name, timestep)
            weight = self.get_condition_weight(cond_name, timestep)

            # 添加加权引导
            result = result + scale * weight * (cond_pred - noise_pred_uncond)

        return result

    def _apply_multiplicative(
        self,
        noise_pred_uncond: torch.Tensor,
        noise_pred_conds: Dict[str, torch.Tensor],
        timestep: int,
        condition_dropout: Optional[Dict[str, bool]] = None
    ) -> torch.Tensor:
        """
        应用乘性（几何平均）组合。

        每个条件按其权重比例贡献。
        """
        log_sum = torch.zeros_like(noise_pred_uncond)
        weight_sum = 0.0

        for cond_name, cond_pred in noise_pred_conds.items():
            if cond_name not in self._condition_map:
                continue

            if condition_dropout and condition_dropout.get(cond_name, False):
                continue

            scale = self.get_condition_scale(cond_name, timestep)
            weight = self.get_condition_weight(cond_name, timestep)

            # 单独CFG
            guided = noise_pred_uncond + scale * (cond_pred - noise_pred_uncond)

            # 添加到几何平均（在对数空间）
            log_sum = log_sum + weight * guided
            weight_sum += weight

        if weight_sum > 0:
            return log_sum / weight_sum
        return noise_pred_uncond

    def _apply_hierarchical(
        self,
        noise_pred_uncond: torch.Tensor,
        noise_pred_conds: Dict[str, torch.Tensor],
        timestep: int,
        condition_dropout: Optional[Dict[str, bool]] = None
    ) -> torch.Tensor:
        """
        按优先级应用层次化组合。

        较高优先级的条件先应用，建立基础结构。
        较低优先级的条件细化细节。
        """
        # 按优先级排序条件
        sorted_conds = sorted(
            self.conditions,
            key=lambda c: c.priority,
            reverse=True
        )

        result = noise_pred_uncond.clone()

        for cond in sorted_conds:
            if cond.name not in noise_pred_conds:
                continue

            if condition_dropout and condition_dropout.get(cond.name, False):
                continue

            cond_pred = noise_pred_conds[cond.name]
            scale = self.get_condition_scale(cond.name, timestep)

            # 相对于当前结果应用引导
            result = result + scale * (cond_pred - noise_pred_uncond)

        return result

    def _apply_weighted(
        self,
        noise_pred_uncond: torch.Tensor,
        noise_pred_conds: Dict[str, torch.Tensor],
        timestep: int,
        condition_dropout: Optional[Dict[str, bool]] = None
    ) -> torch.Tensor:
        """
        应用带可选softmax归一化的加权组合。
        """
        # 计算单独的引导预测
        guided_preds = []
        weights = []

        for cond_name, cond_pred in noise_pred_conds.items():
            if cond_name not in self._condition_map:
                continue

            if condition_dropout and condition_dropout.get(cond_name, False):
                continue

            scale = self.get_condition_scale(cond_name, timestep)
            weight = self._condition_map[cond_name].weight

            guided = noise_pred_uncond + scale * (cond_pred - noise_pred_uncond)
            guided_preds.append(guided)
            weights.append(weight)

        if not guided_preds:
            return noise_pred_uncond

        # 堆叠并加权
        guided_stack = torch.stack(guided_preds, dim=0)
        weights = torch.tensor(weights, device=noise_pred_uncond.device, dtype=noise_pred_uncond.dtype)

        if self.normalize_weights:
            weights = torch.softmax(weights / self.temperature, dim=0)
        else:
            weights = weights / weights.sum()

        # 加权求和
        weights = weights.view(-1, *([1] * (guided_stack.dim() - 1)))
        result = (guided_stack * weights).sum(dim=0)

        return result

    def get_effective_scale(
        self,
        timestep: int,
        condition_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        计算有效的组合引导尺度。

        参数:
            timestep: 当前时间步
            condition_weights: 可选的覆盖权重

        返回:
            有效的组合尺度
        """
        total_scale = 0.0
        total_weight = 0.0

        for cond in self.conditions:
            scale = self.get_condition_scale(cond.name, timestep)
            weight = condition_weights.get(cond.name, cond.weight) if condition_weights else cond.weight

            total_scale += scale * weight
            total_weight += weight

        if total_weight > 0:
            return total_scale / total_weight
        return 1.0

    def visualize_conditions(
        self,
        timesteps: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ):
        """
        可视化跨时间步的条件尺度。

        参数:
            timesteps: 要可视化的时间步
            save_path: 可选的图像保存路径
        """
        import matplotlib.pyplot as plt

        if timesteps is None:
            timesteps = list(range(0, self.num_timesteps, self.num_timesteps // 100))

        fig, ax = plt.subplots(figsize=(12, 6))

        for cond in self.conditions:
            scales = [self.get_condition_scale(cond.name, t) for t in timesteps]
            ax.plot(timesteps, scales, label=f'{cond.name} (w={cond.weight:.2f})', linewidth=2)

        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Guidance Scale', fontsize=12)
        ax.set_title('Multi-Condition Guidance Scales', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.close()
        return fig

    def __repr__(self) -> str:
        cond_names = [c.name for c in self.conditions]
        return (
            f"MultiConditionCFG("
            f"conditions={cond_names}, "
            f"mode={self.composition_mode.value})"
        )
