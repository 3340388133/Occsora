"""
Adaptive Noise Scheduler
自适应噪声调度器
"""

import torch
import torch.nn as nn
import numpy as np


class AdaptiveNoiseScheduler(nn.Module):
    """
    根据场景复杂度自适应调整噪声调度
    """

    def __init__(self, num_timesteps: int = 1000):
        super().__init__()
        self.num_timesteps = num_timesteps

        # 基础beta调度
        self.register_buffer(
            'betas',
            torch.linspace(0.0001, 0.02, num_timesteps)
        )

        # 复杂度到调度参数的映射
        self.schedule_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [scale, shift]
        )

    def get_adaptive_beta(
        self,
        timestep: int,
        complexity: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            timestep: 当前时间步
            complexity: (B, 3) 场景复杂度
        Returns:
            (B,) 自适应beta值
        """
        base_beta = self.betas[timestep]

        # 根据复杂度调整
        params = self.schedule_net(complexity)
        scale = torch.sigmoid(params[:, 0]) * 0.5 + 0.75  # [0.75, 1.25]
        shift = torch.tanh(params[:, 1]) * 0.005

        adaptive_beta = base_beta * scale + shift
        return adaptive_beta.clamp(0.0001, 0.05)
