"""
Motion Smoothness Loss
运动平滑约束损失
"""

import torch
import torch.nn as nn


class MotionSmoothnessLoss(nn.Module):
    """物体运动应平滑连续"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_occ: torch.Tensor,
        prev_occ: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pred_occ: (B, T, H, W, D)
            prev_occ: 上一时刻占用
        Returns:
            运动平滑损失
        """
        # 帧间差异
        pred_float = pred_occ.float()
        diff = (pred_float[:, 1:] - pred_float[:, :-1]).abs()

        # 二阶导数(加速度)应该小
        accel = diff[:, 1:] - diff[:, :-1]

        return accel.abs().mean()
