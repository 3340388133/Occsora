"""
Collision Loss
碰撞约束损失 - 物体不能重叠
"""

import torch
import torch.nn as nn


class CollisionLoss(nn.Module):
    """物体间不能发生碰撞/重叠"""

    def __init__(self):
        super().__init__()

    def forward(self, pred_occ: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_occ: (B, T, H, W, D) 预测占用
        Returns:
            碰撞损失
        """
        # 统计每个位置的占用数量
        occupied = (pred_occ > 0).float()

        # 理想情况：每个位置最多一个物体
        overlap = (occupied.sum(dim=-1) > 1).float()

        return overlap.mean()
