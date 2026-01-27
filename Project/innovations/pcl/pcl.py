"""
Physics-Consistent Loss
物理一致性约束损失 - 主模块
"""

import torch
import torch.nn as nn
from .ground_loss import GroundContactLoss
from .collision_loss import CollisionLoss
from .motion_loss import MotionSmoothnessLoss


class PhysicsConsistentLoss(nn.Module):
    """组合所有物理约束损失"""

    def __init__(
        self,
        ground_weight: float = 1.0,
        collision_weight: float = 1.0,
        motion_weight: float = 0.5,
    ):
        super().__init__()
        self.ground_loss = GroundContactLoss()
        self.collision_loss = CollisionLoss()
        self.motion_loss = MotionSmoothnessLoss()

        self.weights = {
            'ground': ground_weight,
            'collision': collision_weight,
            'motion': motion_weight,
        }

    def forward(self, pred_occ: torch.Tensor) -> dict:
        """
        Args:
            pred_occ: (B, T, H, W, D) 预测占用
        Returns:
            损失字典
        """
        losses = {}

        losses['ground'] = self.ground_loss(pred_occ) * self.weights['ground']
        losses['collision'] = self.collision_loss(pred_occ) * self.weights['collision']
        losses['motion'] = self.motion_loss(pred_occ) * self.weights['motion']

        losses['total'] = sum(losses.values())

        return losses
