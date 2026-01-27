"""
Ground Contact Loss
地面接触约束损失
"""

import torch
import torch.nn as nn


class GroundContactLoss(nn.Module):
    """车辆等物体必须与地面接触"""

    def __init__(self, vehicle_classes=[1, 2, 3]):
        super().__init__()
        self.vehicle_classes = vehicle_classes

    def forward(self, pred_occ: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_occ: (B, T, H, W, D) 预测占用
        Returns:
            地面约束损失
        """
        # 检测车辆体素
        vehicle_mask = torch.zeros_like(pred_occ, dtype=torch.bool)
        for cls in self.vehicle_classes:
            vehicle_mask |= (pred_occ == cls)

        # 检查是否接触地面(z=0层)
        ground_contact = vehicle_mask[..., 0]
        floating = vehicle_mask.any(dim=-1) & ~ground_contact

        return floating.float().mean()
