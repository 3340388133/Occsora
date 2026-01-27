"""
Scene Complexity Encoder
场景复杂度编码器
"""

import torch
import torch.nn as nn


class SceneComplexityEncoder(nn.Module):
    """
    编码场景复杂度特征

    输出: [道路复杂度, 物体密度, 运动强度]
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.complexity_head = nn.Linear(hidden_dim, 3)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            condition: (B, D) 条件向量
        Returns:
            (B, 3) 复杂度向量
        """
        feat = self.encoder(condition)
        complexity = torch.sigmoid(self.complexity_head(feat))
        return complexity
