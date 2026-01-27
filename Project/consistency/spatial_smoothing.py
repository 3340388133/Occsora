"""
空间平滑模块
============

用于4D占用数据的空间平滑和细化。

提供各种空间平滑技术来提高输出质量，同时保留重要的结构细节。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass


class SmoothingType(Enum):
    """空间平滑类型。"""
    GAUSSIAN = "gaussian"
    BILATERAL = "bilateral"
    GUIDED = "guided"
    ANISOTROPIC = "anisotropic"
    MORPHOLOGICAL = "morphological"
    MEDIAN = "median"


@dataclass
class SpatialSmoothingConfig:
    """空间平滑配置。"""
    smoothing_type: str = "gaussian"
    kernel_size: int = 3
    sigma: float = 1.0
    iterations: int = 1
    preserve_edges: bool = True
    edge_threshold: float = 0.1


class SpatialSmoothingModule:
    """
    4D占用的空间平滑模块。

    应用空间平滑以减少噪声并提高视觉质量，同时保留重要的结构边界。

    平滑类型:
        - Gaussian: 标准高斯模糊
        - Bilateral: 边缘保留双边滤波器
        - Guided: 使用引导图像的导向滤波器
        - Anisotropic: 各向异性扩散（Perona-Malik）
        - Morphological: 形态学操作
        - Median: 中值滤波器用于椒盐噪声

    使用示例:
        >>> smoother = SpatialSmoothingModule(smoothing_type="bilateral")
        >>> smoothed = smoother.smooth(occupancy_volume)
    """

    def __init__(
        self,
        smoothing_type: str = "gaussian",
        kernel_size: int = 3,
        sigma: float = 1.0,
        sigma_color: float = 0.1,
        sigma_spatial: float = 2.0,
        iterations: int = 1,
        preserve_edges: bool = True,
        edge_threshold: float = 0.1,
        device: str = "cuda"
    ):
        """
        初始化空间平滑模块。

        参数:
            smoothing_type: 要应用的平滑类型
            kernel_size: 平滑核大小
            sigma: 高斯的标准差
            sigma_color: 双边滤波器的颜色sigma
            sigma_spatial: 双边滤波器的空间sigma
            iterations: 平滑迭代次数
            preserve_edges: 是否保留边缘
            edge_threshold: 边缘检测阈值
            device: 计算设备
        """
        self.smoothing_type = SmoothingType(smoothing_type)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.sigma_color = sigma_color
        self.sigma_spatial = sigma_spatial
        self.iterations = iterations
        self.preserve_edges = preserve_edges
        self.edge_threshold = edge_threshold
        self.device = device

        # 初始化核（延迟初始化，在使用时根据数据设备创建）
        self.gaussian_kernel = None
        self.sobel_x = None
        self.sobel_y = None

    def _ensure_kernels(self, device: torch.device):
        """确保核在正确的设备上。"""
        if self.gaussian_kernel is None or self.gaussian_kernel.device != device:
            self._init_kernels(device)

    def _init_kernels(self, device: torch.device):
        """初始化平滑核。"""
        # 高斯核
        self.gaussian_kernel = self._create_gaussian_kernel(
            self.kernel_size, self.sigma
        ).to(device)

        # 用于边缘检测的Sobel核
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=device).view(1, 1, 3, 3)

        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    def _create_gaussian_kernel(
        self,
        kernel_size: int,
        sigma: float
    ) -> torch.Tensor:
        """创建2D高斯核。"""
        x = torch.arange(kernel_size) - kernel_size // 2
        x = x.float()
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        gaussian_2d = gaussian_2d / gaussian_2d.sum()
        return gaussian_2d.view(1, 1, kernel_size, kernel_size)

    def smooth(
        self,
        data: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        应用空间平滑。

        参数:
            data: 输入数据 (B, C, H, W) 或 (B, T, C, H, W) 或 (B, H, W, D)
            mask: 可选的选择性平滑掩码
            guidance: 导向滤波器的可选引导图像

        返回:
            平滑后的数据
        """
        # 确保核在正确的设备上
        self._ensure_kernels(data.device)

        # 处理不同的输入维度
        orig_shape = data.shape
        is_temporal = len(orig_shape) == 5

        if is_temporal:
            B, T, C, H, W = data.shape
            data = data.reshape(B * T, C, H, W)
            if mask is not None:
                mask = mask.reshape(B * T, *mask.shape[2:])

        # 应用平滑
        for _ in range(self.iterations):
            if self.smoothing_type == SmoothingType.GAUSSIAN:
                smoothed = self._gaussian_smooth(data)
            elif self.smoothing_type == SmoothingType.BILATERAL:
                smoothed = self._bilateral_smooth(data)
            elif self.smoothing_type == SmoothingType.GUIDED:
                smoothed = self._guided_smooth(data, guidance)
            elif self.smoothing_type == SmoothingType.ANISOTROPIC:
                smoothed = self._anisotropic_smooth(data)
            elif self.smoothing_type == SmoothingType.MORPHOLOGICAL:
                smoothed = self._morphological_smooth(data)
            elif self.smoothing_type == SmoothingType.MEDIAN:
                smoothed = self._median_smooth(data)
            else:
                raise ValueError(f"未知的平滑类型: {self.smoothing_type}")

            # 边缘保留
            if self.preserve_edges:
                edge_mask = self._detect_edges(data)
                smoothed = smoothed * (1 - edge_mask) + data * edge_mask

            data = smoothed

        # 应用掩码
        if mask is not None:
            data = data * mask + data * (1 - mask)

        # 恢复原始形状
        if is_temporal:
            data = data.reshape(orig_shape)

        return data

    def _gaussian_smooth(self, data: torch.Tensor) -> torch.Tensor:
        """应用高斯平滑。"""
        padding = self.kernel_size // 2
        C = data.shape[1]

        # 按通道应用
        kernel = self.gaussian_kernel.repeat(C, 1, 1, 1)
        smoothed = F.conv2d(data, kernel, padding=padding, groups=C)

        return smoothed

    def _bilateral_smooth(self, data: torch.Tensor) -> torch.Tensor:
        """
        应用双边滤波。

        同时考虑空间距离和强度相似性的边缘保留滤波器。
        """
        B, C, H, W = data.shape
        padding = self.kernel_size // 2
        device = data.device

        # 展开以获取邻域访问
        unfolded = F.unfold(data, self.kernel_size, padding=padding)
        unfolded = unfolded.view(B, C, self.kernel_size**2, H, W)

        # 计算空间权重（高斯）- 确保在正确的设备上
        y, x = torch.meshgrid(
            torch.arange(self.kernel_size, device=device),
            torch.arange(self.kernel_size, device=device),
            indexing='ij'
        )
        center = self.kernel_size // 2
        spatial_weights = torch.exp(
            -((x.float() - center)**2 + (y.float() - center)**2) / (2 * self.sigma_spatial**2)
        ).view(1, 1, -1, 1, 1)

        # 计算强度权重
        center_val = data.unsqueeze(2)
        intensity_diff = (unfolded - center_val).abs()
        intensity_weights = torch.exp(
            -intensity_diff**2 / (2 * self.sigma_color**2)
        )

        # 组合权重
        weights = spatial_weights * intensity_weights
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)

        # 加权平均
        smoothed = (unfolded * weights).sum(dim=2)

        return smoothed

    def _guided_smooth(
        self,
        data: torch.Tensor,
        guidance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        应用导向滤波器。

        使用引导图像在平滑时保留边缘。
        """
        if guidance is None:
            guidance = data

        eps = 0.01
        r = self.kernel_size // 2

        # 盒式滤波器函数
        def box_filter(x, r):
            kernel = torch.ones(1, 1, 2*r+1, 2*r+1, device=x.device) / ((2*r+1)**2)
            kernel = kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel, padding=r, groups=x.shape[1])

        # 计算均值
        mean_I = box_filter(guidance, r)
        mean_p = box_filter(data, r)
        mean_Ip = box_filter(guidance * data, r)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = box_filter(guidance * guidance, r)
        var_I = mean_II - mean_I * mean_I

        # 计算a和b
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        # 计算输出
        mean_a = box_filter(a, r)
        mean_b = box_filter(b, r)

        smoothed = mean_a * guidance + mean_b

        return smoothed

    def _anisotropic_smooth(self, data: torch.Tensor) -> torch.Tensor:
        """
        应用各向异性扩散（Perona-Malik）。

        基于梯度幅度的平滑，同时保留边缘。
        """
        # 计算梯度
        grad_x = F.conv2d(data, self.sobel_x.repeat(data.shape[1], 1, 1, 1),
                         padding=1, groups=data.shape[1])
        grad_y = F.conv2d(data, self.sobel_y.repeat(data.shape[1], 1, 1, 1),
                         padding=1, groups=data.shape[1])

        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        # 计算扩散系数
        k = self.edge_threshold
        c = torch.exp(-(grad_mag / k)**2)

        # 应用扩散
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=data.dtype, device=data.device).view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(data.shape[1], 1, 1, 1)

        laplacian = F.conv2d(data, laplacian_kernel, padding=1, groups=data.shape[1])

        # 更新
        dt = 0.25  # 时间步长
        smoothed = data + dt * c * laplacian

        return smoothed

    def _morphological_smooth(self, data: torch.Tensor) -> torch.Tensor:
        """
        应用形态学平滑。

        开运算后跟闭运算用于噪声去除。
        """
        padding = self.kernel_size // 2

        B, C, H, W = data.shape

        smoothed = data.clone()

        for c in range(C):
            channel = data[:, c:c+1]

            # 腐蚀（最小池化）
            eroded = -F.max_pool2d(-channel, self.kernel_size, stride=1, padding=padding)

            # 膨胀（最大池化）
            opened = F.max_pool2d(eroded, self.kernel_size, stride=1, padding=padding)

            # 膨胀
            dilated = F.max_pool2d(opened, self.kernel_size, stride=1, padding=padding)

            # 腐蚀
            closed = -F.max_pool2d(-dilated, self.kernel_size, stride=1, padding=padding)

            smoothed[:, c:c+1] = closed

        return smoothed

    def _median_smooth(self, data: torch.Tensor) -> torch.Tensor:
        """
        应用中值滤波。

        适合去除椒盐噪声。
        """
        B, C, H, W = data.shape
        padding = self.kernel_size // 2

        # 展开以获取邻域
        unfolded = F.unfold(data, self.kernel_size, padding=padding)
        unfolded = unfolded.view(B, C, self.kernel_size**2, H, W)

        # 取中值
        smoothed = unfolded.median(dim=2)[0]

        return smoothed

    def _detect_edges(self, data: torch.Tensor) -> torch.Tensor:
        """检测边缘用于保留。"""
        B, C, H, W = data.shape

        edge_mask = torch.zeros_like(data)

        for c in range(C):
            channel = data[:, c:c+1]
            grad_x = F.conv2d(channel, self.sobel_x, padding=1)
            grad_y = F.conv2d(channel, self.sobel_y, padding=1)
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
            edge_mask[:, c:c+1] = (grad_mag > self.edge_threshold).float()

        return edge_mask

    def __repr__(self) -> str:
        return (
            f"SpatialSmoothingModule("
            f"type={self.smoothing_type.value}, "
            f"kernel={self.kernel_size}, "
            f"sigma={self.sigma})"
        )
