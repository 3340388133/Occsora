"""
SADS Sampler - 场景自适应扩散采样器
====================================

架构级创新：将SADS集成到推理/采样阶段

与原采样器的区别：
1. 推理时根据当前生成状态估计场景复杂度
2. 动态调整每步的去噪强度（beta/alpha）
3. 复杂场景：更强去噪，更多细节保留
4. 简单场景：更快收敛，减少计算

论文贡献：
- 采样策略级创新，而非仅训练技巧
- 推理时自适应，生成质量与效率的平衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Callable
from tqdm import tqdm


class SceneComplexityEstimator(nn.Module):
    """
    场景复杂度估计器（推理时使用）

    从当前去噪状态估计场景复杂度，用于自适应调度
    """

    def __init__(self, input_channels: int = 128, hidden_dim: int = 128):
        super().__init__()

        # 从特征图估计复杂度
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )

        # 输出3维复杂度向量
        self.complexity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) 当前去噪状态
        Returns:
            (B, 3) 复杂度向量 [空间复杂度, 纹理复杂度, 运动复杂度]
        """
        feat = self.encoder(x)  # (B, hidden, 1, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, hidden)
        complexity = self.complexity_head(feat)  # (B, 3)
        return complexity


class AdaptiveBetaScheduler(nn.Module):
    """
    自适应Beta调度器

    根据场景复杂度动态调整beta值
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        # 基础beta调度（线性）
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.register_buffer('base_betas', betas)

        # 复杂度到调度参数的映射网络
        self.adaptation_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 2),  # [scale, shift]
        )

    def get_adaptive_beta(
        self,
        timestep: int,
        complexity: torch.Tensor
    ) -> torch.Tensor:
        """
        获取自适应beta值

        Args:
            timestep: 当前时间步
            complexity: (B, 3) 场景复杂度
        Returns:
            (B,) 自适应beta值
        """
        base_beta = self.base_betas[timestep]

        # 根据复杂度调整
        params = self.adaptation_net(complexity)
        scale = torch.sigmoid(params[:, 0]) * 0.5 + 0.75  # [0.75, 1.25]
        shift = torch.tanh(params[:, 1]) * 0.005  # [-0.005, 0.005]

        adaptive_beta = base_beta * scale + shift
        return adaptive_beta.clamp(0.0001, 0.05)


class SADSSampler(nn.Module):
    """
    场景自适应扩散采样器 - 架构级创新

    核心创新：推理时根据场景复杂度自适应调整采样策略
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        num_inference_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        input_channels: int = 128,
        clip_denoised: bool = False,
        clip_range: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.clip_denoised = bool(clip_denoised)
        self.clip_range = float(clip_range)

        # 场景复杂度估计器（采样发生在latent空间，默认128通道）
        self.complexity_estimator = SceneComplexityEstimator(input_channels=input_channels)

        # 自适应beta调度器
        self.beta_scheduler = AdaptiveBetaScheduler(
            num_timesteps, beta_start, beta_end
        )

        # 预计算基础调度参数
        self._setup_base_schedule(beta_start, beta_end)

    def _setup_base_schedule(self, beta_start: float, beta_end: float):
        """预计算基础扩散调度参数"""
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer(
            'sqrt_alphas_cumprod',
            torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod',
            torch.sqrt(1.0 - alphas_cumprod)
        )

    def _get_timesteps(self) -> torch.Tensor:
        """获取采样时间步序列"""
        step_ratio = self.num_timesteps // self.num_inference_steps
        timesteps = torch.arange(
            0, self.num_inference_steps
        ) * step_ratio
        timesteps = timesteps.flip(0)  # 从大到小
        return timesteps.long()

    def _estimate_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """估计当前状态的场景复杂度"""
        with torch.no_grad():
            complexity = self.complexity_estimator(x)
        return complexity

    def _adaptive_denoise_step(
        self,
        model: Callable,
        x: torch.Tensor,
        t: int,
        t_prev: int,
        complexity: torch.Tensor,
        model_kwargs: Dict,
    ) -> torch.Tensor:
        """
        自适应去噪步骤

        Args:
            model: 扩散模型
            x: 当前噪声状态
            t: 当前时间步
            t_prev: 下一个时间步
            complexity: 场景复杂度
            model_kwargs: 模型额外参数
        """
        # 获取自适应beta
        adaptive_beta = self.beta_scheduler.get_adaptive_beta(t, complexity)

        # 计算自适应alpha
        adaptive_alpha = 1.0 - adaptive_beta

        # 模型预测噪声
        t_tensor = torch.tensor([t] * x.size(0), device=x.device)

        if 'cfg_scale' in model_kwargs:
            # 使用CFG
            model_output = model(x, t_tensor, **model_kwargs)
        else:
            model_output = model(x, t_tensor, model_kwargs.get('y'))

        # 提取噪声预测（处理learn_sigma的情况）
        if model_output.shape[1] > x.shape[1]:
            noise_pred, _ = model_output.chunk(2, dim=1)
        else:
            noise_pred = model_output

        # 自适应去噪
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x.device)

        # 根据复杂度调整去噪强度
        complexity_factor = complexity.mean(dim=1, keepdim=True)  # (B, 1)
        complexity_factor = complexity_factor.view(-1, 1, 1, 1, 1)  # (B, 1, 1, 1, 1)

        # 复杂场景：保留更多细节（smaller step）
        # 简单场景：更快收敛（larger step）
        step_scale = 1.0 - 0.3 * complexity_factor  # [0.7, 1.0]

        # 计算去噪后的x
        pred_x0 = (x - self.sqrt_one_minus_alphas_cumprod[t] * noise_pred) / self.sqrt_alphas_cumprod[t]
        if self.clip_denoised:
            r = max(1e-8, float(self.clip_range))
            pred_x0 = torch.clamp(pred_x0, -r, r)

        # 计算方向指向x_t
        direction = torch.sqrt(1.0 - alpha_cumprod_prev) * noise_pred

        # 自适应步长
        x_prev = torch.sqrt(alpha_cumprod_prev) * pred_x0 + direction * step_scale

        return x_prev

    def _infer_device(
        self,
        model: Callable,
        noise: Optional[torch.Tensor],
    ) -> torch.device:
        if noise is not None:
            return noise.device

        # Prefer sampler buffers if initialized/moved already.
        for attr in ("alphas_cumprod", "betas"):
            buf = getattr(self, attr, None)
            if isinstance(buf, torch.Tensor):
                return buf.device

        # If a module was passed (some callers may), try to infer from parameters.
        if hasattr(model, "parameters"):
            try:
                return next(model.parameters()).device
            except StopIteration:
                pass
            except Exception:
                pass

        # Fall back to the constructor-provided device.
        try:
            return torch.device(self.device)
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        shape: Tuple[int, ...],
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        自适应采样

        Args:
            model: 扩散模型
            shape: 输出形状
            noise: 初始噪声（可选）
            model_kwargs: 模型额外参数
            progress: 是否显示进度条
        Returns:
            生成的样本
        """
        model_kwargs = model_kwargs or {}
        device = self._infer_device(model=model, noise=noise)

        # 移动采样器到正确设备
        self.to(device)

        # 初始噪声
        if noise is None:
            x = torch.randn(shape, device=device)
        else:
            x = noise.to(device)

        # 获取时间步序列
        timesteps = self._get_timesteps().to(device)

        # 采样循环
        iterator = tqdm(timesteps, desc="SADS Sampling") if progress else timesteps

        for i, t in enumerate(iterator):
            t_int = t.item()
            t_prev = timesteps[i + 1].item() if i + 1 < len(timesteps) else 0

            # 估计当前状态的场景复杂度
            complexity = self._estimate_complexity(x)

            # 自适应去噪步骤
            x = self._adaptive_denoise_step(
                model, x, t_int, t_prev, complexity, model_kwargs
            )

            if progress:
                avg_complexity = complexity.mean().item()
                iterator.set_postfix(complexity=f"{avg_complexity:.3f}")

        return x

    @torch.no_grad()
    def sample_with_adaptive_steps(
        self,
        model: Callable,
        shape: Tuple[int, ...],
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
        min_steps: int = 20,
        max_steps: int = 100,
        complexity_threshold: float = 0.5,
        progress: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """
        自适应步数采样 - 根据场景复杂度动态决定采样步数

        复杂场景：更多步数，保证质量
        简单场景：更少步数，提高效率

        Args:
            model: 扩散模型
            shape: 输出形状
            min_steps: 最小采样步数
            max_steps: 最大采样步数
            complexity_threshold: 复杂度阈值
        Returns:
            (生成的样本, 实际使用的步数)
        """
        model_kwargs = model_kwargs or {}
        device = self._infer_device(model=model, noise=noise)
        self.to(device)

        # 初始噪声
        if noise is None:
            x = torch.randn(shape, device=device)
        else:
            x = noise.to(device)

        # 初始复杂度估计（用于决定步数）
        initial_complexity = self._estimate_complexity(x)
        avg_complexity = initial_complexity.mean().item()

        # 根据复杂度决定步数
        complexity_ratio = min(avg_complexity / complexity_threshold, 1.0)
        adaptive_steps = int(min_steps + (max_steps - min_steps) * complexity_ratio)

        print(f"Scene complexity: {avg_complexity:.3f}, Using {adaptive_steps} steps")

        # 重新计算时间步
        step_ratio = self.num_timesteps // adaptive_steps
        timesteps = torch.arange(0, adaptive_steps) * step_ratio
        timesteps = timesteps.flip(0).long().to(device)

        # 采样循环
        iterator = tqdm(timesteps, desc="Adaptive SADS") if progress else timesteps

        for i, t in enumerate(iterator):
            t_int = t.item()
            t_prev = timesteps[i + 1].item() if i + 1 < len(timesteps) else 0

            complexity = self._estimate_complexity(x)

            x = self._adaptive_denoise_step(
                model, x, t_int, t_prev, complexity, model_kwargs
            )

        return x, adaptive_steps
