"""
Noise Scheduler
===============

Noise schedule management for diffusion sampling.

Provides various noise schedules (beta schedules) used in diffusion models.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
from enum import Enum
from dataclasses import dataclass


class BetaScheduleType(Enum):
    """Types of beta schedules."""
    LINEAR = "linear"
    SCALED_LINEAR = "scaled_linear"
    COSINE = "cosine"
    SQUAREDCOS_CAP_V2 = "squaredcos_cap_v2"
    SIGMOID = "sigmoid"


@dataclass
class ScheduleConfig:
    """Configuration for noise schedule."""
    schedule_type: str = "linear"
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    cosine_s: float = 0.008


def get_named_beta_schedule(
    schedule_type: str,
    num_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    **kwargs
) -> np.ndarray:
    """
    Get a pre-defined beta schedule.

    Args:
        schedule_type: Type of schedule
        num_timesteps: Number of timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        **kwargs: Additional arguments

    Returns:
        Beta schedule array
    """
    if schedule_type == "linear":
        return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)

    elif schedule_type == "scaled_linear":
        return np.linspace(
            beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=np.float64
        ) ** 2

    elif schedule_type == "cosine":
        return _cosine_beta_schedule(num_timesteps, kwargs.get('s', 0.008))

    elif schedule_type == "squaredcos_cap_v2":
        return _cosine_beta_schedule(num_timesteps, kwargs.get('s', 0.008))

    elif schedule_type == "sigmoid":
        return _sigmoid_beta_schedule(num_timesteps, beta_start, beta_end)

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def _cosine_beta_schedule(
    num_timesteps: int,
    s: float = 0.008
) -> np.ndarray:
    """
    Cosine schedule from "Improved Denoising Diffusion Probabilistic Models".

    Args:
        num_timesteps: Number of timesteps
        s: Small offset to prevent singularity

    Returns:
        Beta schedule
    """
    steps = num_timesteps + 1
    t = np.linspace(0, num_timesteps, steps, dtype=np.float64)

    alphas_cumprod = np.cos((t / num_timesteps + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def _sigmoid_beta_schedule(
    num_timesteps: int,
    beta_start: float,
    beta_end: float
) -> np.ndarray:
    """
    Sigmoid beta schedule.

    Args:
        num_timesteps: Number of timesteps
        beta_start: Starting beta
        beta_end: Ending beta

    Returns:
        Beta schedule
    """
    betas = np.linspace(-6, 6, num_timesteps, dtype=np.float64)
    betas = 1 / (1 + np.exp(-betas))  # Sigmoid
    betas = betas * (beta_end - beta_start) + beta_start
    return betas


class NoiseScheduler:
    """
    Noise scheduler for diffusion models.

    Manages the noise schedule and provides utilities for:
        - Adding noise to samples
        - Computing alpha/beta values
        - SNR calculations

    Example:
        >>> scheduler = NoiseScheduler(schedule_type="cosine", num_timesteps=1000)
        >>> noisy = scheduler.add_noise(clean_sample, noise, timestep)
    """

    def __init__(
        self,
        schedule_type: str = "linear",
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        **kwargs
    ):
        """
        Initialize noise scheduler.

        Args:
            schedule_type: Type of beta schedule
            num_timesteps: Total diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            **kwargs: Additional schedule arguments
        """
        self.schedule_type = schedule_type
        self.num_timesteps = num_timesteps

        # Get beta schedule
        betas = get_named_beta_schedule(
            schedule_type, num_timesteps, beta_start, beta_end, **kwargs
        )

        self.betas = torch.from_numpy(betas).float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]), self.alphas_cumprod[:-1]
        ])

        # Derived values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # SNR (Signal-to-Noise Ratio)
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)
        self.log_snr = torch.log(self.snr)

        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to samples at given timesteps.

        Args:
            original_samples: Clean samples x_0
            noise: Gaussian noise
            timesteps: Timesteps for each sample

        Returns:
            Noisy samples x_t
        """
        # Move to correct device
        sqrt_alpha = self.sqrt_alphas_cumprod.to(original_samples.device)[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(original_samples.device)[timesteps]

        # Reshape for broadcasting
        while sqrt_alpha.dim() < original_samples.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        noisy_samples = sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise

        return noisy_samples

    def get_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get SNR values for timesteps."""
        return self.snr.to(timesteps.device)[timesteps]

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity for v-prediction.

        Args:
            sample: Original sample
            noise: Noise tensor
            timesteps: Timesteps

        Returns:
            Velocity
        """
        sqrt_alpha = self.sqrt_alphas_cumprod.to(sample.device)[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(sample.device)[timesteps]

        while sqrt_alpha.dim() < sample.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        velocity = sqrt_alpha * noise - sqrt_one_minus_alpha * sample
        return velocity

    def to(self, device: torch.device) -> 'NoiseScheduler':
        """Move scheduler tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.snr = self.snr.to(device)
        return self

    def visualize(self, save_path: Optional[str] = None):
        """Visualize the noise schedule."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        timesteps = np.arange(self.num_timesteps)

        # Beta schedule
        axes[0, 0].plot(timesteps, self.betas.numpy())
        axes[0, 0].set_title('Beta Schedule')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Beta')

        # Alpha cumprod
        axes[0, 1].plot(timesteps, self.alphas_cumprod.numpy())
        axes[0, 1].set_title('Alpha Cumulative Product')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('Alpha Cumprod')

        # SNR
        axes[1, 0].plot(timesteps, self.snr.numpy())
        axes[1, 0].set_title('Signal-to-Noise Ratio')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('SNR')
        axes[1, 0].set_yscale('log')

        # Sqrt values
        axes[1, 1].plot(timesteps, self.sqrt_alphas_cumprod.numpy(), label='sqrt(alpha_cumprod)')
        axes[1, 1].plot(timesteps, self.sqrt_one_minus_alphas_cumprod.numpy(), label='sqrt(1-alpha_cumprod)')
        axes[1, 1].set_title('Sqrt Alpha Values')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.close()
        return fig

    def __repr__(self) -> str:
        return (
            f"NoiseScheduler("
            f"type={self.schedule_type}, "
            f"timesteps={self.num_timesteps})"
        )
