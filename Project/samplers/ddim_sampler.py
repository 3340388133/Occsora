"""
DDIM Sampler
============

Denoising Diffusion Implicit Models (DDIM) sampler implementation.

DDIM enables deterministic sampling with fewer steps by using a
non-Markovian diffusion process.

Reference: https://arxiv.org/abs/2010.02502
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .base_sampler import BaseSampler, SamplerConfig


class DDIMSampler(BaseSampler):
    """
    DDIM (Denoising Diffusion Implicit Models) Sampler.

    DDIM provides a more efficient sampling process by:
        - Using a deterministic mapping (eta=0) or stochastic (eta>0)
        - Enabling significantly fewer sampling steps
        - Maintaining sample quality with 10-50 steps instead of 1000

    The key insight is that the forward diffusion can be reversed
    through a non-Markovian process, allowing for step skipping.

    Attributes:
        eta: Stochasticity parameter (0 = deterministic, 1 = DDPM-like)

    Example:
        >>> sampler = DDIMSampler(num_inference_steps=50, eta=0.0)
        >>> result = sampler.sample(model, shape=(4, 3, 64, 64))
    """

    def __init__(
        self,
        config: Optional[SamplerConfig] = None,
        eta: float = 0.0,
        **kwargs
    ):
        """
        Initialize DDIM sampler.

        Args:
            config: Sampler configuration
            eta: Stochasticity parameter
                 - 0.0: Fully deterministic (recommended for fast sampling)
                 - 1.0: Equivalent to DDPM
                 - Values in between: Partial stochasticity
            **kwargs: Additional config overrides
        """
        super().__init__(config, **kwargs)
        self.eta = eta

        # Pre-compute DDIM-specific values
        self._precompute_ddim_values()

    def _precompute_ddim_values(self):
        """Pre-compute values for DDIM sampling."""
        # These will be computed per-step for flexibility
        pass

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device] = None
    ):
        """
        Set timesteps for DDIM sampling.

        Args:
            num_inference_steps: Number of sampling steps
            device: Device for tensors
        """
        super().set_timesteps(num_inference_steps, device)

        # Compute DDIM timestep schedule
        # Use uniform spacing in the original timestep space
        step_ratio = self.config.num_train_timesteps // num_inference_steps

        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )

        self.timesteps = torch.from_numpy(timesteps)

        if device is not None:
            self.timesteps = self.timesteps.to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: Optional[float] = None,
        use_clipped_model_output: bool = False,
        generator: Optional[torch.Generator] = None,
        variance_noise: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one DDIM denoising step.

        Args:
            model_output: Noise prediction from the model
            timestep: Current timestep
            sample: Current noisy sample x_t
            eta: Override eta for this step
            use_clipped_model_output: Whether to clip model output
            generator: Random generator
            variance_noise: Optional pre-generated variance noise

        Returns:
            Tuple of (x_{t-1}, predicted x_0)
        """
        if eta is None:
            eta = self.eta

        # Get timestep index
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()

        # Find previous timestep
        prev_timestep = self._get_previous_timestep(timestep)

        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else torch.tensor(1.0)
        )

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # Compute predicted original sample (x_0)
        if self.config.prediction_type == "epsilon":
            # Model predicts noise
            pred_original_sample = (
                sample - beta_prod_t.sqrt() * model_output
            ) / alpha_prod_t.sqrt()
        elif self.config.prediction_type == "sample":
            # Model directly predicts x_0
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            # Model predicts velocity
            pred_original_sample = (
                alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
            )
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")

        # Clip predicted x_0 if requested
        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range,
                self.config.clip_sample_range
            )

        # Compute variance
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance.sqrt()

        # Compute "direction pointing to x_t"
        if self.config.prediction_type == "epsilon":
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_epsilon = (
                sample - alpha_prod_t.sqrt() * pred_original_sample
            ) / beta_prod_t.sqrt()
        elif self.config.prediction_type == "v_prediction":
            pred_epsilon = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample

        # Compute x_{t-1}
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).sqrt() * pred_epsilon

        prev_sample = (
            alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
        )

        # Add noise if eta > 0
        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn(
                    sample.shape,
                    generator=generator,
                    device=sample.device,
                    dtype=sample.dtype
                )
            prev_sample = prev_sample + std_dev_t * variance_noise

        self._step_count += 1

        return prev_sample, pred_original_sample

    def _get_previous_timestep(self, timestep: int) -> int:
        """Get the previous timestep in the schedule."""
        if self.timesteps is None:
            return timestep - 1

        # Find current position in timesteps
        timesteps_list = self.timesteps.tolist()
        try:
            idx = timesteps_list.index(timestep)
            if idx + 1 < len(timesteps_list):
                return timesteps_list[idx + 1]
            return -1  # Last step
        except ValueError:
            # Timestep not in list, approximate
            return max(0, timestep - self.config.num_train_timesteps // self.num_inference_steps)

    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        """Compute variance for DDIM step."""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else torch.tensor(1.0)
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to samples for a given timestep.

        Args:
            original_samples: Clean samples x_0
            noise: Noise tensor
            timesteps: Timesteps

        Returns:
            Noisy samples x_t
        """
        # Make sure alphas are on correct device
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting
        while sqrt_alpha.dim() < original_samples.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        noisy_samples = sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise

        return noisy_samples

    def __repr__(self) -> str:
        return (
            f"DDIMSampler("
            f"steps={self.num_inference_steps}, "
            f"eta={self.eta}, "
            f"prediction={self.config.prediction_type})"
        )
