"""
Euler Samplers
==============

Euler and Euler Ancestral samplers for diffusion models.

These are simple but effective first-order samplers that work well
across a variety of diffusion models.
"""

import torch
import numpy as np
from typing import Optional, Tuple

from .base_sampler import BaseSampler, SamplerConfig


class EulerSampler(BaseSampler):
    """
    Euler Sampler for diffusion models.

    A simple first-order deterministic sampler that uses the Euler
    method for solving the reverse diffusion ODE.

    Features:
        - Simple and fast
        - Deterministic generation
        - Good quality with 20-50 steps

    Example:
        >>> sampler = EulerSampler(num_inference_steps=30)
        >>> result = sampler.sample(model, shape=(4, 3, 64, 64))
    """

    def __init__(
        self,
        config: Optional[SamplerConfig] = None,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float('inf'),
        s_noise: float = 1.0,
        **kwargs
    ):
        """
        Initialize Euler sampler.

        Args:
            config: Sampler configuration
            s_churn: Churn parameter for stochasticity
            s_tmin: Minimum sigma for churn
            s_tmax: Maximum sigma for churn
            s_noise: Noise multiplier for churn
            **kwargs: Additional config overrides
        """
        super().__init__(config, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

        # Compute sigmas
        self._compute_sigmas()

    def _compute_sigmas(self):
        """Compute sigma schedule."""
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod).sqrt()

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device] = None
    ):
        """Set timesteps for Euler sampling."""
        self.num_inference_steps = num_inference_steps

        # Compute sigmas for inference
        sigmas = np.array(self.sigmas.cpu().numpy())

        # Sample sigmas uniformly in log space
        log_sigmas = np.log(sigmas)
        log_sigmas_interp = np.linspace(
            log_sigmas[0], log_sigmas[-1], num_inference_steps + 1
        )
        sigmas_interp = np.exp(log_sigmas_interp)

        # Convert to timesteps
        timesteps = []
        for sigma in sigmas_interp[:-1]:  # Exclude final sigma
            dists = (sigmas - sigma) ** 2
            timesteps.append(np.argmin(dists))

        self.timesteps = torch.tensor(timesteps[::-1], dtype=torch.long)
        self.inference_sigmas = torch.tensor(sigmas_interp[::-1], dtype=torch.float32)

        if device is not None:
            self.timesteps = self.timesteps.to(device)
            self.inference_sigmas = self.inference_sigmas.to(device)
            self.sigmas = self.sigmas.to(device)

        self._step_count = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one Euler step.

        Args:
            model_output: Model noise prediction
            timestep: Current timestep
            sample: Current sample
            generator: Random generator

        Returns:
            Tuple of (next sample, predicted x_0)
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()

        # Get current and next sigma
        step_idx = self._step_count
        sigma = self.inference_sigmas[step_idx]
        sigma_next = self.inference_sigmas[step_idx + 1]

        # Convert model output to denoised prediction
        if self.config.prediction_type == "epsilon":
            pred_original = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original = sample * self.alphas_cumprod[timestep].sqrt() - \
                          model_output * self.sigmas[timestep]
        else:
            pred_original = model_output

        # Euler step
        derivative = (sample - pred_original) / sigma
        dt = sigma_next - sigma
        prev_sample = sample + derivative * dt

        # Add churn if configured
        if self.s_churn > 0 and self.s_tmin <= sigma <= self.s_tmax:
            gamma = min(self.s_churn / (len(self.timesteps) - 1), 2 ** 0.5 - 1)
            sigma_hat = sigma * (1 + gamma)

            noise = torch.randn(
                sample.shape, generator=generator,
                device=sample.device, dtype=sample.dtype
            )
            prev_sample = prev_sample + (sigma_hat ** 2 - sigma ** 2).sqrt() * self.s_noise * noise

        self._step_count += 1

        return prev_sample, pred_original

    def __repr__(self) -> str:
        return (
            f"EulerSampler("
            f"steps={self.num_inference_steps}, "
            f"churn={self.s_churn})"
        )


class EulerAncestralSampler(EulerSampler):
    """
    Euler Ancestral Sampler.

    A stochastic variant of the Euler sampler that adds noise at each step,
    similar to ancestral sampling in DDPM.

    Features:
        - Stochastic generation
        - More diverse outputs than deterministic Euler
        - Good for creative applications

    Example:
        >>> sampler = EulerAncestralSampler(num_inference_steps=30, eta=1.0)
        >>> result = sampler.sample(model, shape=(4, 3, 64, 64))
    """

    def __init__(
        self,
        config: Optional[SamplerConfig] = None,
        eta: float = 1.0,
        **kwargs
    ):
        """
        Initialize Euler Ancestral sampler.

        Args:
            config: Sampler configuration
            eta: Noise scale (1.0 = full stochasticity)
            **kwargs: Additional config overrides
        """
        super().__init__(config, **kwargs)
        self.eta = eta

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one Euler Ancestral step.

        Args:
            model_output: Model noise prediction
            timestep: Current timestep
            sample: Current sample
            generator: Random generator

        Returns:
            Tuple of (next sample, predicted x_0)
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()

        step_idx = self._step_count
        sigma = self.inference_sigmas[step_idx]
        sigma_next = self.inference_sigmas[step_idx + 1]

        # Convert model output to denoised prediction
        if self.config.prediction_type == "epsilon":
            pred_original = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original = sample * self.alphas_cumprod[timestep].sqrt() - \
                          model_output * self.sigmas[timestep]
        else:
            pred_original = model_output

        # Compute sigma_up and sigma_down for ancestral step
        sigma_up = (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2).sqrt()
        sigma_down = (sigma_next ** 2 - sigma_up ** 2).sqrt()

        # Apply eta
        sigma_up = sigma_up * self.eta

        # Euler step to sigma_down
        derivative = (sample - pred_original) / sigma
        dt = sigma_down - sigma
        prev_sample = sample + derivative * dt

        # Add ancestral noise
        if sigma_next > 0:
            noise = torch.randn(
                sample.shape, generator=generator,
                device=sample.device, dtype=sample.dtype
            )
            prev_sample = prev_sample + sigma_up * noise

        self._step_count += 1

        return prev_sample, pred_original

    def __repr__(self) -> str:
        return (
            f"EulerAncestralSampler("
            f"steps={self.num_inference_steps}, "
            f"eta={self.eta})"
        )
