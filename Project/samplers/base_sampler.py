"""
Base Sampler Class
==================

Abstract base class for all diffusion samplers.

Provides common functionality including:
    - Noise schedule management
    - Progress tracking
    - Callback support
    - Configuration management
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time


class SamplerType(Enum):
    """Enumeration of available sampler types."""
    DDIM = "ddim"
    DPM_SOLVER = "dpm_solver"
    DPM_SOLVER_PP = "dpm_solver++"
    EULER = "euler"
    EULER_ANCESTRAL = "euler_a"
    HEUN = "heun"
    LMS = "lms"
    HYBRID = "hybrid"


@dataclass
class SamplerConfig:
    """Configuration for samplers."""
    num_inference_steps: int = 50
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    prediction_type: str = "epsilon"  # "epsilon", "v_prediction", "sample"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    rescale_betas_zero_snr: bool = False
    timestep_spacing: str = "leading"  # "leading", "trailing", "linspace"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_inference_steps': self.num_inference_steps,
            'num_train_timesteps': self.num_train_timesteps,
            'beta_start': self.beta_start,
            'beta_end': self.beta_end,
            'beta_schedule': self.beta_schedule,
            'prediction_type': self.prediction_type,
            'clip_sample': self.clip_sample,
            'clip_sample_range': self.clip_sample_range,
        }


@dataclass
class SamplingResult:
    """Container for sampling results."""
    samples: torch.Tensor
    intermediates: Optional[List[torch.Tensor]] = None
    timesteps: Optional[List[int]] = None
    elapsed_time: float = 0.0
    num_function_evaluations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSampler(ABC):
    """
    Abstract base class for diffusion samplers.

    All samplers should inherit from this class and implement
    the `step` and `sample` methods.

    Attributes:
        config: Sampler configuration
        alphas_cumprod: Cumulative product of alphas
        timesteps: Timesteps for sampling
        num_inference_steps: Number of inference steps

    Methods:
        step: Perform one denoising step
        sample: Generate samples from noise
        set_timesteps: Configure sampling timesteps
    """

    def __init__(
        self,
        config: Optional[SamplerConfig] = None,
        **kwargs
    ):
        """
        Initialize the base sampler.

        Args:
            config: Sampler configuration
            **kwargs: Override config parameters
        """
        self.config = config or SamplerConfig()

        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Initialize noise schedule
        self._init_noise_schedule()

        # Tracking
        self._step_count = 0
        self._nfe = 0  # Number of function evaluations
        self._logger = logging.getLogger(self.__class__.__name__)

        # Callbacks
        self._callbacks: List[Callable] = []

    def _init_noise_schedule(self):
        """Initialize the noise schedule (betas, alphas, etc.)."""
        # Generate betas
        if self.config.beta_schedule == "linear":
            betas = np.linspace(
                self.config.beta_start,
                self.config.beta_end,
                self.config.num_train_timesteps,
                dtype=np.float64
            )
        elif self.config.beta_schedule == "scaled_linear":
            # Scaled linear schedule (stable diffusion)
            betas = np.linspace(
                self.config.beta_start ** 0.5,
                self.config.beta_end ** 0.5,
                self.config.num_train_timesteps,
                dtype=np.float64
            ) ** 2
        elif self.config.beta_schedule == "cosine":
            betas = self._cosine_beta_schedule()
        elif self.config.beta_schedule == "squaredcos_cap_v2":
            betas = self._squaredcos_cap_v2_schedule()
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")

        self.betas = torch.from_numpy(betas).float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Previous alphas_cumprod (for DDPM formula)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),
            self.alphas_cumprod[:-1]
        ])

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Log variance
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )

        # Initialize timesteps
        self.timesteps = None
        self.num_inference_steps = None

    def _cosine_beta_schedule(self, s: float = 0.008) -> np.ndarray:
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672

        Args:
            s: Small offset to prevent singularity

        Returns:
            Beta schedule
        """
        steps = self.config.num_train_timesteps + 1
        t = np.linspace(0, self.config.num_train_timesteps, steps, dtype=np.float64)
        alphas_cumprod = np.cos((t / self.config.num_train_timesteps + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)

    def _squaredcos_cap_v2_schedule(self) -> np.ndarray:
        """Squared cosine schedule with cap."""
        return self._cosine_beta_schedule()

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device] = None
    ):
        """
        Set the timesteps for sampling.

        Args:
            num_inference_steps: Number of denoising steps
            device: Device to place timesteps on
        """
        self.num_inference_steps = num_inference_steps

        # Calculate step ratio
        step_ratio = self.config.num_train_timesteps // num_inference_steps

        # Generate timesteps based on spacing strategy
        if self.config.timestep_spacing == "leading":
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            timesteps = np.round(
                np.arange(self.config.num_train_timesteps, 0, -step_ratio)
            )[:num_inference_steps] - 1
        elif self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(
                0, self.config.num_train_timesteps - 1, num_inference_steps
            ).round()
        else:
            raise ValueError(f"Unknown timestep spacing: {self.config.timestep_spacing}")

        self.timesteps = torch.from_numpy(timesteps.astype(np.int64))

        if device is not None:
            self.timesteps = self.timesteps.to(device)

        # Reset counters
        self._step_count = 0
        self._nfe = 0

    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform one denoising step.

        Args:
            model_output: Output from the diffusion model
            timestep: Current timestep
            sample: Current noisy sample
            **kwargs: Additional arguments

        Returns:
            Tuple of (denoised sample, predicted original sample)
        """
        pass

    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        initial_noise: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        progress_callback: Optional[Callable] = None,
        **model_kwargs
    ) -> SamplingResult:
        """
        Generate samples from noise.

        Args:
            model: Diffusion model
            shape: Shape of samples to generate
            condition: Optional conditioning tensor
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            generator: Random generator for reproducibility
            initial_noise: Optional initial noise tensor
            return_intermediates: Whether to return intermediate samples
            progress_callback: Optional progress callback
            **model_kwargs: Additional arguments for the model

        Returns:
            SamplingResult containing generated samples and metadata
        """
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Set timesteps
        if num_inference_steps is not None:
            self.set_timesteps(num_inference_steps, device)
        elif self.timesteps is None:
            self.set_timesteps(self.config.num_inference_steps, device)

        # Initialize noise
        if initial_noise is not None:
            sample = initial_noise.to(device=device, dtype=dtype)
        else:
            sample = torch.randn(
                shape, generator=generator, device=device, dtype=dtype
            )

        # Tracking
        intermediates = [] if return_intermediates else None
        start_time = time.time()

        # Move schedule tensors to device
        self._to_device(device)

        # Sampling loop
        for i, t in enumerate(self.timesteps):
            # Call model
            model_output = self._call_model(
                model, sample, t, condition, guidance_scale, **model_kwargs
            )

            # Perform denoising step
            sample, pred_original = self.step(model_output, t, sample)

            # Store intermediate
            if return_intermediates:
                intermediates.append(sample.cpu().clone())

            # Progress callback
            if progress_callback is not None:
                progress_callback(i, len(self.timesteps), sample)

            # Run custom callbacks
            for callback in self._callbacks:
                callback(i, t, sample)

        elapsed_time = time.time() - start_time

        return SamplingResult(
            samples=sample,
            intermediates=intermediates,
            timesteps=self.timesteps.tolist() if self.timesteps is not None else None,
            elapsed_time=elapsed_time,
            num_function_evaluations=self._nfe,
            metadata={
                'sampler_type': self.__class__.__name__,
                'num_inference_steps': self.num_inference_steps,
                'guidance_scale': guidance_scale,
            }
        )

    def _call_model(
        self,
        model: nn.Module,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor],
        guidance_scale: float,
        **model_kwargs
    ) -> torch.Tensor:
        """
        Call the model with optional CFG.

        Args:
            model: Diffusion model
            sample: Current sample
            timestep: Current timestep
            condition: Conditioning tensor
            guidance_scale: CFG scale
            **model_kwargs: Additional model arguments

        Returns:
            Model output
        """
        self._nfe += 1

        # If model has forward_with_cfg method, use it directly (handles CFG internally)
        if hasattr(model, 'forward_with_cfg'):
            return model.forward_with_cfg(sample, timestep, **model_kwargs)

        if guidance_scale == 1.0 or condition is None:
            # No guidance
            return model(sample, timestep, condition, **model_kwargs)

        # Classifier-free guidance
        # Double batch for unconditional + conditional
        sample_double = torch.cat([sample, sample], dim=0)
        timestep_double = torch.cat([timestep.expand(sample.shape[0])] * 2, dim=0)

        # Prepare conditions (unconditional + conditional)
        uncond = torch.zeros_like(condition)
        cond_double = torch.cat([uncond, condition], dim=0)

        # Forward pass
        model_output = model(sample_double, timestep_double, cond_double, **model_kwargs)

        # Split and apply CFG
        uncond_out, cond_out = model_output.chunk(2, dim=0)
        guided_output = uncond_out + guidance_scale * (cond_out - uncond_out)

        return guided_output

    def _to_device(self, device: torch.device):
        """Move schedule tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)

    def add_callback(self, callback: Callable):
        """Add a callback function."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """Remove a callback function."""
        self._callbacks.remove(callback)

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity for v-prediction.

        Args:
            sample: Original sample x_0
            noise: Noise tensor
            timestep: Timestep

        Returns:
            Velocity
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timestep]

        while sqrt_alpha.dim() < sample.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        velocity = sqrt_alpha * noise - sqrt_one_minus_alpha * sample
        return velocity

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"steps={self.num_inference_steps}, "
            f"beta_schedule={self.config.beta_schedule})"
        )
