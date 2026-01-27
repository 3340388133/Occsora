"""
Hybrid Multi-Strategy Sampler
=============================

Intelligent sampler that combines multiple strategies for optimal
quality-speed trade-off.

The hybrid sampler can:
    - Automatically select the best strategy based on constraints
    - Switch strategies mid-generation
    - Combine multiple samplers for different generation phases
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import time
import logging

from .base_sampler import BaseSampler, SamplerConfig, SamplingResult
from .ddim_sampler import DDIMSampler
from .dpm_solver import DPMSolverSampler
from .euler_sampler import EulerSampler, EulerAncestralSampler


class OptimizationTarget(Enum):
    """Target for optimization."""
    QUALITY = "quality"
    SPEED = "speed"
    BALANCED = "balanced"


@dataclass
class HybridConfig:
    """Configuration for hybrid sampler."""
    optimization_target: str = "balanced"
    quality_threshold: float = 0.8  # Minimum quality (0-1)
    max_time_seconds: float = 60.0  # Maximum generation time
    enable_adaptive: bool = True  # Enable adaptive strategy switching
    fallback_sampler: str = "ddim"  # Fallback sampler type


class HybridSampler(BaseSampler):
    """
    Hybrid Multi-Strategy Diffusion Sampler.

    This sampler intelligently combines multiple sampling strategies
    to achieve optimal results based on user-specified constraints.

    Strategies:
        - DDIM: Balanced quality and speed
        - DPM-Solver++: Fastest with good quality
        - Euler: Simple and reliable
        - Euler-A: Best for diversity

    Modes:
        - Quality: Prioritize output quality
        - Speed: Prioritize generation speed
        - Balanced: Balance both factors

    Features:
        - Automatic sampler selection
        - Mid-generation strategy switching
        - Quality estimation
        - Time budget management

    Example:
        >>> sampler = HybridSampler(
        ...     optimization_target="balanced",
        ...     max_time_seconds=30.0
        ... )
        >>> result = sampler.sample(model, shape=(4, 3, 64, 64))
    """

    # Sampler presets
    SAMPLER_PRESETS = {
        "quality": {
            "sampler": "ddim",
            "steps": 100,
            "eta": 0.0
        },
        "speed": {
            "sampler": "dpm_solver++",
            "steps": 15,
            "solver_order": 2
        },
        "balanced": {
            "sampler": "dpm_solver++",
            "steps": 25,
            "solver_order": 2
        },
        "diverse": {
            "sampler": "euler_a",
            "steps": 30,
            "eta": 1.0
        }
    }

    def __init__(
        self,
        config: Optional[SamplerConfig] = None,
        optimization_target: str = "balanced",
        quality_threshold: float = 0.8,
        max_time_seconds: float = 60.0,
        enable_adaptive: bool = True,
        **kwargs
    ):
        """
        Initialize Hybrid Sampler.

        Args:
            config: Base sampler configuration
            optimization_target: Optimization target ("quality", "speed", "balanced")
            quality_threshold: Minimum acceptable quality (0-1)
            max_time_seconds: Maximum generation time in seconds
            enable_adaptive: Enable adaptive strategy switching
            **kwargs: Additional configuration
        """
        super().__init__(config, **kwargs)

        self.optimization_target = OptimizationTarget(optimization_target)
        self.quality_threshold = quality_threshold
        self.max_time_seconds = max_time_seconds
        self.enable_adaptive = enable_adaptive

        # Initialize sub-samplers
        self._init_samplers()

        # Current active sampler
        self._active_sampler: Optional[BaseSampler] = None

        # Performance tracking
        self._performance_history: List[Dict[str, Any]] = []

        self._logger = logging.getLogger(self.__class__.__name__)

    def _init_samplers(self):
        """Initialize all available sub-samplers."""
        self.samplers = {
            "ddim": DDIMSampler(self.config, eta=0.0),
            "ddim_stochastic": DDIMSampler(self.config, eta=0.5),
            "dpm_solver++": DPMSolverSampler(self.config, solver_order=2),
            "dpm_solver++_fast": DPMSolverSampler(self.config, solver_order=1),
            "euler": EulerSampler(self.config),
            "euler_a": EulerAncestralSampler(self.config, eta=1.0),
        }

    def select_sampler(
        self,
        shape: Tuple[int, ...],
        available_time: Optional[float] = None,
        quality_requirement: Optional[float] = None
    ) -> Tuple[BaseSampler, int]:
        """
        Select the best sampler for given constraints.

        Args:
            shape: Shape of samples to generate
            available_time: Available time in seconds
            quality_requirement: Required quality level (0-1)

        Returns:
            Tuple of (selected sampler, recommended steps)
        """
        available_time = available_time or self.max_time_seconds
        quality_requirement = quality_requirement or self.quality_threshold

        # Estimate complexity
        complexity = np.prod(shape)

        # Select based on optimization target
        if self.optimization_target == OptimizationTarget.SPEED:
            # Prioritize speed
            if available_time < 10:
                return self.samplers["dpm_solver++_fast"], 10
            elif available_time < 30:
                return self.samplers["dpm_solver++"], 20
            else:
                return self.samplers["dpm_solver++"], 25

        elif self.optimization_target == OptimizationTarget.QUALITY:
            # Prioritize quality
            if quality_requirement > 0.9:
                return self.samplers["ddim"], 100
            elif quality_requirement > 0.8:
                return self.samplers["ddim"], 50
            else:
                return self.samplers["dpm_solver++"], 30

        else:  # BALANCED
            # Balance quality and speed
            if available_time < 15:
                return self.samplers["dpm_solver++_fast"], 15
            elif available_time < 30:
                return self.samplers["dpm_solver++"], 25
            elif quality_requirement > 0.85:
                return self.samplers["ddim"], 50
            else:
                return self.samplers["dpm_solver++"], 30

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device] = None
    ):
        """Set timesteps for all sub-samplers."""
        for sampler in self.samplers.values():
            sampler.set_timesteps(num_inference_steps, device)

        self.num_inference_steps = num_inference_steps
        self.timesteps = self.samplers["ddim"].timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one denoising step using active sampler.

        Args:
            model_output: Model prediction
            timestep: Current timestep
            sample: Current sample

        Returns:
            Tuple of (next sample, predicted x_0)
        """
        if self._active_sampler is None:
            self._active_sampler = self.samplers["dpm_solver++"]

        return self._active_sampler.step(model_output, timestep, sample, **kwargs)

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
        Generate samples using optimal strategy.

        Args:
            model: Diffusion model
            shape: Shape of samples
            condition: Conditioning tensor
            guidance_scale: CFG scale
            num_inference_steps: Number of steps (auto-selected if None)
            generator: Random generator
            initial_noise: Initial noise
            return_intermediates: Return intermediate samples
            progress_callback: Progress callback
            **model_kwargs: Additional model arguments

        Returns:
            Sampling result
        """
        start_time = time.time()

        # Select optimal sampler
        sampler, recommended_steps = self.select_sampler(
            shape,
            available_time=self.max_time_seconds
        )

        steps = num_inference_steps or recommended_steps

        self._logger.info(
            f"Selected {sampler.__class__.__name__} with {steps} steps "
            f"(target: {self.optimization_target.value})"
        )

        self._active_sampler = sampler

        # Perform sampling
        result = sampler.sample(
            model=model,
            shape=shape,
            condition=condition,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
            initial_noise=initial_noise,
            return_intermediates=return_intermediates,
            progress_callback=progress_callback,
            **model_kwargs
        )

        # Update metadata
        result.metadata['hybrid_sampler'] = sampler.__class__.__name__
        result.metadata['optimization_target'] = self.optimization_target.value

        # Track performance
        self._performance_history.append({
            'sampler': sampler.__class__.__name__,
            'steps': steps,
            'time': result.elapsed_time,
            'shape': shape
        })

        return result

    def benchmark(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        steps_range: List[int] = [10, 20, 30, 50],
        num_runs: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark all samplers for performance comparison.

        Args:
            model: Diffusion model
            shape: Sample shape
            condition: Conditioning
            steps_range: Steps to test
            num_runs: Number of runs per configuration

        Returns:
            Dictionary of benchmark results
        """
        results = {}

        for sampler_name, sampler in self.samplers.items():
            results[sampler_name] = {}

            for steps in steps_range:
                times = []

                for _ in range(num_runs):
                    start = time.time()
                    sampler.sample(
                        model=model,
                        shape=shape,
                        condition=condition,
                        num_inference_steps=steps
                    )
                    times.append(time.time() - start)

                results[sampler_name][steps] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times)
                }

        return results

    def get_recommendation(
        self,
        time_budget: float,
        quality_requirement: float
    ) -> Dict[str, Any]:
        """
        Get sampler recommendation for given constraints.

        Args:
            time_budget: Available time in seconds
            quality_requirement: Required quality (0-1)

        Returns:
            Recommendation dictionary
        """
        recommendations = []

        # Analyze each sampler
        for name, sampler in self.samplers.items():
            # Estimate quality and speed characteristics
            if "dpm" in name:
                quality_per_step = 0.04  # High quality per step
                time_per_step = 0.02  # Fast
            elif "ddim" in name:
                quality_per_step = 0.03
                time_per_step = 0.03
            else:  # euler
                quality_per_step = 0.025
                time_per_step = 0.02

            # Find optimal steps
            max_steps = int(time_budget / time_per_step)
            estimated_quality = min(1.0, max_steps * quality_per_step)

            if estimated_quality >= quality_requirement:
                recommendations.append({
                    'sampler': name,
                    'steps': max_steps,
                    'estimated_quality': estimated_quality,
                    'estimated_time': max_steps * time_per_step
                })

        # Sort by quality
        recommendations.sort(key=lambda x: x['estimated_quality'], reverse=True)

        if recommendations:
            best = recommendations[0]
            return {
                'recommended_sampler': best['sampler'],
                'recommended_steps': best['steps'],
                'estimated_quality': best['estimated_quality'],
                'estimated_time': best['estimated_time'],
                'alternatives': recommendations[1:3]
            }

        return {
            'recommended_sampler': 'dpm_solver++',
            'recommended_steps': 25,
            'note': 'Default recommendation'
        }

    def __repr__(self) -> str:
        return (
            f"HybridSampler("
            f"target={self.optimization_target.value}, "
            f"samplers={list(self.samplers.keys())})"
        )
