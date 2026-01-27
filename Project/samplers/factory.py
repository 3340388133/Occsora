"""
Sampler Factory
===============

Factory pattern for creating samplers with various configurations.
"""

from typing import Optional, Dict, Any, Type

from .base_sampler import BaseSampler, SamplerConfig
from .ddim_sampler import DDIMSampler
from .dpm_solver import DPMSolverSampler
from .euler_sampler import EulerSampler, EulerAncestralSampler
from .hybrid_sampler import HybridSampler


class SamplerFactory:
    """
    Factory for creating diffusion samplers.

    Provides a unified interface for creating samplers with
    appropriate configurations.

    Example:
        >>> factory = SamplerFactory()
        >>> sampler = factory.create("dpm_solver++", num_inference_steps=25)
        >>> # Or use preset
        >>> sampler = factory.create_preset("fast")
    """

    # Registry of available samplers
    SAMPLERS: Dict[str, Type[BaseSampler]] = {
        "ddim": DDIMSampler,
        "dpm_solver": DPMSolverSampler,
        "dpm_solver++": DPMSolverSampler,
        "euler": EulerSampler,
        "euler_a": EulerAncestralSampler,
        "euler_ancestral": EulerAncestralSampler,
        "hybrid": HybridSampler,
    }

    # Presets for common use cases
    PRESETS: Dict[str, Dict[str, Any]] = {
        "fast": {
            "sampler": "dpm_solver++",
            "num_inference_steps": 15,
            "solver_order": 2
        },
        "balanced": {
            "sampler": "dpm_solver++",
            "num_inference_steps": 25,
            "solver_order": 2
        },
        "quality": {
            "sampler": "ddim",
            "num_inference_steps": 50,
            "eta": 0.0
        },
        "high_quality": {
            "sampler": "ddim",
            "num_inference_steps": 100,
            "eta": 0.0
        },
        "diverse": {
            "sampler": "euler_a",
            "num_inference_steps": 30,
            "eta": 1.0
        },
        "deterministic": {
            "sampler": "ddim",
            "num_inference_steps": 50,
            "eta": 0.0
        },
        "stochastic": {
            "sampler": "euler_a",
            "num_inference_steps": 50,
            "eta": 1.0
        }
    }

    def __init__(self, base_config: Optional[SamplerConfig] = None):
        """
        Initialize the factory.

        Args:
            base_config: Base configuration for all created samplers
        """
        self.base_config = base_config or SamplerConfig()

    @classmethod
    def register(cls, name: str, sampler_class: Type[BaseSampler]):
        """
        Register a new sampler type.

        Args:
            name: Name for the sampler
            sampler_class: Sampler class to register
        """
        cls.SAMPLERS[name] = sampler_class

    @classmethod
    def list_samplers(cls) -> list:
        """Get list of available sampler names."""
        return list(cls.SAMPLERS.keys())

    @classmethod
    def list_presets(cls) -> list:
        """Get list of available preset names."""
        return list(cls.PRESETS.keys())

    def create(
        self,
        sampler_type: str,
        config: Optional[SamplerConfig] = None,
        **kwargs
    ) -> BaseSampler:
        """
        Create a sampler instance.

        Args:
            sampler_type: Type of sampler to create
            config: Optional sampler configuration
            **kwargs: Additional sampler-specific arguments

        Returns:
            Configured sampler instance

        Raises:
            ValueError: If sampler type is unknown
        """
        if sampler_type not in self.SAMPLERS:
            raise ValueError(
                f"Unknown sampler type: {sampler_type}. "
                f"Available: {list(self.SAMPLERS.keys())}"
            )

        sampler_class = self.SAMPLERS[sampler_type]
        config = config or self.base_config

        return sampler_class(config=config, **kwargs)

    def create_preset(
        self,
        preset_name: str,
        config: Optional[SamplerConfig] = None,
        **override_kwargs
    ) -> BaseSampler:
        """
        Create a sampler from a preset configuration.

        Args:
            preset_name: Name of the preset
            config: Optional base configuration
            **override_kwargs: Arguments to override preset values

        Returns:
            Configured sampler instance
        """
        if preset_name not in self.PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available: {list(self.PRESETS.keys())}"
            )

        preset = self.PRESETS[preset_name].copy()
        sampler_type = preset.pop("sampler")

        # Merge with overrides
        preset.update(override_kwargs)

        return self.create(sampler_type, config, **preset)

    def create_for_task(
        self,
        task: str,
        config: Optional[SamplerConfig] = None,
        **kwargs
    ) -> BaseSampler:
        """
        Create a sampler optimized for a specific task.

        Args:
            task: Task type ("image", "video", "4d_occupancy", etc.)
            config: Optional configuration
            **kwargs: Additional arguments

        Returns:
            Task-optimized sampler
        """
        task_presets = {
            "image": "balanced",
            "video": "quality",
            "4d_occupancy": "balanced",
            "fast_preview": "fast",
            "final_render": "high_quality",
        }

        preset = task_presets.get(task, "balanced")
        return self.create_preset(preset, config, **kwargs)

    def __repr__(self) -> str:
        return (
            f"SamplerFactory("
            f"samplers={list(self.SAMPLERS.keys())}, "
            f"presets={list(self.PRESETS.keys())})"
        )


# Convenience function
def create_sampler(
    sampler_type: str = "dpm_solver++",
    num_inference_steps: int = 25,
    **kwargs
) -> BaseSampler:
    """
    Convenience function to create a sampler.

    Args:
        sampler_type: Type of sampler
        num_inference_steps: Number of inference steps
        **kwargs: Additional arguments

    Returns:
        Configured sampler
    """
    factory = SamplerFactory()
    config = SamplerConfig(num_inference_steps=num_inference_steps)
    return factory.create(sampler_type, config, **kwargs)
