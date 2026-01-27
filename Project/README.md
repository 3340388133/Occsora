# OccSora Enhancement Project
# ============================
#
# A comprehensive enhancement toolkit for OccSora 4D Occupancy Generation.
#
# ## Modules
#
# ### 1. Adaptive Classifier-Free Guidance (`adaptive_cfg/`)
# Temporal-aware dynamic CFG scheduling for improved generation quality.
# - Multiple scheduling strategies (linear, cosine, step, adaptive)
# - Per-frame guidance adjustment
# - Multi-condition guidance support
#
# ### 2. Hybrid Multi-Strategy Sampler (`samplers/`)
# Fast, high-quality sampling with automatic strategy selection.
# - DDIM, DPM-Solver++, Euler samplers
# - Quality-speed trade-off optimization
# - Preset configurations for common use cases
#
# ### 3. Spatio-Temporal Consistency Enhancement (`consistency/`)
# Post-processing for improved temporal and spatial consistency.
# - Temporal filtering (EMA, Gaussian, Bilateral, Kalman)
# - Spatial smoothing
# - Object trajectory refinement
# - Multi-scale processing
#
# ## Quick Start
#
# ```python
# from Project.adaptive_cfg import AdaptiveCFGScheduler
# from Project.samplers import SamplerFactory
# from Project.consistency import SpatioTemporalEnhancer
#
# # Adaptive CFG
# scheduler = AdaptiveCFGScheduler(
#     cfg_min=1.0,
#     cfg_max=7.5,
#     schedule_type="cosine"
# )
#
# # Sampler
# factory = SamplerFactory()
# sampler = factory.create_preset("balanced")
#
# # Enhancement
# enhancer = SpatioTemporalEnhancer(
#     enable_temporal_filter=True,
#     enable_spatial_smoothing=True
# )
# result = enhancer.enhance(generated_data)
# ```
#
# ## Run Demo
#
# ```bash
# python Project/demo.py --demo all
# ```
