"""
Enhanced Diffusion Module
=========================

集成自适应CFG和快速采样器的增强版扩散模块。
与原OccSora的gaussian_diffusion.py兼容。
"""

import math
import numpy as np
import torch as th
from typing import Optional, Dict, Any, Callable, Tuple
from tqdm.auto import tqdm

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps

import sys
sys.path.insert(0, '/root/OccSora-main')

from Project.adaptive_cfg import AdaptiveCFGScheduler, TemporalAwareCFG
from Project.consistency import SpatioTemporalEnhancer
from Project.samplers import DPMSolverSampler, DDIMSampler, SamplerFactory


class EnhancedDiffusion(SpacedDiffusion):
    """
    增强版扩散采样器，集成:
    1. 自适应CFG调度
    2. 快速采样 (DDIM/DPM-Solver++)
    3. 时空一致性后处理
    """

    def __init__(
        self,
        *,
        use_timesteps,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        # 增强参数
        use_adaptive_cfg: bool = True,
        cfg_schedule_type: str = "cosine",
        cfg_min: float = 1.0,
        cfg_max: float = 7.5,
        num_frames: int = 16,
        use_temporal_cfg: bool = True,
        use_post_enhancement: bool = True,
    ):
        super().__init__(
            use_timesteps=use_timesteps,
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=loss_type,
        )

        self.use_adaptive_cfg = use_adaptive_cfg
        self.use_temporal_cfg = use_temporal_cfg
        self.use_post_enhancement = use_post_enhancement
        self.num_frames = num_frames

        # 初始化自适应CFG调度器
        if use_adaptive_cfg:
            if use_temporal_cfg:
                self.cfg_scheduler = TemporalAwareCFG(
                    num_frames=num_frames,
                    cfg_min=cfg_min,
                    cfg_max=cfg_max,
                    num_timesteps=self.num_timesteps,
                    keyframe_indices=[0, num_frames // 2, num_frames - 1],
                    motion_adaptive=True,
                )
            else:
                self.cfg_scheduler = AdaptiveCFGScheduler(
                    cfg_min=cfg_min,
                    cfg_max=cfg_max,
                    schedule_type=cfg_schedule_type,
                    num_timesteps=self.num_timesteps,
                )
        else:
            self.cfg_scheduler = None

        # 初始化后处理增强器
        if use_post_enhancement:
            self.enhancer = SpatioTemporalEnhancer(
                enable_temporal_filter=True,
                enable_spatial_smoothing=True,
                enable_multi_scale=False,
                temporal_filter_type="ema",
                temporal_alpha=0.3,
                device="cuda" if th.cuda.is_available() else "cpu",
            )
        else:
            self.enhancer = None

    def get_adaptive_cfg_scale(
        self,
        timestep: int,
        base_cfg_scale: float,
        frame_idx: Optional[int] = None
    ) -> float:
        """获取自适应CFG尺度"""
        if not self.use_adaptive_cfg or self.cfg_scheduler is None:
            return base_cfg_scale

        if self.use_temporal_cfg:
            # TemporalAwareCFG使用get_frame_guidance
            frame_idx = frame_idx if frame_idx is not None else 0
            return self.cfg_scheduler.get_frame_guidance(timestep, frame_idx)
        else:
            # AdaptiveCFGScheduler使用get_guidance_scale
            return self.cfg_scheduler.get_guidance_scale(timestep)

    def p_sample_loop_enhanced(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        use_ddim=False,
        ddim_eta=0.0,
        apply_enhancement=True,
    ):
        """
        增强版采样循环，支持自适应CFG和后处理。

        Args:
            model: 扩散模型
            shape: 输出形状
            noise: 初始噪声
            clip_denoised: 是否裁剪去噪结果
            denoised_fn: 去噪后处理函数
            cond_fn: 条件函数
            model_kwargs: 模型额外参数 (包含cfg_scale)
            device: 设备
            progress: 显示进度条
            use_ddim: 使用DDIM采样
            ddim_eta: DDIM随机性参数
            apply_enhancement: 是否应用后处理增强

        Returns:
            生成的样本
        """
        if device is None:
            device = next(model.parameters()).device

        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            indices = tqdm(indices, desc="Sampling")

        # 获取基础CFG尺度
        base_cfg_scale = model_kwargs.get('cfg_scale', 4.0) if model_kwargs else 4.0

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)

            # 计算自适应CFG尺度
            adaptive_cfg = self.get_adaptive_cfg_scale(i, base_cfg_scale)

            # 更新model_kwargs中的cfg_scale
            if model_kwargs is not None:
                model_kwargs_step = model_kwargs.copy()
                model_kwargs_step['cfg_scale'] = adaptive_cfg
            else:
                model_kwargs_step = {'cfg_scale': adaptive_cfg}

            with th.no_grad():
                if use_ddim:
                    out = self.ddim_sample(
                        model, img, t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs_step,
                        eta=ddim_eta,
                    )
                else:
                    out = self.p_sample(
                        model, img, t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs_step,
                    )
                img = out["sample"]

        # 应用后处理增强
        if apply_enhancement and self.use_post_enhancement and self.enhancer is not None:
            img = self._apply_post_enhancement(img)

        return img

    def _apply_post_enhancement(self, samples: th.Tensor) -> th.Tensor:
        """应用时空一致性后处理增强"""
        if self.enhancer is None:
            return samples

        # 重塑为时序格式 (B, T, C, H, W)
        original_shape = samples.shape
        if len(original_shape) == 5:
            # 已经是 (B, T, C, H, W) 格式
            pass
        elif len(original_shape) == 4:
            # (B*T, C, H, W) -> (B, T, C, H, W)
            samples = samples.unsqueeze(0)

        try:
            result = self.enhancer.enhance(samples)
            enhanced = result.enhanced_data
        except Exception as e:
            print(f"Warning: Post-enhancement failed: {e}")
            enhanced = samples

        # 恢复原始形状
        if len(original_shape) == 4:
            enhanced = enhanced.squeeze(0)

        return enhanced

    def ddim_sample_loop_fast(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        progress=True,
        eta=0.0,
        num_steps=50,
        apply_enhancement=True,
    ):
        """
        快速DDIM采样，支持自定义步数。

        Args:
            num_steps: 采样步数 (默认50，原始为1000)
        """
        if device is None:
            device = next(model.parameters()).device

        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        # 计算跳步间隔
        skip = self.num_timesteps // num_steps
        seq = list(range(0, self.num_timesteps, skip))
        seq_next = [-1] + list(seq[:-1])

        if progress:
            pairs = list(zip(reversed(seq), reversed(seq_next)))
            pairs = tqdm(pairs, desc=f"DDIM Sampling ({num_steps} steps)")
        else:
            pairs = list(zip(reversed(seq), reversed(seq_next)))

        base_cfg_scale = model_kwargs.get('cfg_scale', 4.0) if model_kwargs else 4.0

        for i, j in pairs:
            t = th.tensor([i] * shape[0], device=device)
            t_next = th.tensor([j] * shape[0], device=device) if j >= 0 else None

            # 自适应CFG
            adaptive_cfg = self.get_adaptive_cfg_scale(i, base_cfg_scale)
            model_kwargs_step = model_kwargs.copy() if model_kwargs else {}
            model_kwargs_step['cfg_scale'] = adaptive_cfg

            with th.no_grad():
                out = self.ddim_sample(
                    model, img, t,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs_step,
                    eta=eta,
                )
                img = out["sample"]

        if apply_enhancement and self.enhancer is not None:
            img = self._apply_post_enhancement(img)

        return img

    def dpm_solver_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        progress=True,
        num_steps=25,
        solver_order=2,
        apply_enhancement=True,
    ):
        """
        DPM-Solver++ 风格的快速采样，使用DDIM加速。

        通过减少采样步数实现加速，同时保持自适应CFG。

        Args:
            num_steps: 采样步数 (推荐15-25步)
            solver_order: 未使用，保留接口兼容
        """
        # 使用DDIM快速采样实现类似DPM-Solver的加速效果
        return self.ddim_sample_loop_fast(
            model=model,
            shape=shape,
            noise=noise,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=0.0,  # 确定性采样
            num_steps=num_steps,
            apply_enhancement=apply_enhancement,
        )


def create_enhanced_diffusion(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    # 增强参数
    use_adaptive_cfg=True,
    cfg_schedule_type="cosine",
    cfg_min=1.0,
    cfg_max=7.5,
    num_frames=16,
    use_temporal_cfg=True,
    use_post_enhancement=True,
):
    """
    创建增强版扩散模型的工厂函数。
    与原create_diffusion兼容，增加了自适应CFG和后处理选项。
    """
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]

    return EnhancedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
            if not learn_sigma else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        use_adaptive_cfg=use_adaptive_cfg,
        cfg_schedule_type=cfg_schedule_type,
        cfg_min=cfg_min,
        cfg_max=cfg_max,
        num_frames=num_frames,
        use_temporal_cfg=use_temporal_cfg,
        use_post_enhancement=use_post_enhancement,
    )
