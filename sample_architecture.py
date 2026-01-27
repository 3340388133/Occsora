"""
推理脚本 - 架构级创新版本
==========================

使用架构级创新进行推理：
1. DiT_STCA: 时空因果注意力（推理时也生效）
2. SADSSampler: 场景自适应采样（推理时自适应调度）

用法:
    # 标准推理（使用STCA + SADS）
    python sample_architecture.py --model DiT-STCA-XL/2 --use-sads

    # 自适应步数推理
    python sample_architecture.py --model DiT-STCA-XL/2 --adaptive-steps

    # 对比实验（不使用SADS）
    python sample_architecture.py --model DiT-STCA-XL/2 --no-sads
"""

import os
import sys
import argparse
import numpy as np
import torch
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, '/root/OccSora-main')

# 导入架构级创新模型
from models_stca import DiT_STCA_models

# 导入扩散过程
from diffusion import create_diffusion

# 导入架构级SADS采样器
from Project.innovations.sads import SADSSampler

# 导入后处理增强
from Project.consistency import SpatioTemporalEnhancer


def sample_baseline(model, diffusion, z, y, args, device):
    """基线采样（不使用SADS）"""
    print("\n[Baseline Sampling - No SADS]")

    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    start_time = time.time()
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    elapsed = time.time() - start_time

    print(f"Baseline sampling completed in {elapsed:.2f}s")
    return samples


def sample_with_sads(model, z, y, args, device):
    """使用SADS采样器"""
    print("\n[SADS Sampling - Scene-Adaptive]")

    # 创建SADS采样器
    sads_sampler = SADSSampler(
        num_timesteps=1000,
        num_inference_steps=args.num_steps,
        device=device,
    ).to(device)

    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    start_time = time.time()
    samples = sads_sampler.sample(
        model=model.forward_with_cfg,
        shape=z.shape,
        noise=z,
        model_kwargs=model_kwargs,
        progress=True,
    )
    elapsed = time.time() - start_time

    print(f"SADS sampling completed in {elapsed:.2f}s")
    return samples


def sample_adaptive_steps(model, z, y, args, device):
    """自适应步数采样"""
    print("\n[Adaptive Steps Sampling]")
    print(f"Step range: {args.min_steps} - {args.max_steps}")

    # 创建SADS采样器
    sads_sampler = SADSSampler(
        num_timesteps=1000,
        num_inference_steps=args.num_steps,
        device=device,
    ).to(device)

    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    start_time = time.time()
    samples, actual_steps = sads_sampler.sample_with_adaptive_steps(
        model=model.forward_with_cfg,
        shape=z.shape,
        noise=z,
        model_kwargs=model_kwargs,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        complexity_threshold=args.complexity_threshold,
        progress=True,
    )
    elapsed = time.time() - start_time

    print(f"Adaptive sampling completed in {elapsed:.2f}s (used {actual_steps} steps)")
    return samples


def main():
    parser = argparse.ArgumentParser(description="OccSora Architecture-Level Sampling")

    # 模型参数
    parser.add_argument("--model", type=str, default="DiT-STCA-XL/2",
                        choices=list(DiT_STCA_models.keys()))
    parser.add_argument("--ckpt", type=str, default=None,
                        help="检查点路径")

    # 架构级创新开关
    parser.add_argument("--use-stca", action="store_true", default=False,
                        help="使用STCA架构")
    parser.add_argument("--no-stca", action="store_true",
                        help="不使用STCA")
    parser.add_argument("--use-sads", action="store_true", default=False,
                        help="使用SADS采样器")
    parser.add_argument("--no-sads", action="store_true",
                        help="不使用SADS")

    # 采样参数
    parser.add_argument("--num-steps", type=int, default=50,
                        help="采样步数")
    parser.add_argument("--cfg-scale", type=float, default=4.0,
                        help="CFG尺度")
    parser.add_argument("--seed", type=int, default=42)

    # 自适应步数参数
    parser.add_argument("--adaptive-steps", action="store_true",
                        help="启用自适应步数")
    parser.add_argument("--min-steps", type=int, default=20,
                        help="最小步数")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="最大步数")
    parser.add_argument("--complexity-threshold", type=float, default=0.5,
                        help="复杂度阈值")

    # 后处理
    parser.add_argument("--post-enhance", action="store_true",
                        help="启用时空一致性后处理")

    # 路径
    parser.add_argument("--condition-path", type=str,
                        default="/root/autodl-tmp/OccSora_output/vqvae/step32-2/gt_mode/i_iter_0.npy")
    parser.add_argument("--output-path", type=str,
                        default="/root/OccSora-main/out/samples_architecture.npy")

    args = parser.parse_args()

    # 处理开关
    use_stca = args.use_stca and not args.no_stca
    use_sads = args.use_sads and not args.no_sads

    print("=" * 60)
    print("OccSora Architecture-Level Sampling")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"STCA (Architecture-Level): {use_stca}")
    print(f"SADS (Scene-Adaptive Sampling): {use_sads}")
    print(f"Adaptive Steps: {args.adaptive_steps}")
    print(f"Sampling Steps: {args.num_steps}")
    print(f"CFG Scale: {args.cfg_scale}")
    print("=" * 60)

    # 设置
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = DiT_STCA_models[args.model](use_stca=use_stca).to(device)

    # 加载检查点
    if args.ckpt:
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {args.ckpt}")

    model.eval()

    # 加载条件
    if os.path.exists(args.condition_path):
        data = np.load(args.condition_path)
        print(f"Condition data shape: {data.shape}")
    else:
        print(f"Warning: Condition file not found, using random")
        data = np.zeros((64,), dtype=np.float32)

    # Pad到64帧
    target_frames = 64
    current_frames = len(data)
    if current_frames < target_frames:
        repeats = (target_frames // current_frames) + 1
        data = np.tile(data, repeats)[:target_frames]
    else:
        data = data[:target_frames]

    # 准备batch
    y = np.stack([data, data], axis=0).astype(np.float32)
    y = torch.tensor(y, device=device)

    # 初始噪声
    z = torch.randn((2, 128, 4, 25, 25), device=device)

    # 选择采样方法
    if args.adaptive_steps:
        samples = sample_adaptive_steps(model, z, y, args, device)
    elif use_sads:
        samples = sample_with_sads(model, z, y, args, device)
    else:
        diffusion = create_diffusion(str(args.num_steps))
        samples = sample_baseline(model, diffusion, z, y, args, device)

    # 后处理
    if args.post_enhance:
        print("\n[Post Enhancement]")
        try:
            enhancer = SpatioTemporalEnhancer(
                enable_temporal_filter=True,
                enable_spatial_smoothing=True,
                device=device,
            )
            result = enhancer.enhance(samples)
            samples = result.enhanced_data
            print("Post enhancement applied")
        except Exception as e:
            print(f"Warning: Post-enhancement failed: {e}")

    # 保存结果
    samples, _ = samples.chunk(2, dim=0)
    samples_array = samples.cpu().numpy()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, samples_array)
    print(f"\nSamples saved to: {args.output_path}")


if __name__ == "__main__":
    main()
