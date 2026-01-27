"""
OccSora增强演示
===============

展示如何使用增强模块的演示脚本。

用法:
    python demo.py --input path/to/occupancy.npy --output path/to/enhanced.npy
"""

import argparse
import numpy as np
import torch
import time
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Project.adaptive_cfg import AdaptiveCFGScheduler, TemporalAwareCFG
from Project.samplers import SamplerFactory, DDIMSampler, DPMSolverSampler
from Project.consistency import (
    SpatioTemporalEnhancer,
    TemporalConsistencyFilter,
    SpatialSmoothingModule,
    QualityAssessment
)


def demo_adaptive_cfg():
    """演示自适应CFG调度。"""
    print("\n" + "="*60)
    print("演示: 自适应分类器无关引导调度")
    print("="*60)

    # 使用不同策略创建调度器
    strategies = ["linear", "cosine", "step", "temporal_pyramid"]

    for strategy in strategies:
        scheduler = AdaptiveCFGScheduler(
            cfg_min=1.0,
            cfg_max=7.5,
            schedule_type=strategy,
            num_timesteps=1000
        )

        # 采样一些时间步
        sample_timesteps = [0, 250, 500, 750, 999]
        print(f"\n{strategy.upper()} 调度:")
        for t in sample_timesteps:
            scale = scheduler.get_guidance_scale(t)
            print(f"  t={t:4d}: CFG = {scale:.3f}")

    # 演示时序感知CFG
    print("\n" + "-"*40)
    print("时序感知CFG（16帧）:")

    temporal_cfg = TemporalAwareCFG(
        num_frames=16,
        cfg_max=7.5,
        keyframe_indices=[0, 8, 15]
    )

    timestep = 500
    print(f"\n在t={timestep}时各帧的引导尺度:")
    for frame_idx in [0, 4, 8, 12, 15]:
        scale = temporal_cfg.get_frame_guidance(timestep, frame_idx)
        marker = " (关键帧)" if frame_idx in [0, 8, 15] else ""
        print(f"  帧 {frame_idx:2d}: CFG = {scale:.3f}{marker}")


def demo_samplers():
    """演示采样器选择和使用。"""
    print("\n" + "="*60)
    print("演示: 混合多策略采样器")
    print("="*60)

    factory = SamplerFactory()

    print("\n可用采样器:", factory.list_samplers())
    print("可用预设:", factory.list_presets())

    # 使用不同预设创建采样器
    print("\n" + "-"*40)
    print("采样器推荐:")

    recommendations = [
        ("fast", 10, 0.7),
        ("balanced", 30, 0.8),
        ("quality", 60, 0.95),
    ]

    for preset, time_budget, quality_req in recommendations:
        sampler = factory.create_preset(preset)
        print(f"\n  预设 '{preset}' (时间={time_budget}s, 质量={quality_req}):")
        print(f"    -> {sampler}")


def demo_consistency_enhancement():
    """演示一致性增强管道。"""
    print("\n" + "="*60)
    print("演示: 时空一致性增强")
    print("="*60)

    # 创建合成测试数据
    print("\n生成合成4D占用数据...")
    B, T, C, H, W = 1, 16, 1, 64, 64
    data = torch.randn(B, T, C, H, W)

    # 添加一些人工噪声/不一致性
    noise = torch.randn_like(data) * 0.3
    noisy_data = data + noise

    print(f"数据形状: {noisy_data.shape}")

    # 创建增强器
    enhancer = SpatioTemporalEnhancer(
        enable_temporal_filter=True,
        enable_spatial_smoothing=True,
        enable_multi_scale=True,
        enable_object_tracking=False,  # 为了速度禁用
        temporal_filter_type="ema",
        temporal_alpha=0.3
    )

    print(f"\n增强器: {enhancer}")
    print(f"模块: {enhancer.get_module_info()}")

    # 应用增强
    print("\n应用增强...")
    result = enhancer.enhance(noisy_data)

    print(f"\n增强完成，耗时 {result.elapsed_time:.3f}s")
    print(f"应用的模块: {result.applied_modules}")
    print("\n指标:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}")


def demo_quality_metrics():
    """演示质量指标计算。"""
    print("\n" + "="*60)
    print("演示: 质量评估指标")
    print("="*60)

    # 创建测试数据
    B, T, C, H, W = 1, 8, 1, 32, 32

    # 平滑数据
    smooth_data = torch.zeros(B, T, C, H, W)
    for t in range(T):
        smooth_data[:, t] = torch.randn(B, C, H, W) * 0.1 + t * 0.01

    # 噪声/闪烁数据
    noisy_data = smooth_data + torch.randn_like(smooth_data) * 0.5

    # 质量评估
    qa = QualityAssessment()

    print("\n评估平滑数据质量...")
    smooth_results = qa.assess(smooth_data, detailed=False)
    print(f"  总体质量: {smooth_results['overall_quality']:.4f}")

    print("\n评估噪声数据质量...")
    noisy_results = qa.assess(noisy_data, detailed=False)
    print(f"  总体质量: {noisy_results['overall_quality']:.4f}")

    print(f"\n质量差异: {smooth_results['overall_quality'] - noisy_results['overall_quality']:.4f}")


def demo_visualization():
    """演示可视化功能。"""
    print("\n" + "="*60)
    print("演示: 可视化（保存到文件）")
    print("="*60)

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), "demo_output")
    os.makedirs(output_dir, exist_ok=True)

    # 可视化CFG调度
    scheduler = AdaptiveCFGScheduler(
        cfg_min=1.0,
        cfg_max=7.5,
        schedule_type="cosine",
        num_timesteps=1000
    )

    cfg_path = os.path.join(output_dir, "cfg_schedule.png")
    try:
        scheduler.visualize_schedule(save_path=cfg_path)
        print(f"  CFG调度可视化已保存到: {cfg_path}")
    except Exception as e:
        print(f"  无法保存可视化: {e}")

    # 可视化时序CFG
    temporal_cfg = TemporalAwareCFG(num_frames=16, cfg_max=7.5)

    temporal_path = os.path.join(output_dir, "temporal_cfg.png")
    try:
        temporal_cfg.visualize_frame_guidance(save_path=temporal_path)
        print(f"  时序CFG可视化已保存到: {temporal_path}")
    except Exception as e:
        print(f"  无法保存可视化: {e}")

    print(f"\n输出目录: {output_dir}")


def main():
    """运行所有演示。"""
    parser = argparse.ArgumentParser(description="OccSora增强演示")
    parser.add_argument("--demo", type=str, default="all",
                       choices=["all", "cfg", "samplers", "consistency", "metrics", "viz"],
                       help="运行哪个演示")
    args = parser.parse_args()

    print("\n" + "#"*60)
    print("#" + " "*20 + "OccSora Enhancement" + " "*19 + "#")
    print("#" + " "*20 + "   Demo Script   " + " "*21 + "#")
    print("#"*60)

    demos = {
        "cfg": demo_adaptive_cfg,
        "samplers": demo_samplers,
        "consistency": demo_consistency_enhancement,
        "metrics": demo_quality_metrics,
        "viz": demo_visualization,
    }

    if args.demo == "all":
        for name, func in demos.items():
            func()
    else:
        demos[args.demo]()

    print("\n" + "="*60)
    print("演示完成!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
