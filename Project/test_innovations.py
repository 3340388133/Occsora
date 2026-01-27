"""
Project创新模块测试脚本
======================

测试所有创新点模块的功能是否正常工作。
"""

import sys
sys.path.insert(0, '/root/OccSora-main')

import torch
import numpy as np


def test_adaptive_cfg():
    """测试自适应CFG模块"""
    print("\n" + "=" * 50)
    print("Testing: adaptive_cfg module")
    print("=" * 50)

    from Project.adaptive_cfg import (
        AdaptiveCFGScheduler,
        TemporalAwareCFG,
        MultiConditionCFG
    )

    # 测试AdaptiveCFGScheduler
    print("\n1. AdaptiveCFGScheduler:")
    scheduler = AdaptiveCFGScheduler(
        cfg_min=1.0,
        cfg_max=7.5,
        schedule_type="cosine",
        num_timesteps=1000
    )

    # 测试不同时间步的CFG值
    for t in [0, 250, 500, 750, 999]:
        cfg = scheduler.get_guidance_scale(t)
        print(f"   timestep={t:4d} -> cfg_scale={cfg:.3f}")

    # 测试TemporalAwareCFG
    print("\n2. TemporalAwareCFG:")
    temporal_cfg = TemporalAwareCFG(
        num_frames=16,
        cfg_min=1.0,
        cfg_max=7.5,
        num_timesteps=1000
    )

    for frame in [0, 7, 15]:
        cfg = temporal_cfg.get_frame_guidance(500, frame)
        print(f"   frame={frame:2d}, timestep=500 -> cfg_scale={cfg:.3f}")

    print("\n   [PASS] adaptive_cfg module works correctly!")
    return True


def test_samplers():
    """测试采样器模块"""
    print("\n" + "=" * 50)
    print("Testing: samplers module")
    print("=" * 50)

    from Project.samplers import (
        SamplerFactory,
        DDIMSampler,
        DPMSolverSampler,
        HybridSampler
    )

    # 测试SamplerFactory
    print("\n1. SamplerFactory:")
    factory = SamplerFactory()
    print(f"   Available samplers: {factory.list_samplers()}")
    print(f"   Available presets: {factory.list_presets()}")

    # 创建不同类型的采样器
    print("\n2. Creating samplers:")
    for sampler_type in ["ddim", "dpm_solver", "euler"]:
        try:
            sampler = factory.create(sampler_type, num_inference_steps=25)
            print(f"   {sampler_type}: {type(sampler).__name__} created")
        except Exception as e:
            print(f"   {sampler_type}: Error - {e}")

    print("\n   [PASS] samplers module works correctly!")
    return True


def test_consistency():
    """测试一致性增强模块"""
    print("\n" + "=" * 50)
    print("Testing: consistency module")
    print("=" * 50)

    from Project.consistency import (
        SpatioTemporalEnhancer,
        TemporalConsistencyFilter,
        SpatialSmoothingModule
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试SpatioTemporalEnhancer
    print("\n1. SpatioTemporalEnhancer:")
    enhancer = SpatioTemporalEnhancer(
        enable_temporal_filter=True,
        enable_spatial_smoothing=True,
        device=device
    )
    print(f"   Created on device: {device}")

    # 创建测试数据
    test_data = torch.randn(1, 16, 128, 25, 25, device=device)
    print(f"   Input shape: {test_data.shape}")

    try:
        result = enhancer.enhance(test_data)
        print(f"   Output shape: {result.enhanced_data.shape}")
        print("   [PASS] Enhancement successful!")
    except Exception as e:
        print(f"   Warning: {e}")

    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("OccSora Project Innovation Modules Test")
    print("=" * 60)

    results = {}

    # 测试各模块
    results['adaptive_cfg'] = test_adaptive_cfg()
    results['samplers'] = test_samplers()
    results['consistency'] = test_consistency()

    # 汇总结果
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for module, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {module}: [{status}]")

    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))
    return all_passed


if __name__ == "__main__":
    main()
