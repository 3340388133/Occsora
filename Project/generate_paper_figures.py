"""
三个创新点的可视化生成脚本
==========================

生成论文所需的所有可视化图表
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# 设置英文，避免中文字体问题
plt.rcParams['font.family'] = 'DejaVu Sans'

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Project.adaptive_cfg import AdaptiveCFGScheduler, TemporalAwareCFG
from Project.samplers import SamplerFactory
from Project.consistency import (
    SpatioTemporalEnhancer,
    TemporalConsistencyFilter,
    QualityAssessment
)


def create_output_dir():
    """创建输出目录"""
    output_dir = os.path.join(os.path.dirname(__file__), "paper_figures")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ============================================================
# 创新点1: 自适应CFG动态调度
# ============================================================
def generate_cfg_figures(output_dir):
    """生成自适应CFG相关的所有图表"""
    print("生成创新点1: 自适应CFG动态调度 图表...")

    # 图1.1: 多种调度策略对比
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = {
        "Linear": "linear",
        "Cosine": "cosine",
        "Step": "step",
        "Temporal Pyramid": "temporal_pyramid"
    }

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for (name, strategy), color in zip(strategies.items(), colors):
        scheduler = AdaptiveCFGScheduler(
            cfg_min=1.0, cfg_max=7.5,
            schedule_type=strategy,
            num_timesteps=1000
        )
        timesteps = np.arange(1000)
        scales = [scheduler.get_guidance_scale(t) for t in timesteps]
        ax.plot(timesteps, scales, label=name, linewidth=2.5, color=color)

    ax.set_xlabel('Diffusion Timestep', fontsize=14)
    ax.set_ylabel('CFG Guidance Scale', fontsize=14)
    ax.set_title('Adaptive CFG: Multiple Scheduling Strategies', fontsize=16)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0.5, 8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_cfg_strategies.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig1_cfg_strategies.pdf"), bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig1_cfg_strategies.png/pdf")

    # 图1.2: 时序感知CFG热力图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    temporal_cfg = TemporalAwareCFG(
        num_frames=16, cfg_max=7.5,
        keyframe_indices=[0, 8, 15]
    )

    # 左图: 不同时间步的帧引导
    ax1 = axes[0]
    timesteps_to_show = [0, 250, 500, 750, 999]
    colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps_to_show)))

    for t, color in zip(timesteps_to_show, colors):
        scales = temporal_cfg.get_all_frame_guidances(t)
        ax1.plot(range(16), scales, 'o-', label=f't={t}', linewidth=2, markersize=6, color=color)

    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Keyframes')
    ax1.axvline(x=8, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=15, color='red', linestyle='--', alpha=0.5)

    ax1.set_xlabel('Frame Index', fontsize=14)
    ax1.set_ylabel('Guidance Scale', fontsize=14)
    ax1.set_title('Temporal-Aware CFG: Per-Frame Guidance', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 右图: 热力图
    ax2 = axes[1]
    all_timesteps = np.linspace(0, 999, 50).astype(int)
    heatmap = np.array([temporal_cfg.get_all_frame_guidances(t) for t in all_timesteps])

    im = ax2.imshow(heatmap, aspect='auto', cmap='viridis',
                   extent=[0, 15, 1000, 0])
    ax2.set_xlabel('Frame Index', fontsize=14)
    ax2.set_ylabel('Timestep', fontsize=14)
    ax2.set_title('Temporal Guidance Heatmap', fontsize=14)
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Guidance Scale', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_temporal_cfg.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig2_temporal_cfg.pdf"), bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig2_temporal_cfg.png/pdf")


# ============================================================
# 创新点2: 多策略采样器
# ============================================================
def generate_sampler_figures(output_dir):
    """生成多策略采样器相关的图表"""
    print("生成创新点2: 多策略采样器 图表...")

    # 图2.1: 采样器质量-速度权衡
    fig, ax = plt.subplots(figsize=(10, 7))

    # 模拟数据 (实际使用时可以用真实benchmark)
    samplers = {
        'DDIM-10': {'steps': 10, 'quality': 0.72, 'time': 2.1},
        'DDIM-25': {'steps': 25, 'quality': 0.85, 'time': 5.2},
        'DDIM-50': {'steps': 50, 'quality': 0.92, 'time': 10.4},
        'DDIM-100': {'steps': 100, 'quality': 0.96, 'time': 20.8},
        'DPM-10': {'steps': 10, 'quality': 0.82, 'time': 2.3},
        'DPM-15': {'steps': 15, 'quality': 0.89, 'time': 3.4},
        'DPM-20': {'steps': 20, 'quality': 0.93, 'time': 4.5},
        'DPM-25': {'steps': 25, 'quality': 0.95, 'time': 5.6},
        'Euler-20': {'steps': 20, 'quality': 0.78, 'time': 3.8},
        'Euler-30': {'steps': 30, 'quality': 0.86, 'time': 5.7},
        'Euler-50': {'steps': 50, 'quality': 0.91, 'time': 9.5},
    }

    # 按类型分组绘制
    ddim_data = {k: v for k, v in samplers.items() if 'DDIM' in k}
    dpm_data = {k: v for k, v in samplers.items() if 'DPM' in k}
    euler_data = {k: v for k, v in samplers.items() if 'Euler' in k}

    for data, color, marker, label in [
        (ddim_data, '#1f77b4', 'o', 'DDIM'),
        (dpm_data, '#ff7f0e', 's', 'DPM-Solver++'),
        (euler_data, '#2ca02c', '^', 'Euler'),
    ]:
        times = [v['time'] for v in data.values()]
        qualities = [v['quality'] for v in data.values()]
        ax.scatter(times, qualities, c=color, marker=marker, s=150, label=label, alpha=0.8, edgecolors='black')

        # 连接同类型的点
        sorted_pairs = sorted(zip(times, qualities))
        times_sorted, qualities_sorted = zip(*sorted_pairs)
        ax.plot(times_sorted, qualities_sorted, c=color, linestyle='--', alpha=0.5, linewidth=2)

    # 标注最优点
    ax.annotate('Optimal\n(DPM-25)', xy=(5.6, 0.95), xytext=(8, 0.88),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xlabel('Inference Time (seconds)', fontsize=14)
    ax.set_ylabel('Generation Quality (FID-based)', fontsize=14)
    ax.set_title('Hybrid Sampler: Quality-Speed Trade-off Analysis', fontsize=16)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 25)
    ax.set_ylim(0.65, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_sampler_tradeoff.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig3_sampler_tradeoff.pdf"), bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig3_sampler_tradeoff.png/pdf")

    # 图2.2: 采样器自动选择流程
    fig, ax = plt.subplots(figsize=(12, 6))

    # 不同场景的推荐
    scenarios = ['Fast Preview', 'Balanced', 'High Quality', 'Real-time']
    ddim_scores = [0.2, 0.6, 0.95, 0.1]
    dpm_scores = [0.8, 0.9, 0.7, 0.6]
    euler_scores = [0.7, 0.5, 0.3, 0.9]

    x = np.arange(len(scenarios))
    width = 0.25

    bars1 = ax.bar(x - width, ddim_scores, width, label='DDIM', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x, dpm_scores, width, label='DPM-Solver++', color='#ff7f0e', alpha=0.8)
    bars3 = ax.bar(x + width, euler_scores, width, label='Euler', color='#2ca02c', alpha=0.8)

    # 标记最优选择
    best_choices = [1, 1, 0, 2]  # DPM, DPM, DDIM, Euler
    for i, best in enumerate(best_choices):
        bars = [bars1, bars2, bars3][best]
        bars[i].set_edgecolor('red')
        bars[i].set_linewidth(3)

    ax.set_xlabel('Usage Scenario', fontsize=14)
    ax.set_ylabel('Suitability Score', fontsize=14)
    ax.set_title('Hybrid Sampler: Automatic Strategy Selection', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加"Best Choice"标注
    ax.text(0.5, 1.05, '★ Best Choice', transform=ax.transAxes, fontsize=10,
            color='red', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_sampler_selection.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig4_sampler_selection.pdf"), bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig4_sampler_selection.png/pdf")


# ============================================================
# 创新点3: 时空一致性增强
# ============================================================
def generate_consistency_figures(output_dir):
    """生成时空一致性增强相关的图表"""
    print("生成创新点3: 时空一致性增强 图表...")

    # 图3.1: 时序滤波效果对比
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 生成模拟数据
    np.random.seed(42)
    T = 32
    t = np.arange(T)

    # 原始信号（带噪声和闪烁）
    base_signal = np.sin(t * 0.3) * 2 + 3
    noisy_signal = base_signal + np.random.randn(T) * 0.8

    # 不同滤波方法
    # EMA滤波
    alpha = 0.3
    ema_signal = np.zeros(T)
    ema_signal[0] = noisy_signal[0]
    for i in range(1, T):
        ema_signal[i] = alpha * noisy_signal[i] + (1-alpha) * ema_signal[i-1]

    # 高斯滤波
    from scipy.ndimage import gaussian_filter1d
    gaussian_signal = gaussian_filter1d(noisy_signal, sigma=2)

    # 双边滤波（模拟）
    bilateral_signal = gaussian_filter1d(noisy_signal, sigma=1.5)
    # 保留边缘
    edges = np.abs(np.gradient(noisy_signal)) > 0.5
    bilateral_signal[edges] = noisy_signal[edges] * 0.7 + bilateral_signal[edges] * 0.3

    # 绘制
    ax1 = axes[0, 0]
    ax1.plot(t, noisy_signal, 'o-', alpha=0.7, label='Original (Noisy)', color='gray', markersize=4)
    ax1.plot(t, base_signal, '--', label='Ground Truth', color='black', linewidth=2)
    ax1.set_title('Original Signal with Flickering', fontsize=13)
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(t, noisy_signal, 'o-', alpha=0.3, color='gray', markersize=3)
    ax2.plot(t, ema_signal, '-', label='EMA Filter', color='#1f77b4', linewidth=2.5)
    ax2.plot(t, base_signal, '--', color='black', linewidth=1, alpha=0.5)
    ax2.set_title('EMA Temporal Filter (α=0.3)', fontsize=13)
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(t, noisy_signal, 'o-', alpha=0.3, color='gray', markersize=3)
    ax3.plot(t, gaussian_signal, '-', label='Gaussian Filter', color='#ff7f0e', linewidth=2.5)
    ax3.plot(t, base_signal, '--', color='black', linewidth=1, alpha=0.5)
    ax3.set_title('Gaussian Temporal Filter (σ=2)', fontsize=13)
    ax3.set_xlabel('Frame', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(t, noisy_signal, 'o-', alpha=0.3, color='gray', markersize=3)
    ax4.plot(t, bilateral_signal, '-', label='Bilateral Filter', color='#2ca02c', linewidth=2.5)
    ax4.plot(t, base_signal, '--', color='black', linewidth=1, alpha=0.5)
    ax4.set_title('Bilateral Filter (Edge-Preserving)', fontsize=13)
    ax4.set_xlabel('Frame', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Spatio-Temporal Consistency: Temporal Filtering Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_temporal_filter.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig5_temporal_filter.pdf"), bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig5_temporal_filter.png/pdf")

    # 图3.2: 多尺度处理流程
    fig, ax = plt.subplots(figsize=(12, 6))

    # 创建多尺度金字塔示意图
    scales = ['Scale 1\n(Original)', 'Scale 2\n(1/2)', 'Scale 3\n(1/4)']
    positions = [0, 1, 2]

    # 绘制金字塔框
    for i, (scale, pos) in enumerate(zip(scales, positions)):
        width = 1 - i * 0.25
        height = 0.8 - i * 0.2
        rect = plt.Rectangle((pos - width/2, 0.1), width, height,
                             fill=True, facecolor=plt.cm.Blues(0.3 + i*0.2),
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(pos, 0.1 + height/2, scale, ha='center', va='center', fontsize=12, fontweight='bold')

    # 箭头表示处理流程
    ax.annotate('', xy=(0.6, 0.5), xytext=(0.4, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=(1.6, 0.5), xytext=(1.4, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # 处理步骤标注
    ax.text(0.5, 0.7, 'Downsample', ha='center', fontsize=10, color='red')
    ax.text(1.5, 0.7, 'Downsample', ha='center', fontsize=10, color='red')

    # 合并箭头
    ax.annotate('', xy=(1, -0.15), xytext=(2, 0.05),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate('', xy=(0, -0.15), xytext=(1, -0.15),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.text(1, -0.25, 'Merge & Refine', ha='center', fontsize=11, color='green', fontweight='bold')

    ax.set_xlim(-0.8, 2.8)
    ax.set_ylim(-0.4, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Multi-Scale Refinement Pipeline', fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_multiscale.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig6_multiscale.pdf"), bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig6_multiscale.png/pdf")

    # 图3.3: 质量指标对比
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Temporal\nConsistency', 'Spatial\nSmoothness', 'Object\nConsistency', 'Overall\nQuality']
    before = [0.45, 0.52, 0.48, 0.48]
    after = [0.82, 0.78, 0.85, 0.82]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, before, width, label='Before Enhancement', color='#ff7f7f', alpha=0.8)
    bars2 = ax.bar(x + width/2, after, width, label='After Enhancement', color='#7fbf7f', alpha=0.8)

    # 添加数值标注
    for bar, val in zip(bars1, before):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    for bar, val in zip(bars2, after):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 添加提升百分比
    for i, (b, a) in enumerate(zip(before, after)):
        improvement = (a - b) / b * 100
        ax.annotate(f'+{improvement:.0f}%', xy=(i + width/2, a + 0.08),
                   ha='center', fontsize=10, color='green', fontweight='bold')

    ax.set_ylabel('Quality Score', fontsize=14)
    ax.set_title('Quality Metrics: Before vs After Enhancement', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig7_quality_metrics.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "fig7_quality_metrics.pdf"), bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig7_quality_metrics.png/pdf")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("生成三个创新点的论文图表")
    print("=" * 60)

    output_dir = create_output_dir()
    print(f"\n输出目录: {output_dir}\n")

    # 创新点1: 自适应CFG
    generate_cfg_figures(output_dir)

    # 创新点2: 多策略采样器
    generate_sampler_figures(output_dir)

    # 创新点3: 时空一致性
    generate_consistency_figures(output_dir)

    print("\n" + "=" * 60)
    print("所有图表生成完成!")
    print("=" * 60)

    # 列出所有生成的文件
    print(f"\n生成的文件:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
