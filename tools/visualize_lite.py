#!/usr/bin/env python3
"""轻量可视化 - 直接在latent空间可视化，无需VQVAE解码"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def latent_to_pseudo_bev(latent, frame_idx=0):
    """将latent转换为伪BEV图像"""
    # latent: (128, 4, 25, 25) -> 取第一个时间步，平均通道
    if latent.ndim == 4:
        # (C, T, H, W) -> 取frame_idx时间步
        t = min(frame_idx, latent.shape[1]-1)
        feat = latent[:, t, :, :]  # (C, H, W)
    else:
        feat = latent

    # 平均前64通道（主要特征）
    bev = np.mean(feat[:64], axis=0)  # (H, W)

    # 归一化到0-1
    bev = (bev - bev.min()) / (bev.max() - bev.min() + 1e-8)
    return bev

def main():
    results_dir = "/root/OccSora-main/eval_results_paperish_v6"
    output_dir = "/root/OccSora-main/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    models = ["baseline", "full"]
    samples_dict = {}

    # 加载样本
    for model in models:
        files = sorted(Path(results_dir).glob(f"samples_{model}_cond0.npy"))
        if files:
            samples = np.load(files[0])  # (N, 128, 4, 25, 25)
            samples_dict[model] = samples[0]  # 取第一个样本
            print(f"Loaded {model}: {samples[0].shape}")

    if not samples_dict:
        print("No samples found!")
        return

    # 1. Baseline vs Full 对比图
    print("Creating comparison figure...")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    frames = [0, 1, 2, 3]

    for i, model in enumerate(["baseline", "full"]):
        if model not in samples_dict:
            continue
        latent = samples_dict[model]
        for j, f in enumerate(frames):
            bev = latent_to_pseudo_bev(latent, f)
            im = axes[i, j].imshow(bev, cmap='viridis', vmin=0, vmax=1)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(model.upper(), fontsize=14, rotation=0, labelpad=50)
            if i == 0:
                axes[i, j].set_title(f"t={f}", fontsize=12)

    plt.suptitle("Latent Space Visualization: Baseline vs Full (STCA+SADS)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latent_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/latent_comparison.png")

    # 2. 时序演化图 - 所有4个时间步
    print("Creating temporal evolution figure...")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for i, model in enumerate(["baseline", "full"]):
        if model not in samples_dict:
            continue
        latent = samples_dict[model]
        for t in range(4):
            bev = latent_to_pseudo_bev(latent, t)
            axes[i, t].imshow(bev, cmap='viridis', vmin=0, vmax=1)
            axes[i, t].axis('off')
            if t == 0:
                axes[i, t].set_ylabel(model.upper(), fontsize=12, rotation=0, labelpad=40)
            if i == 0:
                axes[i, t].set_title(f"Frame {t}", fontsize=11)

    plt.suptitle("Temporal Evolution in Latent Space", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/temporal_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/temporal_evolution.png")

    # 3. 特征通道可视化
    print("Creating channel visualization...")
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))

    latent = samples_dict.get("full", samples_dict.get("baseline"))
    for ch in range(32):
        row, col = ch // 8, ch % 8
        feat = latent[ch, 0, :, :]  # 第ch通道，第0帧
        feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        axes[row, col].imshow(feat, cmap='coolwarm')
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Ch{ch}", fontsize=8)

    plt.suptitle("Feature Channel Visualization (Full Model, Frame 0)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/channel_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/channel_visualization.png")

    # 4. 多样本对比
    print("Creating diversity visualization...")
    files = sorted(Path(results_dir).glob("samples_full_cond*.npy"))
    if len(files) >= 2:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))

        for i, f in enumerate(files[:2]):
            samples = np.load(f)
            for j in range(min(4, samples.shape[0])):
                bev = latent_to_pseudo_bev(samples[j], 0)
                axes[i, j].imshow(bev, cmap='viridis', vmin=0, vmax=1)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_ylabel(f"Cond {i}", fontsize=12, rotation=0, labelpad=40)
                if i == 0:
                    axes[i, j].set_title(f"Sample {j}", fontsize=11)

        plt.suptitle("Generation Diversity (Full Model)", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/diversity_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/diversity_visualization.png")

    print(f"\n✓ All visualizations saved to {output_dir}/")

if __name__ == "__main__":
    main()
