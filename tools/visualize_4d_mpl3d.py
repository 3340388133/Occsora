#!/usr/bin/env python3
"""4D占用可视化 - Matplotlib 3D体素渲染
生成 t=0, 5, 10, 15, 20 五个时间步的3D占用图
"""
import os
import sys
import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, "/root/OccSora-main")

# 语义类别颜色 (RGB, 0-1)
COLORS = {
    0: (0, 0, 0),           # noise
    1: (1.0, 0.47, 0.2),    # barrier
    2: (1.0, 0.75, 0.8),    # bicycle
    3: (1.0, 1.0, 0),       # bus
    4: (0, 0.59, 0.96),     # car
    5: (0, 1.0, 1.0),       # construction
    6: (1.0, 0.5, 0),       # motorcycle
    7: (1.0, 0, 0),         # pedestrian
    8: (1.0, 0.94, 0.59),   # traffic cone
    9: (0.53, 0.24, 0),     # trailer
    10: (0.63, 0.13, 0.94), # truck
    11: (1.0, 0, 1.0),      # driveable
    12: (0.55, 0.54, 0.54), # other flat
    13: (0.29, 0, 0.29),    # sidewalk
    14: (0.59, 0.94, 0.31), # terrain
    15: (0.9, 0.9, 0.98),   # manmade
    16: (0, 0.69, 0),       # vegetation
    17: (0.5, 0.5, 0.5),    # empty
}

EMPTY_ID = 17
GROUND_IDS = {11, 12, 13, 14}

CLASS_NAMES = [
    "noise", "barrier", "bicycle", "bus", "car", "construction",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable", "other_flat", "sidewalk", "terrain", "manmade",
    "vegetation", "empty"
]


def decode_latent_to_occ(vq, latent, device):
    """解码latent到occupancy"""
    with torch.no_grad():
        z = latent[:, :64].to(device)
        x = vq.decoder_gpt(vq.post_vq_conv(z))
        B, C, F, H, W = x.shape
        D = 16
        template = vq.class_embeds.weight.T
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        labels = torch.empty((B, F, H, W, D), dtype=torch.int16, device="cpu")
        for b in range(B):
            for f in range(F):
                xf = x[b, f].view(H * W, D, -1)
                sim = torch.matmul(xf, template)
                labels[b, f] = torch.argmax(sim, dim=-1).to(torch.int16).cpu().view(H, W, D)
    return labels.numpy()


def visualize_occ_3d(occ, frame_idx, output_path,
                     hide_ground=True, hide_empty=True,
                     downsample=4, elev=30, azim=45):
    """使用Matplotlib渲染单帧3D占用图"""

    if occ.ndim == 4:
        frame = occ[frame_idx]
    else:
        frame = occ

    H, W, D = frame.shape

    # 下采样
    if downsample > 1:
        frame = frame[::downsample, ::downsample, :]
        H, W, D = frame.shape

    # 筛选体素
    mask = np.ones_like(frame, dtype=bool)
    if hide_empty:
        mask &= (frame != EMPTY_ID)
    if hide_ground:
        for gid in GROUND_IDS:
            mask &= (frame != gid)

    coords = np.argwhere(mask)
    if len(coords) == 0:
        print(f"  Warning: Frame {frame_idx} has no visible voxels!")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"t = {frame_idx} (no visible voxels)")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    labels = frame[mask]
    colors = np.array([COLORS.get(l, (0.5, 0.5, 0.5)) for l in labels])

    # 创建3D图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2] * 2

    ax.scatter(xs, ys, zs, c=colors, s=15, alpha=0.8, edgecolors='none')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"t = {frame_idx}", fontsize=14, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlim(0, H)
    ax.set_ylim(0, W)
    ax.set_zlim(0, D * 2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_combined_figure(occ, timesteps, output_path, model_name, downsample=4):
    """创建多时间步组合图"""
    n = len(timesteps)
    fig = plt.figure(figsize=(5*n, 5))

    for i, t in enumerate(timesteps):
        t = min(t, occ.shape[0] - 1)
        frame = occ[t]

        if downsample > 1:
            frame = frame[::downsample, ::downsample, :]

        H, W, D = frame.shape

        mask = (frame != EMPTY_ID)
        for gid in GROUND_IDS:
            mask &= (frame != gid)

        coords = np.argwhere(mask)
        ax = fig.add_subplot(1, n, i+1, projection='3d')

        if len(coords) > 0:
            labels = frame[mask]
            colors = np.array([COLORS.get(l, (0.5, 0.5, 0.5)) for l in labels])
            xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2] * 2
            ax.scatter(xs, ys, zs, c=colors, s=8, alpha=0.8, edgecolors='none')

        ax.set_title(f"t = {t}", fontsize=12, fontweight='bold')
        ax.view_init(elev=30, azim=45)
        ax.set_xlim(0, H)
        ax.set_ylim(0, W)
        ax.set_zlim(0, D * 2)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)

    plt.suptitle(f"4D Occupancy - {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Combined: {output_path}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="full")
    ap.add_argument("--cond", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default="/root/OccSora-main/mayavi_vis")
    ap.add_argument("--timesteps", type=str, default="0,5,10,15,20")
    ap.add_argument("--use-cached", action="store_true")
    args = ap.parse_args()

    timesteps = [int(t) for t in args.timesteps.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    cache_path = f"/root/OccSora-main/diagnosis_output/decoded_{args.model}_cond{args.cond}.npy"

    if args.use_cached and os.path.exists(cache_path):
        print(f"Loading cached: {cache_path}")
        occ = np.load(cache_path)
    else:
        from mmengine import Config
        from mmengine.registry import MODELS
        import model as _

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading VQVAE...")
        cfg = Config.fromfile("/root/OccSora-main/config/train_vqvae.py")
        vq = MODELS.build(cfg.model)
        ckpt = torch.load("/root/autodl-tmp/OccSoraModel/epoch_125.pth", map_location="cpu")
        vq.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        vq = vq.to(device).eval()

        latent_path = f"/root/OccSora-main/eval_results_paperish_v6/samples_{args.model}_cond{args.cond}.npy"
        if not os.path.exists(latent_path):
            print(f"Error: {latent_path} not found!")
            return

        print(f"Loading latent: {latent_path}")
        latent = torch.from_numpy(np.load(latent_path)[:1]).float()

        print("Decoding...")
        occ = decode_latent_to_occ(vq, latent, device)[0]

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, occ)
        print(f"Cached: {cache_path}")

    print(f"\nShape: {occ.shape}, Labels: {np.unique(occ)}")

    # 单独图片
    for t in timesteps:
        t_idx = min(t, occ.shape[0] - 1)
        out_path = f"{args.output_dir}/{args.model}_t{t}.png"
        print(f"Rendering t={t}...")
        visualize_occ_3d(occ, t_idx, out_path, downsample=4)

    # 组合图
    combined_path = f"{args.output_dir}/{args.model}_combined.png"
    create_combined_figure(occ, timesteps, combined_path, args.model)

    print(f"\nDone! Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
