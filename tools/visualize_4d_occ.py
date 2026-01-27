#!/usr/bin/env python3
"""4D占用可视化 - BEV视图和时序动画"""
import os
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, "/root/OccSora-main")

# 语义类别颜色
COLORS = np.array([
    [0, 0, 0],        # 0: noise
    [255, 120, 50],   # 1: barrier
    [255, 192, 203],  # 2: bicycle
    [255, 255, 0],    # 3: bus
    [0, 150, 245],    # 4: car
    [0, 255, 255],    # 5: construction
    [255, 127, 0],    # 6: motorcycle
    [255, 0, 0],      # 7: pedestrian
    [255, 240, 150],  # 8: traffic cone
    [135, 60, 0],     # 9: trailer
    [160, 32, 240],   # 10: truck
    [255, 0, 255],    # 11: driveable
    [139, 137, 137],  # 12: other flat
    [75, 0, 75],      # 13: sidewalk
    [150, 240, 80],   # 14: terrain
    [230, 230, 250],  # 15: manmade
    [0, 175, 0],      # 16: vegetation
    [128, 128, 128],  # 17: empty
], dtype=np.uint8)

# Class groupings (nuScenes 16+empty in this repo)
EMPTY_ID = 17
# Ground-like / static surfaces that tend to fill the whole map
GROUND_IDS = {11, 12, 13, 14, 15, 16}

VEHICLE_IDS = {2, 3, 4, 5, 6, 9, 10}
PEDESTRIAN_IDS = {7}

GROUP_NAMES = ["empty", "ground", "vehicle", "pedestrian", "other"]
GROUP_COLORS = np.array(
    [
        [128, 128, 128],  # empty
        [255, 0, 255],    # ground
        [0, 150, 245],    # vehicle
        [255, 0, 0],      # pedestrian
        [0, 175, 0],      # other
    ],
    dtype=np.uint8,
)


def _group_id(lbl: np.ndarray) -> np.ndarray:
    lbl = lbl.astype(np.int32)
    out = np.full_like(lbl, 4, dtype=np.int32)  # other
    out[lbl == EMPTY_ID] = 0
    out[np.isin(lbl, list(GROUND_IDS))] = 1
    out[np.isin(lbl, list(VEHICLE_IDS))] = 2
    out[np.isin(lbl, list(PEDESTRIAN_IDS))] = 3
    return out


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

def create_bev_image(occ, frame_idx=0, grouped: bool = False):
    """Create a BEV RGB image.

    Projection strategy:
    1) Prefer non-ground, non-empty voxels (cars/peds/etc.) from top to bottom.
    2) Fill remaining pixels with ground-like classes (driveable/sidewalk/etc.).
    """
    if occ.ndim == 4:
        occ = occ[frame_idx]
    H, W, D = occ.shape

    bev = np.full((H, W), EMPTY_ID, dtype=np.int32)

    # Pass 1: objects first (non-empty & not ground)
    for d in range(D - 1, -1, -1):
        layer = occ[:, :, d].astype(np.int32)
        mask = (bev == EMPTY_ID) & (layer != EMPTY_ID)
        if GROUND_IDS:
            mask &= ~np.isin(layer, list(GROUND_IDS))
        bev[mask] = layer[mask]

    # Pass 2: fill remaining with ground
    for d in range(D - 1, -1, -1):
        layer = occ[:, :, d].astype(np.int32)
        mask = (bev == EMPTY_ID) & (layer != EMPTY_ID)
        bev[mask] = layer[mask]

    empty_count = int((bev == EMPTY_ID).sum())
    total = int(H * W)

    if grouped:
        bev_g = _group_id(bev)
        rgb = GROUP_COLORS[np.clip(bev_g, 0, len(GROUP_NAMES) - 1)]
        return rgb, bev_g, empty_count, total

    rgb = COLORS[np.clip(bev, 0, EMPTY_ID)]
    return rgb, bev, empty_count, total

def main():
    import matplotlib.pyplot as plt
    from mmengine import Config
    from mmengine.registry import MODELS
    import model as _

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="/root/OccSora-main/eval_results_paperish_v6")
    ap.add_argument("--output-dir", type=str, default="/root/OccSora-main/visualizations")
    ap.add_argument("--grouped", action="store_true", help="Use 5-class grouped view (clearer)")
    ap.add_argument("--cond", type=int, default=0)
    args = ap.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载VQVAE
    print("Loading VQVAE...")
    cfg = Config.fromfile("/root/OccSora-main/config/train_vqvae.py")
    vq = MODELS.build(cfg.model)
    ckpt = torch.load("/root/autodl-tmp/OccSoraModel/epoch_125.pth", map_location="cpu")
    vq.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    vq = vq.to(device).eval()

    models = ["baseline", "full"]
    samples_dict = {}

    # 加载并解码样本
    for model in models:
        files = sorted(Path(results_dir).glob(f"samples_{model}_cond{args.cond}.npy"))
        if files:
            latent = torch.from_numpy(np.load(files[0])[:1]).float()
            print(f"Decoding {model}...")
            occ = decode_latent_to_occ(vq, latent, device)[0]
            samples_dict[model] = occ

    if not samples_dict:
        print("No samples found!")
        return

    # 1. 对比图 (Baseline vs Full)
    print("Creating comparison figure...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    frames = [0, 10, 20, 31]

    for i, model in enumerate(["baseline", "full"]):
        if model not in samples_dict:
            continue
        occ = samples_dict[model]
        for j, f in enumerate(frames):
            img, _, empty_cnt, total = create_bev_image(occ, min(f, occ.shape[0]-1), grouped=args.grouped)
            axes[i, j].imshow(img, interpolation='nearest')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(model.upper(), fontsize=14)
            if i == 0:
                axes[i, j].set_title(f"Frame {f}\nempty={empty_cnt}/{total}", fontsize=11)

    plt.suptitle("4D Occupancy Generation: Baseline vs Full (STCA+SADS)", fontsize=16)
    plt.tight_layout()
    out_name = "comparison_baseline_vs_full_grouped.png" if args.grouped else "comparison_baseline_vs_full.png"
    plt.savefig(f"{output_dir}/{out_name}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/{out_name}")

    # 2. 时序演化图
    print("Creating temporal evolution figure...")
    for model, occ in samples_dict.items():
        fig, axes = plt.subplots(2, 8, figsize=(24, 6))
        n_frames = occ.shape[0]
        frame_indices = np.linspace(0, n_frames-1, 16, dtype=int)

        for idx, f in enumerate(frame_indices):
            row, col = idx // 8, idx % 8
            img, _, empty_cnt, total = create_bev_image(occ, f, grouped=args.grouped)
            axes[row, col].imshow(img, interpolation='nearest')
            axes[row, col].axis('off')
            axes[row, col].set_title(f"t={f} (empty={empty_cnt}/{total})", fontsize=9)

        plt.suptitle(f"{model.upper()} - Temporal Evolution (32 frames)", fontsize=14)
        plt.tight_layout()
        out_name = f"temporal_{model}_grouped.png" if args.grouped else f"temporal_{model}.png"
        plt.savefig(f"{output_dir}/{out_name}", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/{out_name}")

    # 3. GIF动画
    print("Creating GIF animations...")
    try:
        import imageio
        for model, occ in samples_dict.items():
            frames_list = []
            for f in range(occ.shape[0]):
                img, _, _, _ = create_bev_image(occ, f, grouped=args.grouped)
                frames_list.append(img)
            out_name = f"{model}_animation_grouped.gif" if args.grouped else f"{model}_animation.gif"
            imageio.mimsave(f"{output_dir}/{out_name}", frames_list, fps=8)
            print(f"Saved: {output_dir}/{out_name}")
    except ImportError:
        print("imageio not installed, skipping GIF generation")

    print(f"\nAll visualizations saved to {output_dir}/")

if __name__ == "__main__":
    main()
