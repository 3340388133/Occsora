#!/usr/bin/env python3
"""4D占用场景可视化 - 分批解码版本，避免内存溢出"""
import os
import sys
import gc
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, "/root/OccSora-main")

# 语义类别颜色 (nuScenes)
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

EMPTY_ID = 17
GROUND_IDS = {11, 12, 13, 14, 15, 16}

# Grouped visualization (5 classes): empty / ground / vehicle / pedestrian / other
VEHICLE_IDS = {2, 3, 4, 5, 6, 9, 10}
PEDESTRIAN_IDS = {7}

GROUP_NAMES = ["empty", "ground", "vehicle", "pedestrian", "other"]
GROUP_COLORS = np.array(
    [
        [128, 128, 128],  # empty
        [255, 0, 255],    # ground (driveable-like)
        [0, 150, 245],    # vehicle (car-like)
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

CLASS_NAMES = [
    'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'empty'
]

def decode_single_frame(vq, latent_128, frame_idx, device):
    """解码单帧，减少内存使用"""
    with torch.no_grad():
        z = latent_128[:, :64].to(device)
        x = vq.decoder_gpt(vq.post_vq_conv(z))  # (1, C, F, H, W)

        B, C, F, H, W = x.shape
        D = 16
        template = vq.class_embeds.weight.T  # (expansion, num_cls)

        # 只取指定帧
        f = min(frame_idx, F-1)
        xf = x[0, :, f, :, :].permute(1, 2, 0).contiguous()  # (H, W, C)
        xf = xf.view(H * W, D, -1)  # (HW, D, expansion)

        # 分块计算避免内存溢出
        labels = torch.empty((H * W, D), dtype=torch.int16, device="cpu")
        chunk_size = 5000
        for start in range(0, H * W, chunk_size):
            end = min(start + chunk_size, H * W)
            chunk = xf[start:end].to(template.device)
            sim = torch.matmul(chunk, template)
            labels[start:end] = torch.argmax(sim, dim=-1).to(torch.int16).cpu()

        labels = labels.view(H, W, D).numpy()

        # 清理显存
        del x, xf, z
        torch.cuda.empty_cache()
        gc.collect()

    return labels

def create_bev_image(occ, grouped: bool = False):
    """Create BEV from a single frame occupancy (H, W, D).

    Returns: (rgb, bev_label, empty_count, total)
    """
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
    from mmengine import Config
    from mmengine.registry import MODELS
    import model as _

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="/root/OccSora-main/eval_results_paperish_v6")
    ap.add_argument("--output-dir", type=str, default="/root/OccSora-main/visualizations")
    ap.add_argument("--grouped", action="store_true", help="Use 5-class grouped view (clearer)")
    ap.add_argument("--cond", type=int, default=0, help="Condition index to visualize (samples_*_cond{cond}.npy)")
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
    print("VQVAE loaded.")

    # 加载latent样本
    models_to_viz = ["baseline", "full"]
    latents = {}

    for model in models_to_viz:
        files = sorted(Path(results_dir).glob(f"samples_{model}_cond{args.cond}.npy"))
        if files:
            samples = np.load(files[0])
            latents[model] = torch.from_numpy(samples[0:1]).float()
            print(f"Loaded {model}: {latents[model].shape}")

    if not latents:
        print("No samples found!")
        return

    # 1. 生成对比图 (Baseline vs Full, 4帧)
    print("\nGenerating comparison figure...")
    frames_to_show = [0, 10, 20, 31]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, model in enumerate(["baseline", "full"]):
        if model not in latents:
            continue
        latent = latents[model]

        for j, f in enumerate(frames_to_show):
            print(f"  Decoding {model} frame {f}...")
            occ = decode_single_frame(vq, latent, f, device)
            rgb, bev_lbl, empty_cnt, total = create_bev_image(occ, grouped=args.grouped)

            axes[i, j].imshow(rgb, interpolation='nearest')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(model.upper(), fontsize=14, rotation=0, labelpad=50, va='center')
            if i == 0:
                axes[i, j].set_title(f"Frame {f}\nempty={empty_cnt}/{total}", fontsize=11)

    plt.suptitle("4D Occupancy Generation: Baseline vs Full (STCA+SADS)", fontsize=16, y=1.02)
    plt.tight_layout()
    out_name = "4d_occ_comparison_grouped.png" if args.grouped else "4d_occ_comparison.png"
    plt.savefig(f"{output_dir}/{out_name}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/{out_name}")

    # 1b. Diff mask (baseline vs full) to make changes obvious
    if "baseline" in latents and "full" in latents:
        print("\nGenerating diff-mask figure...")
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for j, f in enumerate(frames_to_show):
            occ_b = decode_single_frame(vq, latents["baseline"], f, device)
            occ_f = decode_single_frame(vq, latents["full"], f, device)
            _, lbl_b, _, total = create_bev_image(occ_b, grouped=args.grouped)
            _, lbl_f, _, _ = create_bev_image(occ_f, grouped=args.grouped)
            diff = (lbl_b != lbl_f)
            diff_ratio = float(diff.mean())
            axes[j].imshow(diff.astype(np.float32), cmap="gray", vmin=0, vmax=1, interpolation='nearest')
            axes[j].axis('off')
            axes[j].set_title(f"Frame {f}\ndiff={diff_ratio:.2%}", fontsize=11)
        plt.suptitle("BEV Difference Mask: Baseline vs Full", fontsize=14)
        plt.tight_layout()
        out_name = "4d_occ_diffmask_grouped.png" if args.grouped else "4d_occ_diffmask.png"
        plt.savefig(f"{output_dir}/{out_name}", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/{out_name}")

    # 2. 时序演化图 (Full模型, 8帧)
    print("\nGenerating temporal evolution figure...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    frames_seq = [0, 4, 8, 12, 16, 20, 24, 31]

    latent = latents.get("full", latents.get("baseline"))
    for idx, f in enumerate(frames_seq):
        row, col = idx // 4, idx % 4
        print(f"  Decoding frame {f}...")
        occ = decode_single_frame(vq, latent, f, device)
        rgb, _, empty_cnt, total = create_bev_image(occ, grouped=args.grouped)

        axes[row, col].imshow(rgb, interpolation='nearest')
        axes[row, col].axis('off')
        axes[row, col].set_title(f"t={f} (empty={empty_cnt}/{total})", fontsize=11)

    plt.suptitle("Temporal Evolution of Generated 4D Occupancy (Full Model)", fontsize=14)
    plt.tight_layout()
    out_name = "4d_temporal_evolution_grouped.png" if args.grouped else "4d_temporal_evolution.png"
    plt.savefig(f"{output_dir}/{out_name}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/{out_name}")

    # 3. 添加图例
    print("\nGenerating legend...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    ax.axis('off')

    # 只显示常见类别
    show_classes = [4, 7, 11, 13, 14, 15, 16, 17]  # car, ped, driveable, sidewalk, terrain, manmade, veg, empty
    patches = []
    labels = []
    for c in show_classes:
        color = COLORS[c] / 255.0
        patches.append(plt.Rectangle((0, 0), 1, 1, fc=color))
        labels.append(CLASS_NAMES[c])

    ax.legend(patches, labels, loc='center', ncol=8, fontsize=10)
    plt.savefig(f"{output_dir}/legend.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/legend.png")

    print(f"\n✓ All 4D visualizations saved to {output_dir}/")

if __name__ == "__main__":
    main()
