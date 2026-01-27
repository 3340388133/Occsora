#!/usr/bin/env python3
"""4D占用可视化 - PyVista 3D体素渲染
生成 t=0, 5, 10, 15, 20 五个时间步的3D占用图
"""
import os
import sys

# 设置离屏渲染
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, "/root/OccSora-main")

import pyvista as pv
pv.start_xvfb()  # 启动虚拟帧缓冲

# 语义类别颜色 (RGB, 0-255)
COLORS = {
    0: [0, 0, 0],           # noise
    1: [255, 120, 50],      # barrier
    2: [255, 192, 203],     # bicycle
    3: [255, 255, 0],       # bus
    4: [0, 150, 245],       # car (蓝色)
    5: [0, 255, 255],       # construction
    6: [255, 127, 0],       # motorcycle
    7: [255, 0, 0],         # pedestrian (红色)
    8: [255, 240, 150],     # traffic cone
    9: [135, 60, 0],        # trailer
    10: [160, 32, 240],     # truck (紫色)
    11: [255, 0, 255],      # driveable
    12: [139, 137, 137],    # other flat
    13: [75, 0, 75],        # sidewalk
    14: [150, 240, 80],     # terrain
    15: [230, 230, 250],    # manmade
    16: [0, 175, 0],        # vegetation (绿色)
    17: [128, 128, 128],    # empty
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


def visualize_occ_pyvista(occ, frame_idx, output_path,
                          hide_ground=True, hide_empty=True,
                          downsample=2, azimuth=45, elevation=30):
    """使用PyVista渲染单帧3D占用图"""

    # 获取单帧数据
    if occ.ndim == 4:
        frame = occ[frame_idx]
    else:
        frame = occ

    H, W, D = frame.shape

    # 下采样
    if downsample > 1:
        frame = frame[::downsample, ::downsample, :]
        H, W, D = frame.shape

    # 筛选要显示的体素
    mask = np.ones_like(frame, dtype=bool)
    if hide_empty:
        mask &= (frame != EMPTY_ID)
    if hide_ground:
        for gid in GROUND_IDS:
            mask &= (frame != gid)

    # 提取非空体素
    coords = np.argwhere(mask)
    if len(coords) == 0:
        print(f"  Warning: Frame {frame_idx} has no visible voxels!")
        return

    labels = frame[mask]

    # 创建颜色数组
    colors = np.array([COLORS.get(l, [128, 128, 128]) for l in labels], dtype=np.uint8)

    # 创建点云
    points = coords.astype(float)
    points[:, 2] *= 3  # Z方向拉伸

    cloud = pv.PolyData(points)
    cloud['colors'] = colors

    # 创建渲染器
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
    plotter.background_color = 'white'

    # 添加体素（使用立方体glyph）
    cube = pv.Cube(x_length=0.8*downsample, y_length=0.8*downsample, z_length=2.4)
    glyphs = cloud.glyph(geom=cube, scale=False, orient=False)

    plotter.add_mesh(glyphs, scalars='colors', rgb=True,
                     show_scalar_bar=False)

    # 设置视角
    plotter.camera_position = 'iso'
    plotter.camera.azimuth = azimuth
    plotter.camera.elevation = elevation
    plotter.camera.zoom(1.2)

    # 添加标题
    plotter.add_text(f"t = {frame_idx}", position='upper_edge',
                     font_size=16, color='black')

    # 添加坐标轴
    plotter.add_axes()

    # 保存
    plotter.screenshot(output_path)
    plotter.close()
    print(f"  Saved: {output_path}")


def create_combined_figure(image_paths, timesteps, output_path, model_name):
    """组合多个时间步图片"""
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(image_paths)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

    if n == 1:
        axes = [axes]

    for i, (path, t) in enumerate(zip(image_paths, timesteps)):
        if os.path.exists(path):
            img = Image.open(path)
            axes[i].imshow(img)
        axes[i].set_title(f"t = {t}", fontsize=14, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle(f"4D Occupancy Generation - {model_name}",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Combined figure saved: {output_path}")


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

    # 加载或解码数据
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
    print(f"Timesteps: {timesteps}\n")

    # 生成可视化
    image_paths = []
    for t in timesteps:
        t = min(t, occ.shape[0] - 1)
        out_path = f"{args.output_dir}/{args.model}_t{t}.png"
        print(f"Rendering t={t}...")
        visualize_occ_pyvista(occ, t, out_path, downsample=2)
        image_paths.append(out_path)

    # 组合图
    combined_path = f"{args.output_dir}/{args.model}_combined.png"
    create_combined_figure(image_paths, timesteps, combined_path, args.model)

    print(f"\nDone! Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
