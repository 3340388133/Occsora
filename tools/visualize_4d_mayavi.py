#!/usr/bin/env python3
"""4D占用可视化 - Mayavi 3D体素渲染
生成 t=0, 5, 10, 15, 20 五个时间步的3D占用图
"""
import os
import sys

# 设置虚拟显示（必须在导入mayavi之前）
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from pyvirtualdisplay import Display
display = Display(visible=False, size=(1920, 1080))
display.start()

import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, "/root/OccSora-main")

# 语义类别颜色 (RGB, 0-1范围)
COLORS = {
    0: (0, 0, 0),           # noise
    1: (1.0, 0.47, 0.2),    # barrier
    2: (1.0, 0.75, 0.8),    # bicycle
    3: (1.0, 1.0, 0),       # bus
    4: (0, 0.59, 0.96),     # car (蓝色)
    5: (0, 1.0, 1.0),       # construction
    6: (1.0, 0.5, 0),       # motorcycle
    7: (1.0, 0, 0),         # pedestrian (红色)
    8: (1.0, 0.94, 0.59),   # traffic cone
    9: (0.53, 0.24, 0),     # trailer
    10: (0.63, 0.13, 0.94), # truck (紫色)
    11: (1.0, 0, 1.0),      # driveable
    12: (0.55, 0.54, 0.54), # other flat
    13: (0.29, 0, 0.29),    # sidewalk
    14: (0.59, 0.94, 0.31), # terrain
    15: (0.9, 0.9, 0.98),   # manmade
    16: (0, 0.69, 0),       # vegetation (绿色)
    17: (0.5, 0.5, 0.5),    # empty
}

EMPTY_ID = 17
GROUND_IDS = {11, 12, 13, 14}  # 地面类别（可视化时可选择隐藏）

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


def visualize_occ_mayavi(occ, frame_idx, output_path,
                         hide_ground=True, hide_empty=True,
                         downsample=2, azimuth=45, elevation=60):
    """
    使用Mayavi渲染单帧3D占用图

    Args:
        occ: (T, H, W, D) 占用数据
        frame_idx: 时间帧索引
        output_path: 输出图片路径
        hide_ground: 是否隐藏地面
        hide_empty: 是否隐藏空体素
        downsample: 下采样因子（减少渲染点数）
        azimuth: 方位角
        elevation: 仰角
    """
    from mayavi import mlab

    # 获取单帧数据
    if occ.ndim == 4:
        frame = occ[frame_idx]  # (H, W, D)
    else:
        frame = occ

    H, W, D = frame.shape

    # 下采样以加速渲染
    if downsample > 1:
        frame = frame[::downsample, ::downsample, :]
        H, W, D = frame.shape

    # 创建坐标网格
    x, y, z = np.mgrid[0:H, 0:W, 0:D]

    # 筛选要显示的体素
    mask = np.ones_like(frame, dtype=bool)
    if hide_empty:
        mask &= (frame != EMPTY_ID)
    if hide_ground:
        for gid in GROUND_IDS:
            mask &= (frame != gid)

    # 提取非空体素坐标和类别
    xs = x[mask].flatten()
    ys = y[mask].flatten()
    zs = z[mask].flatten()
    labels = frame[mask].flatten()

    if len(xs) == 0:
        print(f"  Warning: Frame {frame_idx} has no visible voxels!")
        # 创建空白图
        mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))
        mlab.text(0.3, 0.5, f"t={frame_idx}: No visible voxels", width=0.4)
        mlab.savefig(output_path)
        mlab.close()
        return

    # 创建颜色数组
    colors = np.array([COLORS.get(l, (0.5, 0.5, 0.5)) for l in labels])

    # 创建Mayavi场景
    mlab.figure(size=(1200, 900), bgcolor=(1, 1, 1))

    # 使用points3d绘制体素（更快）
    pts = mlab.points3d(xs, ys, zs * 3,  # z方向拉伸以便观察
                        scale_factor=downsample * 0.8,
                        scale_mode='none',
                        mode='cube')

    # 设置颜色
    pts.glyph.scale_mode = 'scale_by_vector'
    pts.mlab_source.dataset.point_data.scalars = labels.astype(np.float32)

    # 创建自定义颜色映射
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(18):
        c = COLORS.get(i, (0.5, 0.5, 0.5))
        lut[i] = [int(c[0]*255), int(c[1]*255), int(c[2]*255), 255]
    pts.module_manager.scalar_lut_manager.lut.table = lut

    # 设置视角
    mlab.view(azimuth=azimuth, elevation=elevation, distance='auto')

    # 添加标题
    mlab.title(f"t = {frame_idx}", size=0.3, height=0.95)

    # 添加坐标轴
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z',
              ranges=[0, H, 0, W, 0, D*3],
              nb_labels=5)

    # 保存图片
    mlab.savefig(output_path, magnification=2)
    mlab.close()
    print(f"  Saved: {output_path}")


def create_multi_timestep_figure(occ, timesteps, output_dir, model_name="model"):
    """创建多时间步的组合图"""
    from mayavi import mlab
    import matplotlib.pyplot as plt
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    # 先生成各个时间步的单独图片
    temp_files = []
    for t in timesteps:
        if t >= occ.shape[0]:
            t = occ.shape[0] - 1
        temp_path = f"{output_dir}/temp_t{t}.png"
        visualize_occ_mayavi(occ, t, temp_path,
                            hide_ground=True, hide_empty=True,
                            downsample=2)
        temp_files.append(temp_path)

    # 使用matplotlib组合图片
    fig, axes = plt.subplots(1, len(timesteps), figsize=(20, 5))

    for i, (t, fpath) in enumerate(zip(timesteps, temp_files)):
        if os.path.exists(fpath):
            img = Image.open(fpath)
            axes[i].imshow(img)
            axes[i].set_title(f"t = {t}", fontsize=14, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle(f"4D Occupancy Generation - {model_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    combined_path = f"{output_dir}/{model_name}_timesteps_combined.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Combined figure saved: {combined_path}")

    # 清理临时文件
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Mayavi 4D Occupancy Visualization")
    ap.add_argument("--model", type=str, default="full",
                    choices=["baseline", "full", "sads", "stca"],
                    help="Model type to visualize")
    ap.add_argument("--cond", type=int, default=0, help="Condition index")
    ap.add_argument("--output-dir", type=str, default="/root/OccSora-main/mayavi_vis")
    ap.add_argument("--timesteps", type=str, default="0,5,10,15,20",
                    help="Comma-separated timesteps to visualize")
    ap.add_argument("--use-cached", action="store_true",
                    help="Use cached decoded occupancy if available")
    args = ap.parse_args()

    timesteps = [int(t) for t in args.timesteps.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # 检查是否有缓存的解码数据
    cache_path = f"/root/OccSora-main/diagnosis_output/decoded_{args.model}_cond{args.cond}.npy"

    if args.use_cached and os.path.exists(cache_path):
        print(f"Loading cached occupancy from {cache_path}")
        occ = np.load(cache_path)
    else:
        # 需要解码
        from mmengine import Config
        from mmengine.registry import MODELS
        import model as _

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载VQVAE
        print("Loading VQVAE...")
        cfg = Config.fromfile("/root/OccSora-main/config/train_vqvae.py")
        vq = MODELS.build(cfg.model)
        ckpt = torch.load("/root/autodl-tmp/OccSoraModel/epoch_125.pth", map_location="cpu")
        vq.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        vq = vq.to(device).eval()

        # 加载latent样本
        latent_path = f"/root/OccSora-main/eval_results_paperish_v6/samples_{args.model}_cond{args.cond}.npy"
        if not os.path.exists(latent_path):
            print(f"Error: {latent_path} not found!")
            return

        print(f"Loading latent from {latent_path}")
        latent = torch.from_numpy(np.load(latent_path)[:1]).float()

        print("Decoding latent to occupancy...")
        occ = decode_latent_to_occ(vq, latent, device)[0]  # (T, H, W, D)

        # 保存缓存
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, occ)
        print(f"Cached decoded occupancy to {cache_path}")

    print(f"\nOccupancy shape: {occ.shape}")
    print(f"Unique labels: {np.unique(occ)}")
    print(f"Timesteps to visualize: {timesteps}")

    # 生成可视化
    print("\n--- Generating Mayavi visualizations ---")

    # 单独的时间步图片
    for t in timesteps:
        if t >= occ.shape[0]:
            t = occ.shape[0] - 1
        out_path = f"{args.output_dir}/{args.model}_t{t}.png"
        print(f"Rendering t={t}...")
        visualize_occ_mayavi(occ, t, out_path,
                            hide_ground=True, hide_empty=True,
                            downsample=2)

    # 组合图
    print("\nCreating combined figure...")
    create_multi_timestep_figure(occ, timesteps, args.output_dir, args.model)

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
