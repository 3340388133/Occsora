#!/usr/bin/env python3
"""
3D语义占用网格可视化 - 论文专用版本

功能：
1. 单帧高质量3D语义占用可视化
2. 多方法对比图（Baseline vs Ours vs GT）
3. 多视角展示（前视、侧视、俯视）
4. 自定义颜色图例（Legend）
5. 支持隐藏/显示地面、空体素

用法:
    python tools/visualize_semantic_occ_paper.py \
        --occ-path path/to/occupancy.npy \
        --output-dir paper_figures \
        --mode single  # single / comparison / multiview

作者: OccSora Team
"""

import os
import sys

# 设置环境变量（必须在导入mayavi之前）
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['HOME'] = os.environ.get('HOME', '/tmp')

# 延迟导入mayavi（在需要时导入）
_mlab = None

def get_mlab():
    global _mlab
    if _mlab is None:
        from mayavi import mlab
        mlab.options.offscreen = True
        _mlab = mlab
    return _mlab

import numpy as np
import torch
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/OccSora-main")

# ============================================================
# 语义类别定义（nuScenes/Occ3D标准）
# ============================================================
CLASS_NAMES = [
    "noise",              # 0
    "barrier",            # 1
    "bicycle",            # 2
    "bus",                # 3
    "car",                # 4
    "construction",       # 5
    "motorcycle",         # 6
    "pedestrian",         # 7
    "traffic_cone",       # 8
    "trailer",            # 9
    "truck",              # 10
    "driveable_surface",  # 11
    "other_flat",         # 12
    "sidewalk",           # 13
    "terrain",            # 14
    "manmade",            # 15
    "vegetation",         # 16
    "empty",              # 17
]

# 论文级颜色映射（RGBA, 0-255）
COLORS_RGBA = np.array([
    [  0,   0,   0, 255],    # 0  noise - black
    [255, 120,  50, 255],    # 1  barrier - orange
    [255, 192, 203, 255],    # 2  bicycle - pink
    [255, 255,   0, 255],    # 3  bus - yellow
    [  0, 150, 245, 255],    # 4  car - blue
    [  0, 255, 255, 255],    # 5  construction - cyan
    [255, 127,   0, 255],    # 6  motorcycle - dark orange
    [255,   0,   0, 255],    # 7  pedestrian - red
    [255, 240, 150, 255],    # 8  traffic_cone - light yellow
    [135,  60,   0, 255],    # 9  trailer - brown
    [160,  32, 240, 255],    # 10 truck - purple
    [255,   0, 255, 255],    # 11 driveable_surface - magenta
    [139, 137, 137, 255],    # 12 other_flat - grey
    [ 75,   0,  75, 255],    # 13 sidewalk - dark purple
    [150, 240,  80, 255],    # 14 terrain - light green
    [230, 230, 250, 255],    # 15 manmade - lavender
    [  0, 175,   0, 255],    # 16 vegetation - green
    [128, 128, 128,   0],    # 17 empty - transparent
], dtype=np.uint8)

# Ego车颜色
EGO_COLORS = np.array([
    [  0, 255, 127, 255],    # 18 ego part 1 - spring green
    [255,  99,  71, 255],    # 19 ego part 2 - tomato
    [  0, 191, 255, 255],    # 20 ego part 3 - deep sky blue
], dtype=np.uint8)

EMPTY_ID = 17
GROUND_IDS = {11, 12, 13, 14}  # 地面类别


# ============================================================
# 核心可视化函数
# ============================================================
def get_grid_coords(dims, voxel_size, vox_origin):
    """计算体素网格的世界坐标"""
    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz, indexing='ij')
    coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)

    if isinstance(voxel_size, (int, float)):
        voxel_size = [voxel_size] * 3
    voxel_size = np.array(voxel_size, dtype=np.float32)
    vox_origin = np.array(vox_origin, dtype=np.float32)

    # 转换到世界坐标（体素中心）
    world_coords = coords * voxel_size + vox_origin + voxel_size / 2
    return world_coords


def add_ego_car(voxels, w, h, z):
    """在场景中心添加Ego车辆"""
    car_vox_range = np.array([
        [w//2 - 2 - 4, w//2 - 2 + 4],
        [h//2 - 2 - 4, h//2 - 2 + 4],
        [z//2 - 2 - 3, z//2 - 2 + 3]
    ], dtype=int)

    car_x = np.arange(car_vox_range[0, 0], car_vox_range[0, 1])
    car_y = np.arange(car_vox_range[1, 0], car_vox_range[1, 1])
    car_z = np.arange(car_vox_range[2, 0], car_vox_range[2, 1])

    car_xx, car_yy, car_zz = np.meshgrid(car_x, car_y, car_z, indexing='ij')

    # 车辆分段着色
    car_label = np.zeros([8, 8, 6], dtype=int)
    car_label[:3, :, :2] = 20
    car_label[3:6, :, :2] = 18
    car_label[6:, :, :2] = 19
    car_label[:3, :, 2:4] = 18
    car_label[3:6, :, 2:4] = 19
    car_label[6:, :, 2:4] = 20
    car_label[:3, :, 4:] = 19
    car_label[3:6, :, 4:] = 20
    car_label[6:, :, 4:] = 18

    # 写入体素
    for i, cx in enumerate(car_x):
        for j, cy in enumerate(car_y):
            for k, cz in enumerate(car_z):
                if 0 <= cx < w and 0 <= cy < h and 0 <= cz < z:
                    voxels[cx, cy, cz] = car_label[i, j, k]

    return voxels


def render_semantic_occ(
    voxels,
    output_path,
    voxel_size=0.4,
    vox_origin=(-40, -40, -1),
    hide_ground=False,
    hide_empty=True,
    add_ego=True,
    azimuth=45,
    elevation=60,
    distance=None,
    fig_size=(2560, 1440),
    title=None,
):
    """
    渲染单帧3D语义占用网格

    Args:
        voxels: (H, W, D) 语义标签数组
        output_path: 输出图片路径
        voxel_size: 体素大小（米）
        vox_origin: 体素原点坐标
        hide_ground: 是否隐藏地面类别
        hide_empty: 是否隐藏空体素
        add_ego: 是否添加Ego车辆
        azimuth: 方位角
        elevation: 仰角
        distance: 相机距离（None为自动）
        fig_size: 图像尺寸
        title: 图像标题
    """
    voxels = voxels.copy()
    w, h, z = voxels.shape

    # 添加Ego车
    if add_ego:
        voxels = add_ego_car(voxels, w, h, z)

    # 计算世界坐标
    world_coords = get_grid_coords([w, h, z], voxel_size, vox_origin)
    labels = voxels.flatten()

    # 构建数据数组 [x, y, z, label]
    data = np.column_stack([world_coords, labels])

    # 过滤体素
    mask = np.ones(len(data), dtype=bool)
    if hide_empty:
        mask &= (data[:, 3] != EMPTY_ID)
        mask &= (data[:, 3] != 21)  # 转换后的empty
    if hide_ground:
        for gid in GROUND_IDS:
            mask &= (data[:, 3] != gid)

    # 过滤noise
    mask &= (data[:, 3] > 0)

    filtered_data = data[mask]

    if len(filtered_data) == 0:
        print(f"Warning: No visible voxels!")
        return

    print(f"  Rendering {len(filtered_data)} voxels...")

    # 获取mlab
    mlab = get_mlab()

    # 创建Mayavi场景
    fig = mlab.figure(size=fig_size, bgcolor=(1, 1, 1))

    # 绘制体素
    pts = mlab.points3d(
        filtered_data[:, 1],  # Y -> X显示
        filtered_data[:, 0],  # X -> Y显示
        filtered_data[:, 2],  # Z
        filtered_data[:, 3],  # 标签
        scale_factor=voxel_size * 0.95,
        mode='cube',
        opacity=1.0,
        vmin=1,
        vmax=21,
    )

    # 设置颜色映射
    full_colors = np.vstack([COLORS_RGBA[1:18], EGO_COLORS, [[175, 0, 75, 255]]])
    pts.glyph.scale_mode = 'scale_by_vector'
    pts.module_manager.scalar_lut_manager.lut.table = full_colors

    # 设置视角
    mlab.view(azimuth=azimuth, elevation=elevation, distance=distance)

    # 添加标题
    if title:
        mlab.title(title, size=0.4, height=0.95, color=(0, 0, 0))

    # 保存
    mlab.savefig(output_path, magnification=1)
    mlab.close()
    print(f"  Saved: {output_path}")


def render_comparison(
    occ_list,
    labels,
    output_path,
    voxel_size=0.4,
    vox_origin=(-40, -40, -1),
    hide_ground=False,
    azimuth=45,
    elevation=60,
):
    """
    渲染多方法对比图（横向排列）

    Args:
        occ_list: 占用数据列表 [(H,W,D), ...]
        labels: 方法标签列表 ["Baseline", "Ours", "GT"]
        output_path: 输出路径
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import tempfile

    n = len(occ_list)
    temp_files = []

    # 渲染每个方法
    for i, (occ, label) in enumerate(zip(occ_list, labels)):
        temp_path = tempfile.mktemp(suffix='.png')
        render_semantic_occ(
            occ,
            temp_path,
            voxel_size=voxel_size,
            vox_origin=vox_origin,
            hide_ground=hide_ground,
            azimuth=azimuth,
            elevation=elevation,
            title=None,  # 用matplotlib添加标题
        )
        temp_files.append(temp_path)

    # 用matplotlib组合
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axes = [axes]

    for i, (fpath, label) in enumerate(zip(temp_files, labels)):
        if os.path.exists(fpath):
            img = Image.open(fpath)
            axes[i].imshow(img)
            axes[i].set_title(label, fontsize=18, fontweight='bold', pad=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # 清理临时文件
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)

    print(f"Comparison figure saved: {output_path}")


def render_multiview(
    voxels,
    output_path,
    voxel_size=0.4,
    vox_origin=(-40, -40, -1),
    hide_ground=False,
):
    """
    渲染多视角图（前视、侧视、俯视、斜视）
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import tempfile

    views = [
        ("Front View", 0, 90),      # 前视
        ("Side View", 90, 90),      # 侧视
        ("Top View (BEV)", 0, 0),   # 俯视
        ("Perspective", 45, 60),    # 斜视
    ]

    temp_files = []

    for name, az, el in views:
        temp_path = tempfile.mktemp(suffix='.png')
        render_semantic_occ(
            voxels,
            temp_path,
            voxel_size=voxel_size,
            vox_origin=vox_origin,
            hide_ground=hide_ground,
            azimuth=az,
            elevation=el,
            fig_size=(1920, 1080),
        )
        temp_files.append((temp_path, name))

    # 组合成2x2图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, (fpath, name) in enumerate(temp_files):
        if os.path.exists(fpath):
            img = Image.open(fpath)
            axes[i].imshow(img)
            axes[i].set_title(name, fontsize=14, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # 清理
    for fpath, _ in temp_files:
        if os.path.exists(fpath):
            os.remove(fpath)

    print(f"Multi-view figure saved: {output_path}")


def create_legend(output_path, show_classes=None):
    """
    创建语义类别图例（用于论文）

    Args:
        output_path: 输出路径
        show_classes: 要显示的类别索引列表，None表示全部
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if show_classes is None:
        # 默认显示主要类别（排除noise和empty）
        show_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]

    fig, ax = plt.subplots(figsize=(4, len(show_classes) * 0.4 + 0.5))

    patches = []
    for idx in show_classes:
        color = COLORS_RGBA[idx][:3] / 255.0
        patch = mpatches.Patch(color=color, label=CLASS_NAMES[idx])
        patches.append(patch)

    ax.legend(
        handles=patches,
        loc='center',
        fontsize=10,
        frameon=True,
        ncol=1,
    )
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Legend saved: {output_path}")


# ============================================================
# VQVAE解码函数
# ============================================================
@torch.no_grad()
def decode_latent_to_occ(latent_path, vqvae_ckpt, vqvae_config, device='cuda'):
    """从latent解码到occupancy"""
    from mmengine import Config
    from mmengine.registry import MODELS
    import model as _

    print(f"Loading VQVAE from: {vqvae_ckpt}")
    cfg = Config.fromfile(vqvae_config)
    vqvae = MODELS.build(cfg.model)

    ckpt = torch.load(vqvae_ckpt, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    vqvae.load_state_dict(state_dict, strict=False)
    vqvae = vqvae.to(device).eval()

    print(f"Loading latent from: {latent_path}")
    latent = np.load(latent_path)
    latent_tensor = torch.from_numpy(latent).float().to(device)

    # 处理通道数
    if latent_tensor.shape[1] == 256:
        latent_tensor = latent_tensor[:, :128]
    if latent_tensor.shape[1] == 128:
        z = latent_tensor[:, :64]
    else:
        z = latent_tensor

    # 解码
    x = vqvae.decoder_gpt(vqvae.post_vq_conv(z))
    B, C, F, H, W = x.shape
    D = 16

    template = vqvae.class_embeds.weight.T
    x = x.permute(0, 2, 3, 4, 1).contiguous()

    labels = torch.empty((B, F, H, W, D), dtype=torch.int16, device='cpu')
    for b in range(B):
        for f in range(F):
            xf = x[b, f].view(H * W, D, -1)
            sim = torch.matmul(xf, template)
            labels[b, f] = torch.argmax(sim, dim=-1).to(torch.int16).cpu().view(H, W, D)

    return labels.numpy()


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='3D Semantic Occupancy Visualization for Paper'
    )
    parser.add_argument('--occ-path', type=str,
                        help='Path to occupancy .npy file (H,W,D) or (T,H,W,D)')
    parser.add_argument('--latent-path', type=str,
                        help='Path to latent .npy file (需要VQVAE解码)')
    parser.add_argument('--vqvae-ckpt', type=str,
                        default='/root/autodl-tmp/OccSoraModel/epoch_125.pth')
    parser.add_argument('--vqvae-config', type=str,
                        default='/root/OccSora-main/config/train_vqvae.py')
    parser.add_argument('--output-dir', type=str, default='paper_figures')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'comparison', 'multiview', 'legend'])
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index for 4D data')
    parser.add_argument('--hide-ground', action='store_true',
                        help='Hide ground classes')
    parser.add_argument('--no-ego', action='store_true',
                        help='Do not add ego car')
    parser.add_argument('--azimuth', type=float, default=45)
    parser.add_argument('--elevation', type=float, default=60)
    parser.add_argument('--voxel-size', type=float, default=0.4)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    if args.mode == 'legend':
        create_legend(f"{args.output_dir}/semantic_legend.png")
        return

    if args.latent_path:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        occ = decode_latent_to_occ(
            args.latent_path,
            args.vqvae_ckpt,
            args.vqvae_config,
            device
        )[0]  # (T, H, W, D)
    elif args.occ_path:
        occ = np.load(args.occ_path)
    else:
        print("Error: Please provide --occ-path or --latent-path")
        return

    print(f"Occupancy shape: {occ.shape}")

    # 获取单帧
    if occ.ndim == 4:
        frame_occ = occ[args.frame]
    else:
        frame_occ = occ

    print(f"Frame shape: {frame_occ.shape}")
    print(f"Unique labels: {np.unique(frame_occ)}")

    # 执行可视化
    if args.mode == 'single':
        output_path = f"{args.output_dir}/semantic_occ_frame{args.frame}.png"
        render_semantic_occ(
            frame_occ,
            output_path,
            voxel_size=args.voxel_size,
            hide_ground=args.hide_ground,
            add_ego=not args.no_ego,
            azimuth=args.azimuth,
            elevation=args.elevation,
        )

    elif args.mode == 'multiview':
        output_path = f"{args.output_dir}/semantic_occ_multiview.png"
        render_multiview(
            frame_occ,
            output_path,
            voxel_size=args.voxel_size,
            hide_ground=args.hide_ground,
        )

    # 同时生成图例
    create_legend(f"{args.output_dir}/semantic_legend.png")

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
