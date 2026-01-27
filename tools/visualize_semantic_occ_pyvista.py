#!/usr/bin/env python3
"""
3D语义占用网格可视化 - PyVista版本（更稳定）

功能：
1. 单帧高质量3D语义占用可视化
2. 多方法对比图
3. 多视角展示
4. 语义类别图例

用法:
    python tools/visualize_semantic_occ_pyvista.py \
        --occ-path path/to/occupancy.npy \
        --output-dir paper_figures \
        --mode single
"""

import os
import sys
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import pyvista as pv
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

pv.OFF_SCREEN = True

sys.path.insert(0, "/root/OccSora-main")

# ============================================================
# 语义类别定义
# ============================================================
CLASS_NAMES = [
    "noise", "barrier", "bicycle", "bus", "car", "construction",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain",
    "manmade", "vegetation", "empty"
]

# RGB颜色 (0-255)
COLORS_RGB = {
    0:  (0, 0, 0),           # noise - black
    1:  (255, 120, 50),      # barrier - orange
    2:  (255, 192, 203),     # bicycle - pink
    3:  (255, 255, 0),       # bus - yellow
    4:  (0, 150, 245),       # car - blue
    5:  (0, 255, 255),       # construction - cyan
    6:  (255, 127, 0),       # motorcycle - dark orange
    7:  (255, 0, 0),         # pedestrian - red
    8:  (255, 240, 150),     # traffic_cone - light yellow
    9:  (135, 60, 0),        # trailer - brown
    10: (160, 32, 240),      # truck - purple
    11: (255, 0, 255),       # driveable - magenta
    12: (139, 137, 137),     # other_flat - grey
    13: (75, 0, 75),         # sidewalk - dark purple
    14: (150, 240, 80),      # terrain - light green
    15: (230, 230, 250),     # manmade - lavender
    16: (0, 175, 0),         # vegetation - green
    17: (128, 128, 128),     # empty - grey
}

EMPTY_ID = 17
GROUND_IDS = {11, 12, 13, 14}


def render_semantic_occ(
    voxels,
    output_path,
    voxel_size=0.4,
    vox_origin=(-40, -40, -1),
    hide_ground=False,
    hide_empty=True,
    azimuth=45,
    elevation=30,
    fig_size=(1920, 1080),
    title=None,
):
    """
    使用PyVista渲染3D语义占用网格
    """
    voxels = voxels.copy()
    H, W, D = voxels.shape

    # 创建plotter
    plotter = pv.Plotter(off_screen=True, window_size=fig_size)
    plotter.set_background('white')

    # 遍历每个语义类别分别渲染
    for class_id in range(18):
        if hide_empty and class_id == EMPTY_ID:
            continue
        if hide_ground and class_id in GROUND_IDS:
            continue
        if class_id == 0:  # skip noise
            continue

        # 找到该类别的所有体素
        mask = (voxels == class_id)
        if not np.any(mask):
            continue

        # 获取体素坐标
        coords = np.argwhere(mask)

        # 转换到世界坐标
        world_coords = coords.astype(np.float32) * voxel_size
        world_coords[:, 0] += vox_origin[0]
        world_coords[:, 1] += vox_origin[1]
        world_coords[:, 2] += vox_origin[2]

        # 获取颜色
        color = np.array(COLORS_RGB[class_id]) / 255.0

        # 为每个体素创建立方体
        for coord in world_coords[::2]:  # 下采样加速
            cube = pv.Cube(center=coord, x_length=voxel_size*0.9,
                          y_length=voxel_size*0.9, z_length=voxel_size*0.9)
            plotter.add_mesh(cube, color=color, opacity=1.0)

    # 设置相机
    plotter.camera_position = 'iso'
    plotter.camera.azimuth = azimuth
    plotter.camera.elevation = elevation
    plotter.camera.zoom(1.2)

    # 添加标题
    if title:
        plotter.add_title(title, font_size=16)

    # 保存
    plotter.screenshot(output_path)
    plotter.close()
    print(f"Saved: {output_path}")


def create_legend(output_path, show_classes=None):
    """创建语义类别图例"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if show_classes is None:
        show_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]

    fig, ax = plt.subplots(figsize=(3, len(show_classes) * 0.35))

    patches = []
    for idx in show_classes:
        color = np.array(COLORS_RGB[idx]) / 255.0
        patch = mpatches.Patch(color=color, label=CLASS_NAMES[idx])
        patches.append(patch)

    ax.legend(handles=patches, loc='center', fontsize=9, frameon=True)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Legend saved: {output_path}")


def render_semantic_occ_fast(
    voxels,
    output_path,
    voxel_size=0.4,
    vox_origin=(-40, -40, -1),
    hide_ground=False,
    hide_empty=True,
    azimuth=45,
    elevation=30,
    downsample=4,
    fig_size=(1920, 1080),
):
    """快速渲染版本 - 使用点云代替立方体"""
    voxels = voxels.copy()
    H, W, D = voxels.shape

    # 下采样
    voxels = voxels[::downsample, ::downsample, :]

    # 收集所有点和颜色
    all_points = []
    all_colors = []

    for class_id in range(1, 17):  # 跳过noise和empty
        if hide_ground and class_id in GROUND_IDS:
            continue

        mask = (voxels == class_id)
        if not np.any(mask):
            continue

        coords = np.argwhere(mask).astype(np.float32)
        coords *= downsample * voxel_size
        coords[:, 0] += vox_origin[0]
        coords[:, 1] += vox_origin[1]
        coords[:, 2] += vox_origin[2]

        color = np.array(COLORS_RGB[class_id]) / 255.0
        colors = np.tile(color, (len(coords), 1))

        all_points.append(coords)
        all_colors.append(colors)

    if not all_points:
        print("No voxels to render!")
        return

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)

    # 创建点云
    cloud = pv.PolyData(points)
    cloud['colors'] = (colors * 255).astype(np.uint8)

    # 渲染
    plotter = pv.Plotter(off_screen=True, window_size=fig_size)
    plotter.set_background('white')
    plotter.add_mesh(cloud, scalars='colors', rgb=True,
                     point_size=8, render_points_as_spheres=True)

    plotter.camera_position = 'iso'
    plotter.camera.azimuth = azimuth
    plotter.camera.elevation = elevation
    plotter.camera.zoom(1.0)

    plotter.screenshot(output_path)
    plotter.close()
    print(f"Saved: {output_path}")


def create_legend(output_path, show_classes=None):
    """创建语义类别图例"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if show_classes is None:
        show_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]

    fig, ax = plt.subplots(figsize=(3, len(show_classes) * 0.35))

    patches = []
    for idx in show_classes:
        color = np.array(COLORS_RGB[idx]) / 255.0
        patch = mpatches.Patch(color=color, label=CLASS_NAMES[idx])
        patches.append(patch)

    ax.legend(handles=patches, loc='center', fontsize=9, frameon=True)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Legend saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='3D Semantic Occ Visualization')
    parser.add_argument('--occ-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='paper_figures')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--hide-ground', action='store_true')
    parser.add_argument('--azimuth', type=float, default=45)
    parser.add_argument('--elevation', type=float, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    occ = np.load(args.occ_path)
    print(f"Loaded: {occ.shape}")

    if occ.ndim == 4:
        frame_occ = occ[args.frame]
    else:
        frame_occ = occ

    print(f"Frame: {frame_occ.shape}, Labels: {np.unique(frame_occ)}")

    output_path = f"{args.output_dir}/semantic_occ_frame{args.frame}.png"
    render_semantic_occ_fast(
        frame_occ, output_path,
        hide_ground=args.hide_ground,
        azimuth=args.azimuth,
        elevation=args.elevation,
        downsample=2,
    )

    create_legend(f"{args.output_dir}/semantic_legend.png")
    print(f"Done! Output: {args.output_dir}/")


if __name__ == '__main__':
    main()
