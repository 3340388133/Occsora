#!/usr/bin/env python3
"""
3D语义占用网格可视化 - Matplotlib版本（最稳定）

用法:
    python tools/visualize_semantic_occ_matplotlib.py \
        --occ-path diagnosis_output/decoded_full_cond0.npy \
        --output-dir paper_figures \
        --frame 0
"""

import os
import sys
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from pathlib import Path
import argparse

sys.path.insert(0, "/root/OccSora-main")

# 语义类别
CLASS_NAMES = [
    "noise", "barrier", "bicycle", "bus", "car", "construction",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain",
    "manmade", "vegetation", "empty"
]

# RGB颜色 (0-1)
COLORS = {
    0:  (0, 0, 0),
    1:  (1.0, 0.47, 0.2),      # barrier - orange
    2:  (1.0, 0.75, 0.8),      # bicycle - pink
    3:  (1.0, 1.0, 0),         # bus - yellow
    4:  (0, 0.59, 0.96),       # car - blue
    5:  (0, 1.0, 1.0),         # construction - cyan
    6:  (1.0, 0.5, 0),         # motorcycle
    7:  (1.0, 0, 0),           # pedestrian - red
    8:  (1.0, 0.94, 0.59),     # traffic_cone
    9:  (0.53, 0.24, 0),       # trailer - brown
    10: (0.63, 0.13, 0.94),    # truck - purple
    11: (1.0, 0, 1.0),         # driveable - magenta
    12: (0.55, 0.54, 0.54),    # other_flat - grey
    13: (0.29, 0, 0.29),       # sidewalk
    14: (0.59, 0.94, 0.31),    # terrain - light green
    15: (0.9, 0.9, 0.98),      # manmade
    16: (0, 0.69, 0),          # vegetation - green
    17: (0.5, 0.5, 0.5),       # empty
}

EMPTY_ID = 17
GROUND_IDS = {11, 12, 13, 14}


def render_semantic_occ(
    voxels,
    output_path,
    hide_ground=False,
    hide_empty=True,
    downsample=4,
    azimuth=45,
    elevation=30,
    figsize=(12, 10),
):
    """使用Matplotlib渲染3D语义占用"""
    voxels = voxels.copy()

    # 下采样
    voxels = voxels[::downsample, ::downsample, :]
    H, W, D = voxels.shape

    # 收集点和颜色
    all_x, all_y, all_z, all_c = [], [], [], []

    for class_id in range(1, 17):
        if hide_ground and class_id in GROUND_IDS:
            continue

        mask = (voxels == class_id)
        if not np.any(mask):
            continue

        coords = np.argwhere(mask)
        color = COLORS[class_id]

        all_x.extend(coords[:, 0])
        all_y.extend(coords[:, 1])
        all_z.extend(coords[:, 2])
        all_c.extend([color] * len(coords))

    if not all_x:
        print("No voxels to render!")
        return

    # 绘制
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(all_x, all_y, all_z, c=all_c, s=15, marker='s', alpha=0.9)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_box_aspect([H, W, D*2])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
