#!/usr/bin/env python3
"""4D占用可视化 - VTK 3D体素渲染
生成 t=0, 5, 10, 15, 20 五个时间步的3D占用图
"""
import os
import sys
import numpy as np
import torch
import vtk

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


def visualize_occ_vtk(occ, frame_idx, output_path,
                      hide_ground=True, hide_empty=True,
                      downsample=2, azimuth=45, elevation=30):
    """使用VTK渲染单帧3D占用图"""

    if occ.ndim == 4:
        frame = occ[frame_idx]
    else:
        frame = occ

    H, W, D = frame.shape

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
        return

    labels = frame[mask]

    # 创建VTK点
    points = vtk.vtkPoints()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")

    for i, (coord, label) in enumerate(zip(coords, labels)):
        x, y, z = coord[0], coord[1], coord[2] * 3
        points.InsertNextPoint(x, y, z)
        c = COLORS.get(int(label), (0.5, 0.5, 0.5))
        colors.InsertNextTuple3(int(c[0]*255), int(c[1]*255), int(c[2]*255))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().SetScalars(colors)
