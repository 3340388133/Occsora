#!/usr/bin/env python
"""
在本地Mac上用Mayavi可视化OccSora occupancy数据
（官方visualize_demo.py的简化版本）

安装依赖:
    pip install mayavi numpy pillow

用法:
    python visualize_with_mayavi_local.py

或者直接查看某一帧:
    python visualize_with_mayavi_local.py --frame 5
"""

import numpy as np
import argparse
from pathlib import Path

def get_grid_coords(dims, resolution):
    """
    计算体素网格坐标
    :param dims: 网格维度 [x, y, z]
    :return coords_grid: 体素中心坐标
    """
    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid


def visualize_frame_mayavi(voxels, vox_origin=[-40, -40, -1], voxel_size=[0.4, 0.4, 0.4],
                          save_path=None, show_interactive=True):
    """
    用Mayavi可视化单帧occupancy（官方风格）

    参数:
        voxels: numpy array (200, 200, 16) - occupancy数据
        vox_origin: list - 体素原点坐标
        voxel_size: list - 体素尺寸
        save_path: str - 保存路径（None则不保存）
        show_interactive: bool - 是否显示交互式窗口
    """
    from mayavi import mlab

    w, h, z = voxels.shape  # 200, 200, 16

    # 计算体素坐标
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # 添加ego车辆（可选，和官方一样）
    car_vox_range = np.array([
        [w//2 - 2 - 4, w//2 - 2 + 4],
        [h//2 - 2 - 4, h//2 - 2 + 4],
        [z//2 - 2 - 3, z//2 - 2 + 3]
    ], dtype=int)

    car_x = np.arange(car_vox_range[0, 0], car_vox_range[0, 1])
    car_y = np.arange(car_vox_range[1, 0], car_vox_range[1, 1])
    car_z = np.arange(car_vox_range[2, 0], car_vox_range[2, 1])

    car_xx, car_yy, car_zz = np.meshgrid(car_x, car_y, car_z)
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

    car_grid = np.array([car_xx.flatten(), car_yy.flatten(), car_zz.flatten()]).T
    car_indexes = car_grid[:, 0] * h * z + car_grid[:, 1] * z + car_grid[:, 2]
    grid_coords[car_indexes, 3] = car_label.flatten()

    grid_coords[grid_coords[:, 3] == 17, 3] = 21

    # 过滤有效体素
    fov_grid_coords = grid_coords
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 21)
    ]

    print(f"有效体素数: {len(fov_voxels)}")

    # 创建Mayavi图形
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    voxel_size_avg = sum(voxel_size) / 3

    # 绘制体素（官方方式：cube模式）
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 1],
        fov_voxels[:, 0],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        scale_factor=1.0 * voxel_size_avg,
        mode="cube",      # 关键：cube模式！
        opacity=1.0,
        vmin=1,
        vmax=21,
    )

    # 官方颜色映射
    colors = np.array([
        [255, 120,  50, 255],   # 1: barrier (橙色)
        [255, 192, 203, 255],   # 2: bicycle (粉色)
        [255, 255,   0, 255],   # 3: bus (黄色)
        [  0, 150, 245, 255],   # 4: car (蓝色)
        [  0, 255, 255, 255],   # 5: construction_vehicle (青色)
        [255, 127,   0, 255],   # 6: motorcycle (深橙)
        [255,   0,   0, 255],   # 7: pedestrian (红色)
        [255, 240, 150, 255],   # 8: traffic_cone (浅黄)
        [135,  60,   0, 255],   # 9: trailer (棕色)
        [160,  32, 240, 255],   # 10: truck (紫色)
        [255,   0, 255, 255],   # 11: driveable_surface (紫色!) ← 路面
        [139, 137, 137, 255],   # 12: other_flat (灰色)
        [ 75,   0,  75, 255],   # 13: sidewalk (深紫)
        [150, 240,  80, 255],   # 14: terrain (浅绿)
        [230, 230, 250, 255],   # 15: manmade (白色)
        [  0, 175,   0, 255],   # 16: vegetation (绿色)
        [  0, 255, 127, 255],   # 17/21: ego car part 1
        [255,  99,  71, 255],   # 18: ego car part 2
        [  0, 191, 255, 255],   # 19: ego car part 3
        [175,   0,  75, 255]    # 20: ego car part 4
    ]).astype(np.uint8)

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    # 保存图像
    if save_path:
        mlab.savefig(save_path)
        print(f"已保存到: {save_path}")

    # 显示交互式窗口
    if show_interactive:
        print("\n💡 提示：")
        print("  - 鼠标左键拖拽：旋转")
        print("  - 鼠标右键拖拽：缩放")
        print("  - 鼠标中键拖拽：平移")
        print("  - 关闭窗口继续下一帧\n")
        mlab.show()
    else:
        mlab.close()


def create_gif_from_frames(output_dir, gif_path, duration=300):
    """从PNG帧创建GIF"""
    from PIL import Image
    import glob

    png_files = sorted(glob.glob(str(Path(output_dir) / 'frame_*.png')))

    if not png_files:
        print("没有找到PNG文件！")
        return

    print(f"\n创建GIF动画...")
    images = [Image.open(f) for f in png_files]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    print(f"✓ GIF已保存: {gif_path}")


def main():
    parser = argparse.ArgumentParser(description='用Mayavi可视化OccSora occupancy数据')
    parser.add_argument('--data', type=str, default='scene_0_occupancy.npy',
                        help='Occupancy数据文件路径')
    parser.add_argument('--frame', type=int, default=None,
                        help='只显示指定帧（0-31）')
    parser.add_argument('--save-frames', action='store_true',
                        help='保存所有帧为PNG')
    parser.add_argument('--create-gif', action='store_true',
                        help='创建GIF动画')
    parser.add_argument('--output-dir', type=str, default='mayavi_output',
                        help='输出目录')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='导出起始帧（默认0）')
    parser.add_argument('--end-frame', type=int, default=None,
                        help='导出结束帧（包含，默认最后一帧）')
    parser.add_argument('--step', type=int, default=1,
                        help='导出步长（默认1）')
    parser.add_argument('--duration', type=int, default=120,
                        help='GIF每帧持续时间（毫秒，默认120）')

    args = parser.parse_args()

    # 加载数据
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ 文件不存在: {data_path}")
        print(f"\n💡 请确保在download_package目录下运行:")
        print(f"   cd download_package")
        print(f"   python visualize_with_mayavi_local.py")
        return

    print(f"加载数据: {data_path}")
    occupancy = np.load(data_path)
    print(f"数据shape: {occupancy.shape}")

    n_frames = occupancy.shape[0]

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 模式1: 只显示单帧
    if args.frame is not None:
        if args.frame < 0 or args.frame >= n_frames:
            print(f"❌ 帧索引超出范围: {args.frame} (应该在0-{n_frames-1})")
            return

        print(f"\n显示第 {args.frame} 帧...")
        frame_data = occupancy[args.frame]
        visualize_frame_mayavi(frame_data, show_interactive=True)

    # 模式2: 保存所有帧为PNG
    elif args.save_frames:
        start = max(0, int(args.start_frame))
        end = n_frames - 1 if args.end_frame is None else min(n_frames - 1, int(args.end_frame))
        step = max(1, int(args.step))
        if start > end:
            print(f"❌ 帧范围不合法: start={start} > end={end}")
            return

        frame_indices = list(range(start, end + 1, step))
        print(f"\n保存{len(frame_indices)}帧为PNG... (start={start}, end={end}, step={step})")
        for i, frame_idx in enumerate(frame_indices):
            frame_data = occupancy[frame_idx]
            save_path = output_dir / f'frame_{frame_idx:04d}.png'
            print(f"处理帧 {i+1}/{len(frame_indices)} (frame={frame_idx})...")
            visualize_frame_mayavi(frame_data, save_path=str(save_path),
                                 show_interactive=False)

        print(f"\n✓ 所有帧已保存到: {output_dir}/")

        # 创建GIF
        if args.create_gif:
            gif_path = output_dir / 'animation.gif'
            create_gif_from_frames(output_dir, gif_path, duration=args.duration)

    # 模式3: 交互式浏览所有帧
    else:
        print(f"\n交互式浏览模式 - 共{n_frames}帧")
        print("关闭窗口查看下一帧，Ctrl+C退出\n")

        for frame_idx in range(n_frames):
            try:
                print(f"显示第 {frame_idx}/{n_frames-1} 帧...")
                frame_data = occupancy[frame_idx]
                visualize_frame_mayavi(frame_data, show_interactive=True)
            except KeyboardInterrupt:
                print("\n用户中断")
                break


if __name__ == '__main__':
    main()
