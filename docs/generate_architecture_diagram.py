"""
OccSora Architecture Diagram Generator
生成类似参考图的Diffusion-based World Model架构图
针对OccSora项目的三大创新：STCA、SADS、PCL
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
import matplotlib.lines as mlines
import numpy as np
from matplotlib.collections import PatchCollection

# 设置字体支持
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

# 定义颜色方案
COLORS = {
    'random_noise': '#FFF9C4',      # 浅黄色 - Random Noise
    'linear_reshape': '#C8E6C9',     # 浅绿色 - Linear and Reshape
    'layer_norm': '#A5D6A7',         # 绿色 - Layer Norm
    'multi_head_attn': '#BBDEFB',    # 浅蓝色 - Multi-Head Self-Attention
    'pointwise_ff': '#F8BBD9',       # 浅粉色 - Pointwise Feedforward
    'scale_shift': '#B2DFDB',        # 浅青色 - Scale Shift
    'token': '#FFCCBC',              # 浅橙色 - Token
    'embedding': '#E1BEE7',          # 浅紫色 - Embedding
    'output_token': '#DCEDC8',       # 浅黄绿 - Output Token
    'occupancy': '#FFE0B2',          # 浅橙色 - Occupancy visualization
    'border_dashed': '#FF9800',      # 橙色虚线边框
    'arrow_train': '#4CAF50',        # 绿色箭头 - Train
    'arrow_generate': '#2196F3',     # 蓝色箭头 - Generate
    'loss': '#F44336',               # 红色 - Loss
}

def create_rounded_box(ax, x, y, width, height, color, label='', fontsize=8,
                       edgecolor='black', linewidth=1, linestyle='-', alpha=0.8):
    """创建圆角矩形框"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.05",
                         facecolor=color, edgecolor=edgecolor,
                         linewidth=linewidth, linestyle=linestyle, alpha=alpha)
    ax.add_patch(box)
    if label:
        ax.text(x + width/2, y + height/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', wrap=True)
    return box

def create_vertical_box(ax, x, y, width, height, color, label='', fontsize=7):
    """创建垂直文字的矩形框"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.01,rounding_size=0.02",
                         facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(box)
    if label:
        ax.text(x + width/2, y + height/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', rotation=90)
    return box

def draw_noise_pattern(ax, x, y, width, height):
    """绘制噪声图案（类似马赛克）"""
    np.random.seed(42)
    n_cells = 5
    cell_w = width / n_cells
    cell_h = height / n_cells

    for i in range(n_cells):
        for j in range(n_cells):
            gray = np.random.uniform(0.3, 0.9)
            rect = Rectangle((x + i*cell_w, y + j*cell_h), cell_w, cell_h,
                            facecolor=(gray, gray, gray), edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)

def draw_occupancy_visualization(ax, x, y, width, height, frame_idx=0):
    """绘制占用网格可视化（类似点云俯视图）"""
    np.random.seed(42 + frame_idx)

    # 背景
    rect = Rectangle((x, y), width, height, facecolor='white', edgecolor='black', linewidth=1)
    ax.add_patch(rect)

    # 绘制随机点表示占用
    n_points = 50
    px = x + np.random.uniform(0.1, 0.9, n_points) * width
    py = y + np.random.uniform(0.1, 0.9, n_points) * height
    colors = np.random.choice(['#9C27B0', '#4CAF50', '#FF9800'], n_points)
    ax.scatter(px, py, c=colors, s=3, alpha=0.7)

def draw_arrow(ax, start, end, color='black', style='->', connectionstyle='arc3,rad=0'):
    """绘制箭头"""
    arrow = FancyArrowPatch(start, end, arrowstyle=style,
                           connectionstyle=connectionstyle,
                           color=color, linewidth=1.5,
                           mutation_scale=10)
    ax.add_patch(arrow)

def draw_plus_circle(ax, x, y, radius=0.15):
    """绘制加号圆圈"""
    circle = Circle((x, y), radius, facecolor='white', edgecolor='black', linewidth=1)
    ax.add_patch(circle)
    ax.plot([x-radius*0.6, x+radius*0.6], [y, y], 'k-', linewidth=1.5)
    ax.plot([x, x], [y-radius*0.6, y+radius*0.6], 'k-', linewidth=1.5)

def create_dit_block(ax, x, y, block_width, block_height):
    """创建DiT Block（包含多个层）"""
    layer_height = block_height / 4

    # Multi-Head Self-Attention (Temporal Causal)
    create_rounded_box(ax, x, y + 3*layer_height, block_width, layer_height*0.9,
                      COLORS['multi_head_attn'], fontsize=5)

    # Pointwise Feedforward
    create_rounded_box(ax, x, y + 2*layer_height, block_width, layer_height*0.9,
                      COLORS['pointwise_ff'], fontsize=5)

    # Scale Shift
    create_rounded_box(ax, x, y + layer_height, block_width, layer_height*0.9,
                      COLORS['scale_shift'], fontsize=5)

    # Layer Norm
    create_rounded_box(ax, x, y, block_width, layer_height*0.9,
                      COLORS['layer_norm'], fontsize=5)

def main():
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # 标题
    ax.text(8, 9.5, 'Diffusion-based World Model', fontsize=16, fontweight='bold',
            ha='center', va='center')

    # ==================== 左侧输入部分 ====================

    # Random Noise (顶部)
    create_rounded_box(ax, 0.3, 7.5, 1.5, 1.2, COLORS['random_noise'], 'Random\nNoise',
                      edgecolor=COLORS['border_dashed'], linestyle='--', fontsize=8)
    draw_noise_pattern(ax, 0.5, 7.7, 1.1, 0.8)

    # Token
    create_rounded_box(ax, 0.3, 6.0, 1.5, 0.8, COLORS['token'], 'Token',
                      edgecolor=COLORS['border_dashed'], linestyle='--', fontsize=9)

    # 下方条件输入
    input_y_start = 1.5
    input_spacing = 1.0

    # Ego Car Trajectory
    create_rounded_box(ax, 0.3, input_y_start + 3*input_spacing, 1.5, 0.7,
                      COLORS['token'], 'Ego Car\nTrajectory',
                      edgecolor=COLORS['border_dashed'], linestyle='--', fontsize=7)

    # Time Step
    create_rounded_box(ax, 0.3, input_y_start + 2*input_spacing, 1.5, 0.7,
                      COLORS['token'], 'Time Step',
                      edgecolor=COLORS['border_dashed'], linestyle='--', fontsize=8)

    # Any Trajectory
    create_rounded_box(ax, 0.3, input_y_start + 1*input_spacing, 1.5, 0.7,
                      COLORS['token'], 'Any\nTrajectory',
                      edgecolor=COLORS['border_dashed'], linestyle='--', fontsize=7)

    # Random Noise (底部)
    create_rounded_box(ax, 0.3, input_y_start, 1.5, 0.7, COLORS['random_noise'], 'Random\nNoise',
                      edgecolor=COLORS['border_dashed'], linestyle='--', fontsize=8)

    # Embedding 标签
    ax.text(2.3, 3.5, 'Embedding', fontsize=9, fontweight='bold', rotation=90, va='center')

    # ==================== Linear and Reshape + Layer Norm ====================

    create_vertical_box(ax, 2.5, 6.5, 0.6, 2.5, COLORS['linear_reshape'], 'Linear and\nReshape', fontsize=7)
    create_vertical_box(ax, 3.3, 6.5, 0.5, 2.5, COLORS['layer_norm'], 'Layer Norm', fontsize=7)

    # ==================== DiT Blocks (STCA) ====================

    block_start_x = 4.2
    block_width = 0.8
    block_spacing = 0.15
    n_blocks = 6

    for i in range(n_blocks):
        bx = block_start_x + i * (block_width + block_spacing)
        create_dit_block(ax, bx, 5.8, block_width, 3.0)

        # 添加STCA符号
        ax.text(bx + block_width/2, 8.95, '✕', fontsize=10, ha='center', va='center')
        ax.text(bx + block_width/2, 8.6, '(○○○)', fontsize=6, ha='center', va='center', color='gray')

    # ==================== 右侧输出部分 ====================

    output_x = 10.5

    # Train Token (输出)
    create_rounded_box(ax, output_x + 2, 7.5, 1.8, 1.5, COLORS['output_token'], 'Train Token',
                      edgecolor='black', fontsize=9)
    # 绘制3D网格示意
    for i in range(3):
        for j in range(3):
            rect = Rectangle((output_x + 2.2 + i*0.4, 7.7 + j*0.35), 0.35, 0.3,
                            facecolor=COLORS['occupancy'], edgecolor='gray', linewidth=0.5, alpha=0.7)
            ax.add_patch(rect)

    # Generate Token (输出)
    create_rounded_box(ax, output_x + 2, 5.5, 1.8, 1.5, COLORS['output_token'], 'Generate Token',
                      edgecolor='black', fontsize=9)
    for i in range(3):
        for j in range(3):
            rect = Rectangle((output_x + 2.2 + i*0.4, 5.7 + j*0.35), 0.35, 0.3,
                            facecolor=COLORS['occupancy'], edgecolor='gray', linewidth=0.5, alpha=0.7)
            ax.add_patch(rect)

    # Loss 标签
    ax.text(13.5, 9.2, 'Loss', fontsize=10, fontweight='bold', color=COLORS['loss'])
    ax.annotate('', xy=(13.0, 8.8), xytext=(13.8, 9.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['loss'], lw=1.5))

    # ==================== 底部占用可视化 ====================

    occ_y = 1.5
    occ_width = 1.8
    occ_height = 1.5
    occ_spacing = 0.3

    # 绘制4帧占用网格
    for i in range(4):
        ox = 5.5 + i * (occ_width + occ_spacing)
        draw_occupancy_visualization(ax, ox, occ_y, occ_width, occ_height, frame_idx=i)

    # ==================== 连接箭头 ====================

    # 从Random Noise到Linear
    draw_arrow(ax, (1.8, 8.1), (2.5, 8.1), color='black')

    # 从Token到Linear
    draw_arrow(ax, (1.8, 6.4), (2.5, 6.8), color='black')

    # 从Linear到Layer Norm
    draw_arrow(ax, (3.1, 7.75), (3.3, 7.75), color='black')

    # 从Layer Norm到DiT Blocks
    draw_arrow(ax, (3.8, 7.75), (4.2, 7.75), color='black')

    # DiT Blocks之间的连接
    for i in range(n_blocks - 1):
        bx1 = block_start_x + i * (block_width + block_spacing) + block_width
        bx2 = block_start_x + (i+1) * (block_width + block_spacing)
        draw_arrow(ax, (bx1, 7.3), (bx2, 7.3), color='black')

    # 从最后一个DiT Block到输出
    last_block_x = block_start_x + (n_blocks-1) * (block_width + block_spacing) + block_width
    draw_arrow(ax, (last_block_x, 7.8), (output_x + 2, 8.0), color=COLORS['arrow_train'])
    draw_arrow(ax, (last_block_x, 7.0), (output_x + 2, 6.5), color=COLORS['arrow_generate'])

    # 条件输入的连接（加号）
    plus_x = 4.0
    plus_y = 7.3
    draw_plus_circle(ax, plus_x, plus_y)
    draw_plus_circle(ax, plus_x, 3.5)

    # 从条件输入到加号
    draw_arrow(ax, (1.8, 4.85), (3.8, 4.85), color='black')
    ax.plot([3.8, 3.8], [4.85, 3.5], 'k-', linewidth=1)
    ax.plot([3.8, plus_x-0.15], [3.5, 3.5], 'k-', linewidth=1)

    # 从加号到DiT Blocks（垂直连接）
    ax.plot([plus_x, plus_x], [plus_y + 0.15, 8.5], 'k-', linewidth=1)
    for i in range(n_blocks):
        bx = block_start_x + i * (block_width + block_spacing) + block_width/2
        ax.plot([plus_x, bx], [8.5, 8.5], color='#FF9800', linewidth=1)
        ax.annotate('', xy=(bx, 8.85), xytext=(bx, 8.5),
                   arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1))

    # 从底部占用到输出的连接
    ax.plot([9.5, 9.5], [2.25, 3.0], 'k-', linewidth=1)
    ax.plot([9.5, 11.0], [3.0, 3.0], 'k-', linewidth=1)
    ax.plot([11.0, 11.0], [3.0, 5.5], 'k-', linewidth=1)

    # ==================== 图例 ====================

    legend_x = 12.5
    legend_y = 4.0
    legend_spacing = 0.5

    # 图例标题
    ax.text(legend_x + 1, legend_y + 3.5, 'Legend', fontsize=10, fontweight='bold', ha='center')

    # Train箭头
    ax.annotate('', xy=(legend_x + 0.8, legend_y + 2.8), xytext=(legend_x, legend_y + 2.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow_train'], lw=2))
    ax.text(legend_x + 1.0, legend_y + 2.8, 'Train', fontsize=8, va='center')
    ax.plot([legend_x + 1.6, legend_x + 1.7], [legend_y + 2.8, legend_y + 2.8],
            'k-', linewidth=1, marker='|', markersize=8)
    ax.text(legend_x + 1.8, legend_y + 2.8, 'Frozen', fontsize=7, va='center')

    # Generate箭头
    ax.annotate('', xy=(legend_x + 0.8, legend_y + 2.3), xytext=(legend_x, legend_y + 2.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow_generate'], lw=2))
    ax.text(legend_x + 1.0, legend_y + 2.3, 'Generate', fontsize=8, va='center')
    ax.plot([legend_x + 1.8, legend_x + 1.9], [legend_y + 2.3, legend_y + 2.3],
            'k--', linewidth=1)
    ax.text(legend_x + 2.0, legend_y + 2.3, 'Train', fontsize=7, va='center')

    # 组件图例
    components = [
        ('Multi-Head Self-Attention', COLORS['multi_head_attn']),
        ('Pointwise Feedforward', COLORS['pointwise_ff']),
        ('Scale Shift', COLORS['scale_shift']),
        ('Layer Norm', COLORS['layer_norm']),
    ]

    for i, (name, color) in enumerate(components):
        cy = legend_y + 1.5 - i * legend_spacing
        rect = Rectangle((legend_x, cy - 0.15), 0.6, 0.3, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(legend_x + 0.7, cy, name, fontsize=7, va='center')

    # ==================== 保存图片 ====================

    plt.tight_layout()
    plt.savefig('/root/OccSora-main/docs/occsora_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('/root/OccSora-main/docs/occsora_architecture.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Architecture diagram saved to:")
    print("  - /root/OccSora-main/docs/occsora_architecture.png")
    print("  - /root/OccSora-main/docs/occsora_architecture.pdf")
    plt.close()

if __name__ == '__main__':
    main()
