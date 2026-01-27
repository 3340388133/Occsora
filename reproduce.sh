#!/bin/bash
#####################################################################
# OccSora 完整复现流程脚本
# 使用方法: bash reproduce.sh [step]
# step 可选: env, data, train_vqvae, gen_token, train_dit, sample, visualize, all
#####################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_step() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}[STEP] $1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

print_warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

#####################################################################
# 第一步: 环境安装
#####################################################################
install_env() {
    print_step "安装环境依赖"

    # 检查 conda
    if ! command -v conda &> /dev/null; then
        print_error "请先安装 Anaconda 或 Miniconda"
        exit 1
    fi

    # 创建 conda 环境
    echo "创建 conda 环境 OccSora (Python 3.8)..."
    conda create -n OccSora python=3.8.0 -y

    # 激活环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate OccSora

    # 安装 PyTorch
    echo "安装 PyTorch 2.0.1 (CUDA 11.8)..."
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

    # 安装核心依赖
    echo "安装核心 Python 依赖..."
    pip install mmcv==2.0.1 mmengine==0.8.4
    pip install einops==0.7.0 timm==0.9.7
    pip install diffusers accelerate
    pip install nuscenes-devkit==1.1.10
    pip install scipy scikit-learn pandas tqdm pillow
    pip install pyquaternion shapely

    # 安装 mmdetection3d
    echo "安装 mmdetection3d..."
    pip install openmim
    mim install mmdet3d

    # 安装 Mayavi 可视化依赖
    echo "安装 Mayavi 可视化依赖..."
    pip install mayavi==4.8.1 vtk==9.2.6
    pip install pyqt5 pyvirtualdisplay

    # 安装 xvfb (Linux 服务器无显示器时需要)
    if command -v apt-get &> /dev/null; then
        echo "安装 xvfb..."
        sudo apt-get update && sudo apt-get install -y xvfb
    fi

    echo -e "${GREEN}环境安装完成!${NC}"
    echo "请运行: conda activate OccSora"
}

#####################################################################
# 第二步: 数据准备
#####################################################################
prepare_data() {
    print_step "数据准备"

    # 检查 data 目录
    mkdir -p data

    echo "请按以下步骤准备数据:"
    echo ""
    echo "1. 下载 nuScenes 数据集 (https://www.nuscenes.org/)"
    echo "   然后创建软链接:"
    echo "   ln -s /your/nuscenes/path data/nuscenes"
    echo ""
    echo "2. 下载 Occ3D 语义占用数据"
    echo "   放到 data/nuscenes/gts/ 目录"
    echo ""
    echo "3. 下载 pickle 文件 (从 TPVFormer 仓库)"
    echo "   - nuscenes_infos_train_temporal_v3_scene.pkl"
    echo "   - nuscenes_infos_val_temporal_v3_scene.pkl"
    echo "   放到 data/ 目录"
    echo ""

    # 检查数据是否存在
    check_data
}

check_data() {
    print_step "检查数据完整性"

    local all_ok=true

    # 检查 nuscenes
    if [ -L "data/nuscenes" ] || [ -d "data/nuscenes" ]; then
        echo -e "${GREEN}[OK] data/nuscenes${NC}"
    else
        echo -e "${RED}[MISSING] data/nuscenes${NC}"
        all_ok=false
    fi

    # 检查 gts
    if [ -d "data/nuscenes/gts" ]; then
        echo -e "${GREEN}[OK] data/nuscenes/gts${NC}"
    else
        echo -e "${RED}[MISSING] data/nuscenes/gts${NC}"
        all_ok=false
    fi

    # 检查 pickle 文件
    if [ -f "data/nuscenes_infos_train_temporal_v3_scene.pkl" ]; then
        echo -e "${GREEN}[OK] nuscenes_infos_train_temporal_v3_scene.pkl${NC}"
    else
        echo -e "${RED}[MISSING] nuscenes_infos_train_temporal_v3_scene.pkl${NC}"
        all_ok=false
    fi

    if [ -f "data/nuscenes_infos_val_temporal_v3_scene.pkl" ]; then
        echo -e "${GREEN}[OK] nuscenes_infos_val_temporal_v3_scene.pkl${NC}"
    else
        echo -e "${RED}[MISSING] nuscenes_infos_val_temporal_v3_scene.pkl${NC}"
        all_ok=false
    fi

    if [ "$all_ok" = true ]; then
        echo -e "${GREEN}数据检查通过!${NC}"
    else
        echo -e "${RED}数据不完整，请补充缺失的数据${NC}"
        return 1
    fi
}

#####################################################################
# 第三步: 训练 VQVAE
#####################################################################
train_vqvae() {
    print_step "训练 VQVAE"

    echo "开始训练 VQVAE..."
    echo "需要: A100 80G GPU"
    echo "预计时间: 约 24-48 小时"
    echo ""

    mkdir -p out/vqvae

    python train_1.py \
        --py-config config/train_vqvae.py \
        --work-dir out/vqvae

    echo -e "${GREEN}VQVAE 训练完成!${NC}"
    echo "权重保存在: out/vqvae/"
}

#####################################################################
# 第四步: 生成 Token 数据
#####################################################################
generate_tokens() {
    print_step "生成 Token 数据"

    echo "使用训练好的 VQVAE 生成 token..."

    python step02.py \
        --py-config config/train_vqvae.py \
        --work-dir out/vqvae

    echo -e "${GREEN}Token 生成完成!${NC}"
}

#####################################################################
# 第五步: 训练 DiT 扩散模型
#####################################################################
train_dit() {
    print_step "训练 DiT 扩散模型"

    echo "开始训练 DiT..."
    echo "需要: 8x A100 80G GPU"
    echo "预计时间: 约 3-7 天"
    echo ""

    # 获取 GPU 数量
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "检测到 ${NUM_GPUS} 个 GPU"

    torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} train_2.py \
        --model DiT-XL/2 \
        --data-path out/token_data

    echo -e "${GREEN}DiT 训练完成!${NC}"
}

#####################################################################
# 第六步: 采样生成
#####################################################################
run_sample() {
    print_step "采样生成 4D 占用"

    # 检查 checkpoint
    CKPT_PATH=${1:-"checkpoints/latest.pt"}

    if [ ! -f "$CKPT_PATH" ]; then
        print_warn "未找到 checkpoint: $CKPT_PATH"
        echo "请指定正确的 checkpoint 路径"
        echo "用法: bash reproduce.sh sample /path/to/checkpoint.pt"
        return 1
    fi

    echo "使用 checkpoint: $CKPT_PATH"

    mkdir -p out

    python sample.py \
        --model DiT-XL/2 \
        --image-size 256 \
        --ckpt "$CKPT_PATH" \
        --gt-mode-dir out/gt_mode_occstats \
        --cond-idx 0 \
        --num-sampling-steps 1000 \
        --cfg-scale 4.0

    echo -e "${GREEN}采样完成!${NC}"
    echo "结果保存在: out/samples_array.npy"
}

#####################################################################
# 第七步: Mayavi 可视化
#####################################################################
run_visualize() {
    print_step "Mayavi 3D 可视化"

    # 设置无显示器环境
    export QT_QPA_PLATFORM=offscreen
    export ETS_TOOLKIT=qt4

    SCENE_IDX=${1:-"0 1"}

    echo "可视化场景: $SCENE_IDX"

    python visualize_demo.py \
        --py-config config/train_vqvae.py \
        --work-dir out/vqvae \
        --scene-idx $SCENE_IDX

    echo -e "${GREEN}可视化完成!${NC}"
    echo "结果保存在: out/vqvae/vis*/"
}

#####################################################################
# 第八步: 评估 - 完整指标评估
#####################################################################
run_eval_all() {
    print_step "完整指标评估 (mIoU, FID, FVD, CD, 时序一致性, 物理一致性)"

    mkdir -p eval_results

    # 参数
    NUM_SAMPLES=${1:-4}
    NUM_STEPS=${2:-50}
    COND_INDICES=${3:-"0,1,2,3"}

    echo "评估参数:"
    echo "  样本数: $NUM_SAMPLES"
    echo "  采样步数: $NUM_STEPS"
    echo "  条件索引: $COND_INDICES"

    python evaluate_all_metrics.py \
        --num-samples $NUM_SAMPLES \
        --num-steps $NUM_STEPS \
        --cond-indices "$COND_INDICES" \
        --out-dir eval_results \
        --vqvae-ckpt out/vqvae/epoch_125.pth \
        --vqvae-config config/train_vqvae.py \
        --gt-mode-dir out/gt_mode_occstats \
        --token-dir out/token

    echo -e "${GREEN}完整评估完成!${NC}"
    echo "结果保存在: eval_results/"
    echo "  - metrics.json (汇总指标)"
    echo "  - metrics_table.md (Markdown表格)"
    echo "  - metrics_table.tex (LaTeX表格)"
}

#####################################################################
# 第九步: Occ3D 格式评估
#####################################################################
run_eval_occ3d() {
    print_step "Occ3D 格式 mIoU 评估"

    NUM_SAMPLES=${1:-1000}
    GT_DIR=${2:-"data/nuscenes/gts"}

    echo "评估 Occ3D 格式..."
    echo "  GT目录: $GT_DIR"
    echo "  样本数: $NUM_SAMPLES"

    python eval_occ3d.py \
        --gt-dir "$GT_DIR" \
        --num-samples $NUM_SAMPLES \
        --output occ3d_results.json

    echo -e "${GREEN}Occ3D 评估完成!${NC}"
    echo "结果保存在: occ3d_results.json"
}

#####################################################################
# 第十步: 多数据集评估
#####################################################################
run_eval_multi() {
    print_step "多数据集评估 (Occ3D + nuScenes)"

    mkdir -p multi_dataset_results

    VQVAE_CKPT=${1:-"out/vqvae/epoch_125.pth"}

    echo "多数据集评估..."
    echo "  VQVAE: $VQVAE_CKPT"

    python eval_multi_dataset.py \
        --datasets occ3d nuscenes \
        --models baseline full \
        --vqvae-ckpt "$VQVAE_CKPT" \
        --output multi_dataset_results

    echo -e "${GREEN}多数据集评估完成!${NC}"
    echo "结果保存在: multi_dataset_results/"
    echo "  - results.json"
    echo "  - report.md"
}

#####################################################################
# 快速评估 (仅基础指标)
#####################################################################
run_eval_quick() {
    print_step "快速评估 (基础指标)"

    mkdir -p eval_results

    echo "运行快速评估..."

    python eval_occsora_quick.py 2>/dev/null || python -c "
import os
import json
import numpy as np

print('快速评估: 检查生成结果...')

# 检查是否有生成的样本
sample_path = 'out/samples_array.npy'
if os.path.exists(sample_path):
    samples = np.load(sample_path)
    print(f'  样本形状: {samples.shape}')
    print(f'  样本范围: [{samples.min():.4f}, {samples.max():.4f}]')
    print(f'  样本均值: {samples.mean():.4f}')
    print(f'  样本标准差: {samples.std():.4f}')

    # 计算简单的多样性指标
    if len(samples.shape) >= 2 and samples.shape[0] > 1:
        flat = samples.reshape(samples.shape[0], -1)
        dists = []
        for i in range(min(10, samples.shape[0])):
            for j in range(i+1, min(10, samples.shape[0])):
                d = np.sqrt(np.mean((flat[i] - flat[j])**2))
                dists.append(d)
        if dists:
            print(f'  样本多样性 (RMSE): {np.mean(dists):.4f}')

    print('\\n快速评估完成!')
else:
    print('  未找到生成样本，请先运行 sample 命令')
"

    echo -e "${GREEN}快速评估完成!${NC}"
}

#####################################################################
# 快速测试可视化 (使用现有数据)
#####################################################################
quick_visualize() {
    print_step "快速测试 Mayavi 可视化"

    export QT_QPA_PLATFORM=offscreen
    export ETS_TOOLKIT=qt4

    python -c "
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from pyvirtualdisplay import Display
display = Display(visible=False, size=(1920, 1080))
display.start()

from mayavi import mlab
import numpy as np

mlab.options.offscreen = True

# 创建测试数据
x, y, z = np.mgrid[-5:5:50j, -5:5:50j, -5:5:50j]
values = np.sin(x*y*z) / (x*y*z + 0.1)

fig = mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))
mlab.contour3d(x, y, z, values)

os.makedirs('mayavi_vis', exist_ok=True)
mlab.savefig('mayavi_vis/test_mayavi.png')
mlab.close()
print('Mayavi 测试成功! 结果保存在 mayavi_vis/test_mayavi.png')
"
}

#####################################################################
# 主函数
#####################################################################
show_help() {
    echo "OccSora 完整复现流程脚本"
    echo ""
    echo "用法: bash reproduce.sh [command] [args...]"
    echo ""
    echo "========== 环境与数据 =========="
    echo "  env            - 安装环境依赖"
    echo "  data           - 准备数据 (显示说明)"
    echo "  check          - 检查数据完整性"
    echo ""
    echo "========== 训练流程 =========="
    echo "  train_vqvae    - 训练 VQVAE (需要 A100 80G)"
    echo "  gen_token      - 生成 Token 数据"
    echo "  train_dit      - 训练 DiT 扩散模型 (需要 8x A100)"
    echo ""
    echo "========== 推理与可视化 =========="
    echo "  sample         - 采样生成 4D 占用"
    echo "                   参数: [checkpoint路径]"
    echo "  visualize      - Mayavi 3D 可视化"
    echo "                   参数: [scene_idx...]"
    echo "  quick_vis      - 快速测试 Mayavi"
    echo ""
    echo "========== 评估指标 =========="
    echo "  eval_all       - 完整评估 (mIoU, FID, FVD, CD, 时序/物理一致性)"
    echo "                   参数: [num_samples] [num_steps] [cond_indices]"
    echo "  eval_occ3d     - Occ3D 格式 mIoU 评估"
    echo "                   参数: [num_samples] [gt_dir]"
    echo "  eval_multi     - 多数据集评估 (Occ3D + nuScenes)"
    echo "                   参数: [vqvae_ckpt]"
    echo "  eval_quick     - 快速评估 (基础指标)"
    echo ""
    echo "========== 完整流程 =========="
    echo "  all            - 完整流程 (训练+采样+可视化)"
    echo "  all_with_eval  - 完整流程 + 评估"
    echo ""
    echo "========== 示例 =========="
    echo "  bash reproduce.sh env"
    echo "  bash reproduce.sh train_vqvae"
    echo "  bash reproduce.sh sample checkpoints/model.pt"
    echo "  bash reproduce.sh visualize 0 1 2"
    echo "  bash reproduce.sh eval_all 4 50 0,1,2,3"
    echo "  bash reproduce.sh eval_occ3d 1000 data/nuscenes/gts"
}

main() {
    cd "$(dirname "$0")"

    case "$1" in
        env)
            install_env
            ;;
        data)
            prepare_data
            ;;
        check)
            check_data
            ;;
        train_vqvae)
            train_vqvae
            ;;
        gen_token)
            generate_tokens
            ;;
        train_dit)
            train_dit
            ;;
        sample)
            run_sample "$2"
            ;;
        visualize)
            shift
            run_visualize "$*"
            ;;
        quick_vis)
            quick_visualize
            ;;
        eval_all)
            run_eval_all "$2" "$3" "$4"
            ;;
        eval_occ3d)
            run_eval_occ3d "$2" "$3"
            ;;
        eval_multi)
            run_eval_multi "$2"
            ;;
        eval_quick)
            run_eval_quick
            ;;
        all)
            check_data
            train_vqvae
            generate_tokens
            train_dit
            run_sample
            run_visualize
            ;;
        all_with_eval)
            check_data
            train_vqvae
            generate_tokens
            train_dit
            run_sample
            run_eval_all
            run_visualize
            ;;
        help|--help|-h|"")
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
