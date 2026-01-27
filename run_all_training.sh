#!/bin/bash
#===============================================================================
# OccSora 创新 - 完整训练流程
#===============================================================================
#
# 这个脚本会自动执行以下步骤：
# 1. 环境检查
# 2. 基线模型训练 (Baseline)
# 3. STCA架构训练 (Architecture-Level STCA)
# 4. SADS采样训练 (Scene-Adaptive Diffusion)
# 5. 完整创新模型训练 (STCA + SADS + PCL)
# 6. 推理测试对比
#
# 用法:
#   chmod +x run_all_training.sh
#   ./run_all_training.sh
#
#===============================================================================

set -euo pipefail  # 遇到错误立即退出（含管道），并禁止未定义变量

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目路径
PROJECT_DIR="/root/OccSora-main"

# 输出路径（默认写到可用大盘；可通过环境变量覆盖）
RUN_DIR="${RUN_DIR:-/root/autodl-tmp/OccSora_runs}"

# 训练后的模型（checkpoints）默认保存到这里
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/model}"

# 日志与采样输出默认放到RUN_DIR下
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"
SAMPLE_DIR="${SAMPLE_DIR:-$RUN_DIR/out}"

# 训练参数（可通过环境变量覆盖）
# - BASELINE_EPOCHS: step1 baseline 训练轮数（默认100）
# - EPOCHS: step2-4 训练轮数（默认100；可设置 EPOCHS=5 做快速验证）
# - START_STEP: 从第几步开始执行（默认1；例如 START_STEP=2 跳过baseline训练）
START_STEP="${START_STEP:-1}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-100}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
SAVE_EVERY="${SAVE_EVERY:-10}"
NUM_STEPS="${NUM_STEPS:-50}"

# 数据路径
DATA_DIR="/root/autodl-tmp/OccSora_output"
CONDITION_PATH="/root/autodl-tmp/OccSora_output/vqvae/step32-2/gt_mode/i_iter_0.npy"

#===============================================================================
# 辅助函数
#===============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}===============================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}===============================================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}[STEP $1]${NC} $2"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}GPU信息:${NC}"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
        return 0
    else
        print_warning "未检测到GPU，将使用CPU训练（速度较慢）"
        return 1
    fi
}

create_dirs() {
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$SAMPLE_DIR"
    mkdir -p "$OUTPUT_DIR/baseline"
    mkdir -p "$OUTPUT_DIR/stca_only"
    mkdir -p "$OUTPUT_DIR/sads_only"
    mkdir -p "$OUTPUT_DIR/full_innovation"
}

#===============================================================================
# 主流程
#===============================================================================

print_header "OccSora 架构级创新 - 完整训练流程"

echo "开始时间: $(date)"
echo "项目目录: $PROJECT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "开始执行步骤: $START_STEP"
echo ""

#-------------------------------------------------------------------------------
# Step 0: 环境检查
#-------------------------------------------------------------------------------
print_step "0" "环境检查"

cd "$PROJECT_DIR"

# 检查Python
python3 --version || { print_error "Python3 未安装"; exit 1; }

# 检查PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || { print_error "PyTorch 未安装"; exit 1; }

# 检查GPU
check_gpu

# 创建目录
create_dirs

print_success "环境检查完成"

#-------------------------------------------------------------------------------
# Step 1: 基线模型训练 (Baseline - 不使用任何创新)
#-------------------------------------------------------------------------------
if [ "$START_STEP" -le 1 ]; then
    print_step "1" "基线模型训练 (Baseline - 无创新)"

    echo "训练配置:"
    echo "  - 模型: DiT-STCA-XL/2 (use_stca=False)"
    echo "  - STCA: 关闭"
    echo "  - SADS: 关闭"
    echo "  - PCL: 关闭"
    echo "  - Epochs: $BASELINE_EPOCHS"
    echo ""

    python3 train_architecture.py \
        --model DiT-STCA-XL/2 \
        --no-stca \
        --epochs "$BASELINE_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR/baseline" \
        --save-every "$SAVE_EVERY" \
        2>&1 | tee "$LOG_DIR/train_baseline.log"

    print_success "基线模型训练完成"
else
    print_step "1" "跳过基线训练（START_STEP=$START_STEP）"
fi

#-------------------------------------------------------------------------------
# Step 2: STCA架构训练 (仅STCA)
#-------------------------------------------------------------------------------
if [ "$START_STEP" -le 2 ]; then
print_step "2" "STCA架构训练 (Architecture-Level STCA)"

echo "训练配置:"
echo "  - 模型: DiT-STCA-XL/2 (use_stca=True)"
echo "  - STCA: 开启 (架构级)"
echo "  - SADS: 关闭"
echo "  - PCL: 关闭"
echo ""

python3 train_architecture.py \
    --model DiT-STCA-XL/2 \
    --use-stca \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR/stca_only" \
    --save-every "$SAVE_EVERY" \
    2>&1 | tee "$LOG_DIR/train_stca_only.log"

print_success "STCA架构训练完成"
else
    print_step "2" "跳过STCA训练（START_STEP=$START_STEP）"
fi

#-------------------------------------------------------------------------------
# Step 3: SADS训练 (STCA + SADS)
#-------------------------------------------------------------------------------
if [ "$START_STEP" -le 3 ]; then
print_step "3" "SADS训练 (STCA + SADS)"

echo "训练配置:"
echo "  - 模型: DiT-STCA-XL/2"
echo "  - STCA: 开启"
echo "  - SADS: 开启 (场景自适应)"
echo "  - PCL: 关闭"
echo ""

python3 train_architecture.py \
    --model DiT-STCA-XL/2 \
    --use-stca \
    --use-sads \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR/sads_only" \
    --save-every "$SAVE_EVERY" \
    2>&1 | tee "$LOG_DIR/train_sads.log"

print_success "SADS训练完成"
else
    print_step "3" "跳过SADS训练（START_STEP=$START_STEP）"
fi

#-------------------------------------------------------------------------------
# Step 4: 完整创新模型训练 (STCA + SADS + PCL)
#-------------------------------------------------------------------------------
if [ "$START_STEP" -le 4 ]; then
print_step "4" "完整创新模型训练 (STCA + SADS + PCL)"

echo "训练配置:"
echo "  - 模型: DiT-STCA-XL/2"
echo "  - STCA: 开启 (架构级时空因果注意力)"
echo "  - SADS: 开启 (场景自适应扩散调度)"
echo "  - PCL: 开启 (物理一致性损失)"
echo ""

python3 train_architecture.py \
    --model DiT-STCA-XL/2 \
    --use-stca \
    --use-sads \
    --use-pcl \
    --pcl-weight 0.1 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR/full_innovation" \
    --save-every "$SAVE_EVERY" \
    2>&1 | tee "$LOG_DIR/train_full.log"

print_success "完整创新模型训练完成"
else
    print_step "4" "跳过完整创新训练（START_STEP=$START_STEP）"
fi

#-------------------------------------------------------------------------------
# Step 5: 推理测试对比
#-------------------------------------------------------------------------------
if [ "$START_STEP" -le 5 ]; then
print_step "5" "推理测试对比"

echo ""
echo "5.1 基线模型推理..."
BASELINE_CKPT="${BASELINE_CKPT:-}"
if [ -z "$BASELINE_CKPT" ]; then
    # 1) Prefer checkpoints under OUTPUT_DIR (autodl-tmp/model)
    CANDIDATE_1="$OUTPUT_DIR/baseline/dit_stca_epoch${BASELINE_EPOCHS}.pt"
    # 2) Fall back to the original repo checkpoints directory if user already trained there
    CANDIDATE_2="$PROJECT_DIR/checkpoints/baseline/dit_stca_epoch${BASELINE_EPOCHS}.pt"

    if [ -f "$CANDIDATE_1" ]; then
        BASELINE_CKPT="$CANDIDATE_1"
    elif [ -f "$CANDIDATE_2" ]; then
        BASELINE_CKPT="$CANDIDATE_2"
    else
        # 3) Last resort: pick the newest baseline checkpoint from either location
        BASELINE_CKPT="$(ls -t "$OUTPUT_DIR"/baseline/*.pt "$PROJECT_DIR"/checkpoints/baseline/*.pt 2>/dev/null | head -n 1 || true)"
    fi
fi

if [ -n "$BASELINE_CKPT" ] && [ -f "$BASELINE_CKPT" ]; then
    python3 sample_architecture.py \
        --model DiT-STCA-XL/2 \
        --no-stca \
        --no-sads \
        --ckpt "$BASELINE_CKPT" \
        --num-steps "$NUM_STEPS" \
        --output-path "$SAMPLE_DIR/samples_baseline.npy" \
        2>&1 | tee "$LOG_DIR/sample_baseline.log"
else
    print_warning "未找到基线检查点（可用 BASELINE_CKPT=/path/to.pt 指定）；跳过 baseline 推理"
fi

echo ""
echo "5.2 STCA模型推理..."
STCA_CKPT="${STCA_CKPT:-$OUTPUT_DIR/stca_only/dit_stca_epoch${EPOCHS}.pt}"
if [ -f "$STCA_CKPT" ]; then
    python3 sample_architecture.py \
        --model DiT-STCA-XL/2 \
        --use-stca \
        --no-sads \
        --ckpt "$STCA_CKPT" \
        --num-steps "$NUM_STEPS" \
        --output-path "$SAMPLE_DIR/samples_stca.npy" \
        2>&1 | tee "$LOG_DIR/sample_stca.log"
else
    print_warning "未找到STCA检查点（$STCA_CKPT）；跳过 STCA 推理（可用 STCA_CKPT=/path/to.pt 指定）"
fi

echo ""
echo "5.3 SADS模型推理..."
SADS_CKPT="${SADS_CKPT:-$OUTPUT_DIR/sads_only/dit_stca_epoch${EPOCHS}.pt}"
if [ -f "$SADS_CKPT" ]; then
    python3 sample_architecture.py \
        --model DiT-STCA-XL/2 \
        --use-stca \
        --use-sads \
        --ckpt "$SADS_CKPT" \
        --num-steps "$NUM_STEPS" \
        --output-path "$SAMPLE_DIR/samples_sads.npy" \
        2>&1 | tee "$LOG_DIR/sample_sads.log"
else
    print_warning "未找到SADS检查点（$SADS_CKPT）；跳过 SADS 推理（可用 SADS_CKPT=/path/to.pt 指定）"
fi

echo ""
echo "5.4 完整创新模型推理..."
FULL_CKPT="${FULL_CKPT:-$OUTPUT_DIR/full_innovation/dit_stca_epoch${EPOCHS}.pt}"
if [ -f "$FULL_CKPT" ]; then
    python3 sample_architecture.py \
        --model DiT-STCA-XL/2 \
        --use-stca \
        --use-sads \
        --ckpt "$FULL_CKPT" \
        --num-steps "$NUM_STEPS" \
        --output-path "$SAMPLE_DIR/samples_full.npy" \
        2>&1 | tee "$LOG_DIR/sample_full.log"
else
    print_warning "未找到Full检查点（$FULL_CKPT）；跳过 Full 推理（可用 FULL_CKPT=/path/to.pt 指定）"
fi

echo ""
echo "5.5 自适应步数推理..."
if [ -f "$FULL_CKPT" ]; then
    python3 sample_architecture.py \
        --model DiT-STCA-XL/2 \
        --use-stca \
        --use-sads \
        --adaptive-steps \
        --min-steps 20 \
        --max-steps 100 \
        --ckpt "$FULL_CKPT" \
        --output-path "$SAMPLE_DIR/samples_adaptive.npy" \
        2>&1 | tee "$LOG_DIR/sample_adaptive.log"
else
    print_warning "未找到Full检查点（$FULL_CKPT）；跳过自适应步数推理"
fi

print_success "推理测试完成"
else
    print_step "5" "跳过推理（START_STEP=$START_STEP）"
fi

#-------------------------------------------------------------------------------
# Step 6: 评估所有指标 (FID, FVD, mIoU, PhysScore, TempScore等)
#-------------------------------------------------------------------------------
if [ "$START_STEP" -le 6 ]; then
print_step "6" "评估所有指标 (FID, FVD, mIoU, PhysScore, TempScore等)"

EVAL_DIR="${EVAL_DIR:-$RUN_DIR/eval_results}"
mkdir -p "$EVAL_DIR"

# 评估参数（可通过环境变量覆盖）
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"
EVAL_STEPS="${EVAL_STEPS:-50}"
EVAL_COND_INDICES="${EVAL_COND_INDICES:-0,1,2,3}"

echo "评估配置:"
echo "  - 样本数: $EVAL_SAMPLES"
echo "  - 采样步数: $EVAL_STEPS"
echo "  - 条件索引: $EVAL_COND_INDICES"
echo "  - 输出目录: $EVAL_DIR"
echo ""

# 检查VQVAE checkpoint和数据目录
VQVAE_CKPT="${VQVAE_CKPT:-/root/autodl-tmp/OccSoraModel/epoch_125.pth}"
VQVAE_CONFIG="${VQVAE_CONFIG:-$PROJECT_DIR/config/train_vqvae.py}"
GT_MODE_DIR="${GT_MODE_DIR:-/root/autodl-tmp/OccSora_output/vqvae/step32-2/gt_mode}"
TOKEN_DIR="${TOKEN_DIR:-/root/autodl-tmp/OccSora_output/vqvae/step32-2/token}"

# 训练后的模型 checkpoint 路径
EVAL_BASELINE_CKPT="${BASELINE_CKPT:-$OUTPUT_DIR/baseline/dit_stca_epoch${BASELINE_EPOCHS}.pt}"
EVAL_STCA_CKPT="${STCA_CKPT:-$OUTPUT_DIR/stca_only/dit_stca_epoch${EPOCHS}.pt}"
EVAL_SADS_CKPT="${SADS_CKPT:-$OUTPUT_DIR/sads_only/dit_stca_epoch${EPOCHS}.pt}"
EVAL_FULL_CKPT="${FULL_CKPT:-$OUTPUT_DIR/full_innovation/dit_stca_epoch${EPOCHS}.pt}"

echo "6.1 运行完整评估..."
echo "  使用模型:"
echo "    Baseline: $EVAL_BASELINE_CKPT"
echo "    STCA: $EVAL_STCA_CKPT"
echo "    SADS: $EVAL_SADS_CKPT"
echo "    Full: $EVAL_FULL_CKPT"
echo ""

# 检查evaluate_all_metrics.py是否存在
if [ -f "$PROJECT_DIR/evaluate_all_metrics.py" ]; then
    python3 "$PROJECT_DIR/evaluate_all_metrics.py" \
        --num-samples "$EVAL_SAMPLES" \
        --num-steps "$EVAL_STEPS" \
        --cond-indices "$EVAL_COND_INDICES" \
        --out-dir "$EVAL_DIR" \
        --vqvae-ckpt "$VQVAE_CKPT" \
        --vqvae-config "$VQVAE_CONFIG" \
        --gt-mode-dir "$GT_MODE_DIR" \
        --token-dir "$TOKEN_DIR" \
        --baseline-ckpt "$EVAL_BASELINE_CKPT" \
        --stca-ckpt "$EVAL_STCA_CKPT" \
        --sads-ckpt "$EVAL_SADS_CKPT" \
        --full-ckpt "$EVAL_FULL_CKPT" \
        2>&1 | tee "$LOG_DIR/evaluate_all.log"
else
    print_warning "未找到 evaluate_all_metrics.py，跳过完整评估"
fi

echo ""
echo "6.2 快速评估..."

# 快速评估（不需要VQVAE）
if [ -f "$PROJECT_DIR/eval_occsora_quick.py" ]; then
    python3 "$PROJECT_DIR/eval_occsora_quick.py" \
        2>&1 | tee "$LOG_DIR/eval_quick.log"
fi

echo ""
echo "评估结果保存在: $EVAL_DIR"
ls -la "$EVAL_DIR"/*.json 2>/dev/null || echo "  (未找到评估结果文件)"
ls -la "$EVAL_DIR"/*.md 2>/dev/null || echo "  (未找到Markdown报告)"

print_success "评估完成"
else
    print_step "6" "跳过评估（START_STEP=$START_STEP）"
fi

#-------------------------------------------------------------------------------
# 完成
#-------------------------------------------------------------------------------
print_header "训练流程完成"

echo "结束时间: $(date)"
echo ""
echo "输出文件:"
echo "  检查点目录: $OUTPUT_DIR"
echo "  日志目录: $LOG_DIR"
echo "  采样结果: $SAMPLE_DIR"
echo "  评估结果: ${EVAL_DIR:-$RUN_DIR/eval_results}"
echo ""
echo "检查点文件:"
ls -la "$OUTPUT_DIR"/*/dit_stca_epoch*.pt 2>/dev/null || echo "  (未找到检查点文件)"
echo ""
echo "采样结果:"
ls -la "$SAMPLE_DIR"/*.npy 2>/dev/null || echo "  (未找到采样结果)"
echo ""
echo "评估结果:"
ls -la "${EVAL_DIR:-$RUN_DIR/eval_results}"/*.json 2>/dev/null || echo "  (未找到评估结果)"
echo ""

print_success "所有训练、推理和评估任务已完成!"
