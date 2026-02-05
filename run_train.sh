#!/bin/bash
# =============================================================================
# Video → ECG 批量训练脚本
# =============================================================================
#
# 用法:
#   ./run_experiments.sh              # 跑所有 schemes (C, D, E, F, G)
#   ./run_experiments.sh e f          # 只跑 Scheme E 和 F
#   ./run_experiments.sh e            # 只跑 Scheme E
#
# 环境变量:
#   CUDA_VISIBLE_DEVICES=0 ./run_experiments.sh   # 指定 GPU
#   SPLIT=user ./run_experiments.sh               # 用 user 划分 (默认 random)
#   QUALITY=good ./run_experiments.sh             # 只用 good 数据 (默认 good,moderate)
#   PATIENCE=30 ./run_experiments.sh              # 设置 patience (默认 20)
#
# 组合使用:
#   CUDA_VISIBLE_DEVICES=1 SPLIT=random QUALITY=good PATIENCE=15 ./run_experiments.sh e f
#
# =============================================================================

set -e

# 默认参数 (可通过环境变量覆盖)
SPLIT="${SPLIT:-random}"
QUALITY="${QUALITY:-good,moderate}"
PATIENCE="${PATIENCE:-20}"
EPOCHS="${EPOCHS:-200}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志文件
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/experiment_${TIMESTAMP}.log"

# 打印带颜色的消息
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$MAIN_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$MAIN_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$MAIN_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$MAIN_LOG"
}

# 打印配置
print_config() {
    echo ""
    echo "=========================================="
    echo "  Video → ECG 批量训练"
    echo "=========================================="
    echo "  Split:     $SPLIT"
    echo "  Quality:   $QUALITY"
    echo "  Patience:  $PATIENCE"
    echo "  Epochs:    $EPOCHS"
    echo "  GPU:       ${CUDA_VISIBLE_DEVICES:-all}"
    echo "  Log:       $MAIN_LOG"
    echo "=========================================="
    echo ""
}

# 训练单个 scheme
train_scheme() {
    local scheme=$1
    local config="configs/scheme_${scheme}.yaml"

    if [[ ! -f "$config" ]]; then
        log_error "Config not found: $config"
        return 1
    fi

    local scheme_log="$LOG_DIR/scheme_${scheme}_${TIMESTAMP}.log"

    log_info "Starting Scheme $scheme..."
    log_info "  Config: $config"
    log_info "  Log: $scheme_log"

    local start_time=$(date +%s)

    # 运行训练
    if python models/train.py \
        --config "$config" \
        --split "$SPLIT" \
        --quality-filter "$QUALITY" \
        --patience "$PATIENCE" \
        --epochs "$EPOCHS" \
        2>&1 | tee "$scheme_log"; then

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))

        log_success "Scheme $scheme completed in ${minutes}m ${seconds}s"
        return 0
    else
        log_error "Scheme $scheme failed! Check $scheme_log for details"
        return 1
    fi
}

# 主函数
main() {
    print_config

    # 确定要跑的 schemes
    local schemes=()
    if [[ $# -eq 0 ]]; then
        # 默认跑所有 (按推荐顺序)
        schemes=(e f c d g)
        log_info "Running ALL schemes: ${schemes[*]}"
    else
        # 从命令行参数获取
        for arg in "$@"; do
            local scheme=$(echo "$arg" | tr '[:upper:]' '[:lower:]')
            if [[ "$scheme" =~ ^[cdefg]$ ]]; then
                schemes+=("$scheme")
            else
                log_warning "Unknown scheme: $arg (valid: c, d, e, f, g)"
            fi
        done
        log_info "Running selected schemes: ${schemes[*]}"
    fi

    if [[ ${#schemes[@]} -eq 0 ]]; then
        log_error "No valid schemes to run!"
        exit 1
    fi

    # 记录开始时间
    local total_start=$(date +%s)
    local success_count=0
    local fail_count=0
    local failed_schemes=()

    # 逐个训练
    for scheme in "${schemes[@]}"; do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_info "Training Scheme ${scheme^^} ($(($success_count + $fail_count + 1))/${#schemes[@]})"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        if train_scheme "$scheme"; then
            ((success_count++))
        else
            ((fail_count++))
            failed_schemes+=("$scheme")
        fi
    done

    # 打印总结
    local total_end=$(date +%s)
    local total_duration=$((total_end - total_start))
    local total_minutes=$((total_duration / 60))
    local total_seconds=$((total_duration % 60))

    echo ""
    echo "=========================================="
    echo "  实验完成!"
    echo "=========================================="
    echo "  总耗时:    ${total_minutes}m ${total_seconds}s"
    echo "  成功:      $success_count"
    echo "  失败:      $fail_count"
    if [[ $fail_count -gt 0 ]]; then
        echo "  失败列表:  ${failed_schemes[*]}"
    fi
    echo "  日志目录:  $LOG_DIR/"
    echo "=========================================="
    echo ""

    # 列出生成的 checkpoints
    echo "生成的 checkpoints:"
    ls -la checkpoints/scheme_*/*/best_model.pt 2>/dev/null || echo "  (无)"

    if [[ $fail_count -gt 0 ]]; then
        exit 1
    fi
}

# 运行
main "$@"
