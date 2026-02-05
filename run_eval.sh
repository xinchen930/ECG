#!/bin/bash
# =============================================================================
# Video → ECG 批量评估脚本
# =============================================================================
#
# 用法:
#   ./run_eval.sh              # 评估所有已训练的模型
#   ./run_eval.sh e f          # 只评估 Scheme E 和 F
#   ./run_eval.sh e            # 只评估 Scheme E
#
# 说明:
#   自动查找 checkpoints/scheme_*/*/best_model.pt 并评估
#
# =============================================================================

set -e

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo ""
echo "=========================================="
echo "  Video → ECG 模型评估"
echo "=========================================="
echo ""

# 确定要评估的 schemes
schemes=()
if [[ $# -eq 0 ]]; then
    # 自动查找所有已训练的 scheme
    for dir in checkpoints/scheme_*/; do
        if [[ -d "$dir" ]]; then
            scheme=$(basename "$dir" | sed 's/scheme_//')
            schemes+=("$scheme")
        fi
    done
    if [[ ${#schemes[@]} -eq 0 ]]; then
        log_error "No checkpoints found in checkpoints/"
        exit 1
    fi
    log_info "Found schemes: ${schemes[*]}"
else
    for arg in "$@"; do
        schemes+=("$(echo "$arg" | tr '[:upper:]' '[:lower:]')")
    done
    log_info "Evaluating schemes: ${schemes[*]}"
fi

# 评估每个 scheme 的所有 checkpoint
total=0
success=0

for scheme in "${schemes[@]}"; do
    scheme_dir="checkpoints/scheme_${scheme}"

    if [[ ! -d "$scheme_dir" ]]; then
        log_warning "Scheme $scheme: no checkpoints found"
        continue
    fi

    # 遍历该 scheme 下的所有实验
    for exp_dir in "$scheme_dir"/*/; do
        if [[ ! -d "$exp_dir" ]]; then
            continue
        fi

        exp_name=$(basename "$exp_dir")
        checkpoint="$exp_dir/best_model.pt"
        config="$exp_dir/config.yaml"

        if [[ ! -f "$checkpoint" ]]; then
            log_warning "Scheme ${scheme^^} ($exp_name): no best_model.pt"
            continue
        fi

        if [[ ! -f "$config" ]]; then
            log_warning "Scheme ${scheme^^} ($exp_name): no config.yaml"
            continue
        fi

        ((total++))

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_info "Evaluating: Scheme ${scheme^^} / $exp_name"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        if python models/run_eval.py --config "$config" --checkpoint "$checkpoint"; then
            log_success "Scheme ${scheme^^} ($exp_name) done"
            ((success++))
        else
            log_error "Scheme ${scheme^^} ($exp_name) failed"
        fi
    done
done

echo ""
echo "=========================================="
echo "  评估完成: $success / $total"
echo "=========================================="
