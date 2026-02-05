#!/bin/bash
# 一键创建 ECG 环境 (支持 Blackwell GPU / CUDA 12.6)
#
# 用法:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# 或者直接复制命令执行

set -e

ENV_NAME="ecg"

echo "========================================="
echo "  ECG 环境安装脚本 (CUDA 12.6 / Blackwell)"
echo "========================================="

# 检查是否已存在环境
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  环境 '${ENV_NAME}' 已存在"
    read -p "是否删除重建？ (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        conda env remove -n ${ENV_NAME} -y
    else
        echo "退出安装"
        exit 0
    fi
fi

echo ""
echo "📦 Step 1: 创建 conda 环境 (Python 3.10 + 基础包)..."
conda create -n ${ENV_NAME} python=3.10 numpy pandas scipy matplotlib pyyaml -y

echo ""
echo "🔧 Step 2: 激活环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo ""
echo "🔥 Step 3: 安装 PyTorch (Blackwell/SM_100 需用 Nightly + CUDA 12.8)..."
echo "   若为 Blackwell GPU (RTX 5090/5080 等)，请用 Nightly；否则可改用 cu126 stable。"
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo ""
echo "📹 Step 4: 安装 OpenCV..."
pip install opencv-python-headless

echo ""
echo "========================================="
echo "✅ 安装完成！"
echo ""
echo "使用方法:"
echo "  conda activate ${ENV_NAME}"
echo "  python models/train.py --config configs/scheme_f.yaml"
echo "========================================="
