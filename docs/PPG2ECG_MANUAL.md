# PPG2ECG 模型运行手册

> 最后更新：2025-02-06
>
> 目的：在 BIDMC 公开数据集上验证 PPG → ECG 重建是否可行

---

## 目录

1. [概述](#1-概述)
2. [环境准备](#2-环境准备)
3. [数据准备](#3-数据准备)
4. [模型 1: PPG2ECG](#4-模型-1-ppg2ecg)
5. [模型 2: CardioGAN](#5-模型-2-cardiogan)
6. [结果判断](#6-结果判断)
7. [常见问题](#7-常见问题)
8. [实现说明](#8-实现说明)

---

## 1. 概述

### 1.1 两个模型对比

| | PPG2ECG | CardioGAN |
|---|---------|-----------|
| **论文** | IEEE Sensors 2020 | AAAI 2021 |
| **架构** | Encoder-Decoder + STN + Attention | GAN (Attention U-Net + Dual Discriminators) |
| **采样率** | 125 Hz | 128 Hz (内部重采样) |
| **窗口** | 256 samples (~2秒) | 512 samples (~4秒) |
| **损失函数** | QRS-enhanced L1 | Adversarial + L1 |
| **参数量** | ~2M | ~15M |
| **训练难度** | 简单 | 中等 (GAN需要调参) |

### 1.2 执行顺序

```
1. 下载 BIDMC 数据集 (手动)
2. 预处理数据
3. 先跑 PPG2ECG (更简单)
4. 如果 PPG2ECG 失败，跑 CardioGAN
```

---

## 2. 环境准备

### 2.1 激活环境

```bash
# 服务器
conda activate torch

# 或本地
conda activate anoshift
```

### 2.2 依赖确认

```bash
# 必需
pip install torch numpy scipy pyyaml matplotlib

# 可选 (用于 WFDB 格式数据)
pip install wfdb pandas
```

---

## 3. 数据准备

### 3.1 下载 BIDMC 数据集

1. 访问 https://physionet.org/content/bidmc/1.0.0/
2. 接受数据使用协议
3. 下载 CSV 文件 (或全部文件)
4. 解压到 `external_data/bidmc/`

**预期目录结构**：
```
external_data/bidmc/
├── bidmc_01_Signals.csv
├── bidmc_02_Signals.csv
├── ...
└── bidmc_53_Signals.csv
```

### 3.2 运行预处理

```bash
cd /path/to/ECG_recon_git

# 预处理 (生成 train.pt 和 test.pt)
python scripts/prepare_bidmc.py \
    --data_dir external_data/bidmc \
    --output_dir external_data/bidmc_processed
```

**预期输出**：
```
external_data/bidmc_processed/
├── train.pt        # 训练集 (~42 subjects)
├── test.pt         # 测试集 (~11 subjects)
└── metadata.json   # 元数据
```

### 3.3 验证数据

```bash
python -c "
import torch
train = torch.load('external_data/bidmc_processed/train.pt', weights_only=False)
print(f'Train PPG shape: {train[\"ppg\"].shape}')
print(f'Train ECG shape: {train[\"ecg\"].shape}')
"
```

预期输出类似：
```
Train PPG shape: torch.Size([N, 1, 512])
Train ECG shape: torch.Size([N, 1, 512])
```

---

## 4. 模型 1: PPG2ECG

### 4.1 模型信息

- **论文**: "Reconstructing QRS Complex from PPG by Transformed Attentional Neural Networks"
- **来源**: https://github.com/james77777778/ppg2ecg-pytorch
- **文件**: `models/ppg2ecg.py`, `models/train_ppg2ecg.py`
- **配置**: `configs/ppg2ecg_bidmc.yaml`

### 4.2 训练命令

```bash
# 基本训练
python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml

# 调整参数
python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.0002
```

### 4.3 训练参数 (默认)

| 参数 | 值 | 说明 |
|------|----|----- |
| epochs | 300 | 最大训练轮数 |
| batch_size | 256 | 批大小 |
| learning_rate | 0.0001 | 学习率 |
| patience | 50 | 早停耐心值 |
| input_size | 256 | 输入窗口长度 |

### 4.4 评估命令

```bash
# 仅评估 (不训练)
python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml --eval_only

# 指定 checkpoint
python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml \
    --eval_only \
    --checkpoint checkpoints/ppg2ecg_bidmc/best_model.pt
```

### 4.5 输出文件

```
checkpoints/ppg2ecg_bidmc/
├── best_model.pt           # 最佳模型权重
├── config.yaml             # 训练配置
├── training_history.json   # 训练历史
└── training_curves.png     # 训练曲线图
```

### 4.6 预期结果

| 指标 | 目标 | 论文值 |
|------|------|--------|
| Pearson r | > 0.7 | 0.844 |
| RMSE | < 0.3 | - |

---

## 5. 模型 2: CardioGAN

### 5.1 模型信息

- **论文**: "CardioGAN: Attentive Generative Adversarial Network with Dual Discriminators for Synthesis of ECG from PPG"
- **来源**: https://github.com/pritamqu/ppg2ecg-cardiogan
- **文件**: `models/cardiogan.py`, `models/train_cardiogan.py`
- **配置**: `configs/cardiogan_bidmc.yaml`

### 5.2 训练命令

```bash
# 基本训练
python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml

# 调整参数
python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001
```

### 5.3 训练参数 (默认)

| 参数 | 值 | 说明 |
|------|----|----- |
| epochs | 50 | 最大训练轮数 |
| batch_size | 16 | 批大小 (GAN需要较小batch) |
| learning_rate | 0.0002 | 学习率 |
| beta1 | 0.5 | Adam momentum (GAN标准值) |
| patience | 20 | 早停耐心值 |
| input_size | 512 | 输入窗口长度 |
| alpha | 3.0 | 对抗损失权重 |
| lambda_recon | 30.0 | 重建损失权重 |

### 5.4 评估命令

```bash
# 仅评估
python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml --eval_only
```

### 5.5 输出文件

```
checkpoints/cardiogan_bidmc/
├── best_generator.pt       # 最佳生成器权重 (用于推理)
├── best_model.pt           # 完整模型 (用于恢复训练)
├── config.yaml             # 训练配置
├── training_history.json   # 训练历史
└── training_curves.png     # 训练曲线图
```

### 5.6 预期结果

| 指标 | 目标 | 说明 |
|------|------|------|
| Pearson r | > 0.7 | 论文在多数据集上达到 |
| HR误差 | < 3 BPM | 论文值 2.89 BPM |

---

## 6. 结果判断

### 6.1 判断流程

```
┌─────────────────────────────────────────┐
│  BIDMC 上运行 PPG2ECG                    │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────▼─────────┐
        │ Pearson r > 0.7?  │
        └────┬─────────┬────┘
             │ Yes     │ No
             ▼         ▼
    ┌────────────┐  ┌────────────────┐
    │ 模型 OK    │  │ 尝试 CardioGAN │
    │ 下一步:    │  └───────┬────────┘
    │ 我们数据   │          │
    └────────────┘  ┌───────▼────────┐
                    │ Pearson r > 0.7?│
                    └────┬──────┬────┘
                         │ Yes  │ No
                         ▼      ▼
                ┌──────────┐ ┌──────────────┐
                │ 模型 OK  │ │ 检查复现细节 │
                └──────────┘ │ 或换其他方法 │
                             └──────────────┘
```

### 6.2 成功标准

- **成功**: Pearson r > 0.7
- **中等**: Pearson r 0.5 ~ 0.7 (可调参)
- **失败**: Pearson r < 0.5 (需检查实现)

### 6.3 后续步骤

**如果 BIDMC 成功**:
1. 在我们的视频数据上验证
2. 创建 `scripts/video_to_ppg.py` 提取视频 PPG
3. 运行评估，记录结果

**如果 BIDMC 失败**:
1. 检查数据预处理
2. 检查模型实现
3. 尝试调整超参数
4. 考虑其他方法

---

## 7. 常见问题

### Q1: CUDA out of memory

**解决方案**:
```bash
# 减小 batch_size
python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml --batch_size 64

# 或 CardioGAN
python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml --batch_size 8
```

### Q2: 找不到数据文件

**检查**:
```bash
ls external_data/bidmc_processed/
# 应该看到 train.pt, test.pt, metadata.json
```

如果没有，运行预处理:
```bash
python scripts/prepare_bidmc.py
```

### Q3: CardioGAN 训练不稳定

GAN 训练可能不稳定，尝试:
- 减小学习率: `--lr 0.0001`
- 增大 batch_size: `--batch_size 32`
- 减小 alpha: 修改 config 中 `loss.alpha: 1.0`

### Q4: 如何恢复训练

```bash
# CardioGAN 支持从 checkpoint 恢复
# 需要手动加载 best_model.pt 中的 optimizer state
```

---

## 8. 实现说明

### 8.1 与原论文的差异

**PPG2ECG**:
- ✅ 架构完全一致 (Encoder-Decoder + STN + Attention)
- ✅ QRS-enhanced loss 一致
- ✅ 超参数一致 (lr=0.0001, batch=256)
- ⚠️ 使用 PyTorch 而非原 repo 的 TensorFlow
- ⚠️ 数据预处理可能有细微差异

**CardioGAN**:
- ✅ Generator 架构一致 (Attention U-Net)
- ✅ Dual Discriminators 一致 (Time + Frequency)
- ✅ 损失函数权重一致 (α=3, λ=30)
- ⚠️ 使用 PyTorch 而非原 repo 的 TensorFlow 2.2
- ⚠️ 省略了 CycleGAN 的逆向生成器 (G_ECG2PPG)
- ⚠️ 简化了部分预处理 (直接重采样而非原始滤波器)

### 8.2 文件清单

```
models/
├── ppg2ecg.py              # PPG2ECG 模型定义
├── train_ppg2ecg.py        # PPG2ECG 训练脚本
├── cardiogan.py            # CardioGAN 模型定义
└── train_cardiogan.py      # CardioGAN 训练脚本

configs/
├── ppg2ecg_bidmc.yaml      # PPG2ECG BIDMC 配置
└── cardiogan_bidmc.yaml    # CardioGAN BIDMC 配置

scripts/
└── prepare_bidmc.py        # BIDMC 数据预处理

external_data/
├── bidmc/                  # 原始 BIDMC 数据 (手动下载)
└── bidmc_processed/        # 预处理后数据
    ├── train.pt
    ├── test.pt
    └── metadata.json
```

### 8.3 快速命令汇总

```bash
# 1. 预处理数据
python scripts/prepare_bidmc.py

# 2. 训练 PPG2ECG (推荐先跑)
CUDA_VISIBLE_DEVICES=0 python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml

# 3. 评估 PPG2ECG
python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml --eval_only

# 4. 训练 CardioGAN (备选)
CUDA_VISIBLE_DEVICES=0 python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml

# 5. 评估 CardioGAN
python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml --eval_only
```

---

## 附录: 相关链接

- PPG2ECG 论文: https://ieeexplore.ieee.org/document/9109576
- PPG2ECG 代码: https://github.com/james77777778/ppg2ecg-pytorch
- CardioGAN 论文: https://arxiv.org/abs/2010.00104
- CardioGAN 代码: https://github.com/pritamqu/ppg2ecg-cardiogan
- BIDMC 数据集: https://physionet.org/content/bidmc/1.0.0/
