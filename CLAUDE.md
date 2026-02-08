# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**目标：从手机视频(PPG) + IMU数据重建心电图(ECG)**

深度学习项目，训练模型从手机传感器数据恢复ECG波形：
- **输入**：手机摄像头视频（提取PPG信号）+ IMU数据（加速度计、陀螺仪）
- **输出/标签**：同步采集的ECG波形（ground truth）

## 数据处理流程

```
1. ECG预处理
   ECG_dataset/*.bs → 解析 → CSV/Parquet（带时间戳）

2. 时间对齐
   phone_dataset时间戳 → 提取对应时间段的ECG数据 → 配对样本

3. Phone数据预处理
   video → PPG信号提取
   IMU JSON → 结构化数组
   → 深度学习输入格式
```

## Running the Project

**环境**：使用 conda 环境 **torch**（服务器）或 anoshift（本地）。代码中未写死环境名，无需改任何文件。

```bash
conda activate torch
# 安装依赖（若未装）
pip install numpy pandas scipy pyyaml opencv-python-headless torch torchvision
```

**已安装依赖**：numpy, pandas, scipy, matplotlib, plotly, fastparquet, pyarrow, opencv-python-headless, torch, torchvision, pyyaml

### 如何开始训练、测试

在**项目根目录**（即 `ECG/`，保证存在 `training_data/samples` 和 `configs/`）下执行：

**训练**（训练结束后会自动在 test 集上评估并打印 RMSE/MAE/Pearson r）：
```bash
conda activate torch
cd /path/to/ECG

# 基本用法
python models/train.py --config configs/scheme_f.yaml

# 指定服务器类型，自动应用最优参数（推荐）
python models/train.py --config configs/scheme_f.yaml --server 3090
python models/train.py --config configs/scheme_f.yaml --server a6000

# 指定 GPU 卡
CUDA_VISIBLE_DEVICES=1 python models/train.py --config configs/scheme_f.yaml --server 3090

# 数据划分方式（推荐先用 random 验证可行性）
python models/train.py --config configs/scheme_e.yaml --split random                  # 随机划分（简单）
python models/train.py --config configs/scheme_e.yaml --split user                    # 用户划分（困难，最终评估）

# 数据质量过滤
python models/train.py --config configs/scheme_e.yaml --quality-filter good           # 只用高质量 (80样本)
python models/train.py --config configs/scheme_e.yaml --quality-filter good,moderate  # 排除 poor (88样本)

# Early stopping 控制（默认 patience=20-30）
python models/train.py --config configs/scheme_e.yaml --patience 15                   # 调整 patience

# Early stopping 数据源（默认用 test set，调试模式）
python models/train.py --config configs/scheme_e.yaml                  # 默认：test set 做早停（调试）
python models/train.py --config configs/scheme_e.yaml --use-val        # 严格模式：validation set 做早停
```

**仅做测试**（用已有 checkpoint 在 test 集上评估，不训练）：
```bash
python models/run_eval.py --config configs/scheme_c.yaml --checkpoint checkpoints/scheme_c/best_model.pt
```

**检查数据与 shape**：
```bash
python models/dataset.py configs/scheme_c.yaml
```

## Current Progress (Task 1: Video → ECG)

**状态：Phase 1-3 代码已完成，待训练验证。**

已实现八套方案，通过 config 切换。**所有方案均支持可选 IMU 融合**（`use_imu: true`）：

**1D 信号方案（轻量，适合快速验证）：**

| | Scheme D | Scheme E | Scheme E-RGB |
|---|---|---|---|
| 类型 | 1D TCN | 1D UNet | 1D UNet (3ch) |
| 输入 | RGB均值 | **红色通道** | RGB三通道 |
| 参数量 | 276K | ~500K | ~500K |
| 显存 | ~2 GB | ~3 GB | ~3 GB |
| 配置文件 | `scheme_d.yaml` | `scheme_e.yaml` | `scheme_e_rgb.yaml` |

**2D 视频方案（中等复杂度）：**

| | Scheme C | Scheme F | Scheme I-Direct | Scheme I-TwoStage |
|---|---|---|---|---|
| 类型 | MTTS-CAN | EfficientPhys | STMap直接 | STMap两阶段 |
| 架构 | 双分支+TSM | 时空注意力 | 2D CNN→ECG | 2D CNN→PPG→ECG |
| 输入 | 差分帧 36×36 | 视频帧 64×64 | STMap 8×8 | STMap 8×8 |
| 参数量 | 2.8M | ~1.5M | ~200-400K | ~400-800K |
| 配置文件 | `scheme_c.yaml` | `scheme_f.yaml` | `scheme_i_direct.yaml` | `scheme_i_twostage.yaml` |

**3D 视频方案（最强但最耗显存）：**

| | Scheme G | Scheme H |
|---|---|---|
| 类型 | PhysNet | **PhysFormer** |
| 架构 | 3D CNN | TD-Transformer + CDC |
| 输入 | 视频帧 64×64 | 视频帧 128×128 |
| 参数量 | ~3-5M | ~8-12M |
| 显存 | ~20-25 GB | ~20-30 GB |
| 配置文件 | `scheme_g.yaml` | `scheme_h.yaml` |

> 💡 使用 `--server 3090` 或 `--server a6000` 参数自动应用最优训练参数（batch_size、AMP、梯度累积等）

> 💡 在任意 config 中设置 `data.use_imu: true` 即可启用 IMU 融合
>
> 💡 Scheme E 使用**红色通道**（接触式 PPG 最佳通道，数据验证 8/10 HR检测 vs 绿色 3/10）
>
> 💡 Scheme F/G/H 是 end-to-end 方案，直接处理视频帧
>
> 💡 Scheme H (PhysFormer) 使用 Center-Difference Convolution 检测帧间微小亮度变化，理论上最适合 PPG 提取
>
> 💡 Scheme I (STMap) 保留空间信息的同时比全帧方案轻量得多
>
> 💡 设置 `data.quality_filter: "good"` 可过滤低质量样本（详见 `docs/data_quality_report_v2.md`）
>
> ⚠️ Scheme A/B (ResNet) 已移除：全局平均池化会丢弃 PPG 所需的微小亮度变化信息
>
> ⚠️ Scheme G/H 显存较大，3090 需小 batch + 梯度累积
>
> ⚠️ 数据质量：98个样本中有10个 poor 样本（HR误差>20BPM），建议过滤

### 已完成文件

```
models/
├── __init__.py
├── dataset.py            # PyTorch Dataset（gap-aware窗口、多输入模式：red/green/rgb/stmap）
├── video_ecg_model.py    # 模型定义（C/D/E/F/G + CompositeLoss + CompositeLossV2）
├── physformer_ecg.py     # PhysFormer-ECG（TD-Transformer + CDC, Scheme H）
├── stmap_builder.py      # STMap构建器（grid/multi-scale/frequency模式）
├── stmap_ecg.py          # STMap→ECG模型（direct + two-stage, Scheme I）
├── train.py              # 训练脚本（自动检测CUDA/MPS、early stopping、PPG辅助loss）
├── run_eval.py           # 仅测试（加载 checkpoint 在 test 集评估）
└── evaluate.py           # 评估（RMSE, MAE, Pearson r）

configs/
├── scheme_c.yaml         # MTTS-CAN (差分帧 + 注意力, 2.8M)
├── scheme_d.yaml         # 1D TCN (RGB均值, 276K)
├── scheme_e.yaml         # 1D UNet (红色通道, ~500K)
├── scheme_e_rgb.yaml     # 1D UNet (RGB三通道, ~500K)
├── scheme_f.yaml         # EfficientPhys (时空注意力, ~1.5M)
├── scheme_g.yaml         # PhysNet (3D CNN, ~3-5M)
├── scheme_h.yaml         # PhysFormer (TD-Transformer + CDC, ~8-12M)
├── scheme_i_direct.yaml  # STMap直接→ECG (~200-400K)
├── scheme_i_twostage.yaml # STMap→PPG→ECG (~400-800K, 多任务)
└── server_presets.yaml   # 服务器预设参数 (3090/A6000 自动配置)

scripts/
├── data_quality_check_v2.py       # 数据质量检查
└── data_quality_deep_analysis.py  # 深度PPG/ECG交叉分析

docs/
├── research_report.md        # 方法综述（19种Video→PPG + 6种PPG→ECG）
├── data_quality_report_v2.md # 数据质量报告
└── project_structure.md      # 项目结构文档
```

### 数据管线验证结果

- 98 pairs → 1042 windows（10s窗口，5s步长）
- 用户级划分：train=882, val=66, test=94 windows
- Scheme C 输入：`(299, 6, 36, 36)` [+ 可选 IMU] → 输出 `(2500,)`
- Scheme D 输入：`(300, 3)` [+ 可选 IMU] → 输出 `(2500,)`
- Scheme E 输入：`(300, 1)` [+ 可选 IMU] → 输出 `(2500,)`
- Scheme F/G 输入：`(300, 3, 64, 64)` [+ 可选 IMU] → 输出 `(2500,)`
- Scheme H 输入：`(300, 3, 128, 128)` [+ 可选 IMU] → 输出 `(2500,)`
- Scheme I 输入：`(300, 3, 8, 8)` STMap [+ 可选 IMU] → 输出 `(2500,)`

（推荐运行顺序：E → I-direct → F → H，从轻到重逐步验证。）

> 📖 **详细文档**：各方案原理、配置说明、验证步骤见 [SCHEMES.md](SCHEMES.md)

### 待改进

- 尚未跑过完整训练，需在 GPU 服务器上验证
- 数据增强尚未实现
- 新数据（70-100样本，高分辨率~40MB/video）即将采集
- 后续批次：Transfer learning (rPPG预训练)、PhysMamba、CardioGAN-style训练

---

## PPG2ECG Baseline Validation (新增)

**目的**：在公开数据集上验证 PPG → ECG 是否可行，排除数据质量问题

**论文**：*"Reconstructing QRS Complex from PPG by Transformed Attentional Neural Networks"* (IEEE Sensors 2020)

### 文件结构

```
external_data/
├── bidmc/                      # BIDMC 原始数据 (需手动下载)
└── bidmc_processed/            # 预处理后数据
    ├── train.pt
    ├── test.pt
    └── metadata.json

scripts/
└── prepare_bidmc.py            # BIDMC 数据预处理脚本

models/
├── ppg2ecg.py                  # PPG2ECG 模型 (Encoder-Decoder + STN + Attention)
└── train_ppg2ecg.py            # PPG2ECG 训练脚本

configs/
└── ppg2ecg_bidmc.yaml          # BIDMC 训练配置
```

### 如何运行

**Step 1: 下载 BIDMC 数据集**
```bash
# 手动下载：访问 https://physionet.org/content/bidmc/1.0.0/
# 下载 bidmc_csv.zip 或全部文件
# 解压到 external_data/bidmc/
```

**Step 2: 预处理数据**
```bash
python scripts/prepare_bidmc.py --data_dir external_data/bidmc --output_dir external_data/bidmc_processed
```

**Step 3: 训练**
```bash
# 基本训练
python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml

# 调整参数
python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml --epochs 100 --batch_size 128
```

**Step 4: 评估**
```bash
python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml --eval_only --checkpoint checkpoints/ppg2ecg_bidmc/best_model.pt
```

### 预期指标

| 指标 | 目标值 | 论文值 |
|------|--------|--------|
| Pearson r | > 0.7 | 0.844 |
| RMSE | < 0.3 | - |
| MAE | < 0.2 | - |

### 判断逻辑

```
BIDMC 成功 (r > 0.7) → 模型 OK，继续在我们数据上验证
BIDMC 失败 (r < 0.5) → 检查复现细节，或尝试 CardioGAN
```

### 模型架构

- **Encoder**: Conv1d (1→32→64→128→256→512), stride=2, PReLU
- **Decoder**: ConvTranspose1d 镜像结构, Tanh 输出
- **STN**: Spatial Transformer Network，校准 PPG 时序偏移
- **Attention**: 多头注意力，聚焦 QRS 复合波区域
- **Loss**: QRS-enhanced L1 loss (对 R 峰区域加权)

### 训练配置 (BIDMC)

- 输入: `[batch, 1, 256]` (256 samples @ 125 Hz ≈ 2 秒)
- 批大小: 256
- 学习率: 0.0001
- 优化器: Adam
- Scheduler: CosineAnnealingLR
- Epochs: 300
- Early stopping patience: 50

## Architecture

### Data Pipeline

```
Raw .bs file (binary ECG)
    ↓
parse_blt_ecg_bs_u8()      # Extract uint8 waveforms + timestamps
    ↓
export_ecg_df_no_marker()  # Create DataFrame with filtered columns
    ↓
Output: CSV, Parquet, gaps.csv, metadata.json
```

### Key Functions in ecg_data_v5.ipynb

| Function | Purpose |
|----------|---------|
| `parse_blt_ecg_bs_u8()` | Parse .bs binary files, extract 20-second segments |
| `ecg_filter_padded()` | Apply notch (50Hz) + bandpass filtering |
| `compute_gap_events()` | Detect time discontinuities between segments |
| `export_ecg_df_no_marker()` | Generate analysis-ready DataFrame |
| `audit_authenticity()` | SHA256 verification chain for data integrity |
| `plot_ecg_window_df()` | Static matplotlib plot with ECG grid |
| `interactive_ecg_plot_df()` | Zoomable Plotly visualization |

### ECG Data Specifications

- **Format:** Binary .bs files (pattern: `II_YYYYMMDD_X_Y.bs`)
- **Sampling rate:** 250 Hz
- **Segment length:** 20 seconds (5000 samples each)
- **ADC resolution:** uint8 (0-255) → int16 (-128 to 127)
- **Lead:** ECG Lead II
- **Mains frequency:** 50 Hz (notch filter target)

### Filtering Modes

| Mode | Bandpass | Use Case |
|------|----------|----------|
| `monitor` | 0.67-40 Hz | Clinical display |
| `diagnostic` | 0.05-100 Hz | Morphological analysis |
| `st` | 0.05-40 Hz | ST-segment research |

### Output DataFrame Columns

`timestamp, t_rel_s_true, segment_index, sample_in_segment, ecg_u8_raw, ecg_counts_raw_int, ecg_counts_filt_monitor, ecg_counts_filt_diagnostic, ecg_counts_filt_st, [ecg_mV]`

## Dataset Structure

```
ECG_dataset/           # ECG原始数据（ground truth）
  ├── {day}/*.bs       # 二进制文件，按日期组织
  └── 命名：II_YYYYMMDD_X_Y.bs

phone_dataset/         # 手机采集的多模态数据（模型输入）
  └── {用户名} YYYY-MM-DD HH_MM.zip
      ├── camera_0_*.mp4     # 手指捂住摄像头视频 → PPG提取 → ECG重建
      ├── camera_1_*.mp4     # 正常录像 → 呼吸曲线恢复
      ├── imu_data_*.json    # 加速度计+陀螺仪
      ├── user_info_*.json   # 受试者信息
      └── annotation_*.json  # 标注（含时间戳，用于对齐ECG）

training_data/samples/  # 预处理后的训练样本（98个）
  └── pair_XXXX/
      ├── video_0.mp4        # = camera_0（手指PPG视频，红色通道主导，已校验 R>G,B）
      ├── ecg.csv            # ECG波形 ground truth (250 Hz)
      ├── imu.csv            # 加速度计+陀螺仪 (~100 Hz)
      ├── annotation.json    # 心率、血压、血氧、呼吸、状态
      ├── user_info.json     # 性别、身高、体重、心脏病史
      └── metadata.json      # 元数据+统计信息
```

## 待完成任务

1. ~~**ECG批量转换**~~：已完成
2. ~~**时间对齐**~~：已完成
3. ~~**Phone数据预处理**~~：已完成
4. ~~**Task 1 代码实现**~~：已完成（Scheme C + D）
5. **GPU 服务器上训练验证 Scheme C/D**：待进行
6. **引入 PhysNet (3D CNN)**：待实现
7. **消融实验与对比实验**：待进行

## Design Principles

- **Non-destructive processing:** Raw data preserved; filtered columns added in parallel
- **Byte-level authenticity:** SHA256 chain verifies: input file → extracted bytes → DataFrame
- **Gap tracking:** Time discontinuities logged separately, not modified in data
- **时间同步**：ECG与Phone数据通过时间戳精确对齐
