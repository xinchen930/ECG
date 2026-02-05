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
python models/train.py --config configs/scheme_c.yaml
# 或 scheme_d
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

已实现三套方案（移除了不适合的 ResNet 方案），通过 config 切换。**所有方案均支持可选 IMU 融合**（`use_imu: true`）：

| | Scheme C | Scheme D | Scheme E |
|---|---|---|---|
| 类型 | MTTS-CAN | 1D TCN | 1D UNet |
| 编码器 | 双分支+TSM+注意力 | TCN (Dilated Conv) | UNet编解码器 |
| 输入分辨率 | 36×36 | 1D信号 | 1D信号 |
| 输入形式 | 差分帧+原始帧 | RGB均值 | **绿色通道** |
| 核心思想 | 差分帧捕捉帧间微小变化 | 直接处理时序信号 | 跳跃连接保留细节 |
| 参数量 (无IMU) | 2.8M | 276K | **~500K** |
| 参数量 (有IMU) | 2.9M | 302K | ~530K |
| Batch size | 16 | 64 | 64 |
| 显存估算 | ~15 GB | ~2 GB | **~3 GB** |
| 3090可跑 | ⚠️ batch=8 | ✅ | ✅ |
| 配置文件 | `scheme_c.yaml` | `scheme_d.yaml` | `scheme_e.yaml` |

> 💡 在任意 config 中设置 `data.use_imu: true` 即可启用 IMU 融合
>
> 💡 Scheme E 使用绿色通道（对血红蛋白最敏感）作为 PPG 信号
>
> ⚠️ Scheme A/B (ResNet) 已移除：ResNet 为图像分类设计，会丢弃 PPG 所需的微小亮度变化信息

### 已完成文件

```
models/
├── __init__.py
├── dataset.py            # PyTorch Dataset（10s窗口切分、用户级划分、可选IMU/差分帧/1D信号/绿色通道）
├── video_ecg_model.py    # 模型定义（C/D/E三种架构 + CompositeLoss）
├── train.py              # 训练脚本（自动检测CUDA/MPS、early stopping）
├── run_eval.py           # 仅测试（加载 checkpoint 在 test 集评估）
└── evaluate.py           # 评估（RMSE, MAE, Pearson r）

configs/
├── scheme_c.yaml         # MTTS-CAN (差分帧 + 注意力)
├── scheme_d.yaml         # 1D Signal (TCN, 276K params)
└── scheme_e.yaml         # 1D UNet (绿色通道PPG, ~500K params)
```

### 数据管线验证结果

- 98 pairs → 1042 windows（10s窗口，5s步长）
- 用户级划分：train=882, val=66, test=94 windows
- Scheme C 输入：`(299, 6, 36, 36)` [+ 可选 IMU `(1000, 6)`] → 输出 `(2500,)`
- Scheme D 输入：`(300, 3)` [+ 可选 IMU `(1000, 6)`] → 输出 `(2500,)`
- Scheme E 输入：`(300, 1)` [+ 可选 IMU `(1000, 6)`] → 输出 `(2500,)`

（训练/测试命令见上文「如何开始训练、测试」；推荐先跑 scheme_e，最轻量且原理合理。）

### 待改进

- 模型架构可参考开源 PPG→ECG / Video→PPG 论文方案优化
- 数据增强尚未实现
- 尚未跑过完整训练，需在 GPU 服务器上验证

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
