# Video → ECG 重建方案总结

本文档详细介绍所有已实现的模型方案，包括原理、配置、运行方法和验证步骤。

---

## 环境安装

### 方式一：Conda（推荐）

```bash
# 创建新环境
conda env create -f environment.yaml

# 激活环境
conda activate ecg
```

### 方式二：pip

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 单独安装 PyTorch (根据 GPU 选择 CUDA 版本)
# Blackwell (RTX PRO 6000 等) 需要 CUDA 12.6:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Ampere/Hopper (A6000, 3090 等) 用 CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## 快速开始

```bash
# 推荐的第一次运行命令（随机划分 + 好数据 + 适中参数）
CUDA_VISIBLE_DEVICES=0 python models/train.py --config configs/scheme_e.yaml \
    --split random \
    --quality-filter good \
    --patience 15 \
    --server 3090
```

---

## 命令行参数汇总

| 参数 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| `--config` | `configs/scheme_*.yaml` | `scheme_c.yaml` | 选择模型方案 |
| `--server` | `3090`, `a6000` | `a6000` | 自动应用服务器最优参数 |
| `--split` | `random`, `user` | `user` | 数据划分方式（random 更简单） |
| `--quality-filter` | `good`, `good,moderate`, `all` | config 中设置 | 数据质量过滤 |
| `--patience` | 整数 | 20-30 (看 config) | Early stopping 耐心值 |
| `--epochs` | 整数 | 200 | 最大训练轮数 |

### 参数详解

**`--split`** 数据划分方式：
- `random`：随机 80/10/10 划分，**推荐先用这个验证模型可行性**
- `user`：按用户划分（无数据泄露），用于最终评估

**`--quality-filter`** 数据质量过滤：
- `good`：只用高质量样本（80个），**推荐**
- `good,moderate`：排除 poor 样本（88个）
- `all`：用全部数据（98个，含 10 个 poor）

**`--patience`** Early stopping：
- 连续多少个 epoch val_loss 不下降就停止
- 默认 20-30，如果训练太早停可以调大到 15-20

**`--server`** 服务器预设：
- `3090`：小 batch、开 AMP、梯度累积
- `a6000`：大 batch、关 AMP

---

## 方案总览

| Scheme | 架构 | 输入形式 | 参数量 | 显存 | 3090 (24G) | A6000 (48G) |
|--------|------|----------|--------|------|:----------:|:-----------:|
| **C** | MTTS-CAN | 差分帧+原始帧 36×36 | 2.8M | ~15 GB | ⚠️ batch=8 | ✅ batch=16 |
| **D** | 1D TCN | RGB均值 (T,3) | 276K | ~2 GB | ✅ batch=64 | ✅ batch=128 |
| **E** | 1D UNet | 绿色通道 (T,1) | ~500K | ~3 GB | ✅ batch=64 | ✅ batch=128 |
| **F** | EfficientPhys | 视频帧 64×64 | ~1.5M | ~10-15 GB | ✅ batch=8-16 | ✅ batch=32 |
| **G** | PhysNet 3D CNN | 视频帧 64×64 | ~3-5M | ~20-25 GB | ⚠️ batch=2-4 | ✅ batch=8-16 |

> ⚠️ = 可以跑但显存紧张，建议开启 `use_amp: true`
>
> ⚠️ **注意**：Scheme D 未做任何修改，之前已跑过效果很差（输出接近直线）。建议优先尝试 Scheme E/F。

---

## GPU 运行指南

### 使用 `--server` 参数（推荐）

通过 `--server` 参数，训练脚本会自动应用对应服务器的最优参数（batch_size、AMP、梯度累积、num_workers），**无需手动修改配置文件**。

```bash
# 3090 服务器
CUDA_VISIBLE_DEVICES=1 python models/train.py --config configs/scheme_f.yaml --server 3090

# A6000 服务器
python models/train.py --config configs/scheme_f.yaml --server a6000
```

### 3090 (24GB) 运行方案

```bash
# 推荐：Scheme E（最轻量且原理合理）
CUDA_VISIBLE_DEVICES=1 python models/train.py --config configs/scheme_e.yaml --server 3090

# Scheme F（End-to-end）
CUDA_VISIBLE_DEVICES=1 python models/train.py --config configs/scheme_f.yaml --server 3090

# Scheme C（MTTS-CAN）
CUDA_VISIBLE_DEVICES=1 python models/train.py --config configs/scheme_c.yaml --server 3090

# Scheme D（Baseline，已测试效果差）
CUDA_VISIBLE_DEVICES=1 python models/train.py --config configs/scheme_d.yaml --server 3090

# Scheme G（⚠️ 不推荐，易 OOM，建议用 A6000）
CUDA_VISIBLE_DEVICES=1 python models/train.py --config configs/scheme_g.yaml --server 3090
```

### A6000 (48GB) 运行方案

A6000 可以运行所有方案，使用更大的 batch_size 加速训练：

```bash
# Scheme G（A6000 专属推荐，3D CNN）
python models/train.py --config configs/scheme_g.yaml --server a6000

# Scheme F（End-to-end，大 batch）
python models/train.py --config configs/scheme_f.yaml --server a6000

# Scheme E（轻量方案的上限验证）
python models/train.py --config configs/scheme_e.yaml --server a6000

# Scheme C（学术验证过的架构）
python models/train.py --config configs/scheme_c.yaml --server a6000

# Scheme D（Baseline）
python models/train.py --config configs/scheme_d.yaml --server a6000
```

---

### 服务器预设参数表

`--server` 参数会从 `configs/server_presets.yaml` 加载以下参数覆盖配置：

| 方案 | GPU | batch_size | grad_accum | use_amp | num_workers | 有效 batch |
|------|-----|------------|------------|---------|-------------|------------|
| **C** | 3090 | 8 | 4 | true | 4 | 32 |
| **C** | A6000 | 32 | 1 | false | 8 | 32 |
| **D** | 3090 | 64 | 1 | false | 4 | 64 |
| **D** | A6000 | 128 | 1 | false | 8 | 128 |
| **E** | 3090 | 64 | 1 | false | 4 | 64 |
| **E** | A6000 | 128 | 1 | false | 8 | 128 |
| **F** | 3090 | 8 | 2 | true | 4 | 16 |
| **F** | A6000 | 32 | 1 | false | 8 | 32 |
| **G** | 3090 | 2 | 8 | true | 2 | 16 |
| **G** | A6000 | 16 | 1 | false | 8 | 16 |

> 💡 不加 `--server` 参数时使用配置文件的默认值

---

## 数据划分与质量过滤

### 数据划分方式

支持两种划分模式：

| 模式 | 命令行参数 | 难度 | 说明 |
|------|-----------|------|------|
| **随机划分** | `--split random` | 简单 | 同一用户可能出现在 train/test，用于验证模型可行性 |
| **用户划分** | `--split user` | 困难 | 严格按用户划分，无数据泄露，用于最终评估 |

```bash
# 推荐：先用随机划分验证模型能否学到东西
python models/train.py --config configs/scheme_e.yaml --split random

# 效果好了再用用户划分做最终评估
python models/train.py --config configs/scheme_e.yaml --split user
```

**用户划分详情**（默认）：

| 集合 | 用户 | 样本数 | 说明 |
|------|------|--------|------|
| **Train** | fzq, fcy, wjy, czq, syw, wcp | ~72 pairs | 训练用 |
| **Val** | nxs | ~7 pairs | 早停判断 |
| **Test** | lrk, fhy | ~19 pairs | 最终评估 |

**随机划分详情**：80% train / 10% val / 10% test，按 pair 随机划分

> 每个 pair 切成 10s 窗口（5s 步长），总计约 1000+ 个训练样本

### 数据质量过滤

98 个样本中有 **10 个 poor 样本**（PPG 心率检测失败，误差 >20 BPM），建议过滤：

```bash
# 只用高质量样本 (80个, 推荐)
python models/train.py --config configs/scheme_e.yaml --quality-filter good

# 排除 poor 样本 (88个)
python models/train.py --config configs/scheme_e.yaml --quality-filter good,moderate

# 用全部数据 (98个)
python models/train.py --config configs/scheme_e.yaml --quality-filter all
```

或在 config 中设置：
```yaml
data:
  quality_filter: "good"  # "good" / "good,moderate" / null
```

> 📊 详细质量分析见 `eval_results/data_quality_report.md`

### Early Stopping 控制

默认 patience=20-30，如果模型训练到十几个 epoch 就停了，可以增大 patience：

```bash
# 增大 patience，允许更长时间不改进
python models/train.py --config configs/scheme_e.yaml --patience 50

# 同时增大最大 epochs
python models/train.py --config configs/scheme_e.yaml --patience 50 --epochs 300

# 组合使用
python models/train.py --config configs/scheme_e.yaml --server 3090 --quality-filter good --patience 50
```

---

## 各方案详解

### Scheme C: MTTS-CAN (差分帧 + 空间注意力)

**原理**：
- 来自 NeurIPS 2020 论文 "Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement"
- 双分支结构：运动分支处理差分帧，外观分支提供空间注意力
- TSM (Temporal Shift Module) 实现无额外计算的时序建模
- 差分帧 `(f[t+1]-f[t])/(f[t+1]+f[t])` 放大帧间微小变化

**输入**：
```
视频帧 → resize(36×36) → 计算差分帧 → concat([diff, raw]) → (T-1, 6, 36, 36)
```

**配置文件**：`configs/scheme_c.yaml`

**运行**：
```bash
# 3090 服务器
CUDA_VISIBLE_DEVICES=1 python models/train.py --config configs/scheme_c.yaml --server 3090

# A6000 服务器
python models/train.py --config configs/scheme_c.yaml --server a6000
```

---

### Scheme D: 1D TCN (RGB均值时序) ⚠️ 已测试效果差

**原理**：
- 最简单的方案，完全抛弃空间信息
- 每帧提取 RGB 三通道均值，得到 (T, 3) 的1D信号
- 使用 TCN (Temporal Convolutional Network) 处理时序
- 优点：超轻量，任何GPU都能跑；缺点：丢失所有空间信息

**输入**：
```
视频帧 → 每帧计算 mean(R), mean(G), mean(B) → (T, 3) → Z-normalize
```

**配置文件**：`configs/scheme_d.yaml`

**运行**：
```bash
# 3090 服务器
python models/train.py --config configs/scheme_d.yaml --server 3090

# A6000 服务器
python models/train.py --config configs/scheme_d.yaml --server a6000
```

> ⚠️ **测试结果**：此方案已测试，输出接近直线，无法重建 ECG 波形。保留仅用于消融实验对比。

---

### Scheme E: 1D UNet (绿色通道 + 跳跃连接)

**原理**：
- 绿色通道对血红蛋白最敏感，是 PPG 信号的主要来源
- UNet 的跳跃连接保留信号的细节信息，不像纯编码器那样全部压缩
- 比 Scheme D 多了跳跃连接，理论上能保留更多波形细节

**输入**：
```
视频帧 → 每帧提取绿色通道均值 → (T, 1) → Z-normalize
```

**配置文件**：`configs/scheme_e.yaml`

**运行**：
```bash
# 3090 服务器
python models/train.py --config configs/scheme_e.yaml --server 3090

# A6000 服务器
python models/train.py --config configs/scheme_e.yaml --server a6000
```

**推荐**：轻量且原理合理，建议首选尝试。

---

### Scheme F: EfficientPhys (时空注意力)

**原理**：
- 来自 WACV 2023 "EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement"
- 时空分解：先做时间卷积，再做空间卷积，减少计算量
- 空间注意力：自动学习关注皮肤区域，忽略背景
- End-to-end：直接从视频帧学习，无需手动提取 PPG

**输入**：
```
视频帧 → resize(64×64) → (T, 3, 64, 64) → 网络自动提取特征
```

**配置文件**：`configs/scheme_f.yaml`

**运行**：
```bash
# 3090 服务器
CUDA_VISIBLE_DEVICES=1 python models/train.py --config configs/scheme_f.yaml --server 3090

# A6000 服务器
python models/train.py --config configs/scheme_f.yaml --server a6000
```

---

### Scheme G: PhysNet (3D CNN)

**原理**：
- 来自 BMVC 2019 "Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks"
- 经典的 3D CNN 架构，把视频当作时空立方体处理
- 3D 卷积同时捕捉空间和时间模式
- 残差连接帮助梯度流动

**输入**：
```
视频帧 → resize(64×64) → (T, 3, 64, 64) → 3D卷积处理
```

**配置文件**：`configs/scheme_g.yaml`

**运行**：
```bash
# A6000 服务器（推荐）
python models/train.py --config configs/scheme_g.yaml --server a6000

# 3090 服务器（⚠️ 可能 OOM）
CUDA_VISIBLE_DEVICES=1 python models/train.py --config configs/scheme_g.yaml --server 3090
```

**注意**：3090 即使使用预设参数仍可能 OOM，推荐在 A6000 上跑。

---

## 运行与验证

### 1. 检查数据管线

在训练前，先验证数据加载是否正确：

```bash
# 检查任意方案的数据 shape
python models/dataset.py configs/scheme_e.yaml
```

预期输出：
```
Loaded 98 sample pairs
Split: train=72 pairs, val=7, test=19
Total windows: 1042
Windows: train=882, val=66, test=94

Sample shapes:
  video: torch.Size([300, 1])  # Scheme E
  ecg:   torch.Size([2500])
```

### 2. 训练模型

```bash
# 激活环境
conda activate torch

# 训练（会自动在 test 集评估）
# 3090 服务器
python models/train.py --config configs/scheme_e.yaml --server 3090

# A6000 服务器
python models/train.py --config configs/scheme_e.yaml --server a6000
```

训练过程会输出：
- 每个 epoch 的 train/val loss
- Early stopping 触发时会保存 best model
- 训练结束后自动在 test 集评估

### 3. Checkpoint 保存路径

模型保存路径包含关键参数，方便追溯：

```
checkpoints/{scheme}/{split}_{quality}_p{patience}/
    ├── best_model.pt        # 最优模型权重
    ├── config.yaml          # 完整训练配置（可复现）
    ├── training_history.json # 训练曲线数据
    └── training_curves.png   # 训练曲线图
```

**示例**：
```
checkpoints/scheme_e/random_good_p15/     # --split random --quality-filter good --patience 15
checkpoints/scheme_e/user_good+moderate_p20/  # --split user --quality-filter good,moderate --patience 20
checkpoints/scheme_f/random_all_p30/      # --split random --quality-filter all --patience 30
```

> 路径命名规则：`{split}_{quality}_p{patience}`
> - split: `random` 或 `user`
> - quality: `good`, `good+moderate`, 或 `all`
> - p{patience}: early stopping 的 patience 值

### 4. 仅评估（不训练）

```bash
# 指定具体的 checkpoint 路径
python models/run_eval.py \
    --config checkpoints/scheme_e/random_good_p15/config.yaml \
    --checkpoint checkpoints/scheme_e/random_good_p15/best_model.pt
```

### 5. 评估指标

| 指标 | 含义 | 越小/大越好 |
|------|------|-------------|
| RMSE | 均方根误差 | 越小越好 |
| MAE | 平均绝对误差 | 越小越好 |
| Pearson r | 相关系数 | 越大越好 (接近1) |

### 6. 对比实验

建议按以下顺序跑，从轻到重（以 3090 为例）：

```bash
# 1. 最轻量 baseline
python models/train.py --config configs/scheme_d.yaml --server 3090

# 2. 推荐方案
python models/train.py --config configs/scheme_e.yaml --server 3090

# 3. End-to-end 方案
python models/train.py --config configs/scheme_f.yaml --server 3090

# 4. 学术验证过的架构
python models/train.py --config configs/scheme_c.yaml --server 3090

# 5. 3D CNN (建议用 A6000)
python models/train.py --config configs/scheme_g.yaml --server a6000
```

---

## 启用 IMU 融合

在任意配置文件中设置：

```yaml
data:
  use_imu: true
  imu_sr: 100
```

IMU 数据会在模型的 bottleneck 层与视频特征融合。

---

## 常见问题

### Early Stopping 太早（训练十几个 epoch 就停了）

这通常是因为 validation loss 波动导致 patience 耗尽。解决方案：

```bash
# 增大 patience
python models/train.py --config configs/scheme_e.yaml --patience 50

# 或同时增大 epochs
python models/train.py --config configs/scheme_e.yaml --patience 50 --epochs 300
```

> 💡 默认 patience=20-30，对于小数据集可能需要更大的值

### OOM (显存不足)

**推荐**：使用 `--server 3090` 参数自动应用低显存配置。

手动调整（不推荐）：
1. 减小 `batch_size`
2. 增加 `gradient_accumulation_steps` 保持有效 batch 不变
3. 启用 `use_amp: true` (混合精度)

```yaml
train:
  batch_size: 4
  gradient_accumulation_steps: 4  # 有效 batch = 16
  use_amp: true
```

### 训练效果差（输出是直线）

1. 检查数据：可视化 PPG 信号和 ECG 的相关性
2. 换 loss：使用 `loss: composite` 而非 `loss: mse`
3. 尝试不同方案：Scheme E/F 比 D 更有可能捕捉波形细节
4. 降低目标：先试预测心率/R-R间期，再尝试完整波形
5. **过滤低质量数据**：使用 `--quality-filter good` 排除 poor 样本

### 3090 跑不了某方案

使用 `--server 3090` 参数会自动应用以下配置：

| 方案 | 3090 预设 |
|------|-----------|
| C | batch=8, grad_accum=4, use_amp=true |
| D | batch=64, 无压力 |
| E | batch=64, 无压力 |
| F | batch=8, grad_accum=2, use_amp=true |
| G | batch=2, grad_accum=8, use_amp=true (**仍建议用 A6000**) |

---

## 推荐调试流程

目前模型效果较差，建议按以下步骤逐步验证：

### 第一步：验证模型可行性（随机划分 + 好数据）

先用最简单的设置，验证模型能否学到东西：

```bash
# 随机划分（简单） + 只用好数据 + 适中 patience
python models/train.py --config configs/scheme_e.yaml \
    --split random \
    --quality-filter good \
    --patience 15 \
    --server 3090
```

**预期**：如果模型可行，val_loss 应该能持续下降，Pearson r 应该 > 0.3

### 第二步：正式评估（用户划分）

随机划分效果还行后，再用用户划分做正式评估：

```bash
# 用户划分（困难，无数据泄露）
python models/train.py --config configs/scheme_e.yaml \
    --split user \
    --quality-filter good \
    --patience 20 \
    --server 3090
```

**预期**：用户划分效果会比随机划分差，但如果差太多说明模型泛化能力不足

### 第三步：尝试其他方案

如果 Scheme E 效果不好，按顺序尝试其他方案：

```bash
# Scheme F (End-to-end，可能更强)
python models/train.py --config configs/scheme_f.yaml --split random --quality-filter good --server 3090

# Scheme C (学术验证过的架构)
python models/train.py --config configs/scheme_c.yaml --split random --quality-filter good --server 3090
```

---

## 推荐训练顺序

### 3090 推荐顺序
1. **Scheme E** — 最轻量且原理合理，首选
2. **Scheme F** — End-to-end，如果 E 效果不好试这个
3. **Scheme C** — 学术验证过的 rPPG 架构（需调小 batch）
4. ~~**Scheme D**~~ — 已测试，效果很差，仅作对比

### A6000 推荐顺序
1. **Scheme G** — 3D CNN，理论上捕捉时空信息能力最强
2. **Scheme F** — End-to-end，可用大 batch
3. **Scheme E** — 轻量方案的上限验证
4. **Scheme C** — 学术验证过的架构

> **说明**：Scheme D（1D TCN + RGB均值）已在之前测试中效果很差（输出接近直线，无 ECG 波形特征），不再作为主要实验对象，仅保留用于消融实验对比。
