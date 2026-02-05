# Video → ECG 重建方案总结

本文档详细介绍所有已实现的模型方案，包括原理、配置、运行方法和验证步骤。

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

### 3. 仅评估（不训练）

```bash
python models/run_eval.py \
    --config configs/scheme_e.yaml \
    --checkpoint checkpoints/scheme_e/best_model.pt
```

### 4. 评估指标

| 指标 | 含义 | 越小/大越好 |
|------|------|-------------|
| RMSE | 均方根误差 | 越小越好 |
| MAE | 平均绝对误差 | 越小越好 |
| Pearson r | 相关系数 | 越大越好 (接近1) |

### 5. 对比实验

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
