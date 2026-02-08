# 实验记录

> 本文档记录所有训练实验的结果、分析和对应修改。按时间顺序维护。

---

## 环境信息

- **服务器**: 4x NVIDIA RTX PRO 6000 Blackwell (98GB VRAM each)
- **PyTorch**: 2.10.0+cu128, CUDA 13.0
- **Conda 环境**: ecg
- **数据**:
  - Batch 1: pair_0000 ~ pair_0097 (98 样本, 视频 320x568 @ 30fps)
  - Batch 2: pair_0098 ~ pair_0134 (37 样本, 视频 1080x1920 @ 30fps)
  - 总计: 135 样本

## 数据质量检测

| 批次 | Good | Moderate | Poor | 总计 |
|------|------|----------|------|------|
| Batch 1 | 81 | 9 | 8 | 98 |
| Batch 2 | 36 | 0 | 1 | 37 |
| **总计** | **117** | **9** | **9** | **135** |

> Batch 2 质量明显更好 (97% good vs Batch 1 的 83% good)
> 质量评判标准: PPG 提取心率与 GT 心率的偏差 (<10 BPM = good, <20 = moderate, >20 = poor)

---

## Round 1: 初始训练 (2026-02-08)

### 配置
- **Loss**: CompositeLoss (MSE + 0.1×Spectral + 0.1×Pearson)
- **Split**: random (简单模式，同用户可出现在 train/test)
- **数据**: Batch 1 good+moderate (88 样本，917 个窗口中 818 train / 99 test)
- **注**: Batch 2 因无质量标签被排除

### 结果

| 方案 | 类型 | RMSE | MAE | Pearson r | Epochs | 备注 |
|------|------|------|-----|-----------|--------|------|
| **E** | 1D UNet (红通道) | 1.022 | 0.592 | -0.012 | 32 | 6.2M params |
| **D** | 1D TCN (RGB均值) | 1.017 | 0.600 | -0.042 | 31 | 276K params |
| **F** | EfficientPhys (2D) | 1.017 | 0.600 | -0.007 | 21 | 1.9M params |
| **C** | MTTS-CAN (2D) | 1.030 | 0.601 | -0.012 | 21 | 2.8M params |

### 分析与诊断

**核心问题**: 所有方案 Pearson r ≈ 0，RMSE ≈ 1.0（等于 z-normalized ECG 的方差）

**诊断发现三个根本原因**:

1. **模型输出尺度坍塌 (最关键)**
   - 模型对所有输入输出几乎相同的近零波形（预测标准差仅 0.0038，目标 0.99，差 259 倍）
   - 不同输入（真实/噪声/零）产生几乎相同的输出 → 模型没有学到输入-输出映射

2. **损失函数不平衡**
   - MSE 主导训练，Pearson 权重仅 0.1，太小
   - "预测均值" 是 MSE 的局部最小值（loss=1.0），模型被困在这里
   - Pearson loss（尺度不变，本该帮助纠正）权重太低

3. **视频数据信噪比极低**
   - PPG 信号（心跳引起的亮度波动）仅占像素值的 1-6%
   - 空间噪声比 PPG 信号大 2x
   - 经全局池化后信号基本被淹没

### 修改措施

| 修改 | 详情 | 文件 |
|------|------|------|
| **升级 Loss 函数** | 所有 config 从 `composite` (Pearson=0.1) 改为 `composite_v2` (Pearson=2.0, dynamic_weighting=true) | configs/scheme_*.yaml |
| **启用动态权重** | 前 20 epoch Pearson 权重 ×1.5（优先学形状），后期逐渐增加频率 loss | configs/scheme_*.yaml |
| **待做: 输出缩放层** | 添加可学习的 scale/bias 层到模型末端 | models/video_ecg_model.py |
| **待做: 时间均值减除** | 2D 模型输入减去帧的时间均值，放大 PPG 微小波动 | models/dataset.py |

---

## Round 2: 修复 Loss 后训练 (2026-02-08)

### 配置变更
- **Loss**: CompositeLossV2 (MSE=1.0 + **Pearson=2.0** + Spectral=0.1 + STFT=0.1, **dynamic_weighting=true**, warmup=20)
- **其余同 Round 1**: random split, Batch 1 good+moderate

### 结果

> 训练中...

| 方案 | 类型 | RMSE | MAE | Pearson r | Epochs | 备注 |
|------|------|------|-----|-----------|--------|------|
| **E** | 1D UNet (红通道) | - | - | - | - | 训练中 |
| **D** | 1D TCN (RGB均值) | - | - | - | - | 训练中 |
| **F** | EfficientPhys (2D) | - | - | - | - | 训练中 |
| **C** | MTTS-CAN (2D) | - | - | - | - | 训练中 |

### 分析

> 待 Round 2 结果出来后更新

---

## 待执行实验

1. Round 2 完成后:
   - 如果有效 → 在 Batch 2、Batch 1+2 上分别跑
   - 如果无效 → 尝试 Pearson-only loss + 输出缩放层

2. BIDMC 公开数据集 baseline:
   - 在干净的医院级 PPG 数据上验证 PPG→ECG 可行性
   - 排除模型架构问题

3. 跨用户泛化 (user split):
   - Random split 成功后再测 user split

4. 3D 方案 (G, H):
   - PhysNet 和 PhysFormer 需要在 2D 方案验证后再跑

---

## 文件变更日志

| 日期 | 文件 | 变更 |
|------|------|------|
| 2026-02-08 | configs/scheme_*.yaml | 所有方案 loss 从 composite 升级为 composite_v2 (Pearson=2.0, dynamic) |
| 2026-02-08 | eval_results/quality_batch1.csv | 新建: Batch 1 质量检测结果 |
| 2026-02-08 | eval_results/quality_batch2.csv | 新建: Batch 2 质量检测结果 |
| 2026-02-08 | eval_results/ppg_analysis_all_samples.csv | 新建: 全部 135 样本质量汇总 |
| 2026-02-08 | training_data/batch_index.json | 新建: 批次索引 |
| 2026-02-08 | training_data/samples/quality_labels.csv | 新建: 质量标签 |
