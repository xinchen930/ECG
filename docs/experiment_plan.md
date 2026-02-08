# 实验计划与分析思路

## 当前环境

- **GPU**: 4x NVIDIA RTX PRO 6000 Blackwell (98GB VRAM each)
- **数据**: 135 对样本（CLAUDE.md 中写 98 对，实际已增加到 135 对）
- **质量过滤后**: 88 对（默认 good+moderate），切窗后 917 个 10s 窗口

---

## 第一轮：可行性验证（当前正在跑）

**目的**：验证各方案能否正常训练，建立 baseline 指标

| GPU | 方案 | 类型 | Split | 质量过滤 | 状态 |
|-----|------|------|-------|---------|------|
| 0 | Scheme E | 1D UNet (红通道) | random | good+moderate | 训练中 |
| 1 | Scheme D | 1D TCN (RGB均值) | random | good+moderate | 训练中 |
| 2 | Scheme F | EfficientPhys (2D) | random | good+moderate | 训练中 |
| 3 | Scheme C | MTTS-CAN (2D) | random | good+moderate | 训练中 |

**为什么用 random split**：
- Random split 更简单（训练/测试可能来自同一用户），适合先验证模型能否学到信号
- 如果 random split 都学不到，说明模型/数据有根本问题
- 如果 random split 学到了但 user split 不行，说明模型没有泛化能力

---

## 第二轮：泛化能力测试

**目的**：用 user split（用户级划分，无数据泄漏）评估真实泛化能力

| GPU | 方案 | Split | 说明 |
|-----|------|-------|------|
| 0 | Scheme E | user | 最轻量，看 1D 信号能否跨用户泛化 |
| 1 | Scheme D | user | 对比 E，看 RGB 均值 vs 红通道 |
| 2 | Scheme F | user | 2D 方案跨用户泛化 |
| 3 | 第一轮最好方案 | user + IMU | 加入 IMU 看是否有帮助 |

---

## 第三轮：补充实验

根据第一二轮结果决定：

### 3a. 更多方案对比
- Scheme E-RGB（RGB三通道 vs 红通道）
- Scheme I-direct（STMap 方案）
- Scheme G（PhysNet 3D CNN）
- Scheme H（PhysFormer，理论最强）

### 3b. 消融实验
- 有 IMU vs 无 IMU
- 不同质量过滤（all / good+moderate / good only）
- 不同窗口长度（如果支持）

### 3c. BIDMC Baseline（如果自有数据结果差）
- 在公开数据集上验证 PPG→ECG 是否可行
- 排除数据质量问题 vs 模型架构问题

---

## 关键指标与判断逻辑

### 三个核心指标
| 指标 | 含义 | 好的范围 |
|------|------|---------|
| **Pearson r** | 波形相关性 | > 0.5 有希望, > 0.7 很好 |
| **RMSE** | 均方根误差 | 越小越好 |
| **MAE** | 平均绝对误差 | 越小越好 |

### 结果判断树

```
Random Split 结果：
├── r > 0.7  → 模型能学到信号，继续 user split
│   ├── User split r > 0.5 → 有泛化能力，继续优化
│   └── User split r < 0.3 → 过拟合到个体，需要更多数据/更强正则化
├── r = 0.3~0.7 → 部分学到，需要调参/换架构
└── r < 0.3 → 基本没学到
    ├── 检查 loss 是否在下降 → 可能需要更多 epoch
    ├── 跑 BIDMC baseline → 验证 PPG→ECG 本身可行性
    └── 检查数据质量 → 可能需要更严格过滤
```

---

## 下一步方向（根据结果选择）

### 路线 A：结果好（random r > 0.7）
1. User split 验证泛化能力
2. 最优方案 + IMU 融合
3. PhysFormer (Scheme H) — 理论最强架构
4. 数据增强 + 正则化提升泛化
5. 收集更多数据（高质量视频）

### 路线 B：结果一般（random r = 0.3~0.7）
1. 分析哪些样本预测好、哪些差 → 找规律
2. 试 Scheme H (PhysFormer)，更强的时空建模
3. 跑 BIDMC baseline 确认 PPG→ECG 可行性
4. 数据质量分析 → 可能需要重新采集

### 路线 C：结果差（random r < 0.3）
1. 先跑 BIDMC baseline：如果 BIDMC 也差 → 模型代码有 bug
2. BIDMC 好但自有数据差 → 数据质量问题
3. 检查 PPG 信号质量：手指视频是否真的包含 PPG 信息？
4. 考虑 CardioGAN（对抗训练可能对噪声数据更鲁棒）

---

## 当前状态

- 第一轮 4 个实验正在 4 张 GPU 上并行训练
- Scheme E/D（1D 方案）预计几分钟内完成
- Scheme F/C（2D 方案）预计 10-30 分钟完成
- 训练结束后自动在 test 集上评估并打印结果
