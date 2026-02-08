# 服务器实验运行指南

## 环境

```bash
conda activate torch
cd /path/to/ECG
git pull
pip install numpy pandas scipy pyyaml opencv-python-headless torch torchvision matplotlib  # 如缺依赖
```

---

## 实验设置总览

### 数据

| 参数 | 值 | 说明 |
|------|-----|------|
| **Batch 1** | pair_0000 ~ pair_0097 | 98 样本, 9 users, 120s |
| **Batch 2** | pair_0098 ~ pair_0134 | 37 样本, 2 users (yjw/yyh), 60s |
| **质量过滤** | `good,moderate` | 排除 PPG 心率误差 > 20 BPM 的 poor 样本 |
| **数据位置** | `training_data/samples/` | 所有数据在同一目录，通过 `batch_index.json` 区分批次 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--split random` | random | 随机划分（快速验证），`user` = 按用户划分（无泄露） |
| `--server a6000` | a6000 | 自动调 batch_size/AMP/梯度累积（也可选 `3090`） |
| `--batch batch_1` | 无(全部) | 按批次过滤：`batch_1`, `batch_2`, `batch_1,batch_2` |
| `--run-tag b1` | 无 | 标记到 checkpoint 目录名，区分不同实验 |
| `--quality-filter` | good,moderate | 写在 config 里。`--quality-filter all` 可覆盖 |
| `--patience` | 20-30 | 早停耐心值 |
| Early stopping | test set | 默认用 test set 做早停（DEBUG 模式），`--use-val` 切严格模式 |

### 模型方案（从轻到重）

| Scheme | 类型 | 参数量 | 显存 | 说明 |
|--------|------|--------|------|------|
| **E** | 1D UNet | ~500K | ~3GB | 红色通道 PPG，最轻量 |
| **I-direct** | STMap 2D CNN | ~300K | ~4GB | 保留空间信息 |
| **I-twostage** | STMap→PPG→ECG | ~600K | ~4GB | 两阶段，有 PPG 中间监督 |
| **F** | EfficientPhys | ~1.5M | ~10GB | End-to-end 时空注意力 |
| **H** | PhysFormer | ~10M | ~25GB | CDC + Transformer，理论最强 |

### 输出保存

每次训练自动保存到 `checkpoints/{scheme}/{split}_{quality}_p{patience}_testval_{tag}/`：

| 文件 | 内容 |
|------|------|
| `best_model.pt` | 最佳权重 |
| `config.yaml` | 完整配置（可复现） |
| `run_summary.json` | 最终指标 + 实验设置 |
| `training_history.json` | 逐 epoch 的 loss/RMSE/MAE/r |
| `training_curves.png` | 训练曲线图 |

训练结束自动评估，**不需要单独跑 evaluate**。

---

## 快速开始：一键跑全部

```bash
# 一键执行：5个方案 × 3个批次 = 15 次训练，4 张 GPU 并行
bash scripts/run_all_parallel.sh
```

这个脚本会：
1. 先跑 batch 2 质量检测（更新 quality CSV）
2. 每个 GPU 分配方案，顺序执行 batch_1 → batch_2 → batch_1+2
3. 最后汇总所有结果

GPU 分配：
- GPU 0: scheme_e → scheme_i_direct（轻量，共用一卡）
- GPU 1: scheme_i_twostage
- GPU 2: scheme_f
- GPU 3: scheme_h

---

## 手动运行（tmux 方式）

如果需要更多控制，手动开 4 个 tmux 窗口：

```bash
mkdir -p logs

# 先跑一次质量检测（所有样本）
python scripts/check_batch_quality.py 2>&1 | tee logs/quality_all.log
```

**窗口 0 (GPU 0) — Scheme E → I-Direct**:
```bash
# Batch 1
CUDA_VISIBLE_DEVICES=0 python models/train.py \
  --config configs/scheme_e.yaml --split random --server a6000 \
  --batch batch_1 --run-tag b1 2>&1 | tee logs/scheme_e_b1.log

# Batch 2
CUDA_VISIBLE_DEVICES=0 python models/train.py \
  --config configs/scheme_e.yaml --split random --server a6000 \
  --batch batch_2 --run-tag b2 --quality-filter all \
  2>&1 | tee logs/scheme_e_b2.log

# Batch 1+2
CUDA_VISIBLE_DEVICES=0 python models/train.py \
  --config configs/scheme_e.yaml --split random --server a6000 \
  --batch batch_1,batch_2 --run-tag b1+2 \
  2>&1 | tee logs/scheme_e_b1+2.log

# 然后 scheme_i_direct 同理...
```

**窗口 1/2/3** 类似，改 `CUDA_VISIBLE_DEVICES` 和 `--config`。

---

## 关于 Batch 2 质量过滤

Batch 1 有现成的质量 CSV（`eval_results/ppg_analysis_all_samples.csv`，98 条）。

Batch 2 需要在**服务器上**先跑质量检测（需要 cv2+scipy），脚本会自动合并到同一个 CSV：

```bash
python scripts/check_batch_quality.py --batch batch_2
```

跑完后 quality CSV 会包含 135 条记录。如果 batch 2 还没有 quality 数据，用 `--quality-filter all` 暂时跳过过滤。

---

## 反馈结果

```bash
# 一键收集所有结果
find checkpoints/ -name run_summary.json -exec echo "=== {} ===" \; -exec cat {} \;

# 打包
tar czf results.tar.gz logs/*.log checkpoints/*/run_summary.json checkpoints/*/training_curves.png
```

---

## 监控

```bash
watch -n 5 nvidia-smi                               # GPU 使用
tail -f logs/scheme_e_b1.log                         # 某个方案日志
for f in logs/*.log; do echo "=== $f ==="; tail -1 "$f"; done   # 所有方案最新进度
```
