# 服务器实验运行指南

## 环境准备

```bash
conda activate torch
cd /path/to/ECG     # 项目根目录（含 training_data/samples, configs/, models/）
git pull             # 确保代码是最新的
```

**依赖检查**（如果缺少请安装）：
```bash
pip install numpy pandas scipy pyyaml opencv-python-headless torch torchvision matplotlib
```

---

## 实验 1：Batch 1 数据（当前 98 样本）

### 快速验证：4 张 A6000，4 个 tmux 窗口

```bash
# 创建 tmux session
tmux new-session -s exp -d

# 窗口 0: GPU 0 — Scheme E + Scheme I-Direct（最轻量，共用一张卡，顺序执行）
tmux send-keys -t exp "conda activate torch && cd /path/to/ECG && \
CUDA_VISIBLE_DEVICES=0 python models/train.py --config configs/scheme_e.yaml --split random --server a6000 --run-tag batch1 2>&1 | tee logs/scheme_e_batch1.log && \
CUDA_VISIBLE_DEVICES=0 python models/train.py --config configs/scheme_i_direct.yaml --split random --server a6000 --run-tag batch1 2>&1 | tee logs/scheme_i_direct_batch1.log" Enter

# 窗口 1: GPU 1 — Scheme I-TwoStage
tmux new-window -t exp
tmux send-keys -t exp "conda activate torch && cd /path/to/ECG && \
CUDA_VISIBLE_DEVICES=1 python models/train.py --config configs/scheme_i_twostage.yaml --split random --server a6000 --run-tag batch1 2>&1 | tee logs/scheme_i_twostage_batch1.log" Enter

# 窗口 2: GPU 2 — Scheme F（EfficientPhys）
tmux new-window -t exp
tmux send-keys -t exp "conda activate torch && cd /path/to/ECG && \
CUDA_VISIBLE_DEVICES=2 python models/train.py --config configs/scheme_f.yaml --split random --server a6000 --run-tag batch1 2>&1 | tee logs/scheme_f_batch1.log" Enter

# 窗口 3: GPU 3 — Scheme H（PhysFormer，最重）
tmux new-window -t exp
tmux send-keys -t exp "conda activate torch && cd /path/to/ECG && \
CUDA_VISIBLE_DEVICES=3 python models/train.py --config configs/scheme_h.yaml --split random --server a6000 --run-tag batch1 2>&1 | tee logs/scheme_h_batch1.log" Enter
```

**操作步骤**：
```bash
# 先创建日志目录
mkdir -p logs

# 然后逐个复制上面的命令执行，或者直接用下面的脚本
bash scripts/run_batch1.sh
```

### 手动执行（逐个复制到每个 tmux 窗口）

**窗口 0 (GPU 0)**：
```bash
conda activate torch && cd /path/to/ECG
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python models/train.py \
  --config configs/scheme_e.yaml \
  --split random --server a6000 --run-tag batch1 \
  2>&1 | tee logs/scheme_e_batch1.log

CUDA_VISIBLE_DEVICES=0 python models/train.py \
  --config configs/scheme_i_direct.yaml \
  --split random --server a6000 --run-tag batch1 \
  2>&1 | tee logs/scheme_i_direct_batch1.log
```

**窗口 1 (GPU 1)**：
```bash
conda activate torch && cd /path/to/ECG
CUDA_VISIBLE_DEVICES=1 python models/train.py \
  --config configs/scheme_i_twostage.yaml \
  --split random --server a6000 --run-tag batch1 \
  2>&1 | tee logs/scheme_i_twostage_batch1.log
```

**窗口 2 (GPU 2)**：
```bash
conda activate torch && cd /path/to/ECG
CUDA_VISIBLE_DEVICES=2 python models/train.py \
  --config configs/scheme_f.yaml \
  --split random --server a6000 --run-tag batch1 \
  2>&1 | tee logs/scheme_f_batch1.log
```

**窗口 3 (GPU 3)**：
```bash
conda activate torch && cd /path/to/ECG
CUDA_VISIBLE_DEVICES=3 python models/train.py \
  --config configs/scheme_h.yaml \
  --split random --server a6000 --run-tag batch1 \
  2>&1 | tee logs/scheme_h_batch1.log
```

---

## 实验 2：Batch 2 数据（新高分辨率数据）

将新数据处理后放到 `training_data/batch2/` 目录（格式同 `training_data/samples/`），然后：

```bash
# 需要先为 batch2 生成质量报告
# 如果 batch2 格式和 batch1 一样，可以复用 quality_csv
# 否则需要重新跑 visualize.ipynb 的 Cell 0 生成新的 CSV

# 每个窗口同理，只改 --data-dir 和 --run-tag：
CUDA_VISIBLE_DEVICES=0 python models/train.py \
  --config configs/scheme_e.yaml \
  --split random --server a6000 \
  --data-dir training_data/batch2 \
  --run-tag batch2 \
  --quality-filter all \
  2>&1 | tee logs/scheme_e_batch2.log
```

注意：如果 batch2 没有质量 CSV，用 `--quality-filter all` 跳过过滤。

---

## 实验 3：Batch 1+2 合并

将两批数据合并到同一个目录（或创建符号链接）：

```bash
# 方法 1：创建合并目录
mkdir -p training_data/batch1+2
cp -r training_data/samples/pair_* training_data/batch1+2/
cp -r training_data/batch2/pair_* training_data/batch1+2/

# 方法 2：符号链接（节省空间）
mkdir -p training_data/batch1+2
ln -s $(pwd)/training_data/samples/pair_* training_data/batch1+2/
ln -s $(pwd)/training_data/batch2/pair_* training_data/batch1+2/

# 然后训练
CUDA_VISIBLE_DEVICES=0 python models/train.py \
  --config configs/scheme_e.yaml \
  --split random --server a6000 \
  --data-dir training_data/batch1+2 \
  --run-tag batch1+2 \
  --quality-filter all \
  2>&1 | tee logs/scheme_e_batch1+2.log
```

---

## 实验设置说明

### 当前默认配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据划分 | `--split random` | 随机划分（快速验证） |
| 早停方式 | 默认用 test set | DEBUG 模式，`--use-val` 切换到严格模式 |
| 质量过滤 | `good,moderate` | 排除 10 个 poor 样本，使用 88 个样本 |
| 质量 CSV | `eval_results/ppg_analysis_all_samples.csv` | 基于 PPG HR 检测准确度 |
| 服务器预设 | `--server a6000` | 自动调整 batch_size, AMP, 梯度累积 |
| Run tag | `--run-tag batch1` | 区分不同数据批次 |

### 质量过滤说明

质量分类基于 PPG 心率检测 vs ECG 标注心率的偏差：
- **good** (80 samples): HR 误差 < 10 BPM
- **moderate** (8 samples): HR 误差 10-16 BPM
- **poor** (10 samples): HR 误差 30-80 BPM（PPG 信号很弱或受干扰）

默认排除 poor 样本。如需包含全部，加 `--quality-filter all`。

### Checkpoint 保存位置

```
checkpoints/
├── scheme_e/
│   ├── random_good+moderate_p30_testval_batch1/
│   │   ├── best_model.pt        # 最佳模型权重
│   │   ├── config.yaml          # 完整训练配置（可复现）
│   │   ├── training_history.json # 逐 epoch 的 loss 和指标
│   │   ├── training_curves.png   # 训练曲线图
│   │   └── run_summary.json     # 最终结果汇总
│   └── random_good+moderate_p30_testval_batch2/
│       └── ...
├── scheme_f/
│   └── ...
└── scheme_h/
    └── ...
```

---

## 不需要单独跑 evaluate

训练脚本 `train.py` **训练结束后会自动**：
1. 加载 best_model.pt
2. 在 test set 上评估 RMSE / MAE / Pearson r
3. 打印结果到终端（同时被 `tee` 保存到日志）
4. 保存 `run_summary.json`（包含所有指标）

所以 **不需要** 额外跑 `run_eval.py`，除非你想用不同的 checkpoint 评估。

---

## 需要反馈给 Claude 的信息

训练完成后，提供以下内容（新开一个 Claude Code session 即可）：

### 1. 快速汇总（必须）

```bash
# 一键收集所有 run_summary.json
find checkpoints/ -name "run_summary.json" -exec echo "=== {} ===" \; -exec cat {} \;
```

或者直接复制每个方案最后打印的这行：
```
[scheme_e] Test results: RMSE=X.XXXX MAE=X.XXXX Pearson_r=X.XXXX
```

### 2. 训练曲线（有帮助）

把 `checkpoints/*/training_curves.png` 图片发过来（截图或文件）。

### 3. 如果出错

把完整报错信息贴过来，尤其是：
- `RuntimeError: ... shape mismatch` → 数据管线 bug
- `ImportError: ...` → 缺少依赖
- `CUDA out of memory` → 显存不够，需调小 batch
- `NaN` 出现 → 学习率太高或梯度爆炸

### 4. 可选：完整日志

```bash
# 打包所有日志和结果
tar czf experiment_results.tar.gz logs/ checkpoints/*/run_summary.json checkpoints/*/training_curves.png checkpoints/*/training_history.json
```

---

## 监控命令

```bash
# 查看 GPU 使用情况
watch -n 5 nvidia-smi

# 查看某个方案的最新日志
tail -f logs/scheme_e_batch1.log

# 切换 tmux 窗口
tmux select-window -t exp:0   # 窗口 0
tmux select-window -t exp:1   # 窗口 1
# 或 Ctrl+B 然后按数字键

# 查看所有方案的最新进度
for f in logs/*batch1*.log; do echo "=== $(basename $f) ==="; tail -1 "$f"; done
```

---

## 如果需要 3090 服务器

把所有命令中的 `--server a6000` 改为 `--server 3090`。区别：
- A6000: 更大 batch size, 不一定需要 AMP
- 3090: 更小 batch size, AMP 开启, 更多梯度累积
