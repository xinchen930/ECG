# 下一个 Session 的任务

> 更新时间：2025-02-05
> 背景文档：`docs/2025-02-05_analysis.md`

---

## 🎯 立即执行的任务

**目标**：在 BIDMC 公开数据集上验证 PPG2ECG 模型能否 work

**为什么先做这个**：
- 控制变量：排除数据质量问题
- BIDMC 是高质量接触式 PPG（125Hz），如果模型在这上面都不 work，说明模型有问题
- 成功后再迁移到我们的视频数据

---

## 详细步骤拆解

### Step 1: 下载 BIDMC 数据集

**命令**：
```bash
# 创建目录
mkdir -p external_data/bidmc

# 方式 1：使用 wget（需要 PhysioNet 账号凭证）
cd external_data/bidmc
wget -r -N -c -np --user=<username> --password=<password> \
    https://physionet.org/files/bidmc/1.0.0/

# 方式 2：手动下载
# 访问 https://physionet.org/content/bidmc/1.0.0/
# 下载 bidmc_csv.zip 或 bidmc_data/ 文件夹
```

**预期结果**：
- 53 个受试者的数据文件
- 每个文件包含 8 分钟的 PPG + ECG（125 Hz）
- 文件格式：CSV 或 WFDB 格式

**验证命令**：
```bash
ls external_data/bidmc/ | head -10
```

---

### Step 2: 数据预处理脚本

**创建文件**：`scripts/prepare_bidmc.py`

**功能需求**：
1. 读取 BIDMC 数据文件（CSV 或 WFDB 格式）
2. 提取 PPG 和 ECG 信号
3. 重采样到 100 Hz（PPG2ECG 模型要求）
4. 切成 2 秒窗口（200 点）
5. 归一化到 [-1, 1]
6. 按 80/20 划分训练/测试集（按受试者划分，避免数据泄露）
7. 保存为 PyTorch 格式

**输入输出**：
```
输入：external_data/bidmc/
输出：external_data/bidmc_processed/
  ├── train.pt  (42 subjects × ~240 windows each)
  ├── test.pt   (11 subjects × ~240 windows each)
  └── metadata.json  (数据统计信息)
```

**关键代码框架**：
```python
import numpy as np
import torch
from scipy.signal import resample
from scipy.io import loadmat  # 或 wfdb 库

def load_bidmc_subject(file_path):
    """加载单个受试者的 PPG 和 ECG 数据"""
    # TODO: 根据实际文件格式实现
    pass

def preprocess_signal(signal, target_fs=100, window_sec=2):
    """预处理信号：重采样、切窗、归一化"""
    # 重采样
    # 切窗
    # 归一化到 [-1, 1]
    pass

def main():
    # 1. 加载所有受试者数据
    # 2. 划分训练/测试集（按受试者）
    # 3. 预处理并保存
    pass
```

---

### Step 3: 移植 PPG2ECG 模型

**参考仓库**：https://github.com/james77777778/ppg2ecg-pytorch

**创建文件**：`models/ppg2ecg.py`

**模型架构要点**：
```python
# 论文：Reconstructing QRS Complex from PPG by Transformed Attentional Neural Networks

class PPG2ECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Sequence Transformer: 将 PPG 序列映射到特征空间
        # 2. Attention Network: 关注 QRS 相关的时间点
        # 3. Decoder: 生成 ECG 波形

    def forward(self, ppg):
        # ppg: [batch, 1, 200]  范围 [-1, 1]
        # return: [batch, 1, 200]  ECG 波形
        pass
```

**输入输出格式**：
- 输入：`[batch, 1, 200]`，范围 `[-1, 1]`
- 输出：`[batch, 1, 200]`，ECG 波形

**需要检查的论文细节**：
- [ ] 损失函数：MSE？还是 QRS-enhanced loss？
- [ ] 学习率和优化器配置
- [ ] 是否需要数据增强
- [ ] PPG/ECG 对齐方式（是否需要 PAT 校正）

---

### Step 4: 在 BIDMC 上训练

**创建配置文件**：`configs/ppg2ecg_bidmc.yaml`

```yaml
model:
  name: ppg2ecg
  input_length: 200  # 2秒 @ 100Hz

data:
  train_path: external_data/bidmc_processed/train.pt
  test_path: external_data/bidmc_processed/test.pt
  batch_size: 64

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: Adam
  scheduler: CosineAnnealingLR
  patience: 20  # early stopping

loss:
  type: MSE  # 或 QRS-enhanced

output:
  checkpoint_dir: checkpoints/ppg2ecg_bidmc/
  log_dir: logs/ppg2ecg_bidmc/
```

**创建训练脚本**：`models/train_ppg2ecg.py`

**运行命令**：
```bash
python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml
```

**预期指标**（参考论文）：
- RMSE < 0.3
- Pearson r > 0.8
- MAE < 0.2

---

### Step 5: 判断结果

**结果记录文件**：`eval_results/bidmc_baseline.md`

```markdown
# BIDMC Baseline Results

## 实验配置
- 模型：PPG2ECG-PyTorch
- 数据：BIDMC (42 train / 11 test subjects)
- 训练轮数：XX epochs
- 最佳验证 loss：XX

## 测试集结果
- RMSE: XX
- MAE: XX
- Pearson r: XX

## 可视化
[附上 PPG vs 预测 ECG vs 真实 ECG 的对比图]

## 结论
[根据结果判断模型是否 work]
```

**判断逻辑**：
```
如果 BIDMC 上成功（Pearson r > 0.7）：
  ✅ 模型 OK，继续 Step 6 在我们数据上验证

如果 BIDMC 上失败（Pearson r < 0.5）：
  ❌ 检查复现细节：
    - 损失函数是否正确？
    - 数据预处理是否正确？
    - 学习率是否合适？
  → 或尝试 CardioGAN（有预训练权重）
```

---

### Step 6: 在我们的数据上验证

**创建文件**：`scripts/video_to_ppg.py`

**功能需求**：
1. 读取 `video_0.mp4`
2. 提取绿色通道均值（30 Hz）
3. 重采样到 100 Hz
4. 切成 2 秒窗口（200 点）
5. 归一化到 [-1, 1]

**代码框架**：
```python
import cv2
import numpy as np
from scipy.signal import resample

def extract_ppg_from_video(video_path, skip_start=30):
    """从视频提取绿色通道均值作为 PPG"""
    cap = cv2.VideoCapture(video_path)

    # 跳过开头几帧
    for _ in range(skip_start):
        cap.read()

    green_values = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 提取绿色通道均值
        green_values.append(frame[:, :, 1].mean())

    cap.release()
    return np.array(green_values)  # shape: (T,)

def preprocess_ppg(ppg_30hz, target_fs=100, window_sec=2):
    """重采样、切窗、归一化"""
    # 30Hz → 100Hz
    ratio = target_fs / 30
    ppg_100hz = resample(ppg_30hz, int(len(ppg_30hz) * ratio))

    # 切窗
    window_size = target_fs * window_sec  # 200
    windows = []
    for i in range(0, len(ppg_100hz) - window_size, window_size // 2):  # 50% overlap
        window = ppg_100hz[i:i + window_size]
        # 归一化到 [-1, 1]
        window = (window - window.mean()) / (window.std() + 1e-8)
        window = np.clip(window, -3, 3) / 3  # 截断到 [-1, 1]
        windows.append(window)

    return np.array(windows)
```

**创建配置文件**：`configs/ppg2ecg_our_data.yaml`

**运行命令**：
```bash
# 先提取 PPG
python scripts/video_to_ppg.py

# 训练/评估
python models/train_ppg2ecg.py --config configs/ppg2ecg_our_data.yaml
```

**结果记录**：`eval_results/ppg2ecg_our_data.md`

---

## 预期时间线

| 步骤 | 预计时间 | 累计时间 |
|------|----------|----------|
| Step 1: 下载 BIDMC | 5-10 min | 10 min |
| Step 2: 预处理脚本 | 15-20 min | 30 min |
| Step 3: 移植模型 | 30-45 min | 1h 15min |
| Step 4: BIDMC 训练 | 30-60 min | 2h 15min |
| Step 5: 判断结果 | 5 min | 2h 20min |
| Step 6: 我们数据 | 20-30 min | 2h 50min |

---

## 下一个 Session 的 Prompt

**复制以下内容开始下一个 session**：

---

```
我们在做 Video → ECG 重建项目。经过分析，决定先验证 PPG → ECG 这一环是否可行。

**本次任务**：在 BIDMC 公开数据集上验证 PPG2ECG 模型

**详细步骤见**：`docs/PROMPT_ppg2ecg_implementation.md`

**请按以下顺序执行**：

1. 下载 BIDMC 数据集到 `external_data/bidmc/`
   - 来源：https://physionet.org/content/bidmc/1.0.0/

2. 创建 `scripts/prepare_bidmc.py` 预处理数据
   - 重采样到 100Hz，切 2 秒窗口，归一化 [-1,1]
   - 按受试者划分 train/test

3. 从 https://github.com/james77777778/ppg2ecg-pytorch 移植模型到 `models/ppg2ecg.py`
   - 输入：[batch, 1, 200]，范围 [-1, 1]
   - 输出：[batch, 1, 200] ECG 波形

4. 创建 `configs/ppg2ecg_bidmc.yaml` 配置文件

5. 创建 `models/train_ppg2ecg.py` 训练脚本

6. 在 BIDMC 上训练并记录结果到 `eval_results/bidmc_baseline.md`
   - 预期：Pearson r > 0.7

7. 如果成功（r > 0.7），继续：
   - 创建 `scripts/video_to_ppg.py` 提取视频 PPG
   - 在我们数据上验证
   - 记录到 `eval_results/ppg2ecg_our_data.md`

**注意**：
- 保持论文细节，如果省略要明确说明
- 每完成一步，更新 TODO 列表
- 如果遇到问题，先检查数据预处理是否正确

**判断逻辑**：
- BIDMC 成功 + 我们数据失败 → 数据质量问题
- BIDMC 失败 → 模型复现问题，换 CardioGAN
```

---

## 备选方案：CardioGAN

如果 PPG2ECG 效果不好，尝试 CardioGAN：

- **仓库**：https://github.com/pritamqu/ppg2ecg-cardiogan
- **优势**：有预训练权重，在 4 个数据集上验证过
- **缺点**：TensorFlow 2.2，需要单独环境

```bash
# 创建 TensorFlow 环境
conda create -n tf python=3.8
conda activate tf
pip install tensorflow==2.2.0

# 克隆仓库
git clone https://github.com/pritamqu/ppg2ecg-cardiogan.git external_repos/cardiogan
```

---

## 文件清单（预期输出）

```
external_data/
├── bidmc/                          # 下载的原始数据
└── bidmc_processed/                # 预处理后的数据
    ├── train.pt
    ├── test.pt
    └── metadata.json

scripts/
├── prepare_bidmc.py                # BIDMC 预处理
└── video_to_ppg.py                 # 视频转 PPG

models/
├── ppg2ecg.py                      # PPG2ECG 模型
└── train_ppg2ecg.py                # 训练脚本

configs/
├── ppg2ecg_bidmc.yaml              # BIDMC 配置
└── ppg2ecg_our_data.yaml           # 我们数据配置

eval_results/
├── bidmc_baseline.md               # BIDMC 结果
└── ppg2ecg_our_data.md             # 我们数据结果
```
