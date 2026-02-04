"""
诊断脚本：检查输入数据质量和PPG-ECG相关性
"""
import os
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def extract_ppg_from_video(video_path, max_frames=300):
    """从视频提取PPG信号（RGB通道均值）"""
    cap = cv2.VideoCapture(video_path)
    rgb_means = []
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB均值
        rgb_mean = frame.mean(axis=(0, 1))[::-1]  # BGR to RGB
        rgb_means.append(rgb_mean)
        frame_count += 1
    cap.release()

    return np.array(rgb_means)  # (T, 3)


def analyze_sample(pair_dir, window_sec=10, video_fps=30, ecg_sr=250):
    """分析单个样本的PPG和ECG"""
    video_path = os.path.join(pair_dir, "video_0.mp4")
    ecg_path = os.path.join(pair_dir, "ecg.csv")

    # 提取PPG（前10秒）
    n_frames = int(window_sec * video_fps)
    ppg = extract_ppg_from_video(video_path, n_frames)

    # 加载ECG（前10秒）
    ecg_df = pd.read_csv(ecg_path)
    n_ecg = int(window_sec * ecg_sr)
    ecg = ecg_df["ecg_counts_filt_monitor"].values[:n_ecg]

    # Z-normalize
    ppg_norm = (ppg - ppg.mean(axis=0)) / (ppg.std(axis=0) + 1e-8)
    ecg_norm = (ecg - ecg.mean()) / (ecg.std() + 1e-8)

    return ppg_norm, ecg_norm


def check_ppg_quality(ppg):
    """检查PPG信号质量"""
    # 使用红色通道（PPG主要成分）
    red_channel = ppg[:, 0]

    # 计算信号统计
    stats = {
        "mean": red_channel.mean(),
        "std": red_channel.std(),
        "min": red_channel.min(),
        "max": red_channel.max(),
        "range": red_channel.max() - red_channel.min(),
    }

    # 检测周期性（使用自相关）
    autocorr = np.correlate(red_channel, red_channel, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]  # normalize

    # 找第一个峰（排除0点）
    peaks, _ = signal.find_peaks(autocorr[10:], height=0.3)
    if len(peaks) > 0:
        period_frames = peaks[0] + 10
        estimated_hr = 60 * 30 / period_frames  # 假设30fps
        stats["estimated_hr"] = estimated_hr
        stats["has_periodicity"] = True
    else:
        stats["estimated_hr"] = None
        stats["has_periodicity"] = False

    return stats


def main():
    samples_dir = "training_data/samples"
    pairs = sorted([d for d in os.listdir(samples_dir) if d.startswith("pair_")])[:5]

    print("=" * 60)
    print("PPG-ECG 诊断分析")
    print("=" * 60)

    fig, axes = plt.subplots(len(pairs), 2, figsize=(14, 3*len(pairs)))

    for i, pair in enumerate(pairs):
        pair_dir = os.path.join(samples_dir, pair)
        print(f"\n--- {pair} ---")

        try:
            ppg, ecg = analyze_sample(pair_dir)
            stats = check_ppg_quality(ppg)

            print(f"PPG统计: std={stats['std']:.4f}, range={stats['range']:.4f}")
            print(f"周期性: {stats['has_periodicity']}, 估计心率: {stats['estimated_hr']}")

            # 计算PPG红色通道与ECG的相关性
            # 先对齐长度（resample PPG到ECG长度）
            ppg_red = ppg[:, 0]
            ppg_resampled = signal.resample(ppg_red, len(ecg))
            corr, _ = pearsonr(ppg_resampled, ecg)
            print(f"PPG(Red)-ECG相关性: r={corr:.4f}")

            # 绘图
            t_ppg = np.arange(len(ppg)) / 30
            t_ecg = np.arange(len(ecg)) / 250

            ax1, ax2 = axes[i] if len(pairs) > 1 else axes

            ax1.plot(t_ppg, ppg[:, 0], 'r-', label='R', alpha=0.8)
            ax1.plot(t_ppg, ppg[:, 1], 'g-', label='G', alpha=0.5)
            ax1.plot(t_ppg, ppg[:, 2], 'b-', label='B', alpha=0.5)
            ax1.set_ylabel(f'{pair}\nPPG')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)

            ax2.plot(t_ecg, ecg, 'b-', linewidth=0.5)
            ax2.set_ylabel('ECG')
            ax2.grid(True, alpha=0.3)

            if i == 0:
                ax1.set_title('PPG Signal (RGB channels)')
                ax2.set_title('ECG Signal')
            if i == len(pairs) - 1:
                ax1.set_xlabel('Time (s)')
                ax2.set_xlabel('Time (s)')

        except Exception as e:
            print(f"错误: {e}")

    plt.tight_layout()
    save_path = "eval_results/visualize/ppg_ecg_comparison.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\n图片已保存: {save_path}")

    # 汇总分析
    print("\n" + "=" * 60)
    print("关键发现:")
    print("=" * 60)
    print("""
1. PPG和ECG波形形态完全不同:
   - PPG是平滑的脉搏波（类似正弦波）
   - ECG有尖锐的QRS波群

2. 从PPG直接重建ECG波形是极其困难的任务:
   - 两者物理机制不同（血容量 vs 电活动）
   - 形态差异巨大
   - 需要学习高度非线性的映射关系

3. 建议:
   - 降低任务难度：先尝试预测心率而非完整波形
   - 使用更多数据（当前仅~900个训练窗口）
   - 尝试更复杂的模型（如Transformer）
   - 参考相关论文的数据增强策略
""")


if __name__ == "__main__":
    main()
