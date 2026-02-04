"""
分析视频RGB信号与ECG心率的关系
检查视频压缩是否导致PPG信息丢失
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
from scipy import signal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def get_video_info(video_path):
    """获取视频编码信息"""
    cap = cv2.VideoCapture(video_path)
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
        "bitrate": cap.get(cv2.CAP_PROP_BITRATE) if hasattr(cv2, 'CAP_PROP_BITRATE') else None,
    }
    # 解码fourcc
    fourcc_str = "".join([chr((info["fourcc"] >> 8 * i) & 0xFF) for i in range(4)])
    info["codec"] = fourcc_str

    # 文件大小
    info["file_size_mb"] = os.path.getsize(video_path) / (1024 * 1024)
    info["duration_s"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    info["bitrate_kbps"] = (info["file_size_mb"] * 8 * 1024) / info["duration_s"] if info["duration_s"] > 0 else 0

    cap.release()
    return info


def extract_rgb_signal(video_path, max_frames=None):
    """从视频提取RGB通道均值时间序列"""
    cap = cv2.VideoCapture(video_path)

    if max_frames is None:
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rgb_signals = []
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV读取的是BGR格式
        b_mean = frame[:, :, 0].mean()
        g_mean = frame[:, :, 1].mean()
        r_mean = frame[:, :, 2].mean()

        rgb_signals.append([r_mean, g_mean, b_mean])
        frame_idx += 1

    cap.release()
    return np.array(rgb_signals)  # (T, 3) - R, G, B


def bandpass_filter(sig, fs, lowcut=0.7, highcut=4.0, order=3):
    """带通滤波器 (0.7-4Hz 对应 42-240 BPM)"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, sig)


def detect_peaks_and_hr(sig, fs, min_distance_s=0.4):
    """检测峰值并计算心率"""
    min_distance = int(min_distance_s * fs)
    peaks, properties = signal.find_peaks(sig, distance=min_distance, prominence=0.1*sig.std())

    if len(peaks) < 2:
        return peaks, None, None

    # 计算RR间期和心率
    rr_intervals = np.diff(peaks) / fs  # 秒
    hr_per_beat = 60 / rr_intervals  # BPM

    return peaks, rr_intervals, hr_per_beat


def analyze_periodicity(sig, fs):
    """使用FFT分析信号周期性"""
    # 去除直流分量
    sig = sig - sig.mean()

    # FFT
    n = len(sig)
    freqs = np.fft.rfftfreq(n, 1/fs)
    fft_mag = np.abs(np.fft.rfft(sig))

    # 找心率范围内的主频 (0.7-3.5 Hz = 42-210 BPM)
    hr_mask = (freqs >= 0.7) & (freqs <= 3.5)
    if not hr_mask.any():
        return None, None, 0

    hr_freqs = freqs[hr_mask]
    hr_mags = fft_mag[hr_mask]

    peak_idx = np.argmax(hr_mags)
    dominant_freq = hr_freqs[peak_idx]
    dominant_hr = dominant_freq * 60  # BPM

    # 计算信噪比（主峰能量 vs 总能量）
    peak_power = hr_mags[peak_idx] ** 2
    total_power = (hr_mags ** 2).sum()
    snr = peak_power / total_power if total_power > 0 else 0

    return dominant_freq, dominant_hr, snr


def get_ecg_hr(ecg_path, ecg_sr=250):
    """从ECG计算心率"""
    df = pd.read_csv(ecg_path)
    ecg = df["ecg_counts_filt_monitor"].values

    # 检测R波
    ecg_filtered = bandpass_filter(ecg, ecg_sr, 5, 15)  # QRS频率范围
    peaks, _ = signal.find_peaks(ecg_filtered, distance=int(0.4*ecg_sr), prominence=ecg_filtered.std())

    if len(peaks) < 2:
        return None, None

    rr_intervals = np.diff(peaks) / ecg_sr
    hr_values = 60 / rr_intervals

    return peaks, hr_values


def main():
    samples_dir = "training_data/samples"
    pairs = sorted([d for d in os.listdir(samples_dir) if d.startswith("pair_")])

    # 只分析前5个样本
    pairs = pairs[:5]

    print("=" * 70)
    print("视频PPG信号与ECG心率关系分析")
    print("=" * 70)

    results = []

    for pair in pairs:
        pair_dir = os.path.join(samples_dir, pair)
        video_path = os.path.join(pair_dir, "video_0.mp4")
        ecg_path = os.path.join(pair_dir, "ecg.csv")
        meta_path = os.path.join(pair_dir, "metadata.json")

        if not os.path.exists(video_path):
            continue

        print(f"\n{'='*70}")
        print(f"分析: {pair}")
        print("=" * 70)

        # 1. 视频信息
        vinfo = get_video_info(video_path)
        print(f"\n[视频信息]")
        print(f"  分辨率: {vinfo['width']}x{vinfo['height']}")
        print(f"  帧数: {vinfo['frame_count']}, FPS: {vinfo['fps']:.1f}")
        print(f"  时长: {vinfo['duration_s']:.1f}s")
        print(f"  文件大小: {vinfo['file_size_mb']:.2f} MB")
        print(f"  码率: {vinfo['bitrate_kbps']:.0f} kbps")
        print(f"  编码: {vinfo['codec']}")

        # 2. 提取RGB信号（前30秒）
        max_frames = min(int(30 * vinfo['fps']), vinfo['frame_count'])
        rgb = extract_rgb_signal(video_path, max_frames)
        fps = vinfo['fps']

        print(f"\n[RGB信号统计] (前30秒)")
        for i, ch in enumerate(['R', 'G', 'B']):
            print(f"  {ch}: mean={rgb[:,i].mean():.1f}, std={rgb[:,i].std():.4f}, "
                  f"range=[{rgb[:,i].min():.1f}, {rgb[:,i].max():.1f}]")

        # 3. 带通滤波并分析周期性
        print(f"\n[PPG周期性分析] (0.7-4Hz带通滤波后)")
        ppg_results = {}
        for i, ch in enumerate(['R', 'G', 'B']):
            raw = rgb[:, i]
            # Z-normalize
            raw_norm = (raw - raw.mean()) / (raw.std() + 1e-8)
            # 带通滤波
            filtered = bandpass_filter(raw_norm, fps, 0.7, 4.0)

            # FFT分析
            dom_freq, dom_hr, snr = analyze_periodicity(filtered, fps)
            ppg_results[ch] = {
                'raw': raw,
                'filtered': filtered,
                'dominant_hr': dom_hr,
                'snr': snr
            }

            if dom_hr:
                print(f"  {ch}: 主频心率={dom_hr:.1f} BPM, SNR={snr:.3f}")
            else:
                print(f"  {ch}: 未检测到周期性信号")

        # 4. ECG心率
        print(f"\n[ECG心率]")
        ecg_peaks, ecg_hr = get_ecg_hr(ecg_path)
        if ecg_hr is not None and len(ecg_hr) > 0:
            print(f"  平均心率: {ecg_hr.mean():.1f} BPM")
            print(f"  心率范围: [{ecg_hr.min():.1f}, {ecg_hr.max():.1f}] BPM")

        # 读取标注的心率
        with open(meta_path) as f:
            meta = json.load(f)
        annotated_hr = meta.get("heart_rate")
        print(f"  标注心率: {annotated_hr} BPM")

        # 5. 对比
        print(f"\n[PPG vs ECG 心率对比]")
        red_hr = ppg_results['R']['dominant_hr']
        if red_hr and ecg_hr is not None:
            ecg_mean_hr = ecg_hr.mean()
            error = abs(red_hr - ecg_mean_hr)
            error_pct = error / ecg_mean_hr * 100
            print(f"  红色通道PPG心率: {red_hr:.1f} BPM")
            print(f"  ECG平均心率: {ecg_mean_hr:.1f} BPM")
            print(f"  误差: {error:.1f} BPM ({error_pct:.1f}%)")

            if error_pct < 10:
                print(f"  ✓ 心率匹配良好！视频包含有效PPG信息")
            elif error_pct < 20:
                print(f"  △ 心率大致匹配，视频可能有部分PPG信息")
            else:
                print(f"  ✗ 心率不匹配，视频PPG信息可能严重失真")

        results.append({
            'pair': pair,
            'video_info': vinfo,
            'ppg_hr_red': red_hr,
            'ppg_snr_red': ppg_results['R']['snr'],
            'ecg_hr': ecg_hr.mean() if ecg_hr is not None else None,
            'annotated_hr': annotated_hr,
        })

    # 绘制详细图
    print("\n\n生成可视化图表...")
    fig, axes = plt.subplots(len(pairs), 4, figsize=(16, 3*len(pairs)))
    if len(pairs) == 1:
        axes = axes.reshape(1, -1)

    for idx, pair in enumerate(pairs):
        pair_dir = os.path.join(samples_dir, pair)
        video_path = os.path.join(pair_dir, "video_0.mp4")
        ecg_path = os.path.join(pair_dir, "ecg.csv")

        vinfo = get_video_info(video_path)
        fps = vinfo['fps']
        max_frames = min(int(10 * fps), vinfo['frame_count'])  # 10秒

        rgb = extract_rgb_signal(video_path, max_frames)
        t_video = np.arange(len(rgb)) / fps

        # ECG (10秒)
        df = pd.read_csv(ecg_path)
        ecg = df["ecg_counts_filt_monitor"].values[:2500]
        t_ecg = np.arange(len(ecg)) / 250

        # 红色通道原始
        ax = axes[idx, 0]
        ax.plot(t_video, rgb[:, 0], 'r-', linewidth=0.5)
        ax.set_ylabel(f'{pair}\nRed Raw')
        ax.set_title('Red Channel (Raw)' if idx == 0 else '')
        ax.grid(True, alpha=0.3)

        # 红色通道滤波后
        ax = axes[idx, 1]
        red_norm = (rgb[:, 0] - rgb[:, 0].mean()) / (rgb[:, 0].std() + 1e-8)
        red_filt = bandpass_filter(red_norm, fps, 0.7, 4.0)
        ax.plot(t_video, red_filt, 'r-', linewidth=0.5)
        ax.set_ylabel('Red Filtered')
        ax.set_title('Red Channel (0.7-4Hz BP)' if idx == 0 else '')
        ax.grid(True, alpha=0.3)

        # FFT
        ax = axes[idx, 2]
        n = len(red_filt)
        freqs = np.fft.rfftfreq(n, 1/fps)
        fft_mag = np.abs(np.fft.rfft(red_filt))
        hr_mask = (freqs >= 0.5) & (freqs <= 4)
        ax.plot(freqs[hr_mask] * 60, fft_mag[hr_mask], 'b-')  # 转换为BPM
        ax.set_ylabel('FFT Magnitude')
        ax.set_xlabel('Heart Rate (BPM)')
        ax.set_title('Frequency Spectrum' if idx == 0 else '')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(40, 200)

        # ECG
        ax = axes[idx, 3]
        ecg_norm = (ecg - ecg.mean()) / (ecg.std() + 1e-8)
        ax.plot(t_ecg, ecg_norm, 'b-', linewidth=0.5)
        ax.set_ylabel('ECG')
        ax.set_title('ECG Signal' if idx == 0 else '')
        ax.grid(True, alpha=0.3)

        if idx == len(pairs) - 1:
            for a in axes[idx]:
                a.set_xlabel('Time (s)')

    plt.tight_layout()
    save_path = "eval_results/visualize/ppg_ecg_analysis.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图片已保存: {save_path}")

    # 汇总
    print("\n" + "=" * 70)
    print("汇总分析")
    print("=" * 70)

    valid_pairs = [r for r in results if r['ppg_hr_red'] and r['ecg_hr']]
    if valid_pairs:
        errors = [abs(r['ppg_hr_red'] - r['ecg_hr']) for r in valid_pairs]
        print(f"有效样本数: {len(valid_pairs)}/{len(results)}")
        print(f"PPG-ECG心率误差: 平均={np.mean(errors):.1f} BPM, 最大={np.max(errors):.1f} BPM")

        good = sum(1 for e in errors if e < 10)
        print(f"心率误差<10BPM的样本: {good}/{len(valid_pairs)}")

    avg_bitrate = np.mean([r['video_info']['bitrate_kbps'] for r in results])
    avg_snr = np.mean([r['ppg_snr_red'] for r in results if r['ppg_snr_red']])
    print(f"\n平均视频码率: {avg_bitrate:.0f} kbps")
    print(f"平均PPG信号SNR: {avg_snr:.3f}")

    if avg_bitrate < 500:
        print("\n⚠️ 视频码率较低 (<500kbps)，可能导致PPG信息丢失")
    if avg_snr < 0.1:
        print("⚠️ PPG信号SNR较低，周期性不明显")


if __name__ == "__main__":
    main()
