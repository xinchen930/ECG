"""
Quick test: What happens if we bandpass filter the PPG in the dataset pipeline?
Simulate what the model would see with bandpass filtering and check if correlation improves.
Also check the sign of correlation (PPG and ECG may be anti-correlated).
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import cv2
from scipy import signal as scipy_signal
from scipy.stats import pearsonr
from pathlib import Path

sys.path.insert(0, '/home/xinchen/ECG')
sys.path.insert(0, '/home/xinchen/ECG/models')

SAMPLES_DIR = '/home/xinchen/ECG/training_data/samples'


def simulate_model_input(video_path, ecg_path, fps=30, ecg_sr=250,
                         window_sec=10, stride_sec=5,
                         skip_first_sec=2):
    """
    Simulate what the model would see with different preprocessing.
    Returns per-window correlations for:
    1. Current approach: raw red mean -> z-norm
    2. Proposed: bandpass filter -> z-norm
    3. Proposed: skip first N seconds + bandpass + z-norm
    4. Proposed: frame differences
    """
    # Read video
    cap = cv2.VideoCapture(video_path)
    fps_actual = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    red_means = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        red_means.append(frame[:, :, 2].astype(np.float64).mean())
    cap.release()
    red = np.array(red_means)

    # Read ECG
    ecg_df = pd.read_csv(ecg_path)
    ecg = ecg_df['ecg_counts_filt_monitor'].values.astype(np.float64)

    # ECG global stats (for per-pair normalization)
    ecg_mean = ecg.mean()
    ecg_std = ecg.std() + 1e-8

    window_ppg = int(window_sec * fps)
    stride_ppg = int(stride_sec * fps)
    window_ecg = int(window_sec * ecg_sr)
    stride_ecg = int(stride_sec * ecg_sr)

    # Precompute bandpassed signal (full video)
    sos_bp = scipy_signal.butter(4, [0.5, min(5.0, fps/2 - 0.1)], btype='bandpass', fs=fps, output='sos')
    red_bp = scipy_signal.sosfiltfilt(sos_bp, red)

    # Frame differences
    red_diff = np.diff(red)

    results = {'raw': [], 'bandpass': [], 'diff': [], 'bp_skip': []}

    n_windows = min(
        (len(red) - window_ppg) // stride_ppg + 1,
        (len(ecg) - window_ecg) // stride_ecg + 1,
    )

    skip_frames = int(skip_first_sec * fps)

    for w in range(max(0, n_windows)):
        ppg_start = w * stride_ppg
        ecg_start = w * stride_ecg

        # Raw approach (current)
        ppg_raw = red[ppg_start:ppg_start + window_ppg]
        ecg_win = ecg[ecg_start:ecg_start + window_ecg]

        if len(ppg_raw) < window_ppg or len(ecg_win) < window_ecg:
            break

        # ECG normalization (per-pair, same as dataset.py)
        ecg_norm = (ecg_win - ecg_mean) / ecg_std

        # Downsample ECG to 30 Hz for correlation
        ecg_down = scipy_signal.resample(ecg_norm, window_ppg)

        # Method 1: Raw z-norm (current)
        ppg_n = (ppg_raw - ppg_raw.mean()) / (ppg_raw.std() + 1e-8)
        r_raw, _ = pearsonr(ppg_n, ecg_down)
        results['raw'].append(r_raw)

        # Method 2: Bandpass filtered then z-norm
        ppg_bp = red_bp[ppg_start:ppg_start + window_ppg]
        ppg_bp_n = (ppg_bp - ppg_bp.mean()) / (ppg_bp.std() + 1e-8)
        r_bp, _ = pearsonr(ppg_bp_n, ecg_down)
        results['bandpass'].append(r_bp)

        # Method 3: Frame differences (naturally removes drift)
        if ppg_start + window_ppg <= len(red_diff):
            ppg_d = red_diff[ppg_start:ppg_start + window_ppg - 1]
            ppg_d_n = (ppg_d - ppg_d.mean()) / (ppg_d.std() + 1e-8)
            ecg_down_d = scipy_signal.resample(ecg_norm, window_ppg - 1)
            r_d, _ = pearsonr(ppg_d_n, ecg_down_d)
            results['diff'].append(r_d)

        # Method 4: Bandpass + skip first N seconds
        if ppg_start >= skip_frames:
            r_bp2, _ = pearsonr(ppg_bp_n, ecg_down)
            results['bp_skip'].append(r_bp2)

    return results


def main():
    print("=" * 70)
    print("  Bandpass Filter Impact Test")
    print("=" * 70)

    all_pairs = sorted(Path(SAMPLES_DIR).iterdir())

    all_results = {'raw': [], 'bandpass': [], 'diff': [], 'bp_skip': []}
    per_sample = []

    for pair_dir in all_pairs:
        meta_path = pair_dir / "metadata.json"
        video_path = str(pair_dir / "video_0.mp4")
        ecg_path = str(pair_dir / "ecg.csv")
        if not meta_path.exists() or not os.path.exists(video_path) or not os.path.exists(ecg_path):
            continue

        results = simulate_model_input(video_path, ecg_path)

        for key in all_results:
            all_results[key].extend(results[key])

        with open(meta_path) as f:
            meta = json.load(f)

        per_sample.append({
            'pair': pair_dir.name,
            'mean_raw': np.mean(results['raw']) if results['raw'] else 0,
            'mean_bp': np.mean(results['bandpass']) if results['bandpass'] else 0,
            'mean_diff': np.mean(results['diff']) if results['diff'] else 0,
            'n_windows': len(results['raw']),
        })

    # Overall statistics
    print(f"\n{'='*60}")
    print(f"  Overall Statistics (across ALL windows)")
    print(f"{'='*60}")

    for method, name in [
        ('raw', 'Current (raw z-norm)'),
        ('bandpass', 'Bandpass (0.5-5Hz) + z-norm'),
        ('diff', 'Frame differences + z-norm'),
        ('bp_skip', 'Bandpass + skip first 2s'),
    ]:
        arr = np.array(all_results[method])
        if len(arr) == 0:
            continue
        print(f"\n  {name}:")
        print(f"    N windows: {len(arr)}")
        print(f"    Mean r: {arr.mean():+.4f}")
        print(f"    Mean |r|: {np.abs(arr).mean():.4f}")
        print(f"    Median |r|: {np.median(np.abs(arr)):.4f}")
        print(f"    Std r: {arr.std():.4f}")
        print(f"    |r| > 0.1: {(np.abs(arr) > 0.1).sum()}/{len(arr)} ({(np.abs(arr) > 0.1).sum()/len(arr)*100:.0f}%)")
        print(f"    |r| > 0.2: {(np.abs(arr) > 0.2).sum()}/{len(arr)} ({(np.abs(arr) > 0.2).sum()/len(arr)*100:.0f}%)")
        print(f"    |r| > 0.3: {(np.abs(arr) > 0.3).sum()}/{len(arr)} ({(np.abs(arr) > 0.3).sum()/len(arr)*100:.0f}%)")
        print(f"    |r| > 0.5: {(np.abs(arr) > 0.5).sum()}/{len(arr)} ({(np.abs(arr) > 0.5).sum()/len(arr)*100:.0f}%)")

    # Sign analysis
    print(f"\n{'='*60}")
    print(f"  Sign Analysis: Is PPG-ECG correlation positive or negative?")
    print(f"{'='*60}")
    bp = np.array(all_results['bandpass'])
    n_pos = (bp > 0.1).sum()
    n_neg = (bp < -0.1).sum()
    print(f"  Bandpass windows with r > +0.1: {n_pos}")
    print(f"  Bandpass windows with r < -0.1: {n_neg}")
    print(f"  --> {'More negative correlations' if n_neg > n_pos else 'More positive correlations'}")
    print(f"  --> This means PPG and ECG are {'anti-correlated' if n_neg > n_pos else 'correlated'}")
    print(f"      (PPG valley aligns with ECG R-peak if anti-correlated)")

    # Key insight: even with bandpass, the correlation is weak
    print(f"\n{'='*60}")
    print(f"  KEY INSIGHT")
    print(f"{'='*60}")
    raw_mean = np.abs(np.array(all_results['raw'])).mean()
    bp_mean = np.abs(np.array(all_results['bandpass'])).mean()
    improvement = bp_mean / (raw_mean + 1e-10)
    print(f"  Bandpass filtering improves |r| by {improvement:.1f}x (from {raw_mean:.4f} to {bp_mean:.4f})")
    print(f"  BUT even with bandpass, the mean |r| = {bp_mean:.4f} is STILL LOW")
    print(f"  This suggests the fundamental challenge:")
    print(f"    - PPG and ECG have DIFFERENT waveform morphologies")
    print(f"    - PPG is a smoothed, delayed version of the cardiac pressure wave")
    print(f"    - ECG has sharp QRS complexes that PPG does not have")
    print(f"    - A 1D model must LEARN this non-linear transformation")
    print(f"    - Low r does NOT mean the task is impossible —")
    print(f"      it means the model needs to learn complex temporal transformations")
    print(f"  HOWEVER: with raw r ≈ {raw_mean:.3f}, the model has NOTHING to start with")
    print(f"  Bandpass filtering gives it {bp_mean:.3f}, which is at least a signal to learn from")


if __name__ == "__main__":
    main()
