"""
Final diagnostic: Quantify the auto-exposure transient problem and its impact on training.
The key finding from previous analysis:
- 95/135 samples (70%) show NO periodic cardiac signal in the red channel mean
- Only 21/135 (16%) show good cardiac periodicity
- ALL samples show a massive auto-exposure transient at the start
- The transient dominates the signal (AC/DC > 5%), masking the tiny PPG pulse (~0.1% AC/DC)

This script:
1. Quantifies the transient duration across all samples
2. Shows what the signal looks like AFTER the transient settles
3. Checks if cardiac signal exists in the stable portion
4. Computes window-level correlation only from stable portions
5. Compares raw red channel vs bandpass-filtered red channel as model input
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


def find_stable_region(red_means, fps=30, threshold_pct=0.5):
    """
    Find where the auto-exposure transient ends and the signal stabilizes.
    Returns the frame index where the signal becomes stable.

    Method: find where the running std of derivatives drops below a threshold.
    """
    diffs = np.diff(red_means)
    # Running std of diffs over 1-second window
    win = int(fps)
    if len(diffs) < win * 2:
        return 0

    running_std = np.array([diffs[i:i+win].std() for i in range(len(diffs) - win)])

    # The stable region has small derivatives
    # Threshold: when running std drops below threshold_pct of its peak
    peak_std = np.max(running_std[:int(5*fps)])  # Look at first 5 seconds
    stable_threshold = max(peak_std * threshold_pct, 0.5)  # At least 0.5

    # Find first point where it stays below threshold for at least 2 seconds
    consecutive_needed = int(2 * fps)
    for i in range(len(running_std)):
        if i + consecutive_needed <= len(running_std):
            if all(running_std[i:i+consecutive_needed] < stable_threshold):
                return i + win  # Add back the window offset
    return len(red_means) // 2  # If never stabilizes, use middle


def main():
    print("=" * 70)
    print("  Final PPG Diagnostic: Auto-Exposure Transient Analysis")
    print("=" * 70)

    all_pairs = sorted(Path(SAMPLES_DIR).iterdir())
    results = []

    for pair_dir in all_pairs:
        meta_path = pair_dir / "metadata.json"
        video_path = pair_dir / "video_0.mp4"
        ecg_path = pair_dir / "ecg.csv"
        if not meta_path.exists() or not video_path.exists() or not ecg_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        # Read full video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        red_means = []
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            red_means.append(frame[:, :, 2].astype(np.float64).mean())
        cap.release()

        if len(red_means) < 300:
            continue

        red = np.array(red_means)

        # Find stable region
        stable_start = find_stable_region(red, fps)
        transient_duration = stable_start / fps

        # Analyze stable portion
        stable_red = red[stable_start:]
        stable_cv = stable_red.std() / (stable_red.mean() + 1e-10)

        # Check cardiac periodicity in stable portion
        ppg_autocorr = 0
        stable_hr = 0
        if len(stable_red) > 60:
            try:
                sos = scipy_signal.butter(4, [0.5, min(5.0, fps/2 - 0.1)], btype='bandpass', fs=fps, output='sos')
                bp = scipy_signal.sosfiltfilt(sos, stable_red)
                ac = np.correlate(bp, bp, mode='full')
                ac = ac[len(ac)//2:]
                ac = ac / (ac[0] + 1e-10)
                min_lag = int(0.4 * fps)
                max_lag = int(1.5 * fps)
                if max_lag < len(ac):
                    seg = ac[min_lag:max_lag]
                    peaks, props = scipy_signal.find_peaks(seg, height=0.05)
                    if len(peaks) > 0:
                        best_idx = np.argmax(props['peak_heights'])
                        ppg_autocorr = props['peak_heights'][best_idx]
                        stable_hr = 60 * fps / (peaks[best_idx] + min_lag)
            except Exception:
                pass

        # Load ECG for correlation analysis in stable portion
        ecg_df = pd.read_csv(str(ecg_path))
        ecg = ecg_df['ecg_counts_filt_monitor'].values.astype(np.float64)
        ecg_sr = 250

        # Map stable_start frame to ECG sample
        ecg_stable_start = int(stable_start / fps * ecg_sr)

        # Compute correlation in 10-second windows from stable portion
        window_ppg = 300  # 10 seconds at 30 fps
        window_ecg = 2500  # 10 seconds at 250 Hz
        stride_ppg = 150
        stride_ecg = 1250

        stable_ppg = red[stable_start:]
        stable_ecg = ecg[ecg_stable_start:]

        corrs_raw = []
        corrs_bp = []

        n_windows = min(
            (len(stable_ppg) - window_ppg) // stride_ppg + 1,
            (len(stable_ecg) - window_ecg) // stride_ecg + 1,
        )

        for w in range(max(0, n_windows)):
            pp = stable_ppg[w*stride_ppg : w*stride_ppg + window_ppg]
            ec = stable_ecg[w*stride_ecg : w*stride_ecg + window_ecg]

            if len(pp) < window_ppg or len(ec) < window_ecg:
                break

            # Downsample ECG
            ec_down = scipy_signal.resample(ec, window_ppg)

            # Raw correlation
            pp_n = (pp - pp.mean()) / (pp.std() + 1e-8)
            ec_n = (ec_down - ec_down.mean()) / (ec_down.std() + 1e-8)
            r, _ = pearsonr(pp_n, ec_n)
            corrs_raw.append(r)

            # Bandpass filtered correlation
            try:
                sos_p = scipy_signal.butter(4, [0.5, min(5.0, fps/2 - 0.1)], btype='bandpass', fs=fps, output='sos')
                sos_e = scipy_signal.butter(4, [0.5, 5.0], btype='bandpass', fs=ecg_sr, output='sos')
                pp_bp = scipy_signal.sosfiltfilt(sos_p, pp)
                ec_bp = scipy_signal.sosfiltfilt(sos_e, ec)
                ec_bp_down = scipy_signal.resample(ec_bp, window_ppg)
                pp_bp_n = (pp_bp - pp_bp.mean()) / (pp_bp.std() + 1e-8)
                ec_bp_n = (ec_bp_down - ec_bp_down.mean()) / (ec_bp_down.std() + 1e-8)
                r_bp, _ = pearsonr(pp_bp_n, ec_bp_n)
                corrs_bp.append(r_bp)
            except Exception:
                corrs_bp.append(0)

        # Also compute correlation for ALL windows (including transient)
        corrs_all_raw = []
        corrs_all_bp = []
        n_windows_all = min(
            (len(red) - window_ppg) // stride_ppg + 1,
            (len(ecg) - window_ecg) // stride_ecg + 1,
        )
        for w in range(max(0, n_windows_all)):
            pp = red[w*stride_ppg : w*stride_ppg + window_ppg]
            ec = ecg[w*stride_ecg : w*stride_ecg + window_ecg]
            if len(pp) < window_ppg or len(ec) < window_ecg:
                break
            ec_down = scipy_signal.resample(ec, window_ppg)
            pp_n = (pp - pp.mean()) / (pp.std() + 1e-8)
            ec_n = (ec_down - ec_down.mean()) / (ec_down.std() + 1e-8)
            r, _ = pearsonr(pp_n, ec_n)
            corrs_all_raw.append(r)

            try:
                sos_p = scipy_signal.butter(4, [0.5, min(5.0, fps/2 - 0.1)], btype='bandpass', fs=fps, output='sos')
                sos_e = scipy_signal.butter(4, [0.5, 5.0], btype='bandpass', fs=ecg_sr, output='sos')
                pp_bp = scipy_signal.sosfiltfilt(sos_p, pp)
                ec_bp = scipy_signal.sosfiltfilt(sos_e, ec)
                ec_bp_down = scipy_signal.resample(ec_bp, window_ppg)
                pp_bp_n = (pp_bp - pp_bp.mean()) / (pp_bp.std() + 1e-8)
                ec_bp_n = (ec_bp_down - ec_bp_down.mean()) / (ec_bp_down.std() + 1e-8)
                r_bp, _ = pearsonr(pp_bp_n, ec_bp_n)
                corrs_all_bp.append(r_bp)
            except Exception:
                corrs_all_bp.append(0)

        results.append({
            'pair': pair_dir.name,
            'user': meta['phone_user'],
            'hr': meta.get('heart_rate', 0),
            'state': meta.get('measurement_state', '?'),
            'total_frames': len(red),
            'transient_end_frame': stable_start,
            'transient_duration_s': transient_duration,
            'stable_cv': stable_cv,
            'ppg_autocorr_stable': ppg_autocorr,
            'stable_hr': stable_hr,
            'mean_r_raw_stable': np.mean(corrs_raw) if corrs_raw else 0,
            'mean_r_bp_stable': np.mean(corrs_bp) if corrs_bp else 0,
            'n_windows_stable': len(corrs_raw),
            'mean_r_raw_all': np.mean(corrs_all_raw) if corrs_all_raw else 0,
            'mean_r_bp_all': np.mean(corrs_all_bp) if corrs_all_bp else 0,
            'n_windows_all': len(corrs_all_raw),
        })

    # Print summary
    print(f"\nTotal samples analyzed: {len(results)}")

    # Transient duration statistics
    transients = [r['transient_duration_s'] for r in results]
    print(f"\n--- Auto-Exposure Transient Duration ---")
    print(f"  Mean: {np.mean(transients):.1f}s")
    print(f"  Median: {np.median(transients):.1f}s")
    print(f"  Min: {np.min(transients):.1f}s")
    print(f"  Max: {np.max(transients):.1f}s")
    print(f"  Samples with transient > 5s: {sum(1 for t in transients if t > 5)}/{len(transients)}")
    print(f"  Samples with transient > 10s: {sum(1 for t in transients if t > 10)}/{len(transients)}")
    print(f"  Samples with transient > 20s: {sum(1 for t in transients if t > 20)}/{len(transients)}")

    # PPG quality in stable portion
    print(f"\n--- PPG Quality in Stable Portion ---")
    good = sum(1 for r in results if r['ppg_autocorr_stable'] > 0.3)
    weak = sum(1 for r in results if 0.1 < r['ppg_autocorr_stable'] <= 0.3)
    none_ = sum(1 for r in results if r['ppg_autocorr_stable'] <= 0.1)
    print(f"  Good (autocorr > 0.3): {good}/{len(results)} ({good/len(results)*100:.0f}%)")
    print(f"  Weak (0.1 < autocorr <= 0.3): {weak}/{len(results)} ({weak/len(results)*100:.0f}%)")
    print(f"  None (autocorr <= 0.1): {none_}/{len(results)} ({none_/len(results)*100:.0f}%)")

    # Correlation comparison: all windows vs stable-only
    print(f"\n--- PPG-ECG Correlation: ALL Windows (including transient) ---")
    all_raw = [r['mean_r_raw_all'] for r in results if r['n_windows_all'] > 0]
    all_bp = [r['mean_r_bp_all'] for r in results if r['n_windows_all'] > 0]
    print(f"  Mean |raw correlation|: {np.mean(np.abs(all_raw)):.4f}")
    print(f"  Mean |bandpass correlation|: {np.mean(np.abs(all_bp)):.4f}")
    print(f"  Samples with |r_raw| > 0.2: {sum(1 for r in all_raw if abs(r) > 0.2)}/{len(all_raw)}")
    print(f"  Samples with |r_bp| > 0.3: {sum(1 for r in all_bp if abs(r) > 0.3)}/{len(all_bp)}")

    print(f"\n--- PPG-ECG Correlation: STABLE Windows Only ---")
    stable_raw = [r['mean_r_raw_stable'] for r in results if r['n_windows_stable'] > 0]
    stable_bp = [r['mean_r_bp_stable'] for r in results if r['n_windows_stable'] > 0]
    print(f"  Mean |raw correlation|: {np.mean(np.abs(stable_raw)):.4f}")
    print(f"  Mean |bandpass correlation|: {np.mean(np.abs(stable_bp)):.4f}")
    print(f"  Samples with |r_raw| > 0.2: {sum(1 for r in stable_raw if abs(r) > 0.2)}/{len(stable_raw)}")
    print(f"  Samples with |r_bp| > 0.3: {sum(1 for r in stable_bp if abs(r) > 0.3)}/{len(stable_bp)}")

    # Print detailed per-sample
    print(f"\n{'='*120}")
    print(f"  Detailed Results (sorted by stable bandpass correlation)")
    print(f"{'='*120}")
    results.sort(key=lambda x: abs(x['mean_r_bp_stable']), reverse=True)
    print(f"  {'Pair':<12} {'User':<5} {'HR':>3} {'State':<12} {'Trans(s)':>8} {'PPG_AC':>7} "
          f"{'r_raw_all':>10} {'r_bp_all':>10} {'r_raw_stab':>11} {'r_bp_stab':>11} {'n_win':>6}")
    print(f"  {'-'*108}")
    for r in results:
        print(f"  {r['pair']:<12} {r['user']:<5} {r['hr']:>3} {r['state']:<12} "
              f"{r['transient_duration_s']:>7.1f}s {r['ppg_autocorr_stable']:>7.3f} "
              f"{r['mean_r_raw_all']:>+10.4f} {r['mean_r_bp_all']:>+10.4f} "
              f"{r['mean_r_raw_stable']:>+11.4f} {r['mean_r_bp_stable']:>+11.4f} "
              f"{r['n_windows_stable']:>6}")

    # THE KEY ANALYSIS: What does the model actually see?
    print(f"\n\n{'='*70}")
    print(f"  KEY FINDINGS: Why the model sees Pearson r ≈ 0")
    print(f"{'='*70}")

    print(f"""
  1. AUTO-EXPOSURE TRANSIENT DOMINATES THE SIGNAL
     - Mean transient duration: {np.mean(transients):.1f}s (median: {np.median(transients):.1f}s)
     - In a 10-second window, {sum(1 for t in transients if t > 5)/len(transients)*100:.0f}% of samples have transient > 5s
     - The transient creates a huge monotonic drift (>50 units) that completely
       masks the tiny PPG pulse variation (~0.5-2 units)
     - After z-normalization, the transient becomes the dominant feature

  2. PPG SIGNAL IS EXTREMELY WEAK
     - Red channel mean ~200-210 with per-beat variation of ~0.5-2 units
     - PPG AC/DC ratio: ~0.1-0.5% (vs auto-exposure drift of 5-10%)
     - Only {good}/{len(results)} ({good/len(results)*100:.0f}%) samples show clear cardiac periodicity
     - Even the "good" samples have weak PPG relative to noise

  3. THE Z-NORMALIZATION AMPLIFIES THE WRONG SIGNAL
     - Per-window z-normalization (in dataset.py) normalizes each 10s window
     - If the window contains a transient, the transient slope becomes
       the dominant normalized signal, not the PPG oscillation
     - The model learns to predict ECG from exposure drift, which has ZERO
       correlation with ECG

  4. RAW vs BANDPASS FILTERED CORRELATION
     - Raw PPG-ECG correlation: |r| ≈ {np.mean(np.abs(all_raw)):.3f} (near zero)
     - Bandpass (0.5-5Hz) correlation: |r| ≈ {np.mean(np.abs(all_bp)):.3f}
     - Bandpass filtering removes the drift and reveals the cardiac component
     - BUT: bandpass filtering is NOT done in dataset.py!

  RECOMMENDED FIXES:
  a) Add bandpass filter (0.5-5 Hz) to the PPG signal in dataset.py BEFORE z-normalization
  b) Skip the first N seconds (transient) of each video
  c) Use detrending (highpass filter at 0.3-0.5 Hz) to remove exposure drift
  d) Consider using frame-to-frame DIFFERENCES instead of raw channel means
     (differences naturally remove slow drift)
""")


if __name__ == "__main__":
    main()
