"""
Deep PPG signal investigation - Part 2
Focus on:
1. Frame-by-frame pixel value distribution (is auto-exposure creating artifacts?)
2. Is the signal actually saturated at 254/255? (critical: clipping destroys PPG)
3. What does the PPG signal look like when we remove the saturation frames?
4. Histogram of per-frame red channel values
5. Check per-WINDOW normalization vs per-PAIR normalization effects
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


def detailed_video_analysis(video_path, max_frames=600, pair_name=""):
    """Extremely detailed frame-by-frame video analysis."""
    print(f"\n{'='*70}")
    print(f"  Detailed Video Analysis: {pair_name}")
    print(f"{'='*70}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Resolution: {width}x{height}, FPS: {fps:.2f}, Total frames: {total_frames}")

    n_frames = min(max_frames, total_frames)

    red_means = []
    red_medians = []
    red_maxes = []
    red_mins = []
    green_means = []
    blue_means = []
    saturated_fracs = []  # fraction of pixels at 254/255

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        red_ch = frame[:, :, 2].astype(np.float64)  # Red channel (BGR)
        red_means.append(red_ch.mean())
        red_medians.append(np.median(red_ch))
        red_maxes.append(red_ch.max())
        red_mins.append(red_ch.min())
        green_means.append(frame[:, :, 1].mean())
        blue_means.append(frame[:, :, 0].mean())
        # Fraction of saturated pixels
        saturated = (frame[:, :, 2] >= 254).sum() / frame[:, :, 2].size
        saturated_fracs.append(saturated)

    cap.release()

    red_means = np.array(red_means)
    red_medians = np.array(red_medians)
    red_maxes = np.array(red_maxes)
    red_mins = np.array(red_mins)
    green_means = np.array(green_means)
    blue_means = np.array(blue_means)
    saturated_fracs = np.array(saturated_fracs)

    # ========== SATURATION ANALYSIS ==========
    print(f"\n  --- Saturation Analysis ---")
    print(f"  Mean saturation fraction (pixels >= 254): {saturated_fracs.mean():.4f}")
    print(f"  Frames with > 50% saturated pixels: {(saturated_fracs > 0.5).sum()}/{len(saturated_fracs)}")
    print(f"  Frames with > 90% saturated pixels: {(saturated_fracs > 0.9).sum()}/{len(saturated_fracs)}")
    print(f"  Frames with > 99% saturated pixels: {(saturated_fracs > 0.99).sum()}/{len(saturated_fracs)}")

    # ========== RED CHANNEL PATTERN ==========
    print(f"\n  --- Red Channel Mean Over Time ---")
    print(f"  Overall mean: {red_means.mean():.2f}")
    print(f"  Overall std:  {red_means.std():.4f}")
    print(f"  Min frame mean: {red_means.min():.2f}")
    print(f"  Max frame mean: {red_means.max():.2f}")

    # Check for clipping: if most frames are near 254, PPG is clipped!
    near_max = (red_means > 250).sum()
    print(f"  Frames with red mean > 250: {near_max}/{len(red_means)} ({near_max/len(red_means)*100:.1f}%)")
    near_max2 = (red_means > 253).sum()
    print(f"  Frames with red mean > 253: {near_max2}/{len(red_means)} ({near_max2/len(red_means)*100:.1f}%)")

    # ========== TEMPORAL PATTERN (FIRST 60 FRAMES = 2 SECONDS) ==========
    print(f"\n  --- First 60 frames (2 seconds) ---")
    for i in range(min(60, len(red_means))):
        sat_str = " *** SATURATED ***" if saturated_fracs[i] > 0.9 else ""
        print(f"    Frame {i:3d}: R={red_means[i]:7.2f} (med={red_medians[i]:5.0f}, "
              f"max={red_maxes[i]:3.0f}, min={red_mins[i]:3.0f}) "
              f"G={green_means[i]:6.2f} B={blue_means[i]:5.2f} "
              f"sat={saturated_fracs[i]:.3f}{sat_str}")

    # ========== THE KEY INSIGHT: PPG is in the VARIATION of red channel ==========
    print(f"\n  --- PPG Signal Quality (Red Channel Variation) ---")

    # Detrend: remove slow drift (auto-exposure changes)
    if len(red_means) > 30:
        # Use SOS filter for detrending (highpass at 0.5 Hz)
        sos = scipy_signal.butter(4, 0.5, btype='highpass', fs=fps, output='sos')
        red_hp = scipy_signal.sosfiltfilt(sos, red_means)
        print(f"  After highpass (>0.5Hz): std={red_hp.std():.6f}, range=[{red_hp.min():.4f}, {red_hp.max():.4f}]")

        # Bandpass in cardiac range
        sos_bp = scipy_signal.butter(4, [0.5, min(5.0, fps/2 - 0.1)], btype='bandpass', fs=fps, output='sos')
        red_bp = scipy_signal.sosfiltfilt(sos_bp, red_means)
        print(f"  Bandpass (0.5-5Hz): std={red_bp.std():.6f}, range=[{red_bp.min():.4f}, {red_bp.max():.4f}]")

        # Compare cardiac band signal to noise floor
        # Noise: 5-15 Hz band
        if fps > 12:  # Need enough bandwidth
            sos_noise = scipy_signal.butter(4, [5.0, min(14.0, fps/2 - 0.1)], btype='bandpass', fs=fps, output='sos')
            red_noise = scipy_signal.sosfiltfilt(sos_noise, red_means)
            print(f"  Noise band (5-14Hz): std={red_noise.std():.6f}")
            snr_db = 20 * np.log10(red_bp.std() / (red_noise.std() + 1e-10))
            print(f"  SNR (cardiac/noise): {snr_db:.1f} dB")

    # ========== CRITICAL: What percentage of the signal is saturated? ==========
    # When red channel is saturated (at 254/255), the PPG pulse cannot be detected
    # because the camera sensor is clipped
    print(f"\n  --- Saturation vs PPG Quality ---")
    # Identify non-saturated segments
    non_sat_mask = saturated_fracs < 0.5  # frames where < 50% pixels are saturated
    print(f"  Non-saturated frames: {non_sat_mask.sum()}/{len(non_sat_mask)} ({non_sat_mask.sum()/len(non_sat_mask)*100:.1f}%)")

    if non_sat_mask.sum() > 30:
        red_nonsat = red_means[non_sat_mask]
        print(f"  Non-saturated red mean range: [{red_nonsat.min():.2f}, {red_nonsat.max():.2f}]")
        print(f"  Non-saturated red std: {red_nonsat.std():.4f}")
        print(f"  Non-saturated red CV: {red_nonsat.std()/(red_nonsat.mean()+1e-10):.6f}")
    else:
        print(f"  *** Most frames are saturated - PPG extraction may be impossible ***")

    # ========== FRAME 0 ANOMALY CHECK ==========
    if len(red_means) > 1:
        jump = abs(red_means[1] - red_means[0])
        if jump > 10:
            print(f"\n  *** Frame 0 anomaly: jump of {jump:.2f} from frame 0 to frame 1 ***")
            print(f"  Frame 0 may be an initialization frame (different exposure)")

    return {
        'red_means': red_means,
        'green_means': green_means,
        'blue_means': blue_means,
        'saturated_fracs': saturated_fracs,
        'fps': fps,
    }


def analyze_normalization_impact(video_path, ecg_path, fps=30, ecg_sr=250):
    """
    Check how per-window z-normalization affects the PPG signal.
    The dataset normalizes per-window (10 seconds), which may destroy long-term trends
    but should preserve per-beat variation.
    """
    print(f"\n  --- Normalization Impact Analysis ---")

    # Extract full video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    red_means = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        red_means.append(frame[:, :, 2].astype(np.float64).mean())
    cap.release()
    red_means = np.array(red_means)

    # Load ECG
    ecg_df = pd.read_csv(ecg_path)
    ecg = ecg_df['ecg_counts_filt_monitor'].values

    # Window: 10 seconds = 300 frames for PPG, 2500 samples for ECG
    window_ppg = 300
    window_ecg = 2500
    stride_ppg = 150  # 5 seconds
    stride_ecg = 1250

    print(f"  Total PPG frames: {len(red_means)}")
    print(f"  Total ECG samples: {len(ecg)}")

    correlations_raw = []
    correlations_znorm_window = []
    correlations_znorm_global = []
    correlations_bp = []

    # Global z-norm stats
    ppg_global_mean = red_means.mean()
    ppg_global_std = red_means.std() + 1e-8
    ecg_global_mean = ecg.mean()
    ecg_global_std = ecg.std() + 1e-8

    n_windows = min(
        (len(red_means) - window_ppg) // stride_ppg + 1,
        (len(ecg) - window_ecg) // stride_ecg + 1,
    )

    for w in range(n_windows):
        ppg_start = w * stride_ppg
        ecg_start = w * stride_ecg

        ppg_win = red_means[ppg_start:ppg_start + window_ppg]
        ecg_win = ecg[ecg_start:ecg_start + window_ecg]

        if len(ppg_win) < window_ppg or len(ecg_win) < window_ecg:
            break

        # Downsample ECG to match PPG
        ecg_down = scipy_signal.resample(ecg_win, window_ppg)

        # 1. Raw correlation
        r_raw, _ = pearsonr(ppg_win, ecg_down)
        correlations_raw.append(r_raw)

        # 2. Per-window z-norm (what dataset.py does)
        ppg_wn = (ppg_win - ppg_win.mean()) / (ppg_win.std() + 1e-8)
        ecg_wn = (ecg_down - ecg_down.mean()) / (ecg_down.std() + 1e-8)
        r_znorm, _ = pearsonr(ppg_wn, ecg_wn)
        correlations_znorm_window.append(r_znorm)

        # 3. Global z-norm
        ppg_gn = (ppg_win - ppg_global_mean) / ppg_global_std
        ecg_gn = (ecg_down - ecg_global_mean) / ecg_global_std
        r_global, _ = pearsonr(ppg_gn, ecg_gn)
        correlations_znorm_global.append(r_global)

        # 4. Bandpass filtered correlation
        if window_ppg > 30:
            try:
                sos_ppg = scipy_signal.butter(4, [0.5, min(5.0, fps/2 - 0.1)], btype='bandpass', fs=fps, output='sos')
                sos_ecg = scipy_signal.butter(4, [0.5, 5.0], btype='bandpass', fs=ecg_sr, output='sos')
                ppg_bp = scipy_signal.sosfiltfilt(sos_ppg, ppg_win)
                ecg_bp = scipy_signal.sosfiltfilt(sos_ecg, ecg_win)
                ecg_bp_down = scipy_signal.resample(ecg_bp, window_ppg)
                ppg_bp_n = (ppg_bp - ppg_bp.mean()) / (ppg_bp.std() + 1e-8)
                ecg_bp_n = (ecg_bp_down - ecg_bp_down.mean()) / (ecg_bp_down.std() + 1e-8)
                r_bp, _ = pearsonr(ppg_bp_n, ecg_bp_n)
                correlations_bp.append(r_bp)
            except Exception:
                correlations_bp.append(0)

    if correlations_raw:
        print(f"\n  Per-window correlations ({n_windows} windows):")
        print(f"  {'Method':<25} {'Mean r':>8} {'Std r':>8} {'|r|>0.3':>8} {'|r|>0.5':>8}")
        print(f"  {'-'*57}")

        for name, corrs in [
            ("Raw", correlations_raw),
            ("Z-norm (per-window)", correlations_znorm_window),
            ("Z-norm (global)", correlations_znorm_global),
            ("Bandpass filtered", correlations_bp),
        ]:
            arr = np.array(corrs)
            print(f"  {name:<25} {arr.mean():>8.4f} {arr.std():>8.4f} "
                  f"{(np.abs(arr) > 0.3).sum():>8}/{len(arr):<5} "
                  f"{(np.abs(arr) > 0.5).sum():>8}/{len(arr):<5}")


def check_exposure_oscillation(video_path, pair_name=""):
    """
    Check if auto-exposure creates artificial oscillations that look like but aren't PPG.
    Auto-exposure often creates a sawtooth or step pattern, not sinusoidal PPG.
    """
    print(f"\n  --- Auto-Exposure Pattern Check ---")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first 300 frames (10 seconds)
    n = min(300, total_frames)
    red_means = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        red_means.append(frame[:, :, 2].astype(np.float64).mean())
    cap.release()

    red = np.array(red_means)

    # The real PPG signal at 30 Hz has:
    # - Heart rate component (~1 Hz): amplitude ~0.1-2% of DC level
    # - For typical finger PPG at 254 mean, this is ~0.25-5 units variation

    dc_level = red.mean()
    ac_range = red.max() - red.min()
    ac_std = red.std()

    print(f"  DC level (mean): {dc_level:.2f}")
    print(f"  AC range (max-min): {ac_range:.2f}")
    print(f"  AC std: {ac_std:.4f}")
    print(f"  AC/DC ratio (pulsatility): {ac_std/dc_level*100:.4f}%")

    # Typical finger PPG AC/DC ratio: 0.5-5%
    # If it's higher, likely motion artifact or exposure changes
    if ac_std/dc_level*100 > 5:
        print(f"  *** WARNING: AC/DC ratio > 5% — likely NOT pure PPG (exposure/motion artifact) ***")
    elif ac_std/dc_level*100 < 0.05:
        print(f"  *** WARNING: AC/DC ratio < 0.05% — signal too weak for PPG ***")

    # Check if the pattern looks like sawtooth (auto-exposure) vs sinusoidal (PPG)
    # Compute ratio of positive vs negative derivatives
    diffs = np.diff(red)
    n_pos = (diffs > 0).sum()
    n_neg = (diffs < 0).sum()
    print(f"  Rising frames: {n_pos}, Falling frames: {n_neg}")
    print(f"  Rise/fall ratio: {n_pos/(n_neg+1):.3f} (should be ~1.0 for sinusoidal)")

    # Check frame-to-frame difference distribution
    diff_abs_mean = np.abs(diffs).mean()
    diff_abs_max = np.abs(diffs).max()
    print(f"  Mean |frame diff|: {diff_abs_mean:.4f}")
    print(f"  Max |frame diff|: {diff_abs_max:.4f}")

    # Large jumps suggest auto-exposure steps, not PPG
    large_jumps = (np.abs(diffs) > 5).sum()
    print(f"  Large jumps (|diff| > 5): {large_jumps}/{len(diffs)}")
    if large_jumps > 5:
        print(f"  *** WARNING: Multiple large jumps — auto-exposure stepping ***")

    # ========== CRITICAL TEST: Remove trend and check for periodic signal ==========
    # Detrend with aggressive highpass to remove auto-exposure
    if len(red) > 60:
        sos_hp = scipy_signal.butter(4, 0.5, btype='highpass', fs=fps, output='sos')
        red_hp = scipy_signal.sosfiltfilt(sos_hp, red)

        sos_bp = scipy_signal.butter(4, [0.5, min(5.0, fps/2 - 0.1)], btype='bandpass', fs=fps, output='sos')
        red_bp = scipy_signal.sosfiltfilt(sos_bp, red)

        # Check if bandpass signal has clear periodicity
        autocorr = np.correlate(red_bp, red_bp, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        min_lag = int(0.4 * fps)  # 150 BPM
        max_lag = int(1.5 * fps)  # 40 BPM
        if max_lag < len(autocorr):
            segment = autocorr[min_lag:max_lag]
            peaks, props = scipy_signal.find_peaks(segment, height=0.1)
            if len(peaks) > 0:
                best = peaks[np.argmax(props['peak_heights'])]
                lag = best + min_lag
                hr = 60 * fps / lag
                peak_val = autocorr[lag]
                print(f"\n  Bandpass autocorrelation peak: {peak_val:.4f} at HR={hr:.0f} BPM")
                if peak_val > 0.3:
                    print(f"  Good periodic signal found in bandpass-filtered red channel")
                elif peak_val > 0.1:
                    print(f"  Weak periodic signal found")
                else:
                    print(f"  *** Very weak or no periodic signal ***")
            else:
                print(f"\n  *** No autocorrelation peak in cardiac range after bandpass ***")

        # Print bandpass signal values (what PPG should look like)
        print(f"\n  Bandpass filtered signal (first 60 frames):")
        for i in range(min(60, len(red_bp))):
            print(f"    Frame {i:3d}: raw={red[i]:7.2f}  bp={red_bp[i]:+8.4f}")


def main():
    print("=" * 70)
    print("  Deep PPG Signal Investigation")
    print("=" * 70)

    # Analyze a few samples in detail
    samples_to_check = ['pair_0001', 'pair_0033', 'pair_0067', 'pair_0101']

    for pair_name in samples_to_check:
        pair_dir = Path(SAMPLES_DIR) / pair_name
        video_path = str(pair_dir / "video_0.mp4")
        ecg_path = str(pair_dir / "ecg.csv")

        if not os.path.exists(video_path):
            print(f"  Skipping {pair_name} (no video)")
            continue

        meta_path = pair_dir / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n\n{'#'*70}")
        print(f"  {pair_name} (user={meta['phone_user']}, HR={meta.get('heart_rate','?')} BPM)")
        print(f"{'#'*70}")

        # Detailed video analysis
        data = detailed_video_analysis(video_path, max_frames=600, pair_name=pair_name)

        # Auto-exposure check
        check_exposure_oscillation(video_path, pair_name)

        # Normalization impact
        analyze_normalization_impact(video_path, ecg_path)

    # ========== GLOBAL ANALYSIS ==========
    print(f"\n\n{'#'*70}")
    print(f"  GLOBAL ANALYSIS: All Samples")
    print(f"{'#'*70}")

    # Quick check of ALL samples for saturation and PPG quality
    all_pairs = sorted(Path(SAMPLES_DIR).iterdir())
    results = []

    for pair_dir in all_pairs:
        meta_path = pair_dir / "metadata.json"
        video_path = pair_dir / "video_0.mp4"
        if not meta_path.exists() or not video_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        # Quick: read first 300 frames
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        red_vals = []
        sat_fracs = []
        for _ in range(300):
            ret, frame = cap.read()
            if not ret:
                break
            red_ch = frame[:, :, 2]
            red_vals.append(red_ch.mean())
            sat_fracs.append((red_ch >= 254).sum() / red_ch.size)
        cap.release()

        if len(red_vals) < 100:
            continue

        red_arr = np.array(red_vals)
        sat_arr = np.array(sat_fracs)

        # Compute bandpass-filtered signal quality
        ppg_quality = 0
        if len(red_arr) > 60:
            try:
                sos = scipy_signal.butter(4, [0.5, min(5.0, fps/2 - 0.1)], btype='bandpass', fs=fps, output='sos')
                bp = scipy_signal.sosfiltfilt(sos, red_arr)

                # Autocorrelation peak
                ac = np.correlate(bp, bp, mode='full')
                ac = ac[len(ac)//2:]
                ac = ac / (ac[0] + 1e-10)
                min_lag = int(0.4 * fps)
                max_lag = int(1.5 * fps)
                if max_lag < len(ac):
                    seg = ac[min_lag:max_lag]
                    peaks, props = scipy_signal.find_peaks(seg, height=0.05)
                    if len(peaks) > 0:
                        ppg_quality = props['peak_heights'].max()
            except Exception:
                pass

        results.append({
            'pair': pair_dir.name,
            'user': meta['phone_user'],
            'hr': meta.get('heart_rate', 0),
            'state': meta.get('measurement_state', '?'),
            'red_mean': red_arr.mean(),
            'red_std': red_arr.std(),
            'red_cv': red_arr.std() / (red_arr.mean() + 1e-10),
            'sat_frac_mean': sat_arr.mean(),
            'sat_frac_gt50': (sat_arr > 0.5).sum() / len(sat_arr),
            'ppg_autocorr_peak': ppg_quality,
        })

    print(f"\n  Total samples analyzed: {len(results)}")

    # Sort by PPG quality
    results.sort(key=lambda x: x['ppg_autocorr_peak'], reverse=True)

    print(f"\n  {'Pair':<12} {'User':<6} {'HR':>4} {'State':<12} {'R_mean':>7} {'R_std':>7} {'R_CV':>8} "
          f"{'Sat%':>6} {'Sat>50%':>8} {'PPG_AC':>8}")
    print(f"  {'-'*95}")
    for r in results:
        print(f"  {r['pair']:<12} {r['user']:<6} {r['hr']:>4} {r['state']:<12} "
              f"{r['red_mean']:>7.1f} {r['red_std']:>7.2f} {r['red_cv']:>8.4f} "
              f"{r['sat_frac_mean']*100:>5.1f}% {r['sat_frac_gt50']*100:>7.1f}% "
              f"{r['ppg_autocorr_peak']:>8.4f}")

    # Summary statistics
    ppg_good = sum(1 for r in results if r['ppg_autocorr_peak'] > 0.3)
    ppg_weak = sum(1 for r in results if 0.1 < r['ppg_autocorr_peak'] <= 0.3)
    ppg_none = sum(1 for r in results if r['ppg_autocorr_peak'] <= 0.1)
    sat_high = sum(1 for r in results if r['sat_frac_mean'] > 0.5)

    print(f"\n  PPG Signal Quality Distribution:")
    print(f"    Good (autocorr > 0.3): {ppg_good}/{len(results)}")
    print(f"    Weak (0.1 < autocorr <= 0.3): {ppg_weak}/{len(results)}")
    print(f"    None (autocorr <= 0.1): {ppg_none}/{len(results)}")
    print(f"    Highly saturated (>50% pixels): {sat_high}/{len(results)}")

    # Key question: what fraction of signal variation is at clipping ceiling?
    high_mean = sum(1 for r in results if r['red_mean'] > 240)
    print(f"\n  Red channel mean > 240: {high_mean}/{len(results)} ({high_mean/len(results)*100:.0f}%)")
    print(f"  --> If red channel is near 254, clipping destroys PPG pulsatile signal!")


if __name__ == "__main__":
    main()
