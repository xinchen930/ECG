"""
Diagnostic script to investigate why 1D models (Scheme E/D) produce Pearson r ≈ 0.

This script checks:
1. PPG signal quality (from video red channel mean)
2. PPG-ECG correlation at raw signal level
3. Temporal alignment (PPG peak vs ECG R-peak)
4. Normalization effects
5. Data loading through actual dataset class
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

# Add project root to path
sys.path.insert(0, '/home/xinchen/ECG')
sys.path.insert(0, '/home/xinchen/ECG/models')

SAMPLES_DIR = '/home/xinchen/ECG/training_data/samples'

###############################################################################
# PART 1: Raw PPG signal extraction and analysis
###############################################################################

def extract_ppg_from_video(video_path, max_frames=300, channel='red'):
    """Extract per-frame channel mean from video as PPG proxy."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_frames = min(max_frames, total_frames)

    red_means = []
    green_means = []
    blue_means = []
    all_pixel_stds = []  # per-frame spatial std of red channel

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV is BGR
        blue_means.append(frame[:, :, 0].mean())
        green_means.append(frame[:, :, 1].mean())
        red_means.append(frame[:, :, 2].mean())
        all_pixel_stds.append(frame[:, :, 2].std())  # spatial std of red channel

    cap.release()

    return {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'n_extracted': len(red_means),
        'red': np.array(red_means),
        'green': np.array(green_means),
        'blue': np.array(blue_means),
        'red_spatial_std': np.array(all_pixel_stds),
    }


def analyze_ppg_signal(ppg, fps, label="PPG"):
    """Analyze a PPG signal: stats, periodicity, frequency content."""
    print(f"\n{'='*60}")
    print(f"  {label} Signal Analysis")
    print(f"{'='*60}")

    # Basic stats
    print(f"  Length: {len(ppg)} samples, FPS: {fps:.1f}")
    print(f"  Duration: {len(ppg)/fps:.1f} seconds")
    print(f"  Raw range: [{ppg.min():.4f}, {ppg.max():.4f}]")
    print(f"  Mean: {ppg.mean():.4f}")
    print(f"  Std:  {ppg.std():.6f}")
    print(f"  Dynamic range (max-min): {ppg.max() - ppg.min():.6f}")

    # Check if signal is essentially flat (std very small relative to mean)
    cv = ppg.std() / (abs(ppg.mean()) + 1e-10)
    print(f"  Coefficient of variation (std/mean): {cv:.6f}")
    if cv < 0.001:
        print(f"  *** WARNING: Signal appears FLAT (CV < 0.001) ***")
    elif cv < 0.01:
        print(f"  *** WARNING: Signal has very low variation (CV < 0.01) ***")

    # After z-normalization (what the model sees)
    ppg_norm = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
    print(f"\n  After z-norm: range [{ppg_norm.min():.4f}, {ppg_norm.max():.4f}]")

    # Autocorrelation to check periodicity
    ppg_centered = ppg - ppg.mean()
    autocorr = np.correlate(ppg_centered, ppg_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags only
    autocorr = autocorr / (autocorr[0] + 1e-10)  # Normalize

    # Find first peak after initial decay (this would indicate heart rate period)
    # Look for peaks in autocorrelation
    min_lag = int(0.3 * fps)  # Min lag: 0.3s (200 BPM)
    max_lag = int(2.0 * fps)  # Max lag: 2.0s (30 BPM)
    if max_lag <= len(autocorr):
        segment = autocorr[min_lag:max_lag]
        if len(segment) > 0:
            peaks, props = scipy_signal.find_peaks(segment, height=0.1)
            if len(peaks) > 0:
                best_peak = peaks[np.argmax(props['peak_heights'])]
                peak_lag = best_peak + min_lag
                estimated_hr = 60.0 * fps / peak_lag
                peak_value = autocorr[peak_lag]
                print(f"\n  Autocorrelation peak: lag={peak_lag} samples ({peak_lag/fps:.3f}s)")
                print(f"  Estimated HR from autocorrelation: {estimated_hr:.1f} BPM")
                print(f"  Autocorrelation peak value: {peak_value:.4f}")
                if peak_value < 0.2:
                    print(f"  *** WARNING: Weak periodicity (autocorr peak < 0.2) ***")
            else:
                print(f"\n  *** NO autocorrelation peak found (no periodicity in {min_lag}-{max_lag} lag range) ***")

    # Frequency spectrum (FFT)
    ppg_detrend = scipy_signal.detrend(ppg)
    freqs = np.fft.rfftfreq(len(ppg_detrend), d=1.0/fps)
    fft_mag = np.abs(np.fft.rfft(ppg_detrend))

    # Find dominant frequency in cardiac range (0.5-3.5 Hz = 30-210 BPM)
    cardiac_mask = (freqs >= 0.5) & (freqs <= 3.5)
    if cardiac_mask.any():
        cardiac_freqs = freqs[cardiac_mask]
        cardiac_power = fft_mag[cardiac_mask]
        dominant_idx = np.argmax(cardiac_power)
        dominant_freq = cardiac_freqs[dominant_idx]
        dominant_power = cardiac_power[dominant_idx]
        total_power = fft_mag.sum()
        cardiac_total_power = cardiac_power.sum()

        print(f"\n  Frequency analysis:")
        print(f"  Dominant cardiac freq: {dominant_freq:.3f} Hz ({dominant_freq*60:.1f} BPM)")
        print(f"  Dominant cardiac power: {dominant_power:.4f}")
        print(f"  Cardiac band power / total power: {cardiac_total_power/total_power:.4f}")

        # SNR: cardiac band vs rest
        non_cardiac_power = total_power - cardiac_total_power
        snr = cardiac_total_power / (non_cardiac_power + 1e-10)
        print(f"  Cardiac SNR (cardiac/non-cardiac): {snr:.4f}")
        if snr < 0.1:
            print(f"  *** WARNING: Very low cardiac SNR ***")

    # Print first 20 values to see temporal pattern
    print(f"\n  First 20 raw values: {ppg[:20].round(4).tolist()}")
    print(f"  Diff (first 20): {np.diff(ppg[:20]).round(6).tolist()}")

    return ppg_norm


###############################################################################
# PART 2: PPG-ECG correlation analysis
###############################################################################

def analyze_ppg_ecg_correlation(ppg, ppg_fps, ecg, ecg_sr, pair_name=""):
    """Check correlation between PPG and ECG at various time lags."""
    print(f"\n{'='*60}")
    print(f"  PPG-ECG Correlation Analysis - {pair_name}")
    print(f"{'='*60}")

    # Duration check
    ppg_duration = len(ppg) / ppg_fps
    ecg_duration = len(ecg) / ecg_sr
    print(f"  PPG: {len(ppg)} samples @ {ppg_fps} Hz = {ppg_duration:.2f}s")
    print(f"  ECG: {len(ecg)} samples @ {ecg_sr} Hz = {ecg_duration:.2f}s")

    # Use minimum duration
    min_duration = min(ppg_duration, ecg_duration)

    # Method 1: Downsample ECG to PPG rate (30Hz)
    ppg_len = int(min_duration * ppg_fps)
    ecg_len = int(min_duration * ecg_sr)
    ppg_clip = ppg[:ppg_len]
    ecg_clip = ecg[:ecg_len]

    # Resample ECG to PPG rate
    ecg_resampled = scipy_signal.resample(ecg_clip, ppg_len)

    # Normalize both
    ppg_n = (ppg_clip - ppg_clip.mean()) / (ppg_clip.std() + 1e-8)
    ecg_n = (ecg_resampled - ecg_resampled.mean()) / (ecg_resampled.std() + 1e-8)

    # Direct correlation (no lag)
    r_direct, p_direct = pearsonr(ppg_n, ecg_n)
    print(f"\n  Direct Pearson r (no lag): {r_direct:.4f} (p={p_direct:.4e})")

    # Cross-correlation to find best lag
    # Try lags from -2s to +2s
    max_lag_samples = int(2.0 * ppg_fps)
    cross_corr = np.correlate(ppg_n, ecg_n, mode='full')
    cross_corr = cross_corr / len(ppg_n)  # Normalize

    center = len(cross_corr) // 2
    lag_range = cross_corr[center - max_lag_samples:center + max_lag_samples]
    lag_values = np.arange(-max_lag_samples, max_lag_samples)

    best_lag_idx = np.argmax(np.abs(lag_range))
    best_lag = lag_values[best_lag_idx]
    best_corr = lag_range[best_lag_idx]
    best_lag_sec = best_lag / ppg_fps

    print(f"  Best cross-correlation: {best_corr:.4f} at lag={best_lag} samples ({best_lag_sec:.3f}s)")

    # Also try correlation with specific physiological lags
    for lag_ms in [0, 100, 200, 300, 400, 500]:
        lag_samples = int(lag_ms / 1000 * ppg_fps)
        if lag_samples > 0 and lag_samples < len(ppg_n):
            r, p = pearsonr(ppg_n[lag_samples:], ecg_n[:-lag_samples])
            print(f"  r at lag {lag_ms:4d}ms: {r:.4f}")

    # Method 2: Upsample PPG to ECG rate (250Hz) - this is what the model needs to learn
    ppg_upsampled = scipy_signal.resample(ppg_clip, ecg_len)
    ppg_up_n = (ppg_upsampled - ppg_upsampled.mean()) / (ppg_upsampled.std() + 1e-8)
    ecg_clip_n = (ecg_clip - ecg_clip.mean()) / (ecg_clip.std() + 1e-8)

    r_up, p_up = pearsonr(ppg_up_n, ecg_clip_n)
    print(f"\n  PPG upsampled to 250Hz vs ECG: r={r_up:.4f}")

    # Method 3: Bandpass filter both to cardiac band (0.5-5 Hz), then correlate
    sos_ppg = scipy_signal.butter(4, [0.5, min(5.0, ppg_fps/2 - 0.1)], btype='bandpass', fs=ppg_fps, output='sos')
    sos_ecg = scipy_signal.butter(4, [0.5, 5.0], btype='bandpass', fs=ecg_sr, output='sos')

    ppg_filtered = scipy_signal.sosfiltfilt(sos_ppg, ppg_clip)
    ecg_filtered = scipy_signal.sosfiltfilt(sos_ecg, ecg_clip)

    # Resample filtered ECG to PPG rate
    ecg_filt_resampled = scipy_signal.resample(ecg_filtered, ppg_len)

    ppg_filt_n = (ppg_filtered - ppg_filtered.mean()) / (ppg_filtered.std() + 1e-8)
    ecg_filt_n = (ecg_filt_resampled - ecg_filt_resampled.mean()) / (ecg_filt_resampled.std() + 1e-8)

    r_filt, p_filt = pearsonr(ppg_filt_n, ecg_filt_n)
    print(f"  Bandpass filtered (0.5-5Hz) correlation: r={r_filt:.4f}")

    # Cross-correlation of filtered signals
    cross_filt = np.correlate(ppg_filt_n, ecg_filt_n, mode='full')
    cross_filt = cross_filt / len(ppg_filt_n)
    center_f = len(cross_filt) // 2
    lag_range_f = cross_filt[center_f - max_lag_samples:center_f + max_lag_samples]
    best_lag_idx_f = np.argmax(np.abs(lag_range_f))
    best_lag_f = lag_values[best_lag_idx_f]
    best_corr_f = lag_range_f[best_lag_idx_f]
    print(f"  Best filtered cross-corr: {best_corr_f:.4f} at lag={best_lag_f/ppg_fps:.3f}s")

    # Method 4: Envelope correlation (PPG envelope vs ECG envelope)
    ppg_envelope = np.abs(scipy_signal.hilbert(ppg_filt_n))
    ecg_envelope = np.abs(scipy_signal.hilbert(ecg_filt_n))
    r_env, _ = pearsonr(ppg_envelope, ecg_envelope)
    print(f"  Envelope correlation: r={r_env:.4f}")

    return r_direct, best_corr, r_filt


###############################################################################
# PART 3: ECG R-peak vs PPG peak alignment
###############################################################################

def check_temporal_alignment(ppg, ppg_fps, ecg, ecg_sr, pair_name=""):
    """Check if PPG peaks lag ECG R-peaks by ~200-400ms (pulse transit time)."""
    print(f"\n{'='*60}")
    print(f"  Temporal Alignment Check - {pair_name}")
    print(f"{'='*60}")

    # Find ECG R-peaks
    ecg_filtered = scipy_signal.detrend(ecg)
    sos = scipy_signal.butter(4, [0.5, 40], btype='bandpass', fs=ecg_sr, output='sos')
    ecg_bp = scipy_signal.sosfiltfilt(sos, ecg_filtered)

    # Find peaks with minimum distance of 0.3s (200 BPM max)
    min_dist = int(0.3 * ecg_sr)
    ecg_peaks, _ = scipy_signal.find_peaks(ecg_bp, distance=min_dist, height=np.std(ecg_bp)*0.5)

    if len(ecg_peaks) < 3:
        print(f"  Could not find enough ECG R-peaks ({len(ecg_peaks)} found)")
        return

    ecg_peak_times = ecg_peaks / ecg_sr
    ecg_rr_intervals = np.diff(ecg_peak_times)
    ecg_hr = 60.0 / np.mean(ecg_rr_intervals)
    print(f"  ECG R-peaks found: {len(ecg_peaks)}")
    print(f"  ECG HR from R-R intervals: {ecg_hr:.1f} BPM")

    # Find PPG peaks
    # Bandpass filter PPG
    ppg_detrend = scipy_signal.detrend(ppg)
    sos_ppg = scipy_signal.butter(4, [0.5, min(5.0, ppg_fps/2 - 0.1)], btype='bandpass', fs=ppg_fps, output='sos')
    ppg_bp = scipy_signal.sosfiltfilt(sos_ppg, ppg_detrend)

    min_dist_ppg = int(0.3 * ppg_fps)
    ppg_peaks, _ = scipy_signal.find_peaks(ppg_bp, distance=min_dist_ppg, height=np.std(ppg_bp)*0.3)

    if len(ppg_peaks) < 3:
        print(f"  Could not find enough PPG peaks ({len(ppg_peaks)} found)")
        # Also try inverted PPG (some PPG is inverted)
        ppg_peaks_inv, _ = scipy_signal.find_peaks(-ppg_bp, distance=min_dist_ppg, height=np.std(ppg_bp)*0.3)
        if len(ppg_peaks_inv) > len(ppg_peaks):
            print(f"  Inverted PPG peaks found: {len(ppg_peaks_inv)} (PPG may be inverted)")
            ppg_peaks = ppg_peaks_inv
        if len(ppg_peaks) < 3:
            return

    ppg_peak_times = ppg_peaks / ppg_fps
    ppg_rr_intervals = np.diff(ppg_peak_times)
    ppg_hr = 60.0 / np.mean(ppg_rr_intervals)
    print(f"  PPG peaks found: {len(ppg_peaks)}")
    print(f"  PPG HR from peak intervals: {ppg_hr:.1f} BPM")

    # Compare HR
    hr_diff = abs(ecg_hr - ppg_hr)
    print(f"  HR difference (ECG vs PPG): {hr_diff:.1f} BPM")
    if hr_diff > 10:
        print(f"  *** WARNING: Large HR discrepancy - PPG and ECG may not be aligned ***")

    # Find pulse transit time (PTT) by matching nearest peaks
    ptts = []
    for ecg_t in ecg_peak_times[:20]:  # First 20 peaks
        # Find nearest PPG peak AFTER ECG peak
        ppg_after = ppg_peak_times[ppg_peak_times > ecg_t]
        if len(ppg_after) > 0:
            nearest_ppg = ppg_after[0]
            ptt = nearest_ppg - ecg_t
            if 0 < ptt < 1.0:  # Reasonable PTT range
                ptts.append(ptt)

    if ptts:
        print(f"\n  Pulse Transit Time (PTT) estimates:")
        print(f"  Mean PTT: {np.mean(ptts)*1000:.1f} ms")
        print(f"  Std PTT:  {np.std(ptts)*1000:.1f} ms")
        print(f"  Range:    [{np.min(ptts)*1000:.1f}, {np.max(ptts)*1000:.1f}] ms")
        if 100 < np.mean(ptts)*1000 < 600:
            print(f"  PTT is in physiological range (100-600ms) - alignment looks OK")
        else:
            print(f"  *** WARNING: PTT out of physiological range ***")
    else:
        print(f"  *** Could not compute PTT - no matching peak pairs found ***")


###############################################################################
# PART 4: Dataset class loading verification
###############################################################################

def verify_dataset_loading():
    """Load samples through the actual dataset class and check what the model sees."""
    print(f"\n{'='*60}")
    print(f"  Dataset Class Loading Verification")
    print(f"{'='*60}")

    import yaml
    with open('/home/xinchen/ECG/configs/scheme_e.yaml') as f:
        cfg = yaml.safe_load(f)

    from dataset import create_datasets
    train_ds, val_ds, test_ds = create_datasets(cfg)

    print(f"\n  Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Check several samples
    for i in range(min(5, len(train_ds))):
        video_tensor, ecg_tensor = train_ds[i]
        pair_idx = train_ds.window_index[i][0]
        pair_name = train_ds.records[pair_idx]['pair_name']

        print(f"\n  Sample {i} ({pair_name}):")
        print(f"    Video tensor shape: {video_tensor.shape}")
        print(f"    Video tensor dtype: {video_tensor.dtype}")
        print(f"    Video range: [{video_tensor.min():.4f}, {video_tensor.max():.4f}]")
        print(f"    Video mean: {video_tensor.mean():.4f}")
        print(f"    Video std:  {video_tensor.std():.4f}")
        print(f"    ECG tensor shape: {ecg_tensor.shape}")
        print(f"    ECG range: [{ecg_tensor.min():.4f}, {ecg_tensor.max():.4f}]")
        print(f"    ECG mean: {ecg_tensor.mean():.4f}")
        print(f"    ECG std:  {ecg_tensor.std():.4f}")

        # Check if video tensor is basically constant
        video_np = video_tensor.numpy()
        if video_np.ndim == 2:  # (T, C)
            for c in range(video_np.shape[1]):
                ch = video_np[:, c]
                print(f"    Channel {c}: mean={ch.mean():.4f}, std={ch.std():.4f}, range=[{ch.min():.4f}, {ch.max():.4f}]")
                # After z-norm these should be ~0 mean, ~1 std

        # Pearson correlation between video signal and ECG (downsampled)
        if video_np.ndim == 2 and video_np.shape[1] == 1:
            ppg_signal = video_np[:, 0]  # (300,)
            ecg_signal = ecg_tensor.numpy()  # (2500,)

            # Downsample ECG to 30Hz
            ecg_down = scipy_signal.resample(ecg_signal, len(ppg_signal))
            r, p = pearsonr(ppg_signal, ecg_down)
            print(f"    Pearson r (model input PPG vs downsampled ECG): {r:.4f} (p={p:.4e})")


###############################################################################
# PART 5: Check if video is actually finger-over-camera (should be mostly red)
###############################################################################

def check_video_content(video_path, n_frames=30):
    """Check whether the video looks like finger-over-camera PPG recording."""
    print(f"\n  Checking video content: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"    Resolution: {width}x{height}, FPS: {fps}, Total frames: {total}")

    b_means, g_means, r_means = [], [], []
    for i in range(min(n_frames, total)):
        ret, frame = cap.read()
        if not ret:
            break
        b_means.append(frame[:, :, 0].mean())
        g_means.append(frame[:, :, 1].mean())
        r_means.append(frame[:, :, 2].mean())
    cap.release()

    b, g, r = np.mean(b_means), np.mean(g_means), np.mean(r_means)
    print(f"    Average channel values - R: {r:.1f}, G: {g:.1f}, B: {b:.1f}")

    if r > g and r > b and r > 100:
        print(f"    Video appears to be finger-over-camera (red dominant) - GOOD")
    elif r > 200 and g > 200 and b > 200:
        print(f"    *** WARNING: Video appears OVEREXPOSED (all channels high) ***")
    else:
        print(f"    *** WARNING: Video does not look like finger PPG ***")

    # Check frame-to-frame variation in red channel
    r_arr = np.array(r_means)
    r_diff = np.diff(r_arr)
    print(f"    Red channel temporal std: {r_arr.std():.6f}")
    print(f"    Red channel frame-to-frame diff std: {r_diff.std():.6f}")
    print(f"    Red channel CV: {r_arr.std()/r_arr.mean():.6f}")

    return r, g, b


###############################################################################
# MAIN
###############################################################################

def main():
    print("=" * 70)
    print("  PPG → ECG Diagnostic Analysis")
    print("  Investigating why 1D models produce Pearson r ≈ 0")
    print("=" * 70)

    # Select a few diverse samples
    sample_dirs = sorted(Path(SAMPLES_DIR).iterdir())
    # Pick samples: first, middle, and some specific ones
    test_pairs = []
    for pair_dir in sample_dirs:
        meta_path = pair_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            test_pairs.append(meta)

    print(f"\nTotal available pairs: {len(test_pairs)}")

    # Analyze 5 samples spread across the dataset
    n_pairs = len(test_pairs)
    indices_to_check = [0, n_pairs//4, n_pairs//2, 3*n_pairs//4, n_pairs-1]
    indices_to_check = sorted(set(indices_to_check))[:5]

    all_correlations = []

    for idx in indices_to_check:
        meta = test_pairs[idx]
        pair_name = f"pair_{meta['pair_id']:04d}"
        pair_dir = Path(SAMPLES_DIR) / pair_name
        video_path = str(pair_dir / "video_0.mp4")
        ecg_path = str(pair_dir / "ecg.csv")

        print(f"\n\n{'#'*70}")
        print(f"  ANALYZING: {pair_name} (user={meta['phone_user']}, HR={meta.get('heart_rate', '?')} BPM)")
        print(f"  State: {meta.get('measurement_state', '?')}, Duration: {meta.get('ecg_duration_s', '?')}s")
        print(f"{'#'*70}")

        # Check video content
        check_video_content(video_path)

        # Extract PPG
        ppg_data = extract_ppg_from_video(video_path, max_frames=600)  # 20 seconds

        # Analyze red channel PPG
        ppg_red = ppg_data['red']
        ppg_norm = analyze_ppg_signal(ppg_red, ppg_data['fps'], label=f"Red Channel PPG - {pair_name}")

        # Also briefly check green channel
        ppg_green = ppg_data['green']
        print(f"\n  Green channel: mean={ppg_green.mean():.2f}, std={ppg_green.std():.6f}, CV={ppg_green.std()/(ppg_green.mean()+1e-10):.6f}")

        # Spatial std of red channel (how uniform is the image)
        print(f"  Red spatial std (mean across frames): {ppg_data['red_spatial_std'].mean():.2f}")
        print(f"  (Low spatial std = uniform red frame = good finger coverage)")

        # Load ECG
        ecg_df = pd.read_csv(ecg_path)
        ecg = ecg_df['ecg_counts_filt_monitor'].values
        ecg_sr = 250

        analyze_ppg_signal(ecg[:5000], ecg_sr, label=f"ECG (first 20s) - {pair_name}")

        # PPG-ECG correlation
        r_direct, r_best, r_filt = analyze_ppg_ecg_correlation(
            ppg_red, ppg_data['fps'], ecg, ecg_sr, pair_name
        )
        all_correlations.append({
            'pair': pair_name,
            'user': meta['phone_user'],
            'hr': meta.get('heart_rate'),
            'state': meta.get('measurement_state'),
            'r_direct': r_direct,
            'r_best_lag': r_best,
            'r_filtered': r_filt,
            'ppg_cv': ppg_red.std() / (ppg_red.mean() + 1e-10),
        })

        # Temporal alignment
        check_temporal_alignment(ppg_red, ppg_data['fps'], ecg, ecg_sr, pair_name)

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY: PPG-ECG Correlations Across Samples")
    print(f"{'='*70}")
    print(f"{'Pair':<12} {'User':<6} {'HR':>4} {'State':<12} {'r_direct':>10} {'r_best':>10} {'r_filt':>10} {'PPG_CV':>10}")
    print("-" * 80)
    for c in all_correlations:
        print(f"{c['pair']:<12} {c['user']:<6} {c['hr'] or '?':>4} {c['state'] or '?':<12} "
              f"{c['r_direct']:>10.4f} {c['r_best_lag']:>10.4f} {c['r_filtered']:>10.4f} {c['ppg_cv']:>10.6f}")

    avg_r_direct = np.mean([c['r_direct'] for c in all_correlations])
    avg_r_filt = np.mean([c['r_filtered'] for c in all_correlations])
    print(f"\nAverage direct correlation: {avg_r_direct:.4f}")
    print(f"Average filtered correlation: {avg_r_filt:.4f}")

    if abs(avg_r_direct) < 0.1:
        print("\n*** CRITICAL: Raw PPG-ECG correlation is near zero. ***")
        print("*** The input signal may not contain meaningful PPG information, ***")
        print("*** OR the time alignment is completely wrong, ***")
        print("*** OR the PPG signal is too noisy at this sampling rate. ***")

    # Part 4: Dataset loading verification
    print(f"\n\n{'#'*70}")
    print(f"  DATASET CLASS VERIFICATION")
    print(f"{'#'*70}")
    verify_dataset_loading()

    print(f"\n\n{'='*70}")
    print(f"  DIAGNOSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
