"""
Deep analysis for data quality report v2.
Focuses on:
1. Improved ECG R-peak detection (Pan-Tompkins-style)
2. PPG-ECG cross-correlation (time-domain alignment validation)
3. All-sample statistics aggregation
4. Quality CSV cross-validation
5. Video frame brightness vs ECG R-peak timing
6. Audio track detection using cv2 only
7. Short-sample investigation
8. User info anomalies
"""

import os
import sys
import json
import csv
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "training_data", "samples")
QUALITY_CSV = os.path.join(BASE_DIR, "eval_results", "ppg_analysis_all_samples.csv")

SAMPLE_PAIRS = [
    "pair_0000", "pair_0001", "pair_0009", "pair_0010", "pair_0015",
    "pair_0023", "pair_0034",
    "pair_0003", "pair_0011", "pair_0026",
]


def detect_r_peaks_improved(ecg, sr=250):
    """Improved R-peak detection using derivative-based approach (simplified Pan-Tompkins)."""
    from scipy import signal as sig

    # 1. Bandpass filter 5-15 Hz to isolate QRS
    nyq = sr / 2
    b, a = sig.butter(2, [5.0/nyq, 15.0/nyq], btype='band')
    ecg_bp = sig.filtfilt(b, a, ecg)

    # 2. Differentiate
    ecg_diff = np.diff(ecg_bp)

    # 3. Square
    ecg_sq = ecg_diff ** 2

    # 4. Moving average (150ms window)
    window = int(0.15 * sr)
    ecg_ma = np.convolve(ecg_sq, np.ones(window)/window, mode='same')

    # 5. Find peaks with adaptive threshold
    threshold = np.mean(ecg_ma) + 0.3 * np.std(ecg_ma)
    min_distance = int(0.3 * sr)  # 300ms minimum between beats (200 BPM max)

    peaks, props = sig.find_peaks(ecg_ma, height=threshold, distance=min_distance)

    return peaks


def analyze_ecg_quality_improved():
    """Improved ECG R-peak detection for all 10 samples."""
    print("=" * 70)
    print("1. IMPROVED ECG R-PEAK DETECTION (Pan-Tompkins style)")
    print("=" * 70)

    from scipy import signal as sig

    results = {}
    for pair in SAMPLE_PAIRS:
        ecg_path = os.path.join(SAMPLES_DIR, pair, "ecg.csv")
        meta_path = os.path.join(SAMPLES_DIR, pair, "metadata.json")

        if not os.path.exists(ecg_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        gt_hr = meta.get("heart_rate", 0)

        # Read ECG
        with open(ecg_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [row for row in reader]

        data = np.array(rows, dtype=float)
        ecg_idx = header.index('ecg_counts_filt_monitor') if 'ecg_counts_filt_monitor' in header else 1
        ecg = data[:, ecg_idx]
        sr = 250

        # Use middle 30 seconds for analysis
        mid = len(ecg) // 2
        win_samples = 30 * sr  # 30 seconds
        start = max(0, mid - win_samples // 2)
        end = min(len(ecg), start + win_samples)
        ecg_window = ecg[start:end]

        peaks = detect_r_peaks_improved(ecg_window, sr)

        if len(peaks) > 2:
            rr = np.diff(peaks) / sr
            detected_hr = 60.0 / np.mean(rr)
            rr_std = np.std(rr)
            rr_cv = rr_std / np.mean(rr) * 100  # coefficient of variation (%)
            hr_error = abs(detected_hr - gt_hr)
        else:
            detected_hr = 0
            rr_std = 0
            rr_cv = 0
            hr_error = gt_hr

        results[pair] = {
            'n_peaks': len(peaks),
            'detected_hr': detected_hr,
            'gt_hr': gt_hr,
            'hr_error': hr_error,
            'rr_std': rr_std,
            'rr_cv': rr_cv,
        }

        match = "OK" if hr_error < 10 else "WARN" if hr_error < 20 else "BAD"
        print(f"  {pair}: GT={gt_hr:3d}, Detected={detected_hr:6.1f}, Error={hr_error:5.1f} bpm, "
              f"Peaks={len(peaks):3d}/30s, RR_CV={rr_cv:5.1f}% [{match}]")

    good = sum(1 for r in results.values() if r['hr_error'] < 10)
    print(f"\n  Accuracy (<10 bpm error): {good}/{len(results)}")
    return results


def analyze_ppg_ecg_cross_correlation():
    """Cross-correlate PPG brightness with ECG to verify temporal alignment."""
    print("\n" + "=" * 70)
    print("2. PPG-ECG CROSS-CORRELATION (Temporal Alignment Validation)")
    print("=" * 70)

    try:
        import cv2
        from scipy import signal as sig
    except ImportError:
        print("  OpenCV/scipy not available")
        return {}

    results = {}
    for pair in SAMPLE_PAIRS[:5]:  # Only do 5 samples (slow)
        video_path = os.path.join(SAMPLES_DIR, pair, "video_0.mp4")
        ecg_path = os.path.join(SAMPLES_DIR, pair, "ecg.csv")
        meta_path = os.path.join(SAMPLES_DIR, pair, "metadata.json")

        if not all(os.path.exists(p) for p in [video_path, ecg_path, meta_path]):
            continue

        # Extract red channel mean from video (subsample to be fast)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        r_means = []
        for i in range(3600):  # Max 2 minutes at 30fps
            ret, frame = cap.read()
            if not ret:
                break
            r_means.append(np.mean(frame[:, :, 2]))  # Red channel (BGR order)
        cap.release()

        if len(r_means) < 300:
            continue

        r_sig = np.array(r_means)

        # Bandpass PPG signal
        nyq = fps / 2
        sos = sig.butter(4, [0.7/nyq, min(4.0/nyq, 0.95)], btype='band', output='sos')
        ppg_filt = sig.sosfilt(sos, sig.detrend(r_sig))

        # Read ECG and resample to video fps
        with open(ecg_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [row for row in reader]
        data = np.array(rows, dtype=float)
        ecg_idx = header.index('ecg_counts_filt_monitor') if 'ecg_counts_filt_monitor' in header else 1
        ecg_full = data[:, ecg_idx]

        # Resample ECG from 250 Hz to video fps (30 Hz)
        n_video_samples = len(r_means)
        n_ecg_samples = len(ecg_full)
        ecg_resampled = sig.resample(ecg_full, n_video_samples)

        # Bandpass ECG at same frequencies
        ecg_filt = sig.sosfilt(sos, sig.detrend(ecg_resampled))

        # Use middle 30 seconds
        mid = len(ppg_filt) // 2
        win = int(30 * fps)
        start = max(0, mid - win // 2)
        end = min(len(ppg_filt), start + win)

        ppg_seg = ppg_filt[start:end]
        ecg_seg = ecg_filt[start:end]

        # Compute envelope of ECG (absolute of Hilbert transform)
        ecg_env = np.abs(sig.hilbert(ecg_seg))
        ecg_env_filt = sig.sosfilt(sos, sig.detrend(ecg_env))

        # Cross-correlation between PPG and ECG envelope
        # PPG follows ECG with some delay (Pulse Transit Time ~ 0.1-0.3s)
        max_lag = int(2 * fps)  # +/- 2 seconds
        lags = np.arange(-max_lag, max_lag + 1)

        ppg_norm = (ppg_seg - np.mean(ppg_seg)) / (np.std(ppg_seg) + 1e-8)
        ecg_norm = (ecg_env_filt - np.mean(ecg_env_filt)) / (np.std(ecg_env_filt) + 1e-8)

        corr = np.correlate(ppg_norm, ecg_norm, mode='full')
        # Trim to +/- max_lag
        mid_idx = len(corr) // 2
        corr_trimmed = corr[mid_idx - max_lag:mid_idx + max_lag + 1]
        corr_trimmed = corr_trimmed / len(ppg_seg)  # Normalize

        peak_idx = np.argmax(np.abs(corr_trimmed))
        best_lag = lags[peak_idx]
        best_corr = corr_trimmed[peak_idx]
        lag_seconds = best_lag / fps

        results[pair] = {
            'best_lag_frames': int(best_lag),
            'best_lag_seconds': lag_seconds,
            'max_correlation': float(best_corr),
        }

        quality = "GOOD" if abs(best_corr) > 0.3 else "WEAK" if abs(best_corr) > 0.15 else "NONE"
        print(f"  {pair}: max_corr={best_corr:+.3f}, lag={lag_seconds:+.2f}s ({best_lag:+d} frames) [{quality}]")

    return results


def analyze_all_samples_statistics():
    """Comprehensive statistics from ALL 98 samples metadata."""
    print("\n" + "=" * 70)
    print("3. ALL-SAMPLE STATISTICS (98 samples)")
    print("=" * 70)

    pairs = sorted([d for d in os.listdir(SAMPLES_DIR) if d.startswith("pair_")])

    # Collect metadata
    all_meta = []
    for pair in pairs:
        meta_path = os.path.join(SAMPLES_DIR, pair, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
            meta['pair'] = pair
            all_meta.append(meta)

    # Duration distribution
    durations = [m['overlap_duration_s'] for m in all_meta]
    short_samples = [(m['pair'], m['overlap_duration_s']) for m in all_meta if m['overlap_duration_s'] < 60]

    print(f"\n  Duration distribution:")
    print(f"    < 30s:  {sum(1 for d in durations if d < 30)}")
    print(f"    30-60s: {sum(1 for d in durations if 30 <= d < 60)}")
    print(f"    60-90s: {sum(1 for d in durations if 60 <= d < 90)}")
    print(f"    90-120s: {sum(1 for d in durations if 90 <= d < 120)}")
    print(f"    ~120s:  {sum(1 for d in durations if d >= 120)}")

    if short_samples:
        print(f"\n  Short samples (< 60s):")
        for pair, dur in short_samples:
            print(f"    {pair}: {dur:.1f}s")

    # Heart rate distribution
    hrs = [m['heart_rate'] for m in all_meta]
    print(f"\n  Heart rate distribution:")
    print(f"    < 60 bpm:   {sum(1 for h in hrs if h < 60)}")
    print(f"    60-80 bpm:  {sum(1 for h in hrs if 60 <= h < 80)}")
    print(f"    80-100 bpm: {sum(1 for h in hrs if 80 <= h < 100)}")
    print(f"    100-120 bpm: {sum(1 for h in hrs if 100 <= h < 120)}")
    print(f"    > 120 bpm:  {sum(1 for h in hrs if h >= 120)}")

    # Measurement state by user
    print(f"\n  Measurement states per user:")
    user_states = {}
    for m in all_meta:
        user = m['phone_user']
        state = m['measurement_state']
        user_states.setdefault(user, {}).setdefault(state, 0)
        user_states[user][state] += 1

    for user in sorted(user_states.keys()):
        states = user_states[user]
        total = sum(states.values())
        state_str = ", ".join(f"{k}:{v}" for k, v in sorted(states.items()))
        print(f"    {user} ({total}): {state_str}")

    # User info anomalies
    print(f"\n  User info check:")
    for m in all_meta:
        if m.get('user_height', 0) < 100 or m.get('user_height', 0) > 220:
            print(f"    WARNING: {m['pair']} ({m['phone_user']}): height={m.get('user_height')} cm (suspicious)")
        if m.get('user_weight', 0) > 150:
            print(f"    WARNING: {m['pair']} ({m['phone_user']}): weight={m.get('user_weight')} (unit issue? lbs vs kg?)")
        if m.get('user_birth_year', 0) >= 2020:
            print(f"    WARNING: {m['pair']} ({m['phone_user']}): birth_year={m.get('user_birth_year')} (placeholder?)")

    # ECG sampling rate verification
    print(f"\n  ECG effective sampling rate check:")
    sr_anomalies = []
    for pair in pairs:
        ecg_path = os.path.join(SAMPLES_DIR, pair, "ecg.csv")
        with open(ecg_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            first = next(reader)
            # Read last row
            last = first
            n = 1
            for row in reader:
                last = row
                n += 1
        ts_start = float(first[0])
        ts_end = float(last[0])
        ts_range = (ts_end - ts_start) / 1000.0
        eff_sr = (n - 1) / ts_range if ts_range > 0 else 0

        if abs(eff_sr - 250) > 2:
            sr_anomalies.append((pair, eff_sr, n))

    if sr_anomalies:
        for pair, sr, n in sr_anomalies:
            print(f"    ANOMALY: {pair}: effective SR={sr:.1f} Hz (samples={n})")
    else:
        print(f"    All samples close to 250 Hz (within +/- 2 Hz)")

    return all_meta, short_samples


def cross_validate_quality_csv():
    """Cross-validate quality CSV with computed PPG analysis."""
    print("\n" + "=" * 70)
    print("4. QUALITY CSV CROSS-VALIDATION")
    print("=" * 70)

    if not os.path.exists(QUALITY_CSV):
        print(f"  Quality CSV not found: {QUALITY_CSV}")
        return

    with open(QUALITY_CSV) as f:
        reader = csv.DictReader(f)
        quality_data = {row['pair']: row for row in reader}

    print(f"  Quality CSV entries: {len(quality_data)}")

    # Check consistency with metadata
    pairs = sorted([d for d in os.listdir(SAMPLES_DIR) if d.startswith("pair_")])

    missing_in_csv = [p for p in pairs if p not in quality_data]
    extra_in_csv = [p for p in quality_data if p not in pairs]

    if missing_in_csv:
        print(f"  Pairs missing from CSV: {missing_in_csv}")
    if extra_in_csv:
        print(f"  Extra pairs in CSV: {extra_in_csv}")

    # Quality distribution
    qualities = {}
    for row in quality_data.values():
        q = row.get('quality', 'unknown')
        qualities[q] = qualities.get(q, 0) + 1

    print(f"\n  Quality distribution:")
    for q, c in sorted(qualities.items()):
        print(f"    {q}: {c}")

    # SNR statistics by quality
    print(f"\n  SNR by quality:")
    for q in ['good', 'moderate', 'poor']:
        snrs = [float(row['snr']) for row in quality_data.values() if row.get('quality') == q]
        if snrs:
            print(f"    {q}: mean={np.mean(snrs):.3f}, min={np.min(snrs):.3f}, max={np.max(snrs):.3f}")

    # Check HR consistency between CSV and metadata
    print(f"\n  GT HR consistency (CSV vs metadata):")
    mismatches = 0
    for pair in pairs[:20]:  # Check first 20
        if pair in quality_data:
            csv_hr = float(quality_data[pair]['gt_hr'])
            meta_path = os.path.join(SAMPLES_DIR, pair, "metadata.json")
            with open(meta_path) as f:
                meta_hr = json.load(f)['heart_rate']
            if abs(csv_hr - meta_hr) > 0.5:
                print(f"    MISMATCH: {pair}: CSV={csv_hr}, metadata={meta_hr}")
                mismatches += 1
    if mismatches == 0:
        print(f"    First 20 samples: all consistent")


def analyze_ppg_signal_quality_detailed():
    """Detailed PPG signal analysis: temporal SNR, spectral flatness, periodicity."""
    print("\n" + "=" * 70)
    print("5. DETAILED PPG SIGNAL QUALITY")
    print("=" * 70)

    try:
        import cv2
        from scipy import signal as sig
    except ImportError:
        print("  Dependencies not available")
        return {}

    results = {}
    for pair in SAMPLE_PAIRS:
        video_path = os.path.join(SAMPLES_DIR, pair, "video_0.mp4")
        meta_path = os.path.join(SAMPLES_DIR, pair, "metadata.json")

        if not os.path.exists(video_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        gt_hr = meta.get("heart_rate", 0)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        r_means = []
        g_means = []
        b_means = []
        r_stds = []

        for i in range(3600):
            ret, frame = cap.read()
            if not ret:
                break
            b_ch, g_ch, r_ch = cv2.split(frame)
            r_means.append(np.mean(r_ch))
            g_means.append(np.mean(g_ch))
            b_means.append(np.mean(b_ch))
            r_stds.append(np.std(r_ch))
        cap.release()

        if len(r_means) < 300:
            continue

        r_sig = np.array(r_means)
        g_sig = np.array(g_means)

        # Detrend
        r_det = sig.detrend(r_sig)
        g_det = sig.detrend(g_sig)

        # Bandpass 0.7-4 Hz
        nyq = fps / 2
        sos = sig.butter(4, [0.7/nyq, min(4.0/nyq, 0.95)], btype='band', output='sos')

        r_filt = sig.sosfilt(sos, r_det)
        g_filt = sig.sosfilt(sos, g_det)

        # 1. Temporal SNR: signal power / noise power
        # Use 10-second windows
        win_size = int(10 * fps)
        n_wins = len(r_filt) // win_size

        temporal_snrs_r = []
        temporal_snrs_g = []
        for w in range(n_wins):
            seg = r_filt[w*win_size:(w+1)*win_size]
            freqs, psd = sig.welch(seg, fs=fps, nperseg=min(128, len(seg)//2))
            hr_mask = (freqs >= 0.7) & (freqs <= 4.0)
            if np.any(hr_mask) and np.sum(psd[~hr_mask]) > 0:
                snr = np.sum(psd[hr_mask]) / np.sum(psd[~hr_mask])
                temporal_snrs_r.append(snr)

            seg_g = g_filt[w*win_size:(w+1)*win_size]
            freqs_g, psd_g = sig.welch(seg_g, fs=fps, nperseg=min(128, len(seg_g)//2))
            if np.any(hr_mask) and np.sum(psd_g[~hr_mask]) > 0:
                snr_g = np.sum(psd_g[hr_mask]) / np.sum(psd_g[~hr_mask])
                temporal_snrs_g.append(snr_g)

        # 2. Autocorrelation strength (periodicity measure)
        acf = np.correlate(r_filt, r_filt, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]

        # Find first peak after initial drop (expected at ~1/HR period)
        expected_period = int(60.0 / gt_hr * fps) if gt_hr > 0 else int(fps)
        search_start = int(expected_period * 0.7)
        search_end = int(expected_period * 1.3)
        search_end = min(search_end, len(acf) - 1)

        if search_start < search_end:
            acf_peak = np.max(acf[search_start:search_end])
        else:
            acf_peak = 0

        # 3. Inter-pixel variation (spatial uniformity)
        mean_r_std = np.mean(r_stds)

        results[pair] = {
            'gt_hr': gt_hr,
            'r_mean_brightness': float(np.mean(r_sig)),
            'g_mean_brightness': float(np.mean(g_sig)),
            'r_temporal_snr_mean': float(np.mean(temporal_snrs_r)) if temporal_snrs_r else 0,
            'g_temporal_snr_mean': float(np.mean(temporal_snrs_g)) if temporal_snrs_g else 0,
            'acf_peak': float(acf_peak),
            'spatial_std': float(mean_r_std),
        }

        quality = "GOOD" if acf_peak > 0.3 else "FAIR" if acf_peak > 0.15 else "POOR"
        print(f"  {pair}: R_bright={np.mean(r_sig):.0f}, "
              f"SNR_R={np.mean(temporal_snrs_r):.2f}, SNR_G={np.mean(temporal_snrs_g):.2f}, "
              f"ACF={acf_peak:.3f}, Spatial_std={mean_r_std:.1f} [{quality}]")

    return results


def investigate_short_samples():
    """Investigate samples shorter than 120s."""
    print("\n" + "=" * 70)
    print("6. SHORT SAMPLE INVESTIGATION")
    print("=" * 70)

    pairs = sorted([d for d in os.listdir(SAMPLES_DIR) if d.startswith("pair_")])

    short_pairs = []
    for pair in pairs:
        meta_path = os.path.join(SAMPLES_DIR, pair, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        dur = meta.get("overlap_duration_s", 0)
        if dur < 115:  # Anything notably shorter than 120s
            short_pairs.append((pair, dur, meta))

    if not short_pairs:
        print("  No notably short samples found (all >= 115s)")
        return

    print(f"  Found {len(short_pairs)} samples shorter than 115s:")
    for pair, dur, meta in sorted(short_pairs, key=lambda x: x[1]):
        user = meta.get('phone_user', '?')
        ecg_n = meta.get('ecg_samples', 0)
        state = meta.get('measurement_state', '?')
        print(f"    {pair}: {dur:.1f}s, user={user}, state={state}, ecg_samples={ecg_n}")

    # Check if these are in quality CSV
    if os.path.exists(QUALITY_CSV):
        with open(QUALITY_CSV) as f:
            reader = csv.DictReader(f)
            quality_data = {row['pair']: row for row in reader}

        print(f"\n  Quality labels for short samples:")
        for pair, dur, meta in short_pairs:
            if pair in quality_data:
                q = quality_data[pair]['quality']
                print(f"    {pair}: {q}")


def check_video_audio_python():
    """Check audio tracks using cv2 video metadata."""
    print("\n" + "=" * 70)
    print("7. AUDIO TRACK CHECK")
    print("=" * 70)

    # Since ffprobe is not available, check if the video container has audio by
    # looking at file properties. With only cv2, we can't detect audio.
    # Instead, check file sizes - videos with audio are typically larger.

    import subprocess

    # Try using python to check for audio
    for pair in SAMPLE_PAIRS[:3]:
        video_path = os.path.join(SAMPLES_DIR, pair, "video_0.mp4")
        if not os.path.exists(video_path):
            continue

        file_size = os.path.getsize(video_path)

        # Also check video_1 if exists
        video1_path = os.path.join(SAMPLES_DIR, pair, "video_1.mp4")
        has_video1 = os.path.exists(video1_path)
        v1_size = os.path.getsize(video1_path) if has_video1 else 0

        print(f"  {pair}:")
        print(f"    video_0.mp4: {file_size/1024/1024:.1f} MB")
        if has_video1:
            print(f"    video_1.mp4: {v1_size/1024/1024:.1f} MB")

        # Try to detect audio using subprocess if available
        try:
            # Method 1: Use Python's wave or aifc modules
            # Not applicable for mp4

            # Method 2: Check mp4 box structure manually
            with open(video_path, 'rb') as f:
                content = f.read(min(file_size, 100000))  # First 100KB

            # Look for 'soun' track type in mp4 box structure
            has_audio_marker = b'soun' in content
            has_mp4a = b'mp4a' in content
            has_aac = b'\x00\x00\x00\x00mp4a' in content or b'mp4a' in content

            print(f"    Audio markers: soun={has_audio_marker}, mp4a={has_mp4a}")

        except Exception as e:
            print(f"    Audio check failed: {e}")


def analyze_imu_motion_artifacts():
    """Analyze IMU data for motion artifact correlation with PPG quality."""
    print("\n" + "=" * 70)
    print("8. IMU MOTION ARTIFACT ANALYSIS")
    print("=" * 70)

    results = {}
    for pair in SAMPLE_PAIRS:
        imu_path = os.path.join(SAMPLES_DIR, pair, "imu.csv")
        meta_path = os.path.join(SAMPLES_DIR, pair, "metadata.json")

        if not os.path.exists(imu_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        state = meta.get('measurement_state', '?')

        with open(imu_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [row for row in reader]

        data = np.array(rows, dtype=float)

        # Accelerometer (columns 1-3)
        acc = data[:, 1:4]
        acc_mag = np.sqrt(np.sum(acc**2, axis=1))

        # Gyroscope (columns 4-6)
        gyro = data[:, 4:7]
        gyro_mag = np.sqrt(np.sum(gyro**2, axis=1))

        # Motion metrics
        # 1. Jerk (derivative of acceleration) - indicates sudden movements
        acc_jerk = np.diff(acc_mag) * 100  # Scale by approximate SR
        jerk_rms = np.sqrt(np.mean(acc_jerk**2))

        # 2. Acceleration variability
        acc_var = np.var(acc_mag)

        # 3. Gyro activity (rotation = potential finger pressure changes)
        gyro_rms = np.sqrt(np.mean(gyro_mag**2))

        # 4. Count high-motion periods (acc_mag deviates > 0.05g from 1.0)
        motion_fraction = np.mean(np.abs(acc_mag - 1.0) > 0.05)

        results[pair] = {
            'state': state,
            'acc_mean': float(np.mean(acc_mag)),
            'acc_std': float(np.std(acc_mag)),
            'jerk_rms': float(jerk_rms),
            'gyro_rms': float(gyro_rms),
            'motion_fraction': float(motion_fraction),
        }

        motion_level = "STILL" if motion_fraction < 0.1 else "LOW" if motion_fraction < 0.3 else "HIGH"
        print(f"  {pair} ({state:10s}): acc_std={np.std(acc_mag):.4f}, "
              f"jerk={jerk_rms:.4f}, gyro={gyro_rms:.4f}, "
              f"motion={motion_fraction:.1%} [{motion_level}]")

    return results


if __name__ == "__main__":
    print("ECG Reconstruction - Deep Data Quality Analysis")
    print(f"Samples: {SAMPLES_DIR}")
    print(f"Checking {len(SAMPLE_PAIRS)} representative samples\n")

    # 1. Improved ECG R-peak detection
    ecg_results = analyze_ecg_quality_improved()

    # 2. PPG-ECG cross-correlation
    xcorr_results = analyze_ppg_ecg_cross_correlation()

    # 3. All-sample statistics
    all_meta, short_samples = analyze_all_samples_statistics()

    # 4. Quality CSV validation
    cross_validate_quality_csv()

    # 5. Detailed PPG quality
    ppg_detailed = analyze_ppg_signal_quality_detailed()

    # 6. Short sample investigation
    investigate_short_samples()

    # 7. Audio check
    check_video_audio_python()

    # 8. IMU motion analysis
    imu_results = analyze_imu_motion_artifacts()

    print("\n" + "=" * 70)
    print("DEEP ANALYSIS COMPLETE")
    print("=" * 70)
