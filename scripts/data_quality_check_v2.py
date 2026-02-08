"""
Comprehensive data quality check for ECG reconstruction project.
Checks: video properties, ECG quality, IMU data, audio, time alignment, PPG extractability.
Run from project root: python scripts/data_quality_check_v2.py
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

# Representative samples: good, moderate, poor
SAMPLE_PAIRS = [
    "pair_0000",  # good, fzq, resting
    "pair_0001",  # good, fzq, highKnee (high HR)
    "pair_0009",  # good, nxs, high SNR
    "pair_0010",  # good, fhy
    "pair_0015",  # good, wcp
    "pair_0023",  # moderate, syw
    "pair_0034",  # moderate, fzq
    "pair_0003",  # poor, wjy
    "pair_0011",  # poor, fzq
    "pair_0026",  # poor, wjy (doubled HR)
]


def check_all_metadata():
    """Gather statistics from all 98 samples."""
    print("=" * 70)
    print("1. DATA OVERVIEW - All 98 Samples")
    print("=" * 70)

    pairs = sorted([d for d in os.listdir(SAMPLES_DIR) if d.startswith("pair_")])
    print(f"Total pairs found: {len(pairs)}")

    users = {}
    states = {}
    durations = []
    ecg_samples_list = []
    imu_samples_list = []
    hr_values = []

    for pair in pairs:
        meta_path = os.path.join(SAMPLES_DIR, pair, "metadata.json")
        if not os.path.exists(meta_path):
            print(f"  WARNING: {pair} missing metadata.json")
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        user = meta.get("phone_user", "unknown")
        users.setdefault(user, []).append(pair)

        state = meta.get("measurement_state", "unknown")
        states[state] = states.get(state, 0) + 1

        durations.append(meta.get("overlap_duration_s", 0))
        ecg_samples_list.append(meta.get("ecg_samples", 0))
        imu_samples_list.append(meta.get("imu_samples", 0))
        hr_values.append(meta.get("heart_rate", 0))

    print(f"\nUser distribution:")
    for user in sorted(users.keys()):
        print(f"  {user}: {len(users[user])} samples")

    print(f"\nMeasurement states:")
    for state, count in sorted(states.items(), key=lambda x: -x[1]):
        print(f"  {state}: {count}")

    print(f"\nDuration stats (seconds):")
    print(f"  Min: {min(durations):.1f}, Max: {max(durations):.1f}, Mean: {np.mean(durations):.1f}")

    print(f"\nECG samples: Min={min(ecg_samples_list)}, Max={max(ecg_samples_list)}")
    print(f"IMU samples: Min={min(imu_samples_list)}, Max={max(imu_samples_list)}")

    print(f"\nHeart rate: Min={min(hr_values)}, Max={max(hr_values)}, Mean={np.mean(hr_values):.1f}")

    return users, states, durations, ecg_samples_list, hr_values


def check_video_properties():
    """Check video properties for sample pairs."""
    print("\n" + "=" * 70)
    print("2. VIDEO PROPERTIES")
    print("=" * 70)

    try:
        import cv2
    except ImportError:
        print("OpenCV not available, skipping video checks")
        return {}

    results = {}
    for pair in SAMPLE_PAIRS:
        video_path = os.path.join(SAMPLES_DIR, pair, "video_0.mp4")
        if not os.path.exists(video_path):
            print(f"  {pair}: video_0.mp4 NOT FOUND")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  {pair}: CANNOT OPEN video")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

        # Read a few frames to check color channels
        r_means, g_means, b_means = [], [], []
        frame_count = 0
        while frame_count < min(100, total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            b, g, r = cv2.split(frame)
            r_means.append(np.mean(r))
            g_means.append(np.mean(g))
            b_means.append(np.mean(b))
            frame_count += 1

        cap.release()

        r_dominant = np.mean(r_means) > np.mean(g_means) and np.mean(r_means) > np.mean(b_means)

        results[pair] = {
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'duration': duration,
            'codec': codec_str,
            'file_size_mb': file_size_mb,
            'r_mean': np.mean(r_means),
            'g_mean': np.mean(g_means),
            'b_mean': np.mean(b_means),
            'r_dominant': r_dominant,
            'r_std': np.std(r_means),
            'g_std': np.std(g_means),
            'b_std': np.std(b_means),
        }

        print(f"\n  {pair}:")
        print(f"    Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}, Duration: {duration:.1f}s")
        print(f"    Codec: {codec_str}, Size: {file_size_mb:.1f} MB")
        print(f"    Channel means (R/G/B): {np.mean(r_means):.1f} / {np.mean(g_means):.1f} / {np.mean(b_means):.1f}")
        print(f"    Channel stds  (R/G/B): {np.std(r_means):.2f} / {np.std(g_means):.2f} / {np.std(b_means):.2f}")
        print(f"    Red dominant: {r_dominant}")

    return results


def check_ecg_data():
    """Check ECG CSV data quality."""
    print("\n" + "=" * 70)
    print("3. ECG DATA QUALITY")
    print("=" * 70)

    results = {}
    for pair in SAMPLE_PAIRS:
        ecg_path = os.path.join(SAMPLES_DIR, pair, "ecg.csv")
        if not os.path.exists(ecg_path):
            print(f"  {pair}: ecg.csv NOT FOUND")
            continue

        # Read CSV header and first few rows
        with open(ecg_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = []
            for i, row in enumerate(reader):
                rows.append(row)

        n_rows = len(rows)
        n_cols = len(header)

        # Parse data
        data = np.array(rows, dtype=float)

        # Check columns
        ts_col = data[:, 0] if n_cols > 0 else None
        ecg_col_idx = header.index('ecg_counts_filt_monitor') if 'ecg_counts_filt_monitor' in header else 1
        ecg_data = data[:, ecg_col_idx]

        # Check for NaN
        nan_count = np.sum(np.isnan(ecg_data))

        # Basic stats
        ecg_min = np.nanmin(ecg_data)
        ecg_max = np.nanmax(ecg_data)
        ecg_mean = np.nanmean(ecg_data)
        ecg_std = np.nanstd(ecg_data)

        # Duration check
        if ts_col is not None and len(ts_col) > 1:
            ts_range_s = (ts_col[-1] - ts_col[0]) / 1000.0  # ms to s
            effective_sr = (n_rows - 1) / ts_range_s if ts_range_s > 0 else 0
        else:
            ts_range_s = 0
            effective_sr = 0

        results[pair] = {
            'n_rows': n_rows,
            'n_cols': n_cols,
            'columns': header,
            'nan_count': nan_count,
            'ecg_min': ecg_min,
            'ecg_max': ecg_max,
            'ecg_mean': ecg_mean,
            'ecg_std': ecg_std,
            'ts_range_s': ts_range_s,
            'effective_sr': effective_sr,
        }

        print(f"\n  {pair}:")
        print(f"    Columns ({n_cols}): {header}")
        print(f"    Rows: {n_rows}, NaN: {nan_count}")
        print(f"    ECG range: [{ecg_min:.1f}, {ecg_max:.1f}], mean: {ecg_mean:.1f}, std: {ecg_std:.1f}")
        print(f"    Time range: {ts_range_s:.2f}s, Effective SR: {effective_sr:.1f} Hz")

    return results


def check_imu_data():
    """Check IMU CSV data quality."""
    print("\n" + "=" * 70)
    print("4. IMU DATA QUALITY")
    print("=" * 70)

    results = {}
    for pair in SAMPLE_PAIRS:
        imu_path = os.path.join(SAMPLES_DIR, pair, "imu.csv")
        if not os.path.exists(imu_path):
            print(f"  {pair}: imu.csv NOT FOUND")
            continue

        with open(imu_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = []
            for row in reader:
                rows.append(row)

        n_rows = len(rows)
        data = np.array(rows, dtype=float)

        # Check timestamps
        ts = data[:, 0]
        ts_range_s = (ts[-1] - ts[0]) / 1000.0
        effective_sr = (n_rows - 1) / ts_range_s if ts_range_s > 0 else 0

        # Check accelerometer (typically columns 1-3) and gyroscope (4-6)
        # Find column indices
        acc_cols = [i for i, h in enumerate(header) if 'acc' in h.lower() or 'accel' in h.lower()]
        gyro_cols = [i for i, h in enumerate(header) if 'gyro' in h.lower()]

        if not acc_cols:
            # Try positional: assume timestamp, ax, ay, az, gx, gy, gz
            acc_cols = [1, 2, 3] if n_rows > 0 and len(header) >= 7 else []
            gyro_cols = [4, 5, 6] if n_rows > 0 and len(header) >= 7 else []

        acc_data = data[:, acc_cols] if acc_cols else None
        gyro_data = data[:, gyro_cols] if gyro_cols else None

        # Acceleration magnitude
        if acc_data is not None:
            acc_mag = np.sqrt(np.sum(acc_data**2, axis=1))
            acc_mag_mean = np.mean(acc_mag)
            acc_mag_std = np.std(acc_mag)
            acc_mag_max = np.max(acc_mag)
        else:
            acc_mag_mean = acc_mag_std = acc_mag_max = 0

        # Check timestamp regularity
        ts_diffs = np.diff(ts)
        ts_diff_mean = np.mean(ts_diffs)
        ts_diff_std = np.std(ts_diffs)
        ts_gaps = np.sum(ts_diffs > ts_diff_mean * 3)

        results[pair] = {
            'n_rows': n_rows,
            'columns': header,
            'ts_range_s': ts_range_s,
            'effective_sr': effective_sr,
            'acc_mag_mean': acc_mag_mean,
            'acc_mag_std': acc_mag_std,
            'acc_mag_max': acc_mag_max,
            'ts_diff_mean_ms': ts_diff_mean,
            'ts_diff_std_ms': ts_diff_std,
            'ts_gaps': ts_gaps,
        }

        print(f"\n  {pair}:")
        print(f"    Columns ({len(header)}): {header}")
        print(f"    Rows: {n_rows}, Duration: {ts_range_s:.2f}s, Effective SR: {effective_sr:.1f} Hz")
        print(f"    Acc magnitude: mean={acc_mag_mean:.2f}, std={acc_mag_std:.2f}, max={acc_mag_max:.2f}")
        print(f"    Timestamp interval: mean={ts_diff_mean:.2f}ms, std={ts_diff_std:.2f}ms, gaps(>3x): {ts_gaps}")

    return results


def check_audio():
    """Check if videos contain audio tracks."""
    print("\n" + "=" * 70)
    print("5. AUDIO CHECK")
    print("=" * 70)

    try:
        import cv2
    except ImportError:
        print("OpenCV not available, skipping audio check")
        return {}

    # Use ffprobe if available, otherwise just check with cv2
    import subprocess

    results = {}
    for pair in SAMPLE_PAIRS:
        video_path = os.path.join(SAMPLES_DIR, pair, "video_0.mp4")
        if not os.path.exists(video_path):
            continue

        has_audio = False
        audio_info = "N/A"

        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-select_streams', 'a:0',
                 '-show_entries', 'stream=codec_name,sample_rate,channels,duration',
                 '-of', 'json', video_path],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                probe = json.loads(result.stdout)
                streams = probe.get('streams', [])
                if streams:
                    has_audio = True
                    s = streams[0]
                    audio_info = f"codec={s.get('codec_name','?')}, sr={s.get('sample_rate','?')}Hz, ch={s.get('channels','?')}"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            audio_info = "ffprobe not available"

        results[pair] = {'has_audio': has_audio, 'info': audio_info}
        print(f"  {pair}: Audio={'YES' if has_audio else 'NO'} ({audio_info})")

    return results


def check_time_alignment():
    """Check video-ECG time alignment."""
    print("\n" + "=" * 70)
    print("6. TIME ALIGNMENT CHECK")
    print("=" * 70)

    try:
        import cv2
    except ImportError:
        print("OpenCV not available, skipping alignment check")
        return {}

    results = {}
    for pair in SAMPLE_PAIRS:
        meta_path = os.path.join(SAMPLES_DIR, pair, "metadata.json")
        ecg_path = os.path.join(SAMPLES_DIR, pair, "ecg.csv")
        video_path = os.path.join(SAMPLES_DIR, pair, "video_0.mp4")
        imu_path = os.path.join(SAMPLES_DIR, pair, "imu.csv")

        if not all(os.path.exists(p) for p in [meta_path, ecg_path, video_path, imu_path]):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        # Get video duration
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = video_frames / video_fps if video_fps > 0 else 0
        cap.release()

        # Get ECG duration from metadata
        ecg_duration = meta.get("ecg_duration_s", 0)
        imu_duration = meta.get("imu_duration_s", 0)
        overlap_duration = meta.get("overlap_duration_s", 0)

        # Duration mismatches
        ecg_video_diff = abs(ecg_duration - video_duration)
        ecg_imu_diff = abs(ecg_duration - imu_duration)

        results[pair] = {
            'video_duration': video_duration,
            'ecg_duration': ecg_duration,
            'imu_duration': imu_duration,
            'overlap_duration': overlap_duration,
            'ecg_video_diff': ecg_video_diff,
            'ecg_imu_diff': ecg_imu_diff,
            'video_fps': video_fps,
        }

        print(f"\n  {pair}:")
        print(f"    Video: {video_duration:.2f}s ({video_frames} frames @ {video_fps:.1f} fps)")
        print(f"    ECG:   {ecg_duration:.2f}s ({meta.get('ecg_samples', 0)} samples @ 250 Hz)")
        print(f"    IMU:   {imu_duration:.2f}s ({meta.get('imu_samples', 0)} samples)")
        print(f"    Overlap: {overlap_duration:.2f}s")
        print(f"    ECG-Video diff: {ecg_video_diff:.2f}s, ECG-IMU diff: {ecg_imu_diff:.4f}s")

    return results


def check_ppg_extractability():
    """Extract PPG from videos and validate against ECG heart rate."""
    print("\n" + "=" * 70)
    print("7. PPG EXTRACTABILITY CHECK")
    print("=" * 70)

    try:
        import cv2
        from scipy import signal
    except ImportError:
        print("OpenCV or scipy not available, skipping PPG check")
        return {}

    results = {}
    for pair in SAMPLE_PAIRS:
        video_path = os.path.join(SAMPLES_DIR, pair, "video_0.mp4")
        meta_path = os.path.join(SAMPLES_DIR, pair, "metadata.json")

        if not os.path.exists(video_path) or not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        gt_hr = meta.get("heart_rate", 0)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Extract channel means for all frames (or up to 5000 frames = ~166s @ 30fps)
        r_means, g_means, b_means = [], [], []
        max_frames = min(total_frames, 5000)

        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            b, g, r = cv2.split(frame)
            r_means.append(np.mean(r))
            g_means.append(np.mean(g))
            b_means.append(np.mean(b))
        cap.release()

        actual_frames = len(r_means)
        if actual_frames < 60:
            print(f"  {pair}: Too few frames ({actual_frames}), skipping")
            continue

        # Convert to numpy
        r_sig = np.array(r_means)
        g_sig = np.array(g_means)
        b_sig = np.array(b_means)

        # Detrend
        r_sig = signal.detrend(r_sig)
        g_sig = signal.detrend(g_sig)
        b_sig = signal.detrend(b_sig)

        # Bandpass filter: 0.7-4 Hz (42-240 bpm)
        nyq = fps / 2
        low = 0.7 / nyq
        high = min(4.0 / nyq, 0.95)  # Don't exceed Nyquist

        try:
            sos = signal.butter(4, [low, high], btype='band', output='sos')
        except ValueError:
            print(f"  {pair}: Filter design failed (fps={fps:.1f}), skipping")
            continue

        r_filt = signal.sosfilt(sos, r_sig)
        g_filt = signal.sosfilt(sos, g_sig)

        # Compute power spectrum
        freqs_r, psd_r = signal.welch(r_filt, fs=fps, nperseg=min(256, actual_frames//2))
        freqs_g, psd_g = signal.welch(g_filt, fs=fps, nperseg=min(256, actual_frames//2))

        # Find dominant frequency in HR range
        hr_mask = (freqs_r >= 0.7) & (freqs_r <= 4.0)

        if np.any(hr_mask):
            peak_freq_r = freqs_r[hr_mask][np.argmax(psd_r[hr_mask])]
            peak_freq_g = freqs_g[hr_mask][np.argmax(psd_g[hr_mask])]
            ppg_hr_r = peak_freq_r * 60
            ppg_hr_g = peak_freq_g * 60

            # SNR: peak power / mean power
            snr_r = np.max(psd_r[hr_mask]) / np.mean(psd_r[hr_mask]) if np.mean(psd_r[hr_mask]) > 0 else 0
            snr_g = np.max(psd_g[hr_mask]) / np.mean(psd_g[hr_mask]) if np.mean(psd_g[hr_mask]) > 0 else 0
        else:
            ppg_hr_r = ppg_hr_g = 0
            snr_r = snr_g = 0

        hr_error_r = abs(ppg_hr_r - gt_hr)
        hr_error_g = abs(ppg_hr_g - gt_hr)

        results[pair] = {
            'gt_hr': gt_hr,
            'ppg_hr_red': ppg_hr_r,
            'ppg_hr_green': ppg_hr_g,
            'hr_error_red': hr_error_r,
            'hr_error_green': hr_error_g,
            'snr_red': snr_r,
            'snr_green': snr_g,
            'r_variation': np.std(r_sig),
            'g_variation': np.std(g_sig),
            'fps': fps,
            'n_frames': actual_frames,
        }

        print(f"\n  {pair} (GT HR={gt_hr} bpm):")
        print(f"    Frames: {actual_frames}, FPS: {fps:.1f}")
        print(f"    Red channel:   PPG HR={ppg_hr_r:.1f}, error={hr_error_r:.1f} bpm, SNR={snr_r:.2f}")
        print(f"    Green channel: PPG HR={ppg_hr_g:.1f}, error={hr_error_g:.1f} bpm, SNR={snr_g:.2f}")
        print(f"    Signal variation (R/G): {np.std(r_sig):.2f} / {np.std(g_sig):.2f}")

    return results


def check_ecg_waveform_quality():
    """Check ECG waveform quality - look for R peaks, noise, artifacts."""
    print("\n" + "=" * 70)
    print("8. ECG WAVEFORM QUALITY (R-peak detection)")
    print("=" * 70)

    try:
        from scipy import signal as sig
    except ImportError:
        print("scipy not available, skipping waveform check")
        return {}

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

        # Find monitor-filtered column
        if 'ecg_counts_filt_monitor' in header:
            ecg_idx = header.index('ecg_counts_filt_monitor')
        else:
            ecg_idx = 1

        ecg = data[:, ecg_idx]
        sr = 250  # Hz

        # Normalize
        ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)

        # Detect R-peaks using simple threshold
        # Use 10-second window from middle of recording
        mid = len(ecg) // 2
        window_start = max(0, mid - 1250)  # 5s before
        window_end = min(len(ecg), mid + 1250)  # 5s after
        ecg_window = ecg_norm[window_start:window_end]

        # Find peaks with minimum distance (at least 200ms apart = 50 samples at 250Hz)
        threshold = 0.5 * np.std(ecg_window)
        peaks, props = sig.find_peaks(ecg_window, height=threshold, distance=50)

        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / sr  # in seconds
            detected_hr = 60.0 / np.mean(rr_intervals)
            rr_std = np.std(rr_intervals)
            hr_error = abs(detected_hr - gt_hr)
        else:
            detected_hr = 0
            rr_std = 0
            hr_error = gt_hr

        # Check signal quality
        signal_range = np.max(ecg) - np.min(ecg)

        results[pair] = {
            'n_peaks': len(peaks),
            'detected_hr': detected_hr,
            'gt_hr': gt_hr,
            'hr_error': hr_error,
            'rr_std': rr_std,
            'signal_range': signal_range,
            'ecg_std': np.std(ecg),
        }

        print(f"\n  {pair} (GT HR={gt_hr} bpm):")
        print(f"    R-peaks in 10s window: {len(peaks)}")
        print(f"    Detected HR: {detected_hr:.1f} bpm (error: {hr_error:.1f})")
        print(f"    RR interval std: {rr_std:.4f}s")
        print(f"    Signal range: {signal_range:.1f}, std: {np.std(ecg):.1f}")

    return results


if __name__ == "__main__":
    print("ECG Reconstruction - Comprehensive Data Quality Check v2")
    print(f"Samples directory: {SAMPLES_DIR}")
    print(f"Checking {len(SAMPLE_PAIRS)} representative samples\n")

    # 1. Overview
    users, states, durations, ecg_samples, hr_values = check_all_metadata()

    # 2. Video properties
    video_results = check_video_properties()

    # 3. ECG data
    ecg_results = check_ecg_data()

    # 4. IMU data
    imu_results = check_imu_data()

    # 5. Audio check
    audio_results = check_audio()

    # 6. Time alignment
    alignment_results = check_time_alignment()

    # 7. PPG extractability
    ppg_results = check_ppg_extractability()

    # 8. ECG waveform quality
    ecg_waveform_results = check_ecg_waveform_quality()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nTotal samples: 98")
    print(f"Users: {len(users)}")
    print(f"Duration range: {min(durations):.1f}-{max(durations):.1f}s")

    if video_results:
        fps_vals = [v['fps'] for v in video_results.values()]
        print(f"\nVideo FPS range: {min(fps_vals):.1f}-{max(fps_vals):.1f}")
        r_dom = sum(1 for v in video_results.values() if v['r_dominant'])
        print(f"Red channel dominant: {r_dom}/{len(video_results)}")

    if ppg_results:
        good_r = sum(1 for v in ppg_results.values() if v['hr_error_red'] < 10)
        good_g = sum(1 for v in ppg_results.values() if v['hr_error_green'] < 10)
        print(f"\nPPG HR accuracy (<10 bpm error):")
        print(f"  Red channel:   {good_r}/{len(ppg_results)}")
        print(f"  Green channel: {good_g}/{len(ppg_results)}")

    if ecg_waveform_results:
        good_ecg = sum(1 for v in ecg_waveform_results.values() if v['hr_error'] < 10)
        print(f"\nECG R-peak detection (<10 bpm error): {good_ecg}/{len(ecg_waveform_results)}")

    print("\nDone! Results can be used to write data_quality_report_v2.md")
