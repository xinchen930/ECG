"""
Check PPG quality for a batch of samples and generate/update quality CSV.
Requires: opencv-python, scipy, numpy, pandas

Usage:
    # Check all samples (updates existing CSV)
    python scripts/check_batch_quality.py

    # Check only batch 2 samples
    python scripts/check_batch_quality.py --batch batch_2

    # Check specific pair range
    python scripts/check_batch_quality.py --start 98 --end 134
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal


def bandpass_filter(sig, fs, lowcut=0.5, highcut=4.0, order=3):
    nyq = 0.5 * fs
    b, a = scipy_signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return scipy_signal.filtfilt(b, a, sig)


def get_hr(sig, fs):
    sig = sig - sig.mean()
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(sig))
    mask = (freqs >= 50 / 60) & (freqs <= 180 / 60)
    if not mask.any():
        return None, 0, False
    hr_freqs, hr_mags = freqs[mask], fft_mag[mask]
    idx = np.argmax(hr_mags)
    prominence = hr_mags[idx] / (np.median(hr_mags) + 1e-8)
    return hr_freqs[idx] * 60, hr_mags[idx] ** 2 / (hr_mags ** 2).sum(), prominence > 2.0


def analyze_pair(pair_dir):
    video_path = os.path.join(pair_dir, "video_0.mp4")
    meta_path = os.path.join(pair_dir, "metadata.json")
    pair_name = os.path.basename(pair_dir)

    if not os.path.exists(video_path) or not os.path.exists(meta_path):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    duration_s = total_frames / fps if fps > 0 else 0
    bitrate_kbps = (file_size_mb * 8 * 1024) / duration_s if duration_s > 0 else 0

    # Skip first 1 second, read 10 seconds
    skip = int(1 * fps)
    max_frames = int(10 * fps)
    for _ in range(skip):
        cap.read()

    rgb = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        rgb.append([frame[:, :, 2].mean(), frame[:, :, 1].mean(), frame[:, :, 0].mean()])
    cap.release()

    if len(rgb) < 60:  # Need at least 2 seconds
        return None
    rgb = np.array(rgb)

    red_std = rgb[:, 0].std()
    red_range = rgb[:, 0].max() - rgb[:, 0].min()
    red_norm = (rgb[:, 0] - rgb[:, 0].mean()) / (red_std + 1e-8)
    red_filt = bandpass_filter(red_norm, fps, 0.7, 4.0)
    ppg_hr, snr, is_valid = get_hr(red_filt, fps)

    gt_hr = meta.get("heart_rate")
    user = meta.get("phone_user", "unknown")
    hr_error = abs(ppg_hr - gt_hr) if is_valid and ppg_hr and gt_hr else None
    quality = (
        "good" if hr_error is not None and hr_error < 10
        else ("moderate" if hr_error is not None and hr_error < 20
              else "poor")
    )

    return {
        "pair": pair_name,
        "user": user,
        "bitrate_kbps": round(bitrate_kbps, 1),
        "red_std": round(red_std, 2),
        "red_range": round(red_range, 1),
        "ppg_hr": round(ppg_hr, 1) if ppg_hr else None,
        "gt_hr": gt_hr,
        "hr_error": round(hr_error, 1) if hr_error else None,
        "snr": round(snr, 3),
        "is_valid": is_valid,
        "quality": quality,
    }


def main():
    parser = argparse.ArgumentParser(description="Check PPG quality for training samples")
    parser.add_argument("--samples-dir", default="training_data/samples")
    parser.add_argument("--batch", type=str, default=None,
                        help="Check only this batch (e.g. 'batch_2')")
    parser.add_argument("--start", type=int, default=None, help="Start pair ID")
    parser.add_argument("--end", type=int, default=None, help="End pair ID (inclusive)")
    parser.add_argument("--output", default="eval_results/ppg_analysis_all_samples.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    samples_dir = args.samples_dir

    # Determine which pairs to check
    all_pairs = sorted(
        d for d in os.listdir(samples_dir) if d.startswith("pair_")
    )

    if args.batch:
        batch_index_path = os.path.join(os.path.dirname(samples_dir), "batch_index.json")
        with open(batch_index_path) as f:
            batch_data = json.load(f)
        if args.batch in batch_data:
            pair_ids = set(batch_data[args.batch]["pair_ids"])
            all_pairs = [p for p in all_pairs if int(p.split("_")[1]) in pair_ids]
        else:
            print(f"Unknown batch: {args.batch}, available: {list(batch_data.keys())}")
            sys.exit(1)

    if args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end or 9999
        all_pairs = [
            p for p in all_pairs if start <= int(p.split("_")[1]) <= end
        ]

    print(f"Checking {len(all_pairs)} samples...")

    # Load existing CSV if it exists (to merge)
    existing = {}
    if os.path.exists(args.output):
        df = pd.read_csv(args.output)
        for _, row in df.iterrows():
            existing[row["pair"]] = row.to_dict()

    # Analyze new pairs
    new_count = 0
    for i, pair in enumerate(all_pairs):
        pair_dir = os.path.join(samples_dir, pair)
        result = analyze_pair(pair_dir)
        if result:
            existing[pair] = result
            new_count += 1
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(all_pairs)}...")

    # Save merged results
    all_results = sorted(existing.values(), key=lambda x: x["pair"])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(all_results)} entries to {args.output}")
    print(f"  New/updated: {new_count}")

    # Print quality summary
    for q in ["good", "moderate", "poor"]:
        c = len([r for r in all_results if r.get("quality") == q])
        print(f"  {q}: {c}")


if __name__ == "__main__":
    main()
