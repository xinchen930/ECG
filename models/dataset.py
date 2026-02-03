"""
Video-to-ECG Dataset with windowed sampling and user-level splitting.
Supports optional IMU input (Scheme B).
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_metadata(samples_dir: str):
    """Load metadata for all sample pairs, returning list of dicts."""
    samples_dir = Path(samples_dir)
    records = []
    for pair_dir in sorted(samples_dir.iterdir()):
        meta_path = pair_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        meta["pair_dir"] = str(pair_dir)
        records.append(meta)
    return records


def build_window_index(records, window_sec=10, stride_sec=5, video_fps=30, ecg_sr=250):
    """
    Build an index of (pair_idx, start_frame, end_frame, ecg_start, ecg_end)
    for all valid windows across all sample pairs.
    """
    window_frames = int(window_sec * video_fps)
    stride_frames = int(stride_sec * video_fps)
    window_ecg = int(window_sec * ecg_sr)

    index = []
    for i, rec in enumerate(records):
        video_path = os.path.join(rec["pair_dir"], "video_0.mp4")
        if not os.path.exists(video_path):
            continue
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        n_ecg = rec.get("ecg_samples", 0)

        max_ecg_windows = n_ecg // window_ecg
        max_video_windows = (n_frames - window_frames) // stride_frames + 1
        n_windows = min(max_video_windows, max_ecg_windows)

        for w in range(max(0, n_windows)):
            vf_start = w * stride_frames
            vf_end = vf_start + window_frames
            ecg_start = int(w * stride_sec * ecg_sr)
            ecg_end = ecg_start + window_ecg
            if vf_end <= n_frames and ecg_end <= n_ecg:
                index.append((i, vf_start, vf_end, ecg_start, ecg_end))

    return index


def split_by_user(records, train_users, val_users, test_users):
    """Split record indices by user identity."""
    train_idx, val_idx, test_idx = [], [], []
    for i, rec in enumerate(records):
        user = rec["phone_user"]
        if user in train_users:
            train_idx.append(i)
        elif user in val_users:
            val_idx.append(i)
        elif user in test_users:
            test_idx.append(i)
    return train_idx, val_idx, test_idx


class VideoECGDataset(Dataset):
    """
    PyTorch Dataset yielding (video_frames, ecg_signal) or
    (video_frames, imu_segment, ecg_signal) windows.

    video_frames: (T_v, 3, H, W) float32 tensor, normalized to [0,1]
    imu_segment:  (T_imu, 6) float32 tensor (acc_xyz + gyro_xyz), z-normalized
    ecg_signal:   (T_ecg,) float32 tensor (monitor-filtered ECG counts)
    """

    def __init__(self, records, window_index, img_h=64, img_w=64,
                 ecg_col="ecg_counts_filt_monitor", normalize_ecg=True,
                 use_imu=False, imu_sr=100, window_sec=10,
                 use_diff_frames=False, use_1d_signal=False):
        self.records = records
        self.window_index = window_index
        self.img_h = img_h
        self.img_w = img_w
        self.ecg_col = ecg_col
        self.normalize_ecg = normalize_ecg
        self.use_imu = use_imu
        self.imu_sr = imu_sr
        self.window_sec = window_sec
        self.use_diff_frames = use_diff_frames
        self.use_1d_signal = use_1d_signal
        self.imu_window_len = int(imu_sr * window_sec)

        # Pre-load ECG data for all referenced pairs
        self._ecg_cache = {}
        pair_indices = set(idx[0] for idx in window_index)
        for pi in pair_indices:
            ecg_path = os.path.join(records[pi]["pair_dir"], "ecg.csv")
            df = pd.read_csv(ecg_path)
            self._ecg_cache[pi] = df[ecg_col].values.astype(np.float32)

        # Compute per-pair ECG stats for normalization
        if normalize_ecg:
            self._ecg_mean = {}
            self._ecg_std = {}
            for pi, arr in self._ecg_cache.items():
                self._ecg_mean[pi] = arr.mean()
                self._ecg_std[pi] = arr.std() + 1e-8

        # Pre-load IMU data if needed
        self._imu_cache = {}
        if use_imu:
            imu_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
            self._imu_mean = {}
            self._imu_std = {}
            for pi in pair_indices:
                imu_path = os.path.join(records[pi]["pair_dir"], "imu.csv")
                df = pd.read_csv(imu_path)
                arr = df[imu_cols].values.astype(np.float32)
                self._imu_cache[pi] = arr
                self._imu_mean[pi] = arr.mean(axis=0)
                self._imu_std[pi] = arr.std(axis=0) + 1e-8

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        pair_idx, vf_start, vf_end, ecg_start, ecg_end = self.window_index[idx]
        rec = self.records[pair_idx]

        # --- Load video frames ---
        video_path = os.path.join(rec["pair_dir"], "video_0.mp4")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, vf_start)

        frames = []
        for _ in range(vf_end - vf_start):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (self.img_w, self.img_h))
            frames.append(frame)
        cap.release()

        video = np.stack(frames).astype(np.float32) / 255.0  # (T, H, W, 3)

        if self.use_1d_signal:
            # Extract per-frame ROI statistics: mean of each RGB channel
            # video: (T, H, W, 3) -> signal: (T, 3)
            signal = video.mean(axis=(1, 2))  # (T, 3) - RGB means per frame
            # Z-normalize per channel
            signal_mean = signal.mean(axis=0, keepdims=True)
            signal_std = signal.std(axis=0, keepdims=True) + 1e-8
            signal = (signal - signal_mean) / signal_std
            video_tensor = torch.from_numpy(signal.astype(np.float32))
        elif self.use_diff_frames:
            # Normalized difference frames: (f[t+1]-f[t])/(f[t+1]+f[t]+1e-7)
            diff = (video[1:] - video[:-1]) / (video[1:] + video[:-1] + 1e-7)
            # Z-normalize diff frames
            diff_std = diff.std() + 1e-8
            diff = diff / diff_std
            # Raw frames (z-normalize per-window)
            raw = video[:-1]
            raw_mean = raw.mean()
            raw_std = raw.std() + 1e-8
            raw = (raw - raw_mean) / raw_std
            # Concatenate: (T-1, H, W, 6) -> (T-1, 6, H, W)
            combined = np.concatenate([diff, raw], axis=-1)
            combined = np.transpose(combined, (0, 3, 1, 2))
            video_tensor = torch.from_numpy(combined)
        else:
            video = np.transpose(video, (0, 3, 1, 2))  # (T, 3, H, W)
            video_tensor = torch.from_numpy(video)

        # --- Load ECG ---
        ecg = self._ecg_cache[pair_idx][ecg_start:ecg_end].copy()
        if self.normalize_ecg:
            ecg = (ecg - self._ecg_mean[pair_idx]) / self._ecg_std[pair_idx]
        ecg_tensor = torch.from_numpy(ecg)

        if not self.use_imu:
            return video_tensor, ecg_tensor

        # --- Load IMU ---
        # Map video window position to IMU samples proportionally
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if False else vf_end
        # Use ecg timing ratio to find IMU position
        imu_data = self._imu_cache[pair_idx]
        total_imu = len(imu_data)
        total_ecg = len(self._ecg_cache[pair_idx])
        # proportional mapping
        imu_start = int(ecg_start / total_ecg * total_imu)
        imu_end = imu_start + self.imu_window_len

        if imu_end <= total_imu:
            imu_seg = imu_data[imu_start:imu_end].copy()
        else:
            # Pad with last value
            imu_seg = np.zeros((self.imu_window_len, 6), dtype=np.float32)
            avail = total_imu - imu_start
            if avail > 0:
                imu_seg[:avail] = imu_data[imu_start:total_imu]
                imu_seg[avail:] = imu_data[-1]

        # Z-normalize per-pair
        imu_seg = (imu_seg - self._imu_mean[pair_idx]) / self._imu_std[pair_idx]
        imu_tensor = torch.from_numpy(imu_seg)

        return video_tensor, imu_tensor, ecg_tensor


def create_datasets(cfg):
    """Create train/val/test datasets from config dict."""
    data_cfg = cfg["data"]
    split_cfg = cfg["split"]

    records = load_metadata(data_cfg["samples_dir"])
    print(f"Loaded {len(records)} sample pairs")

    train_pairs, val_pairs, test_pairs = split_by_user(
        records,
        split_cfg["train_users"],
        split_cfg["val_users"],
        split_cfg["test_users"],
    )
    print(f"Split: train={len(train_pairs)} pairs, val={len(val_pairs)}, test={len(test_pairs)}")

    full_index = build_window_index(
        records,
        window_sec=data_cfg["window_seconds"],
        stride_sec=data_cfg["stride_seconds"],
        video_fps=data_cfg["video_fps"],
        ecg_sr=data_cfg["ecg_sr"],
    )
    print(f"Total windows: {len(full_index)}")

    train_set = set(train_pairs)
    val_set = set(val_pairs)
    test_set = set(test_pairs)

    train_index = [w for w in full_index if w[0] in train_set]
    val_index = [w for w in full_index if w[0] in val_set]
    test_index = [w for w in full_index if w[0] in test_set]

    print(f"Windows: train={len(train_index)}, val={len(val_index)}, test={len(test_index)}")

    use_imu = data_cfg.get("use_imu", False)
    imu_sr = data_cfg.get("imu_sr", 100)
    use_diff_frames = data_cfg.get("use_diff_frames", False)
    use_1d_signal = data_cfg.get("use_1d_signal", False)

    kwargs = dict(
        img_h=data_cfg["img_height"],
        img_w=data_cfg["img_width"],
        use_imu=use_imu,
        imu_sr=imu_sr,
        window_sec=data_cfg["window_seconds"],
        use_diff_frames=use_diff_frames,
        use_1d_signal=use_1d_signal,
    )

    train_ds = VideoECGDataset(records, train_index, **kwargs)
    val_ds = VideoECGDataset(records, val_index, **kwargs)
    test_ds = VideoECGDataset(records, test_index, **kwargs)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    import yaml
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/scheme_a.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    train_ds, val_ds, test_ds = create_datasets(cfg)
    print(f"\nSample shapes:")
    sample = train_ds[0]
    if len(sample) == 2:
        video, ecg = sample
        print(f"  video: {video.shape}  (T, C, H, W)")
        print(f"  ecg:   {ecg.shape}    (T_ecg,)")
    else:
        video, imu, ecg = sample
        print(f"  video: {video.shape}  (T, C, H, W)")
        print(f"  imu:   {imu.shape}    (T_imu, 6)")
        print(f"  ecg:   {ecg.shape}    (T_ecg,)")
