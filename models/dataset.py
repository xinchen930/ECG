"""
Video-to-ECG Dataset with windowed sampling and user-level splitting.
Supports optional IMU input (Scheme B).
Supports quality-based filtering of samples.
Supports STMap (Spatio-Temporal Map) input mode (Scheme I).
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_quality_data(quality_csv: str):
    """Load quality data from CSV file, return dict mapping pair name to quality."""
    if not os.path.exists(quality_csv):
        return None
    df = pd.read_csv(quality_csv)
    return dict(zip(df['pair'], df['quality']))


def load_metadata(samples_dir: str, quality_filter: str = None, quality_csv: str = None):
    """
    Load metadata for all sample pairs, returning list of dicts.

    Args:
        samples_dir: Path to the samples directory
        quality_filter: Comma-separated quality levels to include, e.g. "good" or "good,moderate"
                       If None, include all samples.
        quality_csv: Path to quality CSV file (default: eval_results/ppg_analysis_all_samples.csv)
    """
    samples_dir = Path(samples_dir)

    # Load quality data if filtering is requested
    quality_map = None
    if quality_filter:
        if quality_csv is None:
            # Default path relative to project root
            quality_csv = "eval_results/ppg_analysis_all_samples.csv"
        quality_map = load_quality_data(quality_csv)
        if quality_map is None:
            print(f"Warning: quality_csv not found at {quality_csv}, skipping quality filter")
        else:
            allowed_qualities = set(q.strip() for q in quality_filter.split(","))
            print(f"Quality filter: keeping only {allowed_qualities} samples")

    records = []
    filtered_count = 0
    for pair_dir in sorted(samples_dir.iterdir()):
        meta_path = pair_dir / "metadata.json"
        if not meta_path.exists():
            continue

        # Apply quality filter if enabled
        pair_name = pair_dir.name
        if quality_map is not None:
            sample_quality = quality_map.get(pair_name)
            if sample_quality not in allowed_qualities:
                filtered_count += 1
                continue

        with open(meta_path) as f:
            meta = json.load(f)
        meta["pair_dir"] = str(pair_dir)
        meta["pair_name"] = pair_name
        records.append(meta)

    if quality_filter and quality_map:
        print(f"Quality filter: {filtered_count} samples excluded, {len(records)} remaining")

    return records


def _detect_ecg_gaps(ecg_csv_path, gap_threshold_ms=100):
    """
    Detect timestamp gaps in ECG data.

    Args:
        ecg_csv_path: Path to ecg.csv with timestamp_ms column
        gap_threshold_ms: Gap threshold in ms (normal interval ~4ms at 250Hz)

    Returns:
        List of sample indices where gaps occur (gap is between index i and i+1).
        Returns empty list if timestamp_ms column not found.
    """
    try:
        df = pd.read_csv(ecg_csv_path, usecols=["timestamp_ms"])
        ts = df["timestamp_ms"].values
        diffs = np.diff(ts)
        gap_indices = np.where(diffs > gap_threshold_ms)[0]
        return gap_indices.tolist()
    except (KeyError, ValueError):
        # timestamp_ms column missing or unreadable; skip gap detection
        return []


def build_window_index(records, window_sec=10, stride_sec=5, video_fps=30, ecg_sr=250,
                       gap_aware=True, gap_threshold_ms=100):
    """
    Build an index of (pair_idx, start_frame, end_frame, ecg_start, ecg_end)
    for all valid windows across all sample pairs.

    Args:
        records: List of metadata dicts
        window_sec: Window length in seconds
        stride_sec: Stride between windows in seconds
        video_fps: Video frame rate
        ecg_sr: ECG sampling rate
        gap_aware: If True, skip windows that span ECG timestamp gaps
        gap_threshold_ms: Minimum gap size to detect (ms). Normal ECG interval is ~4ms.
    """
    window_frames = int(window_sec * video_fps)
    stride_frames = int(stride_sec * video_fps)
    window_ecg = int(window_sec * ecg_sr)

    index = []
    skipped_gap_windows = 0
    for i, rec in enumerate(records):
        video_path = os.path.join(rec["pair_dir"], "video_0.mp4")
        if not os.path.exists(video_path):
            continue
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        n_ecg = rec.get("ecg_samples", 0)

        # Detect ECG gaps for this pair
        gap_indices = set()
        if gap_aware:
            ecg_csv_path = os.path.join(rec["pair_dir"], "ecg.csv")
            if os.path.exists(ecg_csv_path):
                gap_indices = set(_detect_ecg_gaps(ecg_csv_path, gap_threshold_ms))

        max_ecg_windows = n_ecg // window_ecg
        max_video_windows = (n_frames - window_frames) // stride_frames + 1
        n_windows = min(max_video_windows, max_ecg_windows)

        for w in range(max(0, n_windows)):
            vf_start = w * stride_frames
            vf_end = vf_start + window_frames
            ecg_start = int(w * stride_sec * ecg_sr)
            ecg_end = ecg_start + window_ecg
            if vf_end <= n_frames and ecg_end <= n_ecg:
                # Check if any gap falls within this ECG window
                if gap_aware and gap_indices:
                    has_gap = any(ecg_start <= g < ecg_end for g in gap_indices)
                    if has_gap:
                        skipped_gap_windows += 1
                        continue
                index.append((i, vf_start, vf_end, ecg_start, ecg_end))

    if gap_aware and skipped_gap_windows > 0:
        print(f"Gap-aware windowing: skipped {skipped_gap_windows} windows spanning ECG gaps")

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


def split_random(records, train_ratio=0.9, val_ratio=0.05, seed=42):
    """
    Split record indices randomly (easier task, useful for debugging).

    Args:
        records: List of record dicts
        train_ratio: Fraction for training (default 0.9)
        val_ratio: Fraction for validation (default 0.05, test gets remaining 0.05)
        seed: Random seed for reproducibility
    """
    import random
    n = len(records)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

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
                 use_diff_frames=False, use_1d_signal=False, use_green_channel=False,
                 input_type=None,
                 stmap_grid_h=8, stmap_grid_w=8, stmap_channels="rgb",
                 stmap_multiscale=False, stmap_scales=None):
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
        self.use_green_channel = use_green_channel
        # input_type: flexible channel selection or representation mode
        # Supported 1D modes: "green_channel", "red_channel", "rgb_channels", "all_channels", "rgb_means"
        # STMap mode: "stmap" — builds Spatio-Temporal Map from video frames
        # Falls back to use_green_channel boolean for backward compatibility
        self.input_type = input_type
        self.imu_window_len = int(imu_sr * window_sec)

        # STMap builder (only when input_type == "stmap")
        self.use_stmap = (input_type == "stmap")
        self.stmap_multiscale = stmap_multiscale
        self.stmap_scales = stmap_scales
        if self.use_stmap:
            from stmap_builder import STMapBuilder
            self.stmap_builder = STMapBuilder(
                n_spatial_h=stmap_grid_h,
                n_spatial_w=stmap_grid_w,
                channels=stmap_channels,
            )

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

        # Determine target resolution: 0/None means use native resolution
        target_h = self.img_h
        target_w = self.img_w
        if not target_h or not target_w:
            # Read native resolution from video
            native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            target_h = target_h or native_h
            target_w = target_w or native_w

        frames = []
        for _ in range(vf_end - vf_start):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            else:
                # Only resize if native resolution differs from target
                if frame.shape[0] != target_h or frame.shape[1] != target_w:
                    frame = cv2.resize(frame, (target_w, target_h))
            frames.append(frame)
        cap.release()

        video = np.stack(frames).astype(np.float32) / 255.0  # (T, H, W, 3)

        if self.use_stmap:
            # Build STMap: (T, H, W, 3) -> (T, 3, H, W) -> STMapBuilder -> (C, N_spatial, T)
            video_chw = np.transpose(video, (0, 3, 1, 2))  # (T, 3, H, W)
            if self.stmap_multiscale:
                stmap = self.stmap_builder.build_multiscale(
                    video_chw, scales=self.stmap_scales
                )
            else:
                stmap = self.stmap_builder.build(video_chw)
            # stmap: (C, N_spatial, T) — z-normalize per channel
            for c in range(stmap.shape[0]):
                ch = stmap[c]  # (N_spatial, T)
                ch_mean = ch.mean()
                ch_std = ch.std() + 1e-8
                stmap[c] = (ch - ch_mean) / ch_std
            video_tensor = stmap  # (C, N_spatial, T)
        elif self.use_1d_signal:
            # Extract per-frame ROI statistics
            # video: (T, H, W, 3) in BGR format (OpenCV default)
            # Determine channel selection via input_type (preferred) or legacy use_green_channel
            effective_type = self.input_type
            if effective_type is None:
                # Backward compatibility: map boolean to input_type
                effective_type = "green_channel" if self.use_green_channel else "rgb_means"

            if effective_type == "green_channel":
                # Green channel (index 1 in BGR) - sensitive to blood hemoglobin
                signal = video[:, :, :, 1:2].mean(axis=(1, 2))  # (T, 1)
            elif effective_type == "red_channel":
                # Red channel (index 2 in BGR) - best for finger transmissive PPG
                signal = video[:, :, :, 2:3].mean(axis=(1, 2))  # (T, 1)
            elif effective_type in ("rgb_channels", "all_channels", "rgb_means"):
                # All three BGR channels as separate signals
                signal = video.mean(axis=(1, 2))  # (T, 3)
            else:
                raise ValueError(f"Unknown input_type: {effective_type}. "
                                 f"Use 'green_channel', 'red_channel', 'rgb_channels', 'all_channels', or 'rgb_means'.")
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


def create_datasets(cfg, merge_val_to_train=False):
    """Create train/val/test datasets from config dict.

    Args:
        cfg: Config dict
        merge_val_to_train: If True, merge validation set into training set.
                           Use this when using test set for early stopping (debug mode).
    """
    data_cfg = cfg["data"]
    split_cfg = cfg["split"]

    # Quality filtering
    quality_filter = data_cfg.get("quality_filter", None)
    quality_csv = data_cfg.get("quality_csv", None)

    records = load_metadata(
        data_cfg["samples_dir"],
        quality_filter=quality_filter,
        quality_csv=quality_csv
    )
    print(f"Loaded {len(records)} sample pairs")

    # Split mode: "user" (harder, no data leakage) or "random" (easier, for debugging)
    split_mode = split_cfg.get("mode", "user")
    seed = cfg.get("train", {}).get("seed", 42)

    if split_mode == "random":
        if merge_val_to_train:
            # When using test as val, use 90/0/10 split (no separate validation)
            train_ratio = split_cfg.get("train_ratio", 0.9)
            train_pairs, val_pairs, test_pairs = split_random(
                records, train_ratio=train_ratio, val_ratio=0.0, seed=seed
            )
        else:
            train_ratio = split_cfg.get("train_ratio", 0.9)
            val_ratio = split_cfg.get("val_ratio", 0.05)
            train_pairs, val_pairs, test_pairs = split_random(
                records, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
            )
        print(f"Split mode: RANDOM (easier, same user may appear in train/test)")
    else:
        train_pairs, val_pairs, test_pairs = split_by_user(
            records,
            split_cfg["train_users"],
            split_cfg["val_users"],
            split_cfg["test_users"],
        )
        # Merge val into train if requested
        if merge_val_to_train and val_pairs:
            train_pairs = train_pairs + val_pairs
            val_pairs = []
        print(f"Split mode: USER (harder, no data leakage)")

    print(f"Split: train={len(train_pairs)} pairs, val={len(val_pairs)}, test={len(test_pairs)}")

    gap_aware = data_cfg.get("gap_aware", True)  # Enable gap-aware windowing by default
    gap_threshold_ms = data_cfg.get("gap_threshold_ms", 100)

    full_index = build_window_index(
        records,
        window_sec=data_cfg["window_seconds"],
        stride_sec=data_cfg["stride_seconds"],
        video_fps=data_cfg["video_fps"],
        ecg_sr=data_cfg["ecg_sr"],
        gap_aware=gap_aware,
        gap_threshold_ms=gap_threshold_ms,
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
    use_green_channel = data_cfg.get("use_green_channel", False)
    # input_type overrides use_green_channel when set (preferred config key)
    input_type = data_cfg.get("input_type", None)

    # STMap parameters (only used when input_type == "stmap")
    stmap_grid_h = data_cfg.get("stmap_grid_h", 8)
    stmap_grid_w = data_cfg.get("stmap_grid_w", 8)
    stmap_channels = data_cfg.get("stmap_channels", "rgb")
    stmap_multiscale = data_cfg.get("stmap_multiscale", False)
    stmap_scales = data_cfg.get("stmap_scales", None)  # e.g. [4, 8, 16]

    # Resolution: support null/0/"auto" for native resolution
    img_h = data_cfg.get("img_height", 64)
    img_w = data_cfg.get("img_width", 64)
    if img_h in (None, "auto", 0):
        img_h = 0  # Signal to use native resolution
    if img_w in (None, "auto", 0):
        img_w = 0

    kwargs = dict(
        img_h=img_h,
        img_w=img_w,
        use_imu=use_imu,
        imu_sr=imu_sr,
        window_sec=data_cfg["window_seconds"],
        use_diff_frames=use_diff_frames,
        use_1d_signal=use_1d_signal,
        use_green_channel=use_green_channel,
        input_type=input_type,
        stmap_grid_h=stmap_grid_h,
        stmap_grid_w=stmap_grid_w,
        stmap_channels=stmap_channels,
        stmap_multiscale=stmap_multiscale,
        stmap_scales=stmap_scales,
    )

    train_ds = VideoECGDataset(records, train_index, **kwargs)
    val_ds = VideoECGDataset(records, val_index, **kwargs)
    test_ds = VideoECGDataset(records, test_index, **kwargs)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    import yaml
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/scheme_c.yaml"
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
