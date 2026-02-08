"""
Diagnostic script to identify root cause of near-zero Pearson r in Video->ECG models.

Checks:
1. Model output: Is it a flat line / constant?
2. Data alignment: Are video frames and ECG properly aligned in time?
3. ECG normalization: Are values reasonable?
4. Video data quality: Are video frames informative (not saturated)?
5. Loss function behavior
6. Gradient flow
"""

import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

import json
import numpy as np
import pandas as pd
import torch
import yaml
import cv2
from pathlib import Path

# ====================================================================
# TEST 1: Check ECG data statistics and normalization
# ====================================================================
def test_ecg_data():
    print("=" * 70)
    print("TEST 1: ECG Data Statistics & Normalization")
    print("=" * 70)

    samples_dir = Path("/home/xinchen/ECG/training_data/samples")
    ecg_col = "ecg_counts_filt_monitor"

    all_means, all_stds, all_mins, all_maxs = [], [], [], []
    all_ranges = []

    for pair_dir in sorted(samples_dir.iterdir())[:20]:
        ecg_path = pair_dir / "ecg.csv"
        if not ecg_path.exists():
            continue
        df = pd.read_csv(ecg_path)
        vals = df[ecg_col].values

        m, s = vals.mean(), vals.std()
        mn, mx = vals.min(), vals.max()
        all_means.append(m)
        all_stds.append(s)
        all_mins.append(mn)
        all_maxs.append(mx)
        all_ranges.append(mx - mn)

        print(f"  {pair_dir.name}: mean={m:.2f}, std={s:.2f}, min={mn:.2f}, max={mx:.2f}, range={mx-mn:.2f}")

    print(f"\n  === Across samples (first 20) ===")
    print(f"  Mean of means: {np.mean(all_means):.2f}")
    print(f"  Mean of stds:  {np.mean(all_stds):.2f}")
    print(f"  Mean of ranges: {np.mean(all_ranges):.2f}")
    print(f"  Min of mins: {np.min(all_mins):.2f}")
    print(f"  Max of maxs: {np.max(all_maxs):.2f}")

    # Check: after z-normalization per pair, what do windows look like?
    print(f"\n  === After per-pair z-normalization ===")
    for pair_dir in sorted(samples_dir.iterdir())[:5]:
        ecg_path = pair_dir / "ecg.csv"
        if not ecg_path.exists():
            continue
        df = pd.read_csv(ecg_path)
        vals = df[ecg_col].values.astype(np.float32)
        m, s = vals.mean(), vals.std() + 1e-8
        normed = (vals - m) / s

        # Check windows
        ecg_sr = 250
        window_ecg = 10 * ecg_sr  # 2500
        for w in range(min(3, len(vals) // window_ecg)):
            segment = normed[w * window_ecg: (w+1) * window_ecg]
            print(f"  {pair_dir.name} window {w}: mean={segment.mean():.4f}, std={segment.std():.4f}, "
                  f"min={segment.min():.4f}, max={segment.max():.4f}")

    return all_means, all_stds


# ====================================================================
# TEST 2: Check Video Data Quality
# ====================================================================
def test_video_data():
    print("\n" + "=" * 70)
    print("TEST 2: Video Data Quality")
    print("=" * 70)

    samples_dir = Path("/home/xinchen/ECG/training_data/samples")

    for pair_dir in sorted(samples_dir.iterdir())[:5]:
        video_path = pair_dir / "video_0.mp4"
        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Read first 300 frames (10 seconds)
        frames = []
        for _ in range(min(300, n_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        frames = np.stack(frames).astype(np.float32)
        print(f"\n  {pair_dir.name}: {n_frames} frames, {fps:.1f} fps, {width}x{height}")

        # Per-channel statistics (BGR format from OpenCV)
        for ch_idx, ch_name in enumerate(["Blue", "Green", "Red"]):
            ch = frames[:, :, :, ch_idx]
            print(f"    {ch_name}: mean={ch.mean():.1f}, std={ch.std():.1f}, "
                  f"min={ch.min():.0f}, max={ch.max():.0f}")

        # Temporal variation (the PPG signal)
        # Mean per frame per channel
        temporal_means = frames.mean(axis=(1, 2))  # (T, 3)
        for ch_idx, ch_name in enumerate(["Blue", "Green", "Red"]):
            ch_mean = temporal_means[:, ch_idx]
            ch_std = ch_mean.std()
            ch_range = ch_mean.max() - ch_mean.min()
            print(f"    {ch_name} temporal: mean_of_means={ch_mean.mean():.1f}, "
                  f"std={ch_std:.4f}, range={ch_range:.4f}")

        # Check if frame-to-frame variation is very small (PPG signal amplitude)
        # After normalization to [0,1], what's the temporal std?
        normed_frames = frames / 255.0
        temporal_normed = normed_frames.mean(axis=(1, 2))  # (T, 3)
        for ch_idx, ch_name in enumerate(["Blue", "Green", "Red"]):
            ch = temporal_normed[:, ch_idx]
            print(f"    {ch_name} normed temporal: mean={ch.mean():.6f}, std={ch.std():.6f}, "
                  f"range={ch.max()-ch.min():.6f}")

        # Check if video is basically constant (saturated red from finger PPG)
        first_frame = frames[0]
        last_frame = frames[-1]
        print(f"    Frame diff (first vs last): {np.abs(first_frame - last_frame).mean():.2f}")


# ====================================================================
# TEST 3: Check the signal the model actually receives
# ====================================================================
def test_model_input_signal():
    print("\n" + "=" * 70)
    print("TEST 3: What the Model Actually Receives as Input")
    print("=" * 70)

    # Load config used for training
    config_path = "/home/xinchen/ECG/checkpoints/scheme_f/random_good+moderate_p20_testval_round1_random/config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from dataset import create_datasets
    train_ds, val_ds, test_ds = create_datasets(cfg, merge_val_to_train=True)

    print(f"\n  Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Check a few test samples
    for i in range(min(5, len(test_ds))):
        sample = test_ds[i]
        video, ecg = sample[0], sample[-1]

        print(f"\n  Sample {i}:")
        print(f"    Video shape: {video.shape}, dtype: {video.dtype}")
        print(f"    ECG shape: {ecg.shape}, dtype: {ecg.dtype}")

        # Video statistics
        print(f"    Video: mean={video.mean():.6f}, std={video.std():.6f}, "
              f"min={video.min():.6f}, max={video.max():.6f}")

        # Per-channel (video is (T, C, H, W))
        if video.dim() == 4:
            for ch in range(video.shape[1]):
                ch_data = video[:, ch]
                print(f"    Video ch{ch}: mean={ch_data.mean():.6f}, std={ch_data.std():.6f}, "
                      f"range=[{ch_data.min():.6f}, {ch_data.max():.6f}]")

            # Temporal variation per channel (spatial mean per frame)
            for ch in range(video.shape[1]):
                ch_temporal = video[:, ch].mean(dim=(1, 2))  # (T,)
                print(f"    Video ch{ch} temporal: std={ch_temporal.std():.8f}, "
                      f"range={ch_temporal.max()-ch_temporal.min():.8f}")

        # ECG statistics
        print(f"    ECG: mean={ecg.mean():.6f}, std={ecg.std():.6f}, "
              f"min={ecg.min():.6f}, max={ecg.max():.6f}")

        # Check: is ECG actually oscillatory?
        ecg_np = ecg.numpy()
        from scipy import signal
        freqs, psd = signal.welch(ecg_np, fs=250, nperseg=512)
        # Find dominant frequency
        dominant_freq = freqs[np.argmax(psd)]
        print(f"    ECG dominant freq: {dominant_freq:.2f} Hz "
              f"({dominant_freq*60:.0f} BPM)")


# ====================================================================
# TEST 4: Load model and check predictions
# ====================================================================
def test_model_predictions():
    print("\n" + "=" * 70)
    print("TEST 4: Model Predictions vs Ground Truth")
    print("=" * 70)

    config_path = "/home/xinchen/ECG/checkpoints/scheme_f/random_good+moderate_p20_testval_round1_random/config.yaml"
    ckpt_path = "/home/xinchen/ECG/checkpoints/scheme_f/random_good+moderate_p20_testval_round1_random/best_model.pt"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from dataset import create_datasets
    from video_ecg_model import build_model

    model = build_model(cfg).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    _, _, test_ds = create_datasets(cfg, merge_val_to_train=True)

    print(f"\n  Test samples: {len(test_ds)}")

    all_pred_stats = []
    all_ecg_stats = []

    with torch.no_grad():
        for i in range(min(10, len(test_ds))):
            sample = test_ds[i]
            video, ecg = sample[0], sample[-1]

            video_batch = video.unsqueeze(0).to(device)
            pred = model(video_batch).cpu().squeeze(0).numpy()
            ecg_np = ecg.numpy()

            # Correlation
            r = np.corrcoef(pred, ecg_np)[0, 1]

            print(f"\n  Sample {i}:")
            print(f"    Pred: mean={pred.mean():.6f}, std={pred.std():.6f}, "
                  f"min={pred.min():.6f}, max={pred.max():.6f}")
            print(f"    ECG:  mean={ecg_np.mean():.6f}, std={ecg_np.std():.6f}, "
                  f"min={ecg_np.min():.6f}, max={ecg_np.max():.6f}")
            print(f"    Pearson r: {r:.6f}")

            # Check if prediction is essentially flat
            pred_range = pred.max() - pred.min()
            ecg_range = ecg_np.max() - ecg_np.min()
            print(f"    Pred range: {pred_range:.6f}, ECG range: {ecg_range:.6f}")
            print(f"    Pred range / ECG range: {pred_range / ecg_range:.6f}")

            all_pred_stats.append({
                'mean': pred.mean(), 'std': pred.std(),
                'min': pred.min(), 'max': pred.max(), 'r': r
            })
            all_ecg_stats.append({
                'mean': ecg_np.mean(), 'std': ecg_np.std(),
                'min': ecg_np.min(), 'max': ecg_np.max()
            })

    print(f"\n  === Summary across samples ===")
    pred_stds = [s['std'] for s in all_pred_stats]
    ecg_stds = [s['std'] for s in all_ecg_stats]
    print(f"  Pred std: mean={np.mean(pred_stds):.6f}, range=[{np.min(pred_stds):.6f}, {np.max(pred_stds):.6f}]")
    print(f"  ECG std:  mean={np.mean(ecg_stds):.6f}, range=[{np.min(ecg_stds):.6f}, {np.max(ecg_stds):.6f}]")
    print(f"  Pearson r: mean={np.mean([s['r'] for s in all_pred_stats]):.6f}")

    # Check if all predictions look the same (model ignoring input)
    if len(all_pred_stats) >= 2:
        with torch.no_grad():
            preds = []
            for i in range(min(5, len(test_ds))):
                sample = test_ds[i]
                video = sample[0].unsqueeze(0).to(device)
                pred = model(video).cpu().squeeze(0).numpy()
                preds.append(pred)

            # Compare predictions across different inputs
            print(f"\n  === Cross-prediction similarity ===")
            for i in range(len(preds)):
                for j in range(i+1, len(preds)):
                    r_ij = np.corrcoef(preds[i], preds[j])[0, 1]
                    rmse_ij = np.sqrt(np.mean((preds[i] - preds[j])**2))
                    print(f"  Pred[{i}] vs Pred[{j}]: r={r_ij:.6f}, RMSE={rmse_ij:.6f}")


# ====================================================================
# TEST 5: Check temporal alignment
# ====================================================================
def test_alignment():
    print("\n" + "=" * 70)
    print("TEST 5: Temporal Alignment Check")
    print("=" * 70)

    samples_dir = Path("/home/xinchen/ECG/training_data/samples")

    for pair_dir in sorted(samples_dir.iterdir())[:5]:
        meta_path = pair_dir / "metadata.json"
        ecg_path = pair_dir / "ecg.csv"
        video_path = pair_dir / "video_0.mp4"

        if not all(p.exists() for p in [meta_path, ecg_path, video_path]):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        df = pd.read_csv(ecg_path)

        cap = cv2.VideoCapture(str(video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        video_duration = n_frames / fps if fps > 0 else 0
        ecg_duration = meta.get("ecg_duration_s", 0)
        n_ecg = meta.get("ecg_samples", 0)

        print(f"\n  {pair_dir.name}:")
        print(f"    Video: {n_frames} frames, {fps:.1f} fps, duration={video_duration:.1f}s")
        print(f"    ECG: {n_ecg} samples, 250 Hz, duration={ecg_duration:.1f}s")
        print(f"    Duration match: video={video_duration:.1f}s vs ecg={ecg_duration:.1f}s, "
              f"diff={abs(video_duration - ecg_duration):.1f}s")

        # Check the windowing logic: window 0 maps to what?
        video_fps = 30
        ecg_sr = 250
        window_sec = 10
        stride_sec = 5

        window_frames = window_sec * video_fps  # 300
        stride_frames = stride_sec * video_fps  # 150
        window_ecg = window_sec * ecg_sr  # 2500

        # Window 0:
        vf_start, vf_end = 0, window_frames  # 0-300
        ecg_start, ecg_end = 0, window_ecg  # 0-2500

        video_time_start = vf_start / video_fps
        video_time_end = vf_end / video_fps
        ecg_time_start = ecg_start / ecg_sr
        ecg_time_end = ecg_end / ecg_sr

        print(f"    Window 0: video [{video_time_start:.1f}s - {video_time_end:.1f}s], "
              f"ECG [{ecg_time_start:.1f}s - {ecg_time_end:.1f}s]")

        # BUT: the video fps might not be exactly 30!
        actual_video_fps = fps
        actual_video_window_end_time = window_frames / actual_video_fps
        print(f"    ACTUAL video window 0 end time: {actual_video_window_end_time:.2f}s "
              f"(assumed 10.0s)")

        if abs(actual_video_fps - 30.0) > 1.0:
            print(f"    *** WARNING: Video FPS is {actual_video_fps:.1f}, not 30! "
                  f"This causes temporal misalignment! ***")


# ====================================================================
# TEST 6: Gradient flow check
# ====================================================================
def test_gradient_flow():
    print("\n" + "=" * 70)
    print("TEST 6: Gradient Flow Check")
    print("=" * 70)

    config_path = "/home/xinchen/ECG/checkpoints/scheme_f/random_good+moderate_p20_testval_round1_random/config.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from video_ecg_model import build_model, build_criterion
    from dataset import create_datasets

    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg)

    _, _, test_ds = create_datasets(cfg, merge_val_to_train=True)

    # Single forward-backward pass
    sample = test_ds[0]
    video, ecg = sample[0], sample[-1]

    video_batch = video.unsqueeze(0).to(device)
    ecg_batch = ecg.unsqueeze(0).to(device)

    model.train()
    pred = model(video_batch)
    loss = criterion(pred, ecg_batch)
    loss.backward()

    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Pred shape: {pred.shape}, ECG shape: {ecg_batch.shape}")

    # Check gradients
    total_grad_norm = 0
    zero_grad_layers = []
    large_grad_layers = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            if grad_norm < 1e-10:
                zero_grad_layers.append(name)
            elif grad_norm > 100:
                large_grad_layers.append((name, grad_norm))

    total_grad_norm = total_grad_norm ** 0.5
    print(f"  Total gradient norm: {total_grad_norm:.6f}")

    if zero_grad_layers:
        print(f"  Layers with ZERO gradients ({len(zero_grad_layers)}):")
        for name in zero_grad_layers[:10]:
            print(f"    - {name}")
    else:
        print(f"  All layers have non-zero gradients")

    if large_grad_layers:
        print(f"  Layers with LARGE gradients:")
        for name, norm in large_grad_layers[:10]:
            print(f"    - {name}: {norm:.2f}")


# ====================================================================
# TEST 7: Check what EfficientPhys global_pool does
# ====================================================================
def test_global_pool_issue():
    print("\n" + "=" * 70)
    print("TEST 7: Information Bottleneck Analysis (Global Avg Pool)")
    print("=" * 70)

    config_path = "/home/xinchen/ECG/checkpoints/scheme_f/random_good+moderate_p20_testval_round1_random/config.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from video_ecg_model import build_model
    from dataset import create_datasets

    model = build_model(cfg).to(device)
    ckpt_path = "/home/xinchen/ECG/checkpoints/scheme_f/random_good+moderate_p20_testval_round1_random/best_model.pt"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    _, _, test_ds = create_datasets(cfg, merge_val_to_train=True)

    # Hook into intermediate layers
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    model.stem.register_forward_hook(hook_fn('after_stem'))
    model.global_pool.register_forward_hook(hook_fn('after_global_pool'))
    model.temporal_conv.register_forward_hook(hook_fn('after_temporal_conv'))
    model.head.register_forward_hook(hook_fn('after_head'))

    with torch.no_grad():
        sample = test_ds[0]
        video = sample[0].unsqueeze(0).to(device)
        pred = model(video)

    for name, act in activations.items():
        print(f"\n  {name}: shape={list(act.shape)}")
        print(f"    mean={act.mean():.6f}, std={act.std():.6f}, "
              f"min={act.min():.6f}, max={act.max():.6f}")

        # Check if temporal dimension still has variation
        if act.dim() >= 3:
            # (B, C, T, ...) - check temporal variation
            temporal_dim = 2 if act.dim() >= 4 else -1
            if temporal_dim >= 0 and act.shape[temporal_dim] > 1:
                # Compute temporal std
                temporal_std = act.std(dim=temporal_dim).mean().item()
                print(f"    Temporal std (variation across time): {temporal_std:.8f}")


# ====================================================================
# TEST 8: The critical PPG signal strength test
# ====================================================================
def test_ppg_signal_strength():
    print("\n" + "=" * 70)
    print("TEST 8: PPG Signal Strength vs ECG Target")
    print("=" * 70)

    samples_dir = Path("/home/xinchen/ECG/training_data/samples")

    for pair_dir in sorted(samples_dir.iterdir())[:10]:
        video_path = pair_dir / "video_0.mp4"
        ecg_path = pair_dir / "ecg.csv"

        if not video_path.exists() or not ecg_path.exists():
            continue

        # Extract PPG from red channel
        cap = cv2.VideoCapture(str(video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        red_means = []
        for _ in range(min(300, n_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            red_means.append(frame[:, :, 2].mean())  # Red channel (BGR)
        cap.release()

        ppg = np.array(red_means)

        # Load ECG (first 10 seconds)
        df = pd.read_csv(ecg_path)
        ecg = df["ecg_counts_filt_monitor"].values[:2500]

        # PPG signal quality
        ppg_range = ppg.max() - ppg.min()
        ppg_std = ppg.std()
        ppg_mean = ppg.mean()
        ppg_snr = ppg_std / ppg_mean if ppg_mean > 0 else 0

        # ECG signal quality
        ecg_range = ecg.max() - ecg.min()
        ecg_std = ecg.std()

        print(f"\n  {pair_dir.name}:")
        print(f"    PPG (red): mean={ppg_mean:.1f}, std={ppg_std:.4f}, "
              f"range={ppg_range:.4f}, SNR(std/mean)={ppg_snr:.6f}")
        print(f"    ECG: mean={ecg.mean():.2f}, std={ecg_std:.2f}, range={ecg_range:.2f}")
        print(f"    PPG relative variation: {ppg_std/ppg_mean*100:.4f}%")

        # After /255 normalization and then spatial mean, what's the signal?
        ppg_normed = ppg / 255.0
        ppg_normed_range = ppg_normed.max() - ppg_normed.min()
        print(f"    PPG after /255: range={ppg_normed_range:.6f}, "
              f"this is what the model sees as temporal variation")


# ====================================================================
# TEST 9: Fundamental SNR problem
# ====================================================================
def test_snr_problem():
    print("\n" + "=" * 70)
    print("TEST 9: Signal-to-Noise Ratio Analysis")
    print("=" * 70)

    # The key question: is the PPG signal in the video
    # (tiny brightness fluctuations) distinguishable from noise?

    samples_dir = Path("/home/xinchen/ECG/training_data/samples")
    pair_dir = sorted(samples_dir.iterdir())[0]
    video_path = pair_dir / "video_0.mp4"

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(300):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.astype(np.float32))
    cap.release()

    frames = np.stack(frames)  # (300, H, W, 3)

    # After resize to 64x64 and /255 normalization
    resized = np.stack([cv2.resize(f, (64, 64)) for f in frames])
    normed = resized / 255.0  # This is what the model gets

    print(f"  Input to model: shape={normed.shape}")
    print(f"  Value range: [{normed.min():.4f}, {normed.max():.4f}]")

    # Frame-to-frame difference (temporal difference, what carries PPG info)
    diff = normed[1:] - normed[:-1]
    print(f"\n  Frame-to-frame difference (carrier of PPG signal):")
    print(f"    Mean abs diff: {np.abs(diff).mean():.8f}")
    print(f"    Std of diff:   {diff.std():.8f}")
    print(f"    Max abs diff:  {np.abs(diff).max():.8f}")

    # Compare to the overall value range
    signal_amplitude = np.abs(diff).mean()
    value_range = normed.max() - normed.min()
    print(f"\n  Signal (diff) / Value range: {signal_amplitude / value_range:.8f}")
    print(f"  This is the 'signal' the model needs to extract and map to ECG")

    # Now look at spatial variation (noise/texture) vs temporal variation (PPG)
    spatial_std = normed.std(axis=(1, 2))  # (T, 3) - per frame spatial variation
    temporal_std = normed.mean(axis=(1, 2)).std(axis=0)  # (3,) - temporal variation of spatial means

    print(f"\n  Spatial std (per frame): {spatial_std.mean():.6f}")
    print(f"  Temporal std of spatial mean: {temporal_std}")
    print(f"  Ratio (spatial/temporal): {spatial_std.mean() / temporal_std.mean():.1f}x")
    print(f"  -> Spatial noise dominates temporal PPG signal by {spatial_std.mean() / temporal_std.mean():.0f}x")


# ====================================================================
# TEST 10: Check if model output depends on input at all
# ====================================================================
def test_input_sensitivity():
    print("\n" + "=" * 70)
    print("TEST 10: Does Model Output Depend on Input?")
    print("=" * 70)

    config_path = "/home/xinchen/ECG/checkpoints/scheme_f/random_good+moderate_p20_testval_round1_random/config.yaml"
    ckpt_path = "/home/xinchen/ECG/checkpoints/scheme_f/random_good+moderate_p20_testval_round1_random/best_model.pt"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from video_ecg_model import build_model
    from dataset import create_datasets

    model = build_model(cfg).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    _, _, test_ds = create_datasets(cfg, merge_val_to_train=True)

    with torch.no_grad():
        # Real input
        video_real = test_ds[0][0].unsqueeze(0).to(device)
        pred_real = model(video_real).cpu().numpy().flatten()

        # Random noise input (same shape, same range)
        video_noise = torch.rand_like(video_real)
        pred_noise = model(video_noise).cpu().numpy().flatten()

        # Zero input
        video_zero = torch.zeros_like(video_real)
        pred_zero = model(video_zero).cpu().numpy().flatten()

        # Constant input (all 0.5)
        video_const = torch.ones_like(video_real) * 0.5
        pred_const = model(video_const).cpu().numpy().flatten()

        print(f"  Real input pred:     mean={pred_real.mean():.6f}, std={pred_real.std():.6f}")
        print(f"  Random noise pred:   mean={pred_noise.mean():.6f}, std={pred_noise.std():.6f}")
        print(f"  Zero input pred:     mean={pred_zero.mean():.6f}, std={pred_zero.std():.6f}")
        print(f"  Constant(0.5) pred:  mean={pred_const.mean():.6f}, std={pred_const.std():.6f}")

        # Compare outputs
        r_real_noise = np.corrcoef(pred_real, pred_noise)[0, 1]
        r_real_zero = np.corrcoef(pred_real, pred_zero)[0, 1]
        r_real_const = np.corrcoef(pred_real, pred_const)[0, 1]
        r_noise_zero = np.corrcoef(pred_noise, pred_zero)[0, 1]

        print(f"\n  Correlation between outputs:")
        print(f"    Real vs Random noise: {r_real_noise:.6f}")
        print(f"    Real vs Zero:         {r_real_zero:.6f}")
        print(f"    Real vs Constant:     {r_real_const:.6f}")
        print(f"    Random vs Zero:       {r_noise_zero:.6f}")

        rmse_real_noise = np.sqrt(np.mean((pred_real - pred_noise)**2))
        rmse_real_zero = np.sqrt(np.mean((pred_real - pred_zero)**2))
        print(f"\n  RMSE between outputs:")
        print(f"    Real vs Random: {rmse_real_noise:.6f}")
        print(f"    Real vs Zero:   {rmse_real_zero:.6f}")

        if rmse_real_noise < 0.01 and rmse_real_zero < 0.01:
            print(f"\n  *** CRITICAL: Model output is INDEPENDENT of input! ***")
            print(f"  *** The model has collapsed to producing a constant output ***")


if __name__ == "__main__":
    test_ecg_data()
    test_video_data()
    test_ppg_signal_strength()
    test_snr_problem()
    test_alignment()
    test_model_input_signal()
    test_model_predictions()
    test_input_sensitivity()
    test_gradient_flow()
    test_global_pool_issue()
