"""
Training script for Video-to-ECG reconstruction.
Supports Schemes C/D/E/F/G with optional IMU fusion and server-specific presets.

Usage:
    python models/train.py --config configs/scheme_f.yaml
    python models/train.py --config configs/scheme_f.yaml --server 3090
    python models/train.py --config configs/scheme_f.yaml --server a6000

Server presets (--server):
    3090  : RTX 3090 (24GB) - smaller batch, AMP enabled, gradient accumulation
    a6000 : NVIDIA A6000 (48GB) - larger batch, no AMP needed
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from dataset import create_datasets
from video_ecg_model import build_model, build_criterion
from evaluate import evaluate_model


def train(cfg, use_test_as_val=False):
    """
    Train the model.

    Args:
        cfg: Config dict
        use_test_as_val: If True, use test set for early stopping (debugging mode)
    """
    train_cfg = cfg["train"]
    use_imu = cfg["data"].get("use_imu", False)
    use_amp = train_cfg.get("use_amp", False) and torch.cuda.is_available()
    accum_steps = max(1, train_cfg.get("gradient_accumulation_steps", 1))
    if use_amp:
        print("Using mixed precision (AMP)")
    if accum_steps > 1:
        print(f"Gradient accumulation: {accum_steps} steps (effective batch = {train_cfg['batch_size'] * accum_steps})")

    # Device: prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(train_cfg["seed"])

    # Data - merge val into train when using test for early stopping
    train_ds, val_ds, test_ds = create_datasets(cfg, merge_val_to_train=use_test_as_val)

    num_workers = cfg["data"].get("num_workers", 0)
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"],
                             shuffle=False, num_workers=num_workers, pin_memory=pin)

    # Debug mode: use test set for early stopping (val merged into train)
    if use_test_as_val:
        print("⚠️  DEBUG MODE: Using TEST set for early stopping (val merged into train)")
        early_stop_loader = test_loader
        early_stop_name = "test"
        val_loader = None  # Not used
    else:
        val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"],
                                shuffle=False, num_workers=num_workers, pin_memory=pin)
        early_stop_loader = val_loader
        early_stop_name = "val"

    # Model
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} (trainable: {trainable_params:,})")

    # Loss & optimizer
    criterion = build_criterion(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Checkpoint dir: checkpoints/{scheme}/{split}_{quality}_p{patience}/
    scheme_name = cfg.get("scheme_name", "default")
    split_mode = cfg.get("split", {}).get("mode", "user")
    quality_filter = cfg.get("data", {}).get("quality_filter", "all")
    if quality_filter is None:
        quality_filter = "all"
    else:
        quality_filter = quality_filter.replace(",", "+")  # "good,moderate" -> "good+moderate"
    patience = train_cfg.get("patience", 20)

    debug_suffix = "_testval" if use_test_as_val else ""
    run_tag = cfg.get("run_tag", "")
    tag_suffix = f"_{run_tag}" if run_tag else ""
    run_name = f"{split_mode}_{quality_filter}_p{patience}{debug_suffix}{tag_suffix}"
    save_dir = os.path.join("checkpoints", scheme_name, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoint dir: {save_dir}")

    # Save config for reproducibility
    config_save_path = os.path.join(save_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"Config saved to: {config_save_path}")

    # Check if criterion supports epoch parameter (CompositeLossV2 dynamic weighting)
    from video_ecg_model import CompositeLossV2
    criterion_has_epoch = isinstance(criterion, CompositeLossV2)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"epoch": [], "train_loss": [], "val_rmse": [], "val_mae": [], "val_pearson_r": []}

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if (batch_idx % accum_steps) == 0:
                optimizer.zero_grad()

            # Compute loss; pass epoch for dynamic weighting if supported
            def _compute_loss(pred, ecg):
                if criterion_has_epoch:
                    main_loss = criterion(pred, ecg, epoch=epoch) / accum_steps
                else:
                    main_loss = criterion(pred, ecg) / accum_steps
                # Add PPG auxiliary loss for STMap two-stage model
                if hasattr(model, 'get_ppg_aux_loss'):
                    main_loss = main_loss + model.get_ppg_aux_loss(ecg) / accum_steps
                return main_loss

            if use_imu:
                video, imu, ecg = batch
                video, imu, ecg = video.to(device), imu.to(device), ecg.to(device)
                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        pred = model(video, imu)
                        loss = _compute_loss(pred, ecg)
                else:
                    pred = model(video, imu)
                    loss = _compute_loss(pred, ecg)
            else:
                video, ecg = batch
                video, ecg = video.to(device), ecg.to(device)
                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        pred = model(video)
                        loss = _compute_loss(pred, ecg)
                else:
                    pred = model(video)
                    loss = _compute_loss(pred, ecg)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                if train_cfg.get("grad_clip"):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            train_loss += loss.item() * video.size(0) * accum_steps

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch} batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={loss.item() * accum_steps:.4f}")

        # 末尾不足 accum_steps 的 step 补一次
        if (batch_idx + 1) % accum_steps != 0:
            if train_cfg.get("grad_clip"):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        train_loss /= len(train_ds)
        elapsed = time.time() - t0

        # Validation (or test if use_test_as_val)
        val_metrics = evaluate_model(model, early_stop_loader, device=device, use_imu=use_imu, use_amp=use_amp)
        val_loss = val_metrics["rmse"]

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"{early_stop_name}_rmse={val_metrics['rmse']:.4f} {early_stop_name}_mae={val_metrics['mae']:.4f} "
              f"{early_stop_name}_r={val_metrics['pearson_r']:.4f} | lr={lr:.2e} | {elapsed:.1f}s")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_pearson_r"].append(val_metrics["pearson_r"])

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print(f"  -> Saved best model (val_rmse={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save training curves (per-scheme, so different schemes don't overwrite)
    if history["epoch"]:
        with open(os.path.join(save_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ep = history["epoch"]
        axes[0].plot(ep, history["train_loss"], "b.-", label="train_loss")
        axes[0].plot(ep, history["val_rmse"], "r.-", label="val_rmse")
        axes[0].set_ylabel("Loss / RMSE")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(ep, history["val_pearson_r"], "g.-", label="val_pearson_r")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Pearson r")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)
        plt.suptitle(f"Training curves ({scheme_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  -> Saved {save_dir}/training_curves.png and training_history.json")

    # Final test evaluation
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"),
                                     weights_only=True))
    test_metrics = evaluate_model(model, test_loader, device=device, use_imu=use_imu, use_amp=use_amp)
    print(f"\n[{scheme_name}] Test results: RMSE={test_metrics['rmse']:.4f} "
          f"MAE={test_metrics['mae']:.4f} Pearson_r={test_metrics['pearson_r']:.4f}")
    if use_test_as_val:
        print("  (Note: test set was used for early stopping - this is DEBUG mode, not valid for final evaluation)")

    # Save final summary for easy results collection
    summary = {
        "scheme_name": scheme_name,
        "run_name": run_name,
        "split_mode": split_mode,
        "quality_filter": quality_filter,
        "use_test_as_val": use_test_as_val,
        "run_tag": cfg.get("run_tag", ""),
        "data_dir": cfg["data"]["samples_dir"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_epoch": len(history["epoch"]) - patience_counter if history["epoch"] else 0,
        "total_epochs": len(history["epoch"]),
        "test_metrics": test_metrics,
        "device": device,
    }
    summary_path = os.path.join(save_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> Saved {summary_path}")

    return test_metrics


def load_config_with_server_preset(config_path, server=None):
    """Load config and optionally apply server-specific presets."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if server is None:
        return cfg

    # Load server presets
    presets_path = os.path.join(os.path.dirname(config_path), "server_presets.yaml")
    if not os.path.exists(presets_path):
        print(f"Warning: server_presets.yaml not found at {presets_path}, using base config")
        return cfg

    with open(presets_path) as f:
        presets = yaml.safe_load(f)

    server = server.lower()
    if server not in presets:
        print(f"Warning: unknown server '{server}', available: {list(presets.keys())}")
        return cfg

    scheme_name = cfg.get("scheme_name", "")
    if scheme_name not in presets[server]:
        print(f"Warning: no preset for {scheme_name} on {server}, using base config")
        return cfg

    # Apply server preset overrides to train config
    server_overrides = presets[server][scheme_name]
    print(f"Applying server preset: {server} for {scheme_name}")
    for key, value in server_overrides.items():
        if key == "num_workers":
            cfg["data"]["num_workers"] = value
            print(f"  data.num_workers: {value}")
        else:
            old_val = cfg["train"].get(key)
            cfg["train"][key] = value
            print(f"  train.{key}: {old_val} -> {value}")

    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Video-to-ECG reconstruction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python models/train.py --config configs/scheme_f.yaml
  python models/train.py --config configs/scheme_f.yaml --server 3090
  python models/train.py --config configs/scheme_f.yaml --server a6000

Data split options:
  --split random                  Random split (easier, for debugging/feasibility check)
  --split user                    User-level split (harder, no data leakage, for final eval)

Quality filter options:
  --quality-filter good           Only use good quality samples (80 samples)
  --quality-filter good,moderate  Exclude poor samples (88 samples)
  --quality-filter all            Use all samples including poor (98 samples)

Training control:
  --patience 20                   Set early stopping patience (default: 20-30 from config)
  --epochs 200                    Set max epochs (default: 200 from config)
  --use-test-as-val               DEBUG: Use test set for early stopping (skip validation)
        """,
    )
    parser.add_argument("--config", default="configs/scheme_c.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--server", type=str, default="a6000",
                        choices=["3090", "a6000"],
                        help="Server type for automatic parameter tuning (3090 or a6000)")
    parser.add_argument("--split", type=str, default=None,
                        choices=["random", "user"],
                        help="Data split mode: 'random' (easier) or 'user' (harder, no leakage)")
    parser.add_argument("--quality-filter", type=str, default=None,
                        help="Override quality filter: 'good', 'good,moderate', or 'all' (use all samples)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Maximum training epochs")
    parser.add_argument("--use-test-as-val", action="store_true", default=True,
                        help="Use test set for early stopping (default: True for debugging)")
    parser.add_argument("--use-val", action="store_true",
                        help="Use separate validation set for early stopping (strict mode)")
    parser.add_argument("--batch", type=str, default=None,
                        help="Batch filter: 'batch_1', 'batch_2', 'batch_1,batch_2' (reads batch_index.json)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data.samples_dir (e.g. training_data/batch1, training_data/batch2)")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Extra tag appended to checkpoint dir name (e.g. 'b1', 'b2', 'b1+2')")
    args = parser.parse_args()

    cfg = load_config_with_server_preset(args.config, args.server)

    # Override data directory
    if args.data_dir is not None:
        cfg["data"]["samples_dir"] = args.data_dir
        print(f"Data dir override: {args.data_dir}")

    # Batch filter
    if args.batch is not None:
        cfg["batch"] = args.batch
        print(f"Batch filter: {args.batch}")

    # Override split mode from command line
    if args.split is not None:
        cfg["split"]["mode"] = args.split
        print(f"Split mode override: {args.split}")

    # Override quality_filter from command line
    if args.quality_filter is not None:
        if args.quality_filter.lower() == "all":
            cfg["data"]["quality_filter"] = None
            print("Quality filter: using ALL samples (including poor)")
        else:
            cfg["data"]["quality_filter"] = args.quality_filter
            print(f"Quality filter override: {args.quality_filter}")

    # Override training parameters from command line
    if args.patience is not None:
        cfg["train"]["patience"] = args.patience
        print(f"Patience override: {args.patience}")
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
        print(f"Epochs override: {args.epochs}")

    # Store run tag in config for checkpoint naming
    if args.run_tag is not None:
        cfg["run_tag"] = args.run_tag
        print(f"Run tag: {args.run_tag}")

    # --use-val overrides the default --use-test-as-val=True
    use_test_as_val = args.use_test_as_val and not args.use_val
    if args.use_val:
        print("Strict mode: using separate VALIDATION set for early stopping")
    train(cfg, use_test_as_val=use_test_as_val)
