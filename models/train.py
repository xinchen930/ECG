"""
Training script for Video-to-ECG reconstruction.
Supports both Scheme A (video only) and Scheme B (video + IMU, composite loss).

Usage:
    python models/train.py --config configs/scheme_a.yaml
    python models/train.py --config configs/scheme_b.yaml
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from dataset import create_datasets
from video_ecg_model import build_model, build_criterion
from evaluate import evaluate_model


def train(cfg):
    train_cfg = cfg["train"]
    use_imu = cfg["data"].get("use_imu", False)

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

    # Data
    train_ds, val_ds, test_ds = create_datasets(cfg)

    num_workers = cfg["data"].get("num_workers", 0)
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"],
                            shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"],
                             shuffle=False, num_workers=num_workers, pin_memory=pin)

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

    # Checkpoint dir (per-scheme)
    scheme_name = cfg.get("scheme_name", "default")
    save_dir = os.path.join("checkpoints", scheme_name)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if use_imu:
                video, imu, ecg = batch
                video, imu, ecg = video.to(device), imu.to(device), ecg.to(device)
                pred = model(video, imu)
            else:
                video, ecg = batch
                video, ecg = video.to(device), ecg.to(device)
                pred = model(video)

            loss = criterion(pred, ecg)

            optimizer.zero_grad()
            loss.backward()
            if train_cfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()

            train_loss += loss.item() * video.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch} batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={loss.item():.4f}")

        train_loss /= len(train_ds)
        elapsed = time.time() - t0

        # Validation
        val_metrics = evaluate_model(model, val_loader, device=device, use_imu=use_imu)
        val_loss = val_metrics["rmse"]

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_rmse={val_metrics['rmse']:.4f} val_mae={val_metrics['mae']:.4f} "
              f"val_r={val_metrics['pearson_r']:.4f} | lr={lr:.2e} | {elapsed:.1f}s")

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

    # Final test evaluation
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"),
                                     weights_only=True))
    test_metrics = evaluate_model(model, test_loader, device=device, use_imu=use_imu)
    print(f"\n[{scheme_name}] Test results: RMSE={test_metrics['rmse']:.4f} "
          f"MAE={test_metrics['mae']:.4f} Pearson_r={test_metrics['pearson_r']:.4f}")

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scheme_a.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)
