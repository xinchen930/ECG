"""
Only evaluate on test set (no training).
Loads a checkpoint and prints RMSE, MAE, Pearson r.

Usage:
    python models/run_eval.py --config configs/scheme_a.yaml --checkpoint checkpoints/scheme_a/best_model.pt
"""
import argparse
import os

import torch
import yaml

from dataset import create_datasets
from video_ecg_model import build_model
from evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scheme_a.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_imu = cfg["data"].get("use_imu", False)

    _, _, test_ds = create_datasets(cfg)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    metrics = evaluate_model(model, test_loader, device=device, use_imu=use_imu)
    scheme = cfg.get("scheme_name", "default")
    print(f"[{scheme}] Test — RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} Pearson_r={metrics['pearson_r']:.4f}")


if __name__ == "__main__":
    main()
