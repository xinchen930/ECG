"""Evaluation metrics for ECG reconstruction."""

import numpy as np
import torch


def compute_metrics(pred: np.ndarray, target: np.ndarray):
    """
    Compute reconstruction metrics.
    pred, target: (N, T)
    """
    rmse = np.sqrt(np.mean((pred - target) ** 2, axis=1)).mean()
    mae = np.mean(np.abs(pred - target), axis=1).mean()

    correlations = []
    for i in range(len(pred)):
        r = np.corrcoef(pred[i], target[i])[0, 1]
        if np.isnan(r):
            r = 0.0
        correlations.append(r)
    pearson_r = np.mean(correlations)

    return {"rmse": float(rmse), "mae": float(mae), "pearson_r": float(pearson_r)}


@torch.no_grad()
def evaluate_model(model, dataloader, device="cpu", use_imu=False, use_amp=False):
    """Run model on dataloader, return metrics dict. use_amp 可降低验证时显存."""
    model.eval()
    all_pred, all_target = [], []
    do_amp = use_amp and device == "cuda"

    for batch in dataloader:
        if use_imu:
            video, imu, ecg = batch
            video, imu = video.to(device), imu.to(device)
            if do_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(video, imu).float().cpu().numpy()
            else:
                pred = model(video, imu).cpu().numpy()
        else:
            video, ecg = batch
            video = video.to(device)
            if do_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(video).float().cpu().numpy()
            else:
                pred = model(video).cpu().numpy()
        all_pred.append(pred)
        all_target.append(ecg.numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    return compute_metrics(all_pred, all_target)


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime
    from pathlib import Path

    import yaml

    from dataset import create_datasets
    from video_ecg_model import build_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument(
        "--group_by_scheme",
        action="store_true",
        help="If set, write results under output_dir/<scheme_name>/",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Device: prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load model
    model = build_model(cfg).to(device)
    ckpt_obj = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt_obj.get("model_state_dict", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Data
    use_imu = cfg.get("data", {}).get("use_imu", False)
    _, _, test_ds = create_datasets(cfg)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
    )

    # Evaluate
    metrics = evaluate_model(model, test_loader, device=device, use_imu=use_imu)
    print(f"Test metrics: {metrics}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = Path(args.config).stem
    scheme_name = cfg.get("scheme_name", config_name)

    output_dir = Path(args.output_dir)
    if args.group_by_scheme:
        output_dir = output_dir / scheme_name
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "timestamp": timestamp,
        "device": device,
        "scheme_name": scheme_name,
        "metrics": metrics,
    }

    # JSON output (single-run details)
    json_path = output_dir / f"{config_name}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Append to CSV summary (all runs)
    csv_path = output_dir / "summary.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a") as f:
        if write_header:
            f.write("timestamp,scheme_name,config,checkpoint,rmse,mae,pearson_r\n")
        f.write(
            f"{timestamp},{scheme_name},{config_name},{args.checkpoint},"
            f"{metrics['rmse']:.6f},{metrics['mae']:.6f},{metrics['pearson_r']:.6f}\n"
        )
    print(f"Appended to: {csv_path}")
