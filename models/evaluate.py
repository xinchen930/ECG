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
def evaluate_model(model, dataloader, device="cpu", use_imu=False):
    """Run model on dataloader, return metrics dict."""
    model.eval()
    all_pred, all_target = [], []

    for batch in dataloader:
        if use_imu:
            video, imu, ecg = batch
            video, imu = video.to(device), imu.to(device)
            pred = model(video, imu).cpu().numpy()
        else:
            video, ecg = batch
            video = video.to(device)
            pred = model(video).cpu().numpy()
        all_pred.append(pred)
        all_target.append(ecg.numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    return compute_metrics(all_pred, all_target)
