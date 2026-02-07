"""
Training script for PPG2ECG model on BIDMC dataset.

Paper: "Reconstructing QRS Complex from PPG by Transformed Attentional Neural Networks"
       IEEE Sensors Journal, 2020

Usage:
    # Train on BIDMC dataset
    python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml

    # With specific GPU
    CUDA_VISIBLE_DEVICES=0 python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml

    # Override parameters
    python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml --epochs 100 --batch_size 128
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ppg2ecg import PPG2ECG, PPG2ECG_LSTM, CombinedLoss, count_parameters


# ============================================================================
# Dataset
# ============================================================================

class BIDMCDataset(Dataset):
    """
    BIDMC PPG-ECG Dataset with augmentation.

    Loads preprocessed data from .pt file and applies random offset augmentation.

    Args:
        data_path: Path to .pt file
        full_window: Full window size (512)
        train_window: Training window size (256)
        augment: Enable random offset augmentation
    """
    def __init__(self, data_path: str, full_window: int = 512, train_window: int = 256,
                 augment: bool = True):
        self.full_window = full_window
        self.train_window = train_window
        self.augment = augment

        # Load data
        data = torch.load(data_path, weights_only=False)
        self.ppg = data['ppg']      # [N, 1, full_window]
        self.ecg = data['ecg']      # [N, 1, full_window]
        self.rpeaks = data['rpeaks']  # [N, 1, full_window]

        # Calculate center offset for cropping
        self.center_offset = (full_window - train_window) // 2  # 128
        self.max_offset = self.center_offset  # [-128, 128] range

        print(f"Loaded {len(self)} samples from {data_path}")
        print(f"  PPG shape: {self.ppg.shape}")
        print(f"  ECG shape: {self.ecg.shape}")

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        ppg = self.ppg[idx]      # [1, full_window]
        ecg = self.ecg[idx]      # [1, full_window]
        rpeaks = self.rpeaks[idx]  # [1, full_window]

        # Random offset augmentation (training only)
        if self.augment:
            # Random offset in [-64, 64] (following original implementation)
            offset = np.random.randint(-64, 65)
        else:
            offset = 0

        # Calculate crop indices
        start = self.center_offset + offset
        end = start + self.train_window

        # Crop to training window
        ppg = ppg[:, start:end]
        ecg = ecg[:, start:end]
        rpeaks = rpeaks[:, start:end]

        return ppg, ecg, rpeaks


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, dataloader, device, use_qrs_loss=True, beta=5):
    """
    Evaluate model on a dataset.

    Args:
        model: PPG2ECG model
        dataloader: DataLoader
        device: Device
        use_qrs_loss: Use QRS-weighted loss
        beta: QRS loss beta

    Returns:
        Dict with 'loss', 'rmse', 'mae', 'pearson_r'
    """
    model.eval()
    loss_fn = CombinedLoss(beta=beta, use_qrs_loss=use_qrs_loss)

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for ppg, ecg, rpeaks in dataloader:
            ppg = ppg.to(device)
            ecg = ecg.to(device)
            rpeaks = rpeaks.to(device)

            output = model(ppg)
            pred = output['output']

            loss = loss_fn(pred, ecg, rpeaks if use_qrs_loss else None)
            total_loss += loss.item() * ppg.size(0)

            all_preds.append(pred.cpu())
            all_targets.append(ecg.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    # Flatten: [N, 1, T] -> [N*T]
    pred_flat = all_preds.view(-1).numpy()
    target_flat = all_targets.view(-1).numpy()

    mse = np.mean((pred_flat - target_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_flat - target_flat))

    # Pearson correlation
    pred_centered = pred_flat - np.mean(pred_flat)
    target_centered = target_flat - np.mean(target_flat)
    numerator = np.sum(pred_centered * target_centered)
    denominator = np.sqrt(np.sum(pred_centered ** 2) * np.sum(target_centered ** 2))
    pearson_r = numerator / (denominator + 1e-8)

    return {
        'loss': total_loss / len(dataloader.dataset),
        'rmse': rmse,
        'mae': mae,
        'pearson_r': pearson_r,
    }


# ============================================================================
# Training
# ============================================================================

def train(cfg):
    """
    Train PPG2ECG model.

    Args:
        cfg: Config dict
    """
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Seed
    seed = cfg['training'].get('seed', 2019)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Load datasets
    train_cfg = cfg['training']
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    loss_cfg = cfg['loss']
    output_cfg = cfg['output']

    full_window = data_cfg.get('full_window', 512)
    train_window = model_cfg.get('input_size', 256)

    print(f"\nLoading datasets...")
    train_ds = BIDMCDataset(
        data_cfg['train_path'],
        full_window=full_window,
        train_window=train_window,
        augment=train_cfg.get('data_augmentation', True)
    )
    test_ds = BIDMCDataset(
        data_cfg['test_path'],
        full_window=full_window,
        train_window=train_window,
        augment=False  # No augmentation for test
    )

    num_workers = cfg.get('device', {}).get('num_workers', 4)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Create model
    print(f"\nCreating model...")
    if model_cfg.get('type', 'ppg2ecg') == 'lstm':
        model = PPG2ECG_LSTM(
            input_size=train_window,
            hidden_size=model_cfg.get('hidden_size', 200),
            num_layers=model_cfg.get('num_layers', 2),
            dropout=model_cfg.get('dropout', 0.1),
        )
    else:
        model = PPG2ECG(
            input_size=train_window,
            use_stn=model_cfg.get('use_stn', True),
            use_attention=model_cfg.get('use_attention', True),
        )
    model = model.to(device)

    n_params = count_parameters(model)
    print(f"Model: {model_cfg.get('type', 'ppg2ecg')}")
    print(f"Parameters: {n_params:,}")

    # Loss function
    use_qrs_loss = loss_cfg.get('type', 'qrs_enhanced') == 'qrs_enhanced'
    beta = loss_cfg.get('beta', 5)
    loss_fn = CombinedLoss(beta=beta, use_qrs_loss=use_qrs_loss)
    print(f"Loss: {'QRS-enhanced (beta={})'.format(beta) if use_qrs_loss else 'L1'}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg['learning_rate'],
    )

    # Scheduler
    use_scheduler = train_cfg.get('use_scheduler', False)
    if use_scheduler:
        scheduler_period = train_cfg.get('scheduler_period', train_cfg['epochs'] // 10)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_period,
        )
    else:
        scheduler = None

    # AMP
    use_amp = cfg.get('device', {}).get('use_amp', False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision (AMP)")

    # Checkpoint directory
    save_dir = output_cfg.get('checkpoint_dir', 'checkpoints/ppg2ecg_bidmc/')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoint dir: {save_dir}")

    # Save config
    config_path = os.path.join(save_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Config saved to: {config_path}")

    # Training loop
    best_loss = float("inf")
    patience_counter = 0
    patience = train_cfg.get('patience', 50)
    eval_every = output_cfg.get('eval_every', 5)
    save_every = output_cfg.get('save_every', 50)

    history = {
        'epoch': [],
        'train_loss': [],
        'test_loss': [],
        'test_rmse': [],
        'test_mae': [],
        'test_pearson_r': [],
    }

    print(f"\nStarting training...")
    print(f"  Epochs: {train_cfg['epochs']}")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Learning rate: {train_cfg['learning_rate']}")
    print(f"  Patience: {patience}")
    print()

    for epoch in range(1, train_cfg['epochs'] + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch_idx, (ppg, ecg, rpeaks) in enumerate(train_loader):
            ppg = ppg.to(device)
            ecg = ecg.to(device)
            rpeaks = rpeaks.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(ppg)
                    pred = output['output']
                    loss = loss_fn(pred, ecg, rpeaks if use_qrs_loss else None)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(ppg)
                pred = output['output']
                loss = loss_fn(pred, ecg, rpeaks if use_qrs_loss else None)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * ppg.size(0)

        train_loss /= len(train_ds)
        elapsed = time.time() - t0

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Evaluate
        if epoch % eval_every == 0 or epoch == 1:
            test_metrics = evaluate_model(
                model, test_loader, device,
                use_qrs_loss=use_qrs_loss, beta=beta
            )

            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
                  f"train_loss={train_loss:.4f} | "
                  f"test_rmse={test_metrics['rmse']:.4f} "
                  f"test_mae={test_metrics['mae']:.4f} "
                  f"test_r={test_metrics['pearson_r']:.4f} | "
                  f"lr={lr:.2e} | {elapsed:.1f}s")

            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_metrics['loss'])
            history['test_rmse'].append(test_metrics['rmse'])
            history['test_mae'].append(test_metrics['mae'])
            history['test_pearson_r'].append(test_metrics['pearson_r'])

            # Early stopping on test loss
            if test_metrics['loss'] < best_loss:
                best_loss = test_metrics['loss']
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
                print(f"  -> Saved best model (test_loss={best_loss:.4f}, r={test_metrics['pearson_r']:.4f})")
            else:
                patience_counter += eval_every
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch:3d}/{train_cfg['epochs']} | train_loss={train_loss:.4f} | {elapsed:.1f}s")

        # Periodic checkpoint
        if epoch % save_every == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch{epoch}.pt"))

    # Save training history
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    if history['epoch']:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(history['epoch'], history['train_loss'], 'b.-', label='train_loss')
        axes[0].plot(history['epoch'], history['test_rmse'], 'r.-', label='test_rmse')
        axes[0].set_ylabel('Loss / RMSE')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history['epoch'], history['test_pearson_r'], 'g.-', label='test_pearson_r')
        axes[1].axhline(y=0.7, color='orange', linestyle='--', label='target (r=0.7)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Pearson r')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('PPG2ECG Training on BIDMC')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
        plt.close()
        print(f"Saved training curves to {save_dir}/training_curves.png")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"), weights_only=True))
    final_metrics = evaluate_model(model, test_loader, device, use_qrs_loss=use_qrs_loss, beta=beta)

    print(f"RMSE:      {final_metrics['rmse']:.4f}")
    print(f"MAE:       {final_metrics['mae']:.4f}")
    print(f"Pearson r: {final_metrics['pearson_r']:.4f}")
    print()

    # Success check
    if final_metrics['pearson_r'] > 0.7:
        print("✅ SUCCESS: Pearson r > 0.7")
        print("   Model works on BIDMC. Proceed to test on your video data.")
    elif final_metrics['pearson_r'] > 0.5:
        print("⚠️  MODERATE: Pearson r > 0.5 but < 0.7")
        print("   Consider tuning hyperparameters or trying longer training.")
    else:
        print("❌ NEEDS WORK: Pearson r < 0.5")
        print("   Check: data preprocessing, model architecture, loss function")
        print("   Consider trying CardioGAN as alternative.")

    return final_metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train PPG2ECG model on BIDMC dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml

    # With specific GPU
    CUDA_VISIBLE_DEVICES=0 python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml

    # Override parameters
    python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml --epochs 100 --batch_size 128

    # Test only (evaluate existing checkpoint)
    python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml --eval_only --checkpoint checkpoints/ppg2ecg_bidmc/best_model.pt
        """
    )
    parser.add_argument('--config', type=str, default='configs/ppg2ecg_bidmc.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate, no training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for evaluation')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply overrides
    if args.epochs is not None:
        cfg['training']['epochs'] = args.epochs
        print(f"Override epochs: {args.epochs}")
    if args.batch_size is not None:
        cfg['training']['batch_size'] = args.batch_size
        print(f"Override batch_size: {args.batch_size}")
    if args.lr is not None:
        cfg['training']['learning_rate'] = args.lr
        print(f"Override learning_rate: {args.lr}")

    print("=" * 60)
    print("PPG2ECG Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Model: {cfg['model'].get('type', 'ppg2ecg')}")
    print(f"  use_stn: {cfg['model'].get('use_stn', True)}")
    print(f"  use_attention: {cfg['model'].get('use_attention', True)}")
    print()

    if args.eval_only:
        if args.checkpoint is None:
            args.checkpoint = os.path.join(cfg['output']['checkpoint_dir'], 'best_model.pt')
        print(f"Evaluation only mode")
        print(f"Checkpoint: {args.checkpoint}")

        # Setup device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Load model
        model_cfg = cfg['model']
        train_window = model_cfg.get('input_size', 256)

        if model_cfg.get('type', 'ppg2ecg') == 'lstm':
            model = PPG2ECG_LSTM(input_size=train_window)
        else:
            model = PPG2ECG(
                input_size=train_window,
                use_stn=model_cfg.get('use_stn', True),
                use_attention=model_cfg.get('use_attention', True),
            )
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
        model = model.to(device)

        # Load test data
        data_cfg = cfg['data']
        test_ds = BIDMCDataset(
            data_cfg['test_path'],
            full_window=data_cfg.get('full_window', 512),
            train_window=train_window,
            augment=False
        )
        test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'], shuffle=False)

        # Evaluate
        loss_cfg = cfg['loss']
        metrics = evaluate_model(
            model, test_loader, device,
            use_qrs_loss=loss_cfg.get('type') == 'qrs_enhanced',
            beta=loss_cfg.get('beta', 5)
        )
        print(f"\nTest Results:")
        print(f"  RMSE:      {metrics['rmse']:.4f}")
        print(f"  MAE:       {metrics['mae']:.4f}")
        print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    else:
        train(cfg)


if __name__ == '__main__':
    main()
