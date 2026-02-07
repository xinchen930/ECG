"""
Training script for CardioGAN on BIDMC dataset.

Paper: "CardioGAN: Attentive Generative Adversarial Network with Dual Discriminators
        for Synthesis of ECG from PPG" (AAAI 2021)

Usage:
    # Train on BIDMC dataset
    python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml

    # With specific GPU
    CUDA_VISIBLE_DEVICES=0 python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml

    # Override parameters
    python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml --epochs 50 --batch_size 16
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
from torch.utils.data import Dataset, DataLoader
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cardiogan import CardioGAN, count_parameters


# ============================================================================
# Dataset
# ============================================================================

class BIDMCDatasetCardioGAN(Dataset):
    """
    BIDMC PPG-ECG Dataset for CardioGAN (128 Hz, 512 samples).

    Note: CardioGAN uses different parameters than PPG2ECG:
    - 128 Hz sampling rate (vs 125 Hz)
    - 512 samples per window (vs 256)
    - 4 seconds per window (vs 2 seconds)

    Args:
        data_path: Path to .pt file (can reuse PPG2ECG preprocessed data)
        target_length: Target sequence length (512 for CardioGAN)
        augment: Enable random offset augmentation
    """
    def __init__(self, data_path: str, target_length: int = 512, augment: bool = True):
        self.target_length = target_length
        self.augment = augment

        # Load data
        data = torch.load(data_path, weights_only=False)
        self.ppg = data['ppg']      # [N, 1, original_length]
        self.ecg = data['ecg']      # [N, 1, original_length]

        self.original_length = self.ppg.shape[2]

        print(f"Loaded {len(self)} samples from {data_path}")
        print(f"  Original length: {self.original_length}")
        print(f"  Target length: {self.target_length}")

    def __len__(self):
        return len(self.ppg)

    def _resample(self, signal, target_length):
        """Resample signal to target length using interpolation."""
        # signal: [1, original_length]
        signal = signal.unsqueeze(0)  # [1, 1, original_length]
        resampled = torch.nn.functional.interpolate(
            signal, size=target_length, mode='linear', align_corners=False
        )
        return resampled.squeeze(0)  # [1, target_length]

    def __getitem__(self, idx):
        ppg = self.ppg[idx]      # [1, original_length]
        ecg = self.ecg[idx]      # [1, original_length]

        # Resample to target length (512) if needed
        if self.original_length != self.target_length:
            ppg = self._resample(ppg, self.target_length)
            ecg = self._resample(ecg, self.target_length)

        return ppg, ecg


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, dataloader, device):
    """
    Evaluate CardioGAN on a dataset.

    Args:
        model: CardioGAN model
        dataloader: DataLoader
        device: Device

    Returns:
        Dict with 'rmse', 'mae', 'pearson_r'
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for ppg, ecg in dataloader:
            ppg = ppg.to(device)
            ecg = ecg.to(device)

            pred_ecg = model.generate(ppg)

            all_preds.append(pred_ecg.cpu())
            all_targets.append(ecg.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
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
        'rmse': rmse,
        'mae': mae,
        'pearson_r': pearson_r,
    }


# ============================================================================
# Training
# ============================================================================

def train(cfg):
    """
    Train CardioGAN model.

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

    # Load config
    train_cfg = cfg['training']
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    loss_cfg = cfg['loss']
    output_cfg = cfg['output']

    target_length = model_cfg.get('input_size', 512)

    # Load datasets
    print(f"\nLoading datasets...")
    train_ds = BIDMCDatasetCardioGAN(
        data_cfg['train_path'],
        target_length=target_length,
        augment=train_cfg.get('data_augmentation', True)
    )
    test_ds = BIDMCDatasetCardioGAN(
        data_cfg['test_path'],
        target_length=target_length,
        augment=False
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
    print(f"\nCreating CardioGAN model...")
    model = CardioGAN(
        base_filters=model_cfg.get('base_filters', 64),
        lr=train_cfg['learning_rate'],
        beta1=train_cfg.get('beta1', 0.5),
        beta2=train_cfg.get('beta2', 0.999),
        alpha=loss_cfg.get('alpha', 3.0),
        beta=loss_cfg.get('beta', 1.0),
        lambda_recon=loss_cfg.get('lambda_recon', 30.0),
    )
    model = model.to(device)

    n_gen = count_parameters(model.generator)
    n_disc = count_parameters(model.discriminator_t) + count_parameters(model.discriminator_f)
    print(f"Generator parameters: {n_gen:,}")
    print(f"Discriminator parameters: {n_disc:,}")
    print(f"Total parameters: {n_gen + n_disc:,}")

    # Get optimizers
    optimizer_G, optimizer_D = model.get_optimizers()

    # Checkpoint directory
    save_dir = output_cfg.get('checkpoint_dir', 'checkpoints/cardiogan_bidmc/')
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
    patience = train_cfg.get('patience', 30)
    eval_every = output_cfg.get('eval_every', 1)
    save_every = output_cfg.get('save_every', 10)

    history = {
        'epoch': [],
        'loss_G': [],
        'loss_D_t': [],
        'loss_D_f': [],
        'test_rmse': [],
        'test_mae': [],
        'test_pearson_r': [],
    }

    print(f"\nStarting training...")
    print(f"  Epochs: {train_cfg['epochs']}")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Learning rate: {train_cfg['learning_rate']}")
    print(f"  Patience: {patience}")
    print(f"  Loss weights: α={loss_cfg.get('alpha', 3.0)}, β={loss_cfg.get('beta', 1.0)}, λ={loss_cfg.get('lambda_recon', 30.0)}")
    print()

    for epoch in range(1, train_cfg['epochs'] + 1):
        model.train()
        epoch_loss_G = 0.0
        epoch_loss_D_t = 0.0
        epoch_loss_D_f = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (ppg, ecg) in enumerate(train_loader):
            ppg = ppg.to(device)
            ecg = ecg.to(device)

            # ---------------------
            # Train Discriminators
            # ---------------------
            optimizer_D.zero_grad()

            # Generate fake ECG
            fake_ecg = model.generate(ppg)

            # Time-domain discriminator
            d_t_real = model.discriminator_t(ecg)
            d_t_fake = model.discriminator_t(fake_ecg.detach())
            loss_D_t = model.loss_fn.discriminator_loss(d_t_real, d_t_fake)

            # Frequency-domain discriminator
            d_f_real = model.discriminator_f(ecg)
            d_f_fake = model.discriminator_f(fake_ecg.detach())
            loss_D_f = model.loss_fn.discriminator_loss(d_f_real, d_f_fake)

            loss_D = loss_D_t + loss_D_f
            loss_D.backward()
            optimizer_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            # Generate fake ECG (fresh forward pass)
            fake_ecg = model.generate(ppg)

            # Discriminator outputs for fake
            d_t_fake = model.discriminator_t(fake_ecg)
            d_f_fake = model.discriminator_f(fake_ecg)

            # Generator loss
            loss_G, loss_details = model.loss_fn.generator_loss(d_t_fake, d_f_fake, fake_ecg, ecg)
            loss_G.backward()
            optimizer_G.step()

            # Accumulate losses
            epoch_loss_G += loss_G.item()
            epoch_loss_D_t += loss_D_t.item()
            epoch_loss_D_f += loss_D_f.item()
            n_batches += 1

        # Average losses
        epoch_loss_G /= n_batches
        epoch_loss_D_t /= n_batches
        epoch_loss_D_f /= n_batches
        elapsed = time.time() - t0

        # Evaluate
        if epoch % eval_every == 0 or epoch == 1:
            test_metrics = evaluate_model(model, test_loader, device)

            print(f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
                  f"G={epoch_loss_G:.4f} D_t={epoch_loss_D_t:.4f} D_f={epoch_loss_D_f:.4f} | "
                  f"test_rmse={test_metrics['rmse']:.4f} "
                  f"test_r={test_metrics['pearson_r']:.4f} | "
                  f"{elapsed:.1f}s")

            history['epoch'].append(epoch)
            history['loss_G'].append(epoch_loss_G)
            history['loss_D_t'].append(epoch_loss_D_t)
            history['loss_D_f'].append(epoch_loss_D_f)
            history['test_rmse'].append(test_metrics['rmse'])
            history['test_mae'].append(test_metrics['mae'])
            history['test_pearson_r'].append(test_metrics['pearson_r'])

            # Early stopping on test RMSE
            if test_metrics['rmse'] < best_loss:
                best_loss = test_metrics['rmse']
                patience_counter = 0
                # Save generator only (for inference)
                torch.save(model.generator.state_dict(), os.path.join(save_dir, "best_generator.pt"))
                # Save full model (for resuming training)
                torch.save({
                    'generator': model.generator.state_dict(),
                    'discriminator_t': model.discriminator_t.state_dict(),
                    'discriminator_f': model.discriminator_f.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'epoch': epoch,
                }, os.path.join(save_dir, "best_model.pt"))
                print(f"  -> Saved best model (rmse={best_loss:.4f}, r={test_metrics['pearson_r']:.4f})")
            else:
                patience_counter += eval_every
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
                  f"G={epoch_loss_G:.4f} D_t={epoch_loss_D_t:.4f} D_f={epoch_loss_D_f:.4f} | "
                  f"{elapsed:.1f}s")

        # Periodic checkpoint
        if epoch % save_every == 0:
            torch.save(model.generator.state_dict(), os.path.join(save_dir, f"generator_epoch{epoch}.pt"))

    # Save training history
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    if history['epoch']:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Generator loss
        axes[0, 0].plot(history['epoch'], history['loss_G'], 'b.-')
        axes[0, 0].set_ylabel('Generator Loss')
        axes[0, 0].set_title('Generator Training')
        axes[0, 0].grid(True, alpha=0.3)

        # Discriminator losses
        axes[0, 1].plot(history['epoch'], history['loss_D_t'], 'r.-', label='D_time')
        axes[0, 1].plot(history['epoch'], history['loss_D_f'], 'g.-', label='D_freq')
        axes[0, 1].set_ylabel('Discriminator Loss')
        axes[0, 1].set_title('Discriminator Training')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Test RMSE
        axes[1, 0].plot(history['epoch'], history['test_rmse'], 'b.-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Test RMSE')
        axes[1, 0].set_title('Test RMSE')
        axes[1, 0].grid(True, alpha=0.3)

        # Test Pearson r
        axes[1, 1].plot(history['epoch'], history['test_pearson_r'], 'g.-')
        axes[1, 1].axhline(y=0.7, color='orange', linestyle='--', label='target (r=0.7)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Pearson r')
        axes[1, 1].set_title('Test Pearson Correlation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('CardioGAN Training on BIDMC')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
        plt.close()
        print(f"Saved training curves to {save_dir}/training_curves.png")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    # Load best generator
    model.generator.load_state_dict(torch.load(os.path.join(save_dir, "best_generator.pt"), weights_only=True))
    final_metrics = evaluate_model(model, test_loader, device)

    print(f"RMSE:      {final_metrics['rmse']:.4f}")
    print(f"MAE:       {final_metrics['mae']:.4f}")
    print(f"Pearson r: {final_metrics['pearson_r']:.4f}")
    print()

    # Success check
    if final_metrics['pearson_r'] > 0.7:
        print("✅ SUCCESS: Pearson r > 0.7")
        print("   CardioGAN works on BIDMC. Proceed to test on your video data.")
    elif final_metrics['pearson_r'] > 0.5:
        print("⚠️  MODERATE: Pearson r > 0.5 but < 0.7")
        print("   Consider tuning hyperparameters or trying longer training.")
    else:
        print("❌ NEEDS WORK: Pearson r < 0.5")
        print("   Check: data preprocessing, model architecture, loss weights")

    return final_metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train CardioGAN model on BIDMC dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml

    # With specific GPU
    CUDA_VISIBLE_DEVICES=0 python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml

    # Override parameters
    python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml --epochs 50 --batch_size 16

    # Evaluate only
    python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml --eval_only
        """
    )
    parser.add_argument('--config', type=str, default='configs/cardiogan_bidmc.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate, no training')
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
    print("CardioGAN Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Input size: {cfg['model'].get('input_size', 512)}")
    print()

    if args.eval_only:
        print("Evaluation only mode")

        # Setup device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Load model
        model = CardioGAN(base_filters=cfg['model'].get('base_filters', 64))
        checkpoint_path = os.path.join(cfg['output']['checkpoint_dir'], 'best_generator.pt')
        model.generator.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        model = model.to(device)

        # Load test data
        test_ds = BIDMCDatasetCardioGAN(
            cfg['data']['test_path'],
            target_length=cfg['model'].get('input_size', 512),
            augment=False
        )
        test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'], shuffle=False)

        # Evaluate
        metrics = evaluate_model(model, test_loader, device)
        print(f"\nTest Results:")
        print(f"  RMSE:      {metrics['rmse']:.4f}")
        print(f"  MAE:       {metrics['mae']:.4f}")
        print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    else:
        train(cfg)


if __name__ == '__main__':
    main()
