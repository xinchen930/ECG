"""
Detailed analysis of the per-pair normalization issue
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 70)
print("Per-Pair ECG Normalization Analysis")
print("=" * 70)

samples_dir = Path("/home/xinchen/ECG/training_data/samples")
ecg_col = "ecg_counts_filt_monitor"

# Collect per-pair stats
pair_stats = []
for pair_dir in sorted(samples_dir.iterdir()):
    ecg_path = pair_dir / "ecg.csv"
    if not ecg_path.exists():
        continue
    df = pd.read_csv(ecg_path)
    vals = df[ecg_col].values
    pair_stats.append({
        'name': pair_dir.name,
        'mean': vals.mean(),
        'std': vals.std(),
        'min': vals.min(),
        'max': vals.max(),
    })

means = [s['mean'] for s in pair_stats]
stds = [s['std'] for s in pair_stats]

print(f"\nPer-pair MEAN values (before normalization):")
print(f"  Range: [{min(means):.2f}, {max(means):.2f}]")
print(f"  Most are near zero (bandpass filter removes DC)")

print(f"\nPer-pair STD values (before normalization):")
print(f"  Range: [{min(stds):.2f}, {max(stds):.2f}]")
print(f"  Min: {min(stds):.2f} (smallest ECG signal)")
print(f"  Max: {max(stds):.2f} (largest ECG signal)")
print(f"  Ratio max/min: {max(stds)/min(stds):.1f}x")

print(f"\n  After per-pair z-normalization:")
print(f"  Every pair will have mean~0, std~1")
print(f"  This means a LARGE pair (std=18) and SMALL pair (std=4)")
print(f"  will look IDENTICAL after normalization")
print(f"  -> Model cannot learn that different videos produce different")
print(f"     amplitude ECGs (which is actually what we want - the SHAPE matters)")

# Check: what happens to windows within the same pair?
print(f"\n\n{'='*70}")
print(f"Window-Level ECG Statistics (within same pair)")
print(f"{'='*70}")

ecg_sr = 250
window_ecg = 10 * ecg_sr  # 2500

for pair_dir in sorted(samples_dir.iterdir())[:5]:
    ecg_path = pair_dir / "ecg.csv"
    if not ecg_path.exists():
        continue
    df = pd.read_csv(ecg_path)
    vals = df[ecg_col].values.astype(np.float32)
    pair_mean = vals.mean()
    pair_std = vals.std() + 1e-8
    normed = (vals - pair_mean) / pair_std

    print(f"\n  {pair_dir.name} (raw std={vals.std():.2f}):")
    n_windows = len(vals) // window_ecg
    for w in range(min(4, n_windows)):
        seg = normed[w*window_ecg:(w+1)*window_ecg]
        # The per-window mean should deviate from 0
        print(f"    Window {w}: mean={seg.mean():.4f}, std={seg.std():.4f}, "
              f"range=[{seg.min():.3f}, {seg.max():.3f}]")

# The REAL issue: Model output magnitude
print(f"\n\n{'='*70}")
print(f"Critical: Model Output Scale Problem")
print(f"{'='*70}")
print(f"""
The model predicts values in range [-0.03, 0.065] (std ~0.004)
The ECG target is z-normalized to std ~1.0

The model output is 259x too small!

This happens because:
1. Video input is [0, 1] normalized
2. After multiple conv layers, BN, ReLU, and global avg pooling,
   activations are typically in [0, 0.5] range
3. The final Conv1d(64, 1, 1) head produces a weighted sum
4. With weights ~0.01 each and 64 channels of ~0.1 magnitude,
   output is ~0.01-0.06

The model CAN theoretically scale up (no activation constraint on output),
but the optimization landscape makes it hard:
- Starting from random init, gradients push weights toward MSE-optimal
- MSE-optimal for std~0.004 predictions vs std~1.0 target is:
  predicting the MEAN of target (which is ~0)
- The model gets stuck in this local minimum
- Loss barely decreases: predicting mean gives loss ~1.0 (the variance)
  and the model cannot find a gradient path to larger outputs

KEY INSIGHT: The Pearson loss should help (it's scale-invariant),
but at weight 0.1, MSE (weight 1.0) dominates and pushes toward mean.
""")

# Verify: what's the optimal constant prediction?
print(f"Optimal constant prediction analysis:")
print(f"  If model predicts constant c for all time steps,")
print(f"  MSE loss = E[(c - ecg)^2] = c^2 - 2c*E[ecg] + E[ecg^2]")
print(f"  Since E[ecg]~0 and E[ecg^2]=Var(ecg)~1.0,")
print(f"  MSE(c) = c^2 + 1.0, minimized at c=0")
print(f"  MSE(0) = 1.0, which matches the observed RMSE~1.0!")
print(f"  The model IS predicting ~0, giving RMSE~1.0 - exactly as expected")
print(f"  for a model that has collapsed to mean prediction.")
