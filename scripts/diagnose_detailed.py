"""
Detailed diagnosis: Understanding the magnitude mismatch and model collapse
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

import numpy as np
import torch
import yaml

# Load model and data
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

print("=" * 70)
print("DETAILED DIAGNOSIS: Model Collapse Analysis")
print("=" * 70)

# Get first sample
video, ecg = test_ds[0]
video_batch = video.unsqueeze(0).to(device)
ecg_np = ecg.numpy()

with torch.no_grad():
    pred = model(video_batch).cpu().squeeze(0).numpy()

print(f"\n1. MAGNITUDE MISMATCH:")
print(f"   ECG target std: {ecg_np.std():.4f}")
print(f"   Model pred std: {pred.std():.6f}")
print(f"   Ratio (ECG/pred): {ecg_np.std() / pred.std():.1f}x")
print(f"   Model output is ~{ecg_np.std() / pred.std():.0f}x too small!")

print(f"\n2. MODEL HAS COLLAPSED TO NEAR-CONSTANT OUTPUT:")
print(f"   First 20 pred values: {pred[:20]}")
print(f"   First 20 ECG values:  {ecg_np[:20]}")

print(f"\n3. WHY THE MODEL COLLAPSED - Looking at the architecture:")
print(f"   EfficientPhys has AdaptiveAvgPool3d((None, 1, 1)) - global spatial pooling")
print(f"   After 4 blocks of MaxPool3d((1,2,2)) + global pool on 64x64 input:")
print(f"   64 -> 32 -> 16 -> 8 -> 4 -> AdaptiveAvgPool(1,1)")
print(f"   Each spatial average KILLS the tiny PPG fluctuations!")

# Compute what the global average pool does
print(f"\n4. GLOBAL AVERAGE POOL EFFECT ON PPG SIGNAL:")
# The PPG signal is in the RED channel temporal variation
red_temporal = video[:, 2].mean(dim=(1, 2))  # (300,) - spatial mean of red channel
print(f"   Red channel temporal: mean={red_temporal.mean():.6f}, std={red_temporal.std():.6f}")
print(f"   Red channel spatial std per frame: {video[:, 2].std(dim=(1, 2)).mean():.6f}")
print(f"   Temporal signal is {video[:, 2].std(dim=(1, 2)).mean() / red_temporal.std():.0f}x smaller than spatial variation")

# Check all other checkpoints
print(f"\n5. CHECKING OTHER CHECKPOINTS:")
ckpt_dirs = [
    "/home/xinchen/ECG/checkpoints/scheme_f/random_good+moderate_p20_testval_round1_random",
    "/home/xinchen/ECG/checkpoints/scheme_f/random_good_p20",
]

for ckpt_dir in ckpt_dirs:
    summary_path = os.path.join(ckpt_dir, "run_summary.json")
    if os.path.exists(summary_path):
        import json
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"   {os.path.basename(ckpt_dir)}:")
        print(f"     r={summary['test_metrics']['pearson_r']:.4f}, "
              f"RMSE={summary['test_metrics']['rmse']:.4f}")

# Check all scheme checkpoints
import glob
for scheme_dir in sorted(glob.glob("/home/xinchen/ECG/checkpoints/scheme_*/*/run_summary.json")):
    with open(scheme_dir) as f:
        summary = json.load(f)
    dirname = os.path.dirname(scheme_dir)
    print(f"   {os.path.relpath(dirname, '/home/xinchen/ECG/checkpoints')}:")
    print(f"     r={summary['test_metrics']['pearson_r']:.4f}, "
          f"RMSE={summary['test_metrics']['rmse']:.4f}")

print(f"\n6. ROOT CAUSE ANALYSIS:")
print(f"   The video data IS predominantly red (finger-PPG), values around ~0.8")
print(f"   The PPG signal (heart rate pulsation) is in TINY temporal fluctuations (~0.01 std)")
print(f"   The spatial variation is MUCH larger (~0.12 std)")
print(f"   After global spatial average pooling, the tiny temporal signal survives...")
print(f"   BUT: the temporal variation after all conv layers + global pool = ~0.004 std")
print(f"   This is then mapped through temporal_conv + head to get output")
print(f"   The output range is ~0.09 but ECG target range is ~7.0")
print(f"   => The model CANNOT produce large enough outputs to match ECG!")
print(f"   => It settles for predicting near-zero (the mean of normalized ECG)")

print(f"\n7. SPECIFIC ISSUES TO FIX:")
print(f"   A) The model's final conv1d(decoder_in, 1, kernel_size=1) has no activation")
print(f"      -> Check if it can theoretically produce the right magnitude")
head_weight = model.head[0].weight.data if isinstance(model.head, torch.nn.Sequential) else model.head.weight.data
head_bias = model.head[0].bias.data if isinstance(model.head, torch.nn.Sequential) else model.head.bias.data
print(f"      Head weight: shape={head_weight.shape}, norm={head_weight.norm():.4f}")
print(f"      Head bias: {head_bias.item():.6f}")
print(f"      Max possible output scale (with current weights): {head_weight.abs().sum():.4f}")

# What's the effective learning signal?
print(f"\n   B) LOSS FUNCTION BEHAVIOR:")
from video_ecg_model import CompositeLoss
criterion = CompositeLoss(alpha_freq=0.1, beta_pearson=0.1)

# What loss does predicting mean give?
mean_pred = torch.zeros(1, 2500)  # predicting zero (the mean of normalized ECG)
ecg_target = torch.from_numpy(ecg_np).unsqueeze(0)
loss_mean = criterion(mean_pred, ecg_target)
print(f"      Loss for predicting mean (zero): {loss_mean.item():.4f}")

# What loss does the actual model give?
actual_pred = torch.from_numpy(pred).unsqueeze(0)
loss_actual = criterion(actual_pred, ecg_target)
print(f"      Loss for actual model prediction: {loss_actual.item():.4f}")

# What loss does a slightly varied prediction give?
noisy_pred = mean_pred + torch.randn(1, 2500) * 0.01
loss_noisy = criterion(noisy_pred, ecg_target)
print(f"      Loss for zero + noise(std=0.01): {loss_noisy.item():.4f}")

scaled_pred = mean_pred + torch.randn(1, 2500) * 1.0
loss_scaled = criterion(scaled_pred, ecg_target)
print(f"      Loss for zero + noise(std=1.0): {loss_scaled.item():.4f}")

print(f"\n   C) KEY INSIGHT:")
print(f"      The model learns to predict near-zero because:")
print(f"      1. The ECG is z-normalized per PAIR (mean=0, std~1)")
print(f"      2. Predicting ~zero minimizes MSE (it's the mean)")
print(f"      3. The video signal is too weak for the model to learn")
print(f"         useful features that could improve beyond mean prediction")
print(f"      4. The Pearson loss coefficient (0.1) is too small to force")
print(f"         the model to match the waveform SHAPE rather than amplitude")
