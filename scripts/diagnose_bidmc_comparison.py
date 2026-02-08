"""
Compare BIDMC hospital PPG-ECG correlation with our video PPG-ECG correlation.
BIDMC has clinical pulse oximeter PPG (PLETH) and ECG Lead II at 125 Hz.
This establishes the theoretical upper bound for PPG→ECG correlation.
"""
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import pearsonr
from pathlib import Path

BIDMC_DIR = '/home/xinchen/ECG/external_data/bidmc'


def main():
    print("=" * 70)
    print("  BIDMC Hospital PPG-ECG Correlation (Reference Baseline)")
    print("=" * 70)

    results = []
    for i in range(1, 54):
        path = Path(BIDMC_DIR) / f"bidmc_{i:02d}_Signals.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        ppg = df['PLETH'].values.astype(np.float64)
        ecg = df['II'].values.astype(np.float64)
        sr = 125  # Hz

        # Window-based correlation (10 second windows, 5 second stride)
        window = 10 * sr  # 1250 samples
        stride = 5 * sr  # 625 samples

        corrs_raw = []
        corrs_bp = []

        n_windows = (len(ppg) - window) // stride + 1
        for w in range(n_windows):
            start = w * stride
            ppg_win = ppg[start:start + window]
            ecg_win = ecg[start:start + window]

            # Raw z-norm correlation
            ppg_n = (ppg_win - ppg_win.mean()) / (ppg_win.std() + 1e-8)
            ecg_n = (ecg_win - ecg_win.mean()) / (ecg_win.std() + 1e-8)
            r, _ = pearsonr(ppg_n, ecg_n)
            corrs_raw.append(r)

            # Bandpass filtered
            sos = scipy_signal.butter(4, [0.5, 5.0], btype='bandpass', fs=sr, output='sos')
            ppg_bp = scipy_signal.sosfiltfilt(sos, ppg_win)
            ecg_bp = scipy_signal.sosfiltfilt(sos, ecg_win)
            ppg_bp_n = (ppg_bp - ppg_bp.mean()) / (ppg_bp.std() + 1e-8)
            ecg_bp_n = (ecg_bp - ecg_bp.mean()) / (ecg_bp.std() + 1e-8)
            r_bp, _ = pearsonr(ppg_bp_n, ecg_bp_n)
            corrs_bp.append(r_bp)

        results.append({
            'subject': i,
            'mean_raw': np.mean(corrs_raw),
            'mean_bp': np.mean(corrs_bp),
            'mean_abs_raw': np.mean(np.abs(corrs_raw)),
            'mean_abs_bp': np.mean(np.abs(corrs_bp)),
            'n_windows': len(corrs_raw),
        })

    # Summary
    all_raw = [r['mean_abs_raw'] for r in results]
    all_bp = [r['mean_abs_bp'] for r in results]

    print(f"\n  Subjects analyzed: {len(results)}")
    print(f"\n  Raw PPG-ECG |correlation|:")
    print(f"    Mean: {np.mean(all_raw):.4f}")
    print(f"    Median: {np.median(all_raw):.4f}")
    print(f"    Min: {np.min(all_raw):.4f}")
    print(f"    Max: {np.max(all_raw):.4f}")

    print(f"\n  Bandpass filtered PPG-ECG |correlation|:")
    print(f"    Mean: {np.mean(all_bp):.4f}")
    print(f"    Median: {np.median(all_bp):.4f}")
    print(f"    Min: {np.min(all_bp):.4f}")
    print(f"    Max: {np.max(all_bp):.4f}")

    # Sign analysis
    raw_signed = [r['mean_raw'] for r in results]
    bp_signed = [r['mean_bp'] for r in results]
    print(f"\n  Sign analysis (raw):")
    print(f"    Positive mean r: {sum(1 for r in raw_signed if r > 0)}/{len(raw_signed)}")
    print(f"    Negative mean r: {sum(1 for r in raw_signed if r < 0)}/{len(raw_signed)}")

    print(f"\n  Per-subject breakdown:")
    print(f"  {'Subject':>8} {'|r_raw|':>8} {'|r_bp|':>8} {'r_raw':>8} {'r_bp':>8}")
    print(f"  {'-'*44}")
    for r in sorted(results, key=lambda x: x['mean_abs_bp'], reverse=True):
        print(f"  {r['subject']:>8d} {r['mean_abs_raw']:>8.4f} {r['mean_abs_bp']:>8.4f} "
              f"{r['mean_raw']:>+8.4f} {r['mean_bp']:>+8.4f}")

    # Comparison
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Hospital PPG vs Our Video PPG")
    print(f"{'='*70}")
    print(f"  Hospital PPG (BIDMC, clinical pulse oximeter, 125 Hz):")
    print(f"    Raw |r|: {np.mean(all_raw):.4f}")
    print(f"    Bandpass |r|: {np.mean(all_bp):.4f}")
    print(f"  Our Video PPG (finger-over-camera, 30 Hz):")
    print(f"    Raw |r|: 0.1176  (from previous analysis)")
    print(f"    Bandpass |r|: 0.1422  (from previous analysis)")
    print(f"\n  Gap: {np.mean(all_bp)/0.1422:.1f}x lower correlation in our data")


if __name__ == "__main__":
    main()
