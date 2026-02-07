"""
BIDMC Dataset Preprocessing Script

Downloads and prepares the BIDMC dataset for PPG2ECG training.

BIDMC Dataset:
- Source: https://physionet.org/content/bidmc/1.0.0/
- 53 subjects, 8 minutes each @ 125 Hz
- Contains: PPG, ECG, respiratory signals

This script:
1. Parses BIDMC CSV/WFDB files
2. Extracts PPG and ECG signals
3. Segments into windows (512 samples, ~4 seconds)
4. Detects R-peaks and creates Gaussian expansions for QRS loss
5. Normalizes signals to [-1, 1]
6. Splits by subject: 42 train / 11 test (80/20)
7. Saves as PyTorch tensors

Usage:
    python scripts/prepare_bidmc.py [--data_dir external_data/bidmc] [--output_dir external_data/bidmc_processed]
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    from scipy.signal import find_peaks, resample
    from scipy.ndimage import gaussian_filter1d
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install: pip install torch scipy numpy")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

SAMPLING_RATE = 125  # Hz (BIDMC native)
WINDOW_SIZE = 256    # Training window (256 samples = ~2 seconds)
FULL_WINDOW = 512    # Full window for augmentation (512 samples = ~4 seconds)
TRAIN_RATIO = 0.8    # 80% train, 20% test
RANDOM_SEED = 2019   # Match original implementation


# ============================================================================
# Data Loading
# ============================================================================

def load_bidmc_signals_csv(data_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load BIDMC signals from CSV files.

    Expected structure:
        data_dir/
        ├── bidmc_01_Signals.csv
        ├── bidmc_02_Signals.csv
        └── ...

    Each CSV has columns: Time, PLETH (PPG), II (ECG), RESP, etc.

    Args:
        data_dir: Path to BIDMC data directory

    Returns:
        Dict mapping subject_id to {'ppg': array, 'ecg': array}
    """
    data_dir = Path(data_dir)
    subjects = {}

    # Try to find CSV files
    csv_files = list(data_dir.glob("bidmc_*_Signals.csv"))

    if not csv_files:
        # Try nested structure
        csv_files = list(data_dir.glob("**/bidmc_*_Signals.csv"))

    if not csv_files:
        print(f"No BIDMC CSV files found in {data_dir}")
        print("Expected files like: bidmc_01_Signals.csv")
        print("\nPlease download from: https://physionet.org/content/bidmc/1.0.0/")
        return subjects

    print(f"Found {len(csv_files)} BIDMC signal files")

    for csv_path in sorted(csv_files):
        # Extract subject ID from filename (e.g., bidmc_01_Signals.csv -> 01)
        subject_id = csv_path.stem.split('_')[1]

        try:
            # Load CSV
            import pandas as pd
            df = pd.read_csv(csv_path)

            # Find PPG and ECG columns
            ppg_col = None
            ecg_col = None

            for col in df.columns:
                col_lower = col.lower()
                if 'pleth' in col_lower or 'ppg' in col_lower:
                    ppg_col = col
                elif col_lower == 'ii' or 'ecg' in col_lower:
                    ecg_col = col

            if ppg_col is None or ecg_col is None:
                print(f"  Warning: Missing PPG or ECG in {csv_path.name}")
                print(f"    Columns found: {list(df.columns)}")
                continue

            ppg = df[ppg_col].values.astype(np.float32)
            ecg = df[ecg_col].values.astype(np.float32)

            # Remove NaN values
            valid_mask = ~(np.isnan(ppg) | np.isnan(ecg))
            ppg = ppg[valid_mask]
            ecg = ecg[valid_mask]

            if len(ppg) < FULL_WINDOW:
                print(f"  Warning: Subject {subject_id} too short ({len(ppg)} samples)")
                continue

            subjects[subject_id] = {
                'ppg': ppg,
                'ecg': ecg,
            }
            print(f"  Loaded subject {subject_id}: {len(ppg)} samples ({len(ppg)/SAMPLING_RATE:.1f}s)")

        except Exception as e:
            print(f"  Error loading {csv_path.name}: {e}")

    return subjects


def load_bidmc_signals_wfdb(data_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load BIDMC signals from WFDB format files.

    Expected structure:
        data_dir/
        ├── bidmc01.dat
        ├── bidmc01.hea
        └── ...

    Args:
        data_dir: Path to BIDMC data directory

    Returns:
        Dict mapping subject_id to {'ppg': array, 'ecg': array}
    """
    try:
        import wfdb
    except ImportError:
        print("wfdb not installed. Install with: pip install wfdb")
        return {}

    data_dir = Path(data_dir)
    subjects = {}

    # Find .hea files
    hea_files = list(data_dir.glob("bidmc*.hea"))

    if not hea_files:
        hea_files = list(data_dir.glob("**/bidmc*.hea"))

    if not hea_files:
        return subjects

    print(f"Found {len(hea_files)} WFDB records")

    for hea_path in sorted(hea_files):
        record_name = hea_path.stem
        subject_id = record_name.replace('bidmc', '')

        try:
            record = wfdb.rdrecord(str(hea_path.parent / record_name))

            # Find PPG and ECG signal indices
            ppg_idx = None
            ecg_idx = None

            for i, name in enumerate(record.sig_name):
                name_lower = name.lower()
                if 'pleth' in name_lower or 'ppg' in name_lower:
                    ppg_idx = i
                elif name_lower == 'ii' or 'ecg' in name_lower:
                    ecg_idx = i

            if ppg_idx is None or ecg_idx is None:
                print(f"  Warning: Missing PPG or ECG in {record_name}")
                continue

            ppg = record.p_signal[:, ppg_idx].astype(np.float32)
            ecg = record.p_signal[:, ecg_idx].astype(np.float32)

            subjects[subject_id] = {
                'ppg': ppg,
                'ecg': ecg,
            }
            print(f"  Loaded subject {subject_id}: {len(ppg)} samples ({len(ppg)/SAMPLING_RATE:.1f}s)")

        except Exception as e:
            print(f"  Error loading {record_name}: {e}")

    return subjects


def load_bidmc_signals(data_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load BIDMC signals from either CSV or WFDB format.

    Args:
        data_dir: Path to BIDMC data directory

    Returns:
        Dict mapping subject_id to {'ppg': array, 'ecg': array}
    """
    # Try CSV first
    subjects = load_bidmc_signals_csv(data_dir)

    # If no CSV found, try WFDB
    if not subjects:
        print("Trying WFDB format...")
        subjects = load_bidmc_signals_wfdb(data_dir)

    return subjects


# ============================================================================
# Signal Processing
# ============================================================================

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalize signal to [-1, 1] range.

    Args:
        signal: Input signal

    Returns:
        Normalized signal in [-1, 1]
    """
    signal = signal - np.mean(signal)
    max_abs = np.max(np.abs(signal))
    if max_abs > 0:
        signal = signal / max_abs
    return signal


def detect_rpeaks(ecg: np.ndarray, fs: int = 125) -> np.ndarray:
    """
    Detect R-peaks in ECG signal.

    Uses scipy.signal.find_peaks with physiologically reasonable parameters.

    Args:
        ecg: ECG signal
        fs: Sampling rate

    Returns:
        Array of R-peak indices
    """
    # Normalize
    ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)

    # Find peaks with minimum distance (0.3s = 200 BPM max)
    min_distance = int(0.3 * fs)

    # Use prominence to find significant peaks
    peaks, _ = find_peaks(ecg_norm, distance=min_distance, prominence=0.5)

    return peaks


def expand_rpeaks_gaussian(rpeaks: np.ndarray, length: int, sigma: float = 1.0,
                           fs: int = 125) -> np.ndarray:
    """
    Expand R-peak locations to Gaussian distributions.

    Creates a weight map where R-peak regions have higher values.

    Args:
        rpeaks: R-peak indices
        length: Signal length
        sigma: Gaussian sigma in seconds (default 1.0)
        fs: Sampling rate

    Returns:
        Gaussian-expanded R-peak weights [0, 1]
    """
    expansion = np.zeros(length, dtype=np.float32)

    if len(rpeaks) == 0:
        return expansion

    # Place impulses at R-peak locations
    for peak in rpeaks:
        if 0 <= peak < length:
            expansion[peak] = 1.0

    # Apply Gaussian smoothing
    sigma_samples = sigma * fs
    expansion = gaussian_filter1d(expansion, sigma=sigma_samples)

    # Normalize to [0, 1]
    if np.max(expansion) > 0:
        expansion = expansion / np.max(expansion)

    return expansion


def segment_signals(ppg: np.ndarray, ecg: np.ndarray,
                    full_window: int = 512, stride: int = 256
                    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Segment signals into windows and compute R-peak expansions.

    Args:
        ppg: PPG signal
        ecg: ECG signal
        full_window: Window size
        stride: Stride between windows

    Returns:
        Tuple of (ppg_windows, ecg_windows, rpeak_expansions)
    """
    ppg_windows = []
    ecg_windows = []
    rpeak_expansions = []

    n_samples = len(ppg)
    n_windows = (n_samples - full_window) // stride + 1

    for i in range(n_windows):
        start = i * stride
        end = start + full_window

        ppg_win = ppg[start:end].copy()
        ecg_win = ecg[start:end].copy()

        # Normalize each window independently
        ppg_win = normalize_signal(ppg_win)
        ecg_win = normalize_signal(ecg_win)

        # Detect R-peaks and expand
        rpeaks = detect_rpeaks(ecg_win, fs=SAMPLING_RATE)
        rpeak_exp = expand_rpeaks_gaussian(rpeaks, full_window, sigma=1.0, fs=SAMPLING_RATE)

        ppg_windows.append(ppg_win)
        ecg_windows.append(ecg_win)
        rpeak_expansions.append(rpeak_exp)

    return ppg_windows, ecg_windows, rpeak_expansions


# ============================================================================
# Dataset Creation
# ============================================================================

def create_datasets(subjects: Dict[str, Dict[str, np.ndarray]],
                    train_ratio: float = 0.8,
                    seed: int = 2019
                    ) -> Tuple[Dict, Dict]:
    """
    Create train/test datasets with subject-level split.

    Args:
        subjects: Dict of subject data
        train_ratio: Fraction for training
        seed: Random seed

    Returns:
        Tuple of (train_data, test_data) dicts
    """
    np.random.seed(seed)

    subject_ids = sorted(subjects.keys())
    np.random.shuffle(subject_ids)

    n_train = int(len(subject_ids) * train_ratio)
    train_subjects = subject_ids[:n_train]
    test_subjects = subject_ids[n_train:]

    print(f"\nSubject split: {len(train_subjects)} train, {len(test_subjects)} test")
    print(f"  Train subjects: {train_subjects}")
    print(f"  Test subjects: {test_subjects}")

    def process_subjects(subject_list):
        all_ppg = []
        all_ecg = []
        all_rpeaks = []
        all_subject_ids = []

        for sid in subject_list:
            data = subjects[sid]
            ppg_wins, ecg_wins, rpeak_wins = segment_signals(
                data['ppg'], data['ecg'],
                full_window=FULL_WINDOW,
                stride=WINDOW_SIZE  # Non-overlapping for clean split
            )

            all_ppg.extend(ppg_wins)
            all_ecg.extend(ecg_wins)
            all_rpeaks.extend(rpeak_wins)
            all_subject_ids.extend([sid] * len(ppg_wins))

        return {
            'ppg': np.array(all_ppg, dtype=np.float32),
            'ecg': np.array(all_ecg, dtype=np.float32),
            'rpeaks': np.array(all_rpeaks, dtype=np.float32),
            'subject_ids': all_subject_ids,
        }

    train_data = process_subjects(train_subjects)
    test_data = process_subjects(test_subjects)

    return train_data, test_data


def save_dataset(data: Dict, output_path: str):
    """
    Save dataset as PyTorch tensor file.

    Args:
        data: Dataset dict with 'ppg', 'ecg', 'rpeaks' arrays
        output_path: Output .pt file path
    """
    # Convert to tensors and add channel dimension
    # Shape: [N, full_window] -> [N, 1, full_window]
    tensors = {
        'ppg': torch.from_numpy(data['ppg']).unsqueeze(1),
        'ecg': torch.from_numpy(data['ecg']).unsqueeze(1),
        'rpeaks': torch.from_numpy(data['rpeaks']).unsqueeze(1),
        'subject_ids': data['subject_ids'],
    }

    torch.save(tensors, output_path)
    print(f"Saved {len(data['ppg'])} windows to {output_path}")
    print(f"  PPG shape: {tensors['ppg'].shape}")
    print(f"  ECG shape: {tensors['ecg'].shape}")
    print(f"  R-peaks shape: {tensors['rpeaks'].shape}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prepare BIDMC dataset for PPG2ECG training')
    parser.add_argument('--data_dir', type=str, default='external_data/bidmc',
                        help='Path to raw BIDMC data')
    parser.add_argument('--output_dir', type=str, default='external_data/bidmc_processed',
                        help='Path to save processed data')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Fraction of subjects for training')
    parser.add_argument('--seed', type=int, default=2019,
                        help='Random seed for split')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BIDMC Dataset Preprocessing")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print(f"Window size: {FULL_WINDOW} samples ({FULL_WINDOW/SAMPLING_RATE:.1f}s)")
    print(f"Training window: {WINDOW_SIZE} samples ({WINDOW_SIZE/SAMPLING_RATE:.1f}s)")
    print()

    # Load signals
    print("Loading BIDMC signals...")
    subjects = load_bidmc_signals(args.data_dir)

    if not subjects:
        print("\nNo data loaded. Please check:")
        print(f"  1. Download BIDMC from https://physionet.org/content/bidmc/1.0.0/")
        print(f"  2. Extract to {args.data_dir}")
        print(f"  3. Expected files: bidmc_XX_Signals.csv or bidmc*.dat/.hea")
        sys.exit(1)

    print(f"\nLoaded {len(subjects)} subjects")

    # Create train/test split
    print("\nCreating train/test split...")
    train_data, test_data = create_datasets(
        subjects,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    # Save datasets
    print("\nSaving datasets...")
    save_dataset(train_data, output_dir / 'train.pt')
    save_dataset(test_data, output_dir / 'test.pt')

    # Save metadata
    metadata = {
        'sampling_rate': SAMPLING_RATE,
        'full_window_size': FULL_WINDOW,
        'train_window_size': WINDOW_SIZE,
        'train_subjects': len(set(train_data['subject_ids'])),
        'test_subjects': len(set(test_data['subject_ids'])),
        'train_windows': len(train_data['ppg']),
        'test_windows': len(test_data['ppg']),
        'train_ratio': args.train_ratio,
        'seed': args.seed,
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to {output_dir / 'metadata.json'}")
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"\nTrain set: {metadata['train_windows']} windows from {metadata['train_subjects']} subjects")
    print(f"Test set: {metadata['test_windows']} windows from {metadata['test_subjects']} subjects")
    print(f"\nNext steps:")
    print(f"  1. Run training: python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml")


if __name__ == '__main__':
    main()
