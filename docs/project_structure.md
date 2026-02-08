# ECG Reconstruction Project - Structure Report

> Generated: 2026-02-07
>
> Project root: `/Users/zhangxinchen/code/ECG_recon_git`

---

## 1. Directory Tree Overview

```
ECG_recon_git/
├── .claude/
│   └── settings.local.json
├── .cursor/
│   └── debug.log
├── .gitignore                          (29 lines)
├── CLAUDE.md                           (337 lines)
├── SCHEMES.md                          (644 lines)
├── check_video0_red.py                 (54 lines)
├── environment.yaml                    (49 lines)
├── requirements.txt                    (45 lines)
├── run_eval.sh                         (109 lines)
├── run_train.sh                        (195 lines)
├── setup_env.sh                        (55 lines)
│
├── configs/
│   ├── cardiogan_bidmc.yaml            (56 lines)
│   ├── ppg2ecg_bidmc.yaml              (53 lines)
│   ├── scheme_c.yaml                   (41 lines)
│   ├── scheme_d.yaml                   (43 lines)
│   ├── scheme_e.yaml                   (46 lines)
│   ├── scheme_f.yaml                   (44 lines)
│   ├── scheme_g.yaml                   (44 lines)
│   └── server_presets.yaml             (80 lines)
│
├── docs/
│   ├── 2025-02-05_analysis.md          (539 lines)
│   ├── PPG2ECG_MANUAL.md               (420 lines)
│   └── PROMPT_ppg2ecg_implementation.md (396 lines)
│
├── models/
│   ├── __init__.py                     (0 lines)
│   ├── analyze_video_ppg.py            (349 lines)
│   ├── cardiogan.py                    (542 lines)
│   ├── dataset.py                      (422 lines)
│   ├── diagnose.py                     (175 lines)
│   ├── evaluate.py                     (151 lines)
│   ├── ppg2ecg.py                      (391 lines)
│   ├── run_eval.py                     (49 lines)
│   ├── train.py                        (402 lines)
│   ├── train_cardiogan.py              (554 lines)
│   ├── train_ppg2ecg.py                (571 lines)
│   ├── video_ecg_model.py              (1058 lines)
│   └── visualize.ipynb                 (753 lines)
│
├── scripts/
│   ├── data_quality_check_v2.py        (681 lines)
│   └── prepare_bidmc.py                (537 lines)
│
├── checkpoints/                        (gitignored)
│   ├── scheme_a/
│   │   └── best_model.pt
│   ├── scheme_c/random_good_p20/
│   │   ├── best_model.pt, config.yaml, training_curves.png, training_history.json
│   ├── scheme_d/
│   │   ├── best_model.pt
│   │   └── random_good_p30/ (best_model.pt, config.yaml, ...)
│   ├── scheme_e/
│   │   ├── best_model.pt
│   │   └── random_good_p30/ (best_model.pt, config.yaml, ...)
│   ├── scheme_f/
│   │   ├── best_model.pt
│   │   └── random_good_p20/ (best_model.pt, config.yaml, ...)
│   └── scheme_g/random_good_p20/
│       ├── best_model.pt, config.yaml, training_curves.png, training_history.json
│
├── eval_results/
│   ├── data_quality_report.md
│   ├── ppg_analysis_all_samples.csv
│   ├── scheme_a/  (scheme_a_*.json, summary.csv)
│   ├── scheme_d/  (scheme_d_*.json, summary.csv)
│   └── visualize/
│       ├── all_samples.png, ppg_quality_check*.png, sample_*.png
│       ├── scheme_d/ (all_samples.png, ppg_ecg_analysis.png, ...)
│       ├── scheme_e/ (all_samples.png, pred_vs_real_ecg.png, ...)
│       └── scheme_f/ (all_samples.png, pred_vs_real_ecg.png, ...)
│
├── external_data/                      (gitignored, currently EMPTY)
│   ├── bidmc/                          (empty - BIDMC raw data not downloaded)
│   └── bidmc_processed/                (empty - no preprocessed data)
│
└── training_data/
    ├── dataset_info.json
    └── samples/                        (98 pair directories: pair_0000 .. pair_0097)
        └── pair_XXXX/
            ├── video_0.mp4, ecg.csv, imu.csv
            ├── annotation.json, user_info.json, metadata.json
```

**Total source lines**: ~9,885 (excluding data, checkpoints, eval outputs)

---

## 2. models/ -- Model Definitions and Training Scripts

### 2.1 `__init__.py` (0 lines)

- **Function**: Empty package marker.
- **Status**: OK.

---

### 2.2 `dataset.py` (422 lines)

- **Function**: PyTorch `Dataset` for the Video-to-ECG task. Handles windowed sampling (10 s windows, 5 s stride), user-level and random data splitting, and quality filtering.
- **Key APIs**:
  - `load_quality_data(report_path)` -- loads quality labels from CSV
  - `load_metadata(pair_dir)` -- reads per-sample metadata
  - `build_window_index(pair_dirs, cfg)` -- computes (sample, start_frame, end_frame) tuples
  - `split_by_user(pairs, ...)` / `split_random(pairs, ...)` -- data splitting
  - `VideoECGDataset` class -- `__getitem__` returns `(input_tensor, ecg_target)` with optional IMU
  - `create_datasets(cfg, merge_val_to_train=False)` -- one-call factory
- **Input modes**: diff frames (Scheme C), 1D RGB mean (D), green channel (E), raw video frames (F/G), optional IMU fusion.
- **Status**: Active, well-structured. No issues found.

---

### 2.3 `video_ecg_model.py` (1058 lines)

- **Function**: All five model architectures and the composite loss function.
- **Architectures**:
  | Class | Scheme | Description |
  |-------|--------|-------------|
  | `MTTSCANECGModel` | C | Dual-branch (appearance + motion) with TSM + attention |
  | `Signal1DECGModel` | D | 1D Temporal Convolutional Network (dilated causal convolutions) |
  | `UNet1DECGModel` | E | 1D U-Net with skip connections |
  | `EfficientPhysECGModel` | F | Spatiotemporal attention (temporal difference conv + spatial attention) |
  | `PhysNetECGModel` | G | 3D CNN (PhysNet-inspired) |
- **Shared components**: `IMUEncoder`, `TemporalDecoder`, `CompositeLoss` (MSE + spectral + Pearson).
- **Factory functions**: `build_model(cfg)`, `build_criterion(cfg)`.
- **Status**: Active, core file.
  - **Issue**: Contains leftover debug logging in `TemporalShift.forward()` and `DualBranchEncoder.forward()` that writes to `/home/xinchen/ECG/.cursor/debug.log`. These should be removed for production use.

---

### 2.4 `train.py` (402 lines)

- **Function**: Main training script for all Video-to-ECG schemes (C/D/E/F/G). Handles server presets, quality filtering, split mode override, AMP, gradient accumulation, and early stopping.
- **CLI**: `python models/train.py --config <yaml> [--server 3090|a6000] [--split random|user] [--quality-filter good] [--patience N] [--use-val]`
- **Key behavior**:
  - Default: test set used for early stopping (debug mode); `--use-val` switches to strict mode.
  - Loads server presets from `configs/server_presets.yaml` via `--server` flag.
  - Saves checkpoint to `checkpoints/<scheme>/<split>_<quality>_p<patience>/`.
- **Status**: Active.
  - **Issue**: Contains `_dbg()` function that writes to hardcoded `/home/xinchen/ECG/.cursor/debug.log`. Should be removed.

---

### 2.5 `run_eval.py` (49 lines)

- **Function**: Evaluation-only script. Loads a saved checkpoint, runs inference on the test set, prints RMSE / MAE / Pearson r.
- **CLI**: `python models/run_eval.py --config <yaml> --checkpoint <path>`
- **Status**: Active, lightweight. No issues.

---

### 2.6 `evaluate.py` (151 lines)

- **Function**: Evaluation metrics (RMSE, MAE, Pearson r, per-sample Pearson) and a `evaluate_model(model, dataloader, device)` function. Also has a `__main__` block for standalone evaluation with JSON/CSV output.
- **Status**: Active. No issues.

---

### 2.7 `ppg2ecg.py` (391 lines)

- **Function**: PPG2ECG model for BIDMC baseline validation (IEEE Sensors 2020 paper reproduction).
- **Architecture**: Encoder-Decoder (Conv1d stride-2 downsampling, ConvTranspose1d upsampling) + Spatial Transformer Network (STN) + Multi-Head Attention.
- **Classes**: `PPG2ECG`, `PPG2ECG_LSTM` (LSTM baseline), `QRSLoss`, `CombinedLoss`.
- **Input/Output**: `[batch, 1, 256]` -> `[batch, 1, 256]`.
- **Status**: Complete implementation. No issues found.

---

### 2.8 `train_ppg2ecg.py` (571 lines)

- **Function**: Training script for PPG2ECG on BIDMC dataset.
- **Contains**: `BIDMCDataset` class (with random offset augmentation), full training loop, early stopping, CosineAnnealing scheduler, AMP support, evaluation with Pearson r.
- **CLI**: `python models/train_ppg2ecg.py --config configs/ppg2ecg_bidmc.yaml [--eval_only] [--checkpoint <path>]`
- **Status**: Complete. No issues.
  - **Note**: Cannot be tested until BIDMC data is downloaded to `external_data/bidmc/`.

---

### 2.9 `cardiogan.py` (542 lines)

- **Function**: CardioGAN model (AAAI 2021 paper reproduction). Attention U-Net Generator with Dual Discriminators (Time domain + Frequency domain).
- **Classes**: `ConvBlock`, `ConvTransposeBlock`, `AttentionGate`, `Generator`, `DiscriminatorTime`, `DiscriminatorFreq`, `CardioGANLoss`, `CardioGAN`.
- **Input/Output**: `[batch, 1, 512]` -> `[batch, 1, 512]`.
- **Status**: Complete implementation. No issues.

---

### 2.10 `train_cardiogan.py` (554 lines)

- **Function**: Training script for CardioGAN on BIDMC dataset.
- **Contains**: `BIDMCDatasetCardioGAN` class (with internal resampling 125->128 Hz to 512 samples), GAN training loop with separate Generator/Discriminator optimizers, early stopping.
- **CLI**: `python models/train_cardiogan.py --config configs/cardiogan_bidmc.yaml [--eval_only]`
- **Status**: Complete. Same dependency on BIDMC data as `train_ppg2ecg.py`.

---

### 2.11 `analyze_video_ppg.py` (349 lines)

- **Function**: Analysis/diagnostic script that examines video RGB signals and correlates them with ECG heart rate.
- **Key functions**: `get_video_info()`, `extract_rgb_signal()`, `bandpass_filter()`, `detect_peaks_and_hr()`, `analyze_periodicity()`, `get_ecg_hr()`.
- **Status**: Utility script for exploratory analysis. No issues.

---

### 2.12 `diagnose.py` (175 lines)

- **Function**: Diagnostic script that checks PPG-ECG quality and correlation for the first 5 samples in the dataset.
- **Key functions**: `extract_ppg_from_video()`, `analyze_sample()`, `check_ppg_quality()`.
- **Status**: Utility/debugging tool. No issues.

---

### 2.13 `visualize.ipynb` (753 lines / Jupyter notebook)

- **Function**: Jupyter notebook for visualizing model predictions vs ground-truth ECG waveforms.
- **Status**: Utility. Not reviewed in detail.

---

## 3. configs/ -- Configuration Files

### 3.1 Scheme Configs

| File | Lines | Scheme | Input | Resolution | Batch | LR | Notes |
|------|-------|--------|-------|------------|-------|----|-------|
| `scheme_c.yaml` | 41 | MTTS-CAN | diff frames + raw frames | 36x36 | 16 | 1e-3 | Composite loss |
| `scheme_d.yaml` | 43 | 1D TCN | RGB means | 1D | 64 | 1e-3 | Composite loss |
| `scheme_e.yaml` | 46 | 1D UNet | green channel | 1D | 64 | 1e-3 | quality_filter: good,moderate |
| `scheme_f.yaml` | 44 | EfficientPhys | video frames | 64x64 | 8 | 1e-4 | AMP enabled |
| `scheme_g.yaml` | 44 | PhysNet | video frames | 64x64 | 4 | 1e-4 | AMP required, heavy VRAM |

- **Status**: All active, well-organized. No issues.
- **Note**: Schemes C/D/E use `lr: 1e-3` which may be high for some optimizers (consider 3e-4).

### 3.2 Baseline Configs

| File | Lines | Model | Dataset | Window | Batch | LR |
|------|-------|-------|---------|--------|-------|----|
| `ppg2ecg_bidmc.yaml` | 53 | PPG2ECG | BIDMC | 256 @ 125 Hz | 256 | 1e-4 |
| `cardiogan_bidmc.yaml` | 56 | CardioGAN | BIDMC | 512 @ 128 Hz | 16 | 2e-4 |

- **Status**: Complete. Cannot be used until BIDMC data is downloaded.

### 3.3 `server_presets.yaml` (80 lines)

- **Function**: GPU-specific parameter overrides for RTX 3090 and A6000. Controls batch_size, AMP, gradient_accumulation, num_workers per scheme.
- **Status**: Active. No issues.

---

## 4. scripts/ -- Data Processing Scripts

### 4.1 `data_quality_check_v2.py` (681 lines)

- **Function**: Comprehensive data quality checker. Examines 10 representative samples across 8 quality dimensions: video properties (resolution, FPS, duration), ECG signal quality (noise, morphology), IMU data validity, audio presence, time alignment, PPG extractability, ECG waveform analysis.
- **Output**: Quality classification (good / moderate / poor) per sample.
- **Status**: Active utility. No issues.

### 4.2 `prepare_bidmc.py` (537 lines)

- **Function**: BIDMC dataset preprocessor. Loads CSV or WFDB-format signals, segments into overlapping windows, detects R-peaks, creates Gaussian label expansions, normalizes signals to [-1, 1], and splits by subject into `train.pt` and `test.pt`.
- **CLI**: `python scripts/prepare_bidmc.py [--data_dir ...] [--output_dir ...]`
- **Status**: Complete. Depends on BIDMC data download.

---

## 5. docs/ -- Documentation

### 5.1 `2025-02-05_analysis.md` (539 lines)

- **Function**: Comprehensive analysis of project status. Includes model source survey (PPG2ECG literature), experiment plan with Phase 1-4 roadmap, PPG2ECG and CardioGAN comparison.
- **Status**: Historical reference document. Content is dated 2025-02-05.

### 5.2 `PPG2ECG_MANUAL.md` (420 lines)

- **Function**: Step-by-step manual for running PPG2ECG and CardioGAN models on the BIDMC dataset. Includes environment setup, data preparation, training/evaluation commands, expected results, FAQ, and implementation notes.
- **Status**: Active reference. Well-organized.

### 5.3 `PROMPT_ppg2ecg_implementation.md` (396 lines)

- **Function**: Implementation prompt/plan document that was used to guide the creation of the PPG2ECG baseline validation code. Describes architecture details, loss functions, data preprocessing steps.
- **Status**: Historical planning document. Useful as architecture reference.

---

## 6. Root-Level Files

### 6.1 `CLAUDE.md` (337 lines)

- **Function**: Project instructions for Claude Code. Contains project overview, data pipeline description, running instructions, architecture summary, current progress, and task list.
- **Status**: Active, comprehensive. Serves as the primary onboarding document.

### 6.2 `SCHEMES.md` (644 lines)

- **Function**: Detailed documentation of all 5 model schemes (C/D/E/F/G). Includes architectural details, GPU memory guides, training commands, debugging workflow, and comparison tables.
- **Status**: Active reference.

### 6.3 `requirements.txt` (45 lines)

- **Function**: pip dependency list. Core packages: numpy, pandas, scipy, opencv-python-headless, matplotlib, pyyaml, torch, torchvision, wfdb.
- **Status**: Active. No issues.

### 6.4 `environment.yaml` (49 lines)

- **Function**: Conda environment specification. Python 3.10, CUDA 12.6 (pytorch-cuda=12.6), plus pip packages.
- **Status**: Active. Targets Blackwell GPU / CUDA 12.6.

### 6.5 `setup_env.sh` (55 lines)

- **Function**: One-click environment setup shell script. Creates conda env `torch`, installs PyTorch with CUDA 12.6, and pip packages.
- **Status**: Active utility.

### 6.6 `run_train.sh` (195 lines)

- **Function**: Batch training launcher. Iterates over multiple scheme configs and runs training with configurable server presets, split modes, and quality filters.
- **Status**: Active utility.

### 6.7 `run_eval.sh` (109 lines)

- **Function**: Batch evaluation launcher. Runs evaluation on saved checkpoints for multiple schemes.
- **Status**: Active utility.

### 6.8 `check_video0_red.py` (54 lines)

- **Function**: Validates that all `video_0.mp4` files in training_data/samples have a dominant red channel (expected for PPG finger videos where the finger covers the camera with flash on).
- **Status**: Utility/one-time check. No issues.

### 6.9 `.gitignore` (29 lines)

- **Function**: Excludes `__pycache__`, `external_data/`, `checkpoints/`, IDE files, OS files.
- **Status**: Appropriate for the project.

---

## 7. Data Directories

### 7.1 `training_data/samples/` -- 98 pair directories

- Each `pair_XXXX/` contains: `video_0.mp4`, `ecg.csv`, `imu.csv`, `annotation.json`, `user_info.json`, `metadata.json`.
- 98 pairs -> ~1042 windows (10 s window, 5 s stride).
- Quality breakdown: ~80 good/moderate samples, ~10 poor samples (HR error > 20 BPM).

### 7.2 `external_data/` -- EMPTY

- `bidmc/` and `bidmc_processed/` directories exist but contain no files.
- **Action needed**: Download BIDMC dataset from https://physionet.org/content/bidmc/1.0.0/ to run PPG2ECG / CardioGAN baselines.

### 7.3 `checkpoints/` -- Trained Models

| Scheme | Subdirectory | Contents |
|--------|-------------|----------|
| A | `scheme_a/` | `best_model.pt` |
| C | `scheme_c/random_good_p20/` | best_model.pt, config.yaml, training_curves.png, training_history.json |
| D | `scheme_d/` + `random_good_p30/` | Two checkpoints (root + subfolder) |
| E | `scheme_e/` + `random_good_p30/` | Two checkpoints (root + subfolder) |
| F | `scheme_f/` + `random_good_p20/` | Two checkpoints (root + subfolder) |
| G | `scheme_g/random_good_p20/` | best_model.pt, config.yaml, training_curves.png, training_history.json |

- **Note**: Scheme A checkpoint exists but Scheme A (ResNet) has been removed from the codebase. This checkpoint is orphaned.

### 7.4 `eval_results/` -- Evaluation Outputs

| Path | Description |
|------|-------------|
| `data_quality_report.md` | Data quality assessment |
| `ppg_analysis_all_samples.csv` | PPG analysis results for all 98 samples |
| `scheme_a/` | Scheme A evaluation JSON + summary CSV |
| `scheme_d/` | Scheme D evaluation JSON + summary CSV |
| `visualize/` | Visualization PNGs (scheme_d, scheme_e, scheme_f subfolders) |

---

## 8. Issues and Recommendations

### 8.1 Debug Logging Artifacts (Should Clean Up)

| File | Location | Issue |
|------|----------|-------|
| `models/video_ecg_model.py` | `TemporalShift.forward()`, `DualBranchEncoder.forward()` | Writes debug logs to hardcoded path `/home/xinchen/ECG/.cursor/debug.log` |
| `models/train.py` | `_dbg()` function | Writes debug logs to hardcoded path `/home/xinchen/ECG/.cursor/debug.log` |

These are leftover development artifacts. The hardcoded server path will fail silently on other machines but adds unnecessary I/O. Recommend removing.

### 8.2 Orphaned Checkpoint

- `checkpoints/scheme_a/best_model.pt` exists, but Scheme A (ResNet) has been removed from the model definitions in `video_ecg_model.py`. This checkpoint cannot be loaded by any current code. Consider deleting.

### 8.3 External Data Not Downloaded

- BIDMC dataset directories (`external_data/bidmc/`, `external_data/bidmc_processed/`) are empty. The PPG2ECG and CardioGAN baseline experiments cannot be run until this data is downloaded and preprocessed.

### 8.4 Learning Rate Consistency

- Schemes C/D/E use `lr: 1e-3`, which may be aggressive. The MEMORY.md notes suggest considering `3e-4`. Schemes F/G correctly use `lr: 1e-4` for larger models.

### 8.5 No Data Augmentation

- As noted in CLAUDE.md, data augmentation has not been implemented for the Video-to-ECG pipeline. With only 98 samples (~1042 windows), augmentation would likely improve generalization.

### 8.6 `.cursor/debug.log` in Project Root

- A `.cursor/debug.log` file exists in the project root. This is a development artifact and should be added to `.gitignore` or deleted.

---

## 9. Summary Statistics

| Category | Count |
|----------|-------|
| Python source files | 14 |
| YAML config files | 8 |
| Shell scripts | 3 |
| Documentation files (md) | 6 |
| Jupyter notebooks | 1 |
| Total source lines | ~9,885 |
| Training samples | 98 pairs |
| Trained checkpoints | 6 schemes |
| Model architectures | 7 (5 Video-ECG + PPG2ECG + CardioGAN) |
