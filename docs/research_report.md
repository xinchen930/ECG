# Video-to-ECG Reconstruction: Comprehensive Method Survey

> **Project Context**: Contact PPG video (finger on camera, ~30fps, red-dominant) + IMU --> ECG waveform (250 Hz, Lead II)
> **Data**: 98 paired samples, ~1042 10s windows after segmentation
> **Date**: 2026-02

---

## Table of Contents

1. [Method Taxonomy Tree](#1-method-taxonomy-tree)
2. [Direction 1: Video to PPG Methods](#2-direction-1-video-to-ppg-methods)
   - [2.1 Traditional Signal Processing](#21-traditional-signal-processing-methods)
   - [2.2 Deep Learning Methods](#22-deep-learning-methods)
   - [2.3 PPG Representation Comparison](#23-ppg-representation-comparison)
3. [Direction 2: PPG to ECG Reconstruction](#3-direction-2-ppg-to-ecg-reconstruction)
4. [Direction 3: Video to ECG End-to-End](#4-direction-3-video-to-ecg-end-to-end)
5. [Direction 4: Multi-Modal Fusion](#5-direction-4-multi-modal-fusion)
6. [Direction 5: Loss Functions and Training Tricks](#6-direction-5-loss-functions-and-training-tricks)
7. [Direction 6: Pretrained Backbone Selection](#7-direction-6-pretrained-backbone-selection)
8. [Recommended Implementation Plan](#8-recommended-implementation-plan)
9. [Key Design Decisions](#9-key-design-decisions)
10. [References and Open-Source Code](#10-references-and-open-source-code)

---

## 1. Method Taxonomy Tree

```
Video-to-ECG Reconstruction
|
+-- [A] Two-Stage Pipeline: Video --> PPG --> ECG
|   |
|   +-- Stage 1: Video --> PPG
|   |   +-- Traditional Signal Processing
|   |   |   +-- GREEN (2008) - Green channel mean
|   |   |   +-- ICA (2011) - Independent Component Analysis
|   |   |   +-- CHROM (2013) - Chrominance-based
|   |   |   +-- PBV (2014) - Blood Volume Pulse signature
|   |   |   +-- POS (2016) - Plane Orthogonal to Skin
|   |   |   +-- LGI (2018) - Local Group Invariance
|   |   |   +-- OMIT (2023) - Face2PPG unsupervised
|   |   |
|   |   +-- Deep Learning (2D CNN)
|   |   |   +-- DeepPhys (2018) - Dual-branch attention CNN
|   |   |   +-- TS-CAN / MTTS-CAN (2020) - Temporal Shift + Attention
|   |   |   +-- EfficientPhys (2023) - Simplified TSM + Attention
|   |   |   +-- BigSmall (2023) - Dual-resolution multi-task
|   |   |
|   |   +-- Deep Learning (3D CNN)
|   |   |   +-- PhysNet (2019) - 3D spatiotemporal CNN
|   |   |   +-- iBVPNet (2024) - 3D CNN for iBVP dataset
|   |   |
|   |   +-- Deep Learning (Transformer)
|   |   |   +-- PhysFormer (2022) - Temporal Difference Transformer
|   |   |   +-- PhysFormer++ (2023) - SlowFast TD Transformer
|   |   |   +-- RhythmFormer (2024) - Hierarchical Temporal Periodic Transformer
|   |   |
|   |   +-- Deep Learning (SSM/Mamba)
|   |   |   +-- PhysMamba (2024) - SlowFast TD Mamba
|   |   |
|   |   +-- Deep Learning (Other)
|   |       +-- ContrastPhys (2022) - Contrastive self-supervised
|   |       +-- FactorizePhys (2024) - NMF attention
|   |       +-- Dual-GAN rPPG - GAN-based extraction
|   |
|   +-- Stage 2: PPG --> ECG
|       +-- Encoder-Decoder
|       |   +-- PPG2ECG (Tian 2020) - Attentional Enc-Dec + STN
|       |   +-- W-Net (2024) - Dual U-Net
|       |
|       +-- GAN-based
|       |   +-- CardioGAN (Sarkar 2021) - Attention U-Net + Dual Discriminator
|       |   +-- P2E-WGAN (Vo 2021) - Conditional WGAN-GP
|       |   +-- P2E-LSGAN - Least-Square GAN
|       |
|       +-- U-Net based
|       |   +-- 1D UNet (our Scheme E)
|       |   +-- Attention U-Net variants
|       |
|       +-- Transformer-based
|       |   +-- Cross-attention PPG2ECG (2024+)
|       |
|       +-- Diffusion-based (emerging 2024+)
|           +-- DiffECG, CardioWave (conceptual)
|
+-- [B] End-to-End: Video --> ECG (Direct)
|   |
|   +-- 3D CNN + Temporal Decoder
|   |   +-- PhysNet-ECG (our Scheme G)
|   |   +-- EfficientPhys-ECG (our Scheme F)
|   |
|   +-- MTTS-CAN + Temporal Decoder (our Scheme C)
|   |
|   +-- Hybrid CNN-Transformer (proposed)
|   |   +-- PhysFormer backbone + ECG decoder
|   |   +-- Video Swin Transformer + ECG head
|   |
|   +-- Multi-task: PPG + ECG simultaneous prediction
|
+-- [C] Representation-Level Approaches
    |
    +-- 1D: Channel means (GREEN, RGB average)
    +-- 2D: STMap (Spatio-Temporal Map)
    +-- 3D: Pixel-level PPG volume
    +-- Feature-level: Neural network intermediate features
```

---

## 2. Direction 1: Video to PPG Methods

### 2.1 Traditional Signal Processing Methods

These methods are primarily designed for remote PPG (face video), but the core signal processing principles are relevant. For **contact PPG** (finger on camera), these serve as baselines or preprocessing steps.

#### GREEN (Verkruysse et al., 2008)

| Item | Detail |
|------|--------|
| **Paper** | "Remote plethysmographic imaging using ambient light" |
| **Principle** | Extracts PPG from the green channel mean intensity over ROI. Green light (520-570nm) is most absorbed by hemoglobin, so blood volume changes create the strongest pulsatile signal in the green channel. |
| **Formula** | `PPG(t) = mean(G_channel(t))` averaged over ROI pixels |
| **Input** | Video frames, ROI mask |
| **Output** | 1D PPG signal at video framerate |
| **Pros** | Extremely simple; good baseline; works well for contact PPG |
| **Cons** | Susceptible to motion artifacts; ignores spatial information; limited by single channel |
| **Contact PPG Suitability** | **HIGH** - In contact mode, green channel signal is very strong |
| **Recommendation** | 3/5 - Good as the simplest baseline, but likely insufficient for ECG reconstruction |

**Important note for our project**: In contact finger PPG, the **red channel is dominant** (camera LED is red, light transmitted through finger tissue is mostly red). Unlike rPPG where green is preferred, for contact PPG we should consider red channel or all RGB channels.

#### ICA (Poh et al., 2011)

| Item | Detail |
|------|--------|
| **Paper** | "Advancements in noncontact multiparameter physiological measurements using a webcam" |
| **Principle** | Models observed RGB channels as linear mixtures of independent source signals. Applies Independent Component Analysis (FastICA) to separate the PPG source from noise/motion artifacts. |
| **Formula** | `X = A * S` where X = observed RGB, A = mixing matrix, S = independent sources. ICA recovers S, and the component with strongest cardiac periodicity is selected as PPG. |
| **Input** | RGB channel mean traces (3 x T) |
| **Output** | 1D PPG signal |
| **Pros** | Better noise separation than single channel; can separate motion from cardiac |
| **Cons** | Assumes linearity; component selection can fail; batch processing (not real-time) |
| **Contact PPG Suitability** | MEDIUM - Less needed when SNR is already high |
| **Recommendation** | 2/5 - Overly complex for contact PPG |

#### CHROM (de Haan & Jeanne, 2013)

| Item | Detail |
|------|--------|
| **Paper** | "Robust pulse rate from chrominance-based rPPG" |
| **Principle** | Projects RGB signals onto a chrominance plane orthogonal to specular reflections. Uses linear combination of chrominance signals to cancel out motion artifacts. |
| **Formula** | `X = 3R/mean(R) - 2`, `Y = 1.5R/mean(R) + G/mean(G) - 1.5B/mean(B)`, `PPG = X - alpha*Y` where `alpha = std(X)/std(Y)` |
| **Input** | Spatially averaged RGB traces |
| **Output** | 1D PPG signal |
| **Pros** | Good motion robustness; real-time capable; well-understood mathematically |
| **Cons** | Assumes skin-like spectral properties; designed for reflective (face) not transmissive (finger) mode |
| **Contact PPG Suitability** | LOW-MEDIUM - The chrominance model assumes reflected light from skin surface, while contact PPG is transmissive |
| **Recommendation** | 2/5 - Not ideal for contact finger video |

#### POS (Wang et al., 2017)

| Item | Detail |
|------|--------|
| **Paper** | "Algorithmic principles of remote PPG" |
| **Principle** | Projects RGB signal onto a plane orthogonal to the skin tone direction. Uses the Blood Volume Pulse (BVP) vector as a projection axis, computed from temporal statistics. |
| **Formula** | `S1 = G - B`, `S2 = -2R + G + B`, `PPG = S1 + alpha*S2` where `alpha = std(S1)/std(S2)` computed in temporal windows |
| **Input** | Spatially averaged RGB traces |
| **Output** | 1D PPG signal |
| **Pros** | State-of-the-art among traditional methods; robust to illumination changes |
| **Cons** | Same reflective model assumptions as CHROM |
| **Contact PPG Suitability** | LOW-MEDIUM |
| **Recommendation** | 2/5 - Better for rPPG than contact PPG |

#### PBV (de Haan & van Leest, 2014)

| Item | Detail |
|------|--------|
| **Paper** | "Improved motion robustness of remote-PPG by using the blood volume pulse signature" |
| **Principle** | Uses a predefined signature vector for how blood volume changes affect RGB channels. Projects observed signals onto this signature. |
| **Input/Output** | RGB traces --> 1D PPG |
| **Pros** | Physically motivated; good for static subjects |
| **Cons** | Requires accurate skin model; calibration-dependent |
| **Contact PPG Suitability** | LOW - Designed for reflective mode |
| **Recommendation** | 1/5 |

#### LGI (Pilz et al., 2018)

| Item | Detail |
|------|--------|
| **Paper** | "Local Group Invariance for heart rate estimation from face videos in the wild" |
| **Principle** | Uses local group invariance for spatiotemporal consistency in PPG extraction, making it robust to face movement and deformation. |
| **Contact PPG Suitability** | LOW - Face-specific |
| **Recommendation** | 1/5 |

#### OMIT / Face2PPG (Alvarez et al., 2023)

| Item | Detail |
|------|--------|
| **Paper** | "Face2PPG: An Unsupervised Pipeline for Blood Volume Pulse Extraction From Faces" |
| **Principle** | Unsupervised pipeline combining traditional methods with learned ROI selection |
| **Contact PPG Suitability** | LOW - Face-specific |
| **Recommendation** | 1/5 |

#### Summary: Traditional Methods for Contact Finger PPG

For contact finger PPG, the traditional methods designed for rPPG are **mostly not directly applicable** because:
1. They assume **reflective** light from skin surface (rPPG), while finger contact is **transmissive**
2. The chrominance/skin-tone models don't apply when the camera is saturated with red light from the LED
3. However, **GREEN channel mean** remains a valid simple baseline
4. **Red channel mean** may actually be more informative for our setup since the LED is red

**Recommended traditional baselines for our project**:
- Red channel mean (strongest signal in transmissive mode)
- Green channel mean (classical baseline)
- All RGB channels as multi-channel input (let the model learn the optimal combination)

---

### 2.2 Deep Learning Methods

#### DeepPhys (Chen & McDuff, 2018)

| Item | Detail |
|------|--------|
| **Paper** | "DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks" |
| **Year** | 2018 |
| **Architecture** | Dual-branch 2D CNN: Motion branch (temporal difference frames) + Appearance branch (raw frames). Appearance generates spatial attention masks that gate motion features. |
| **Input** | Pairs of (diff_frame, raw_frame), each 3-channel, typically 36x36 |
| **Output** | Per-frame scalar (PPG amplitude) |
| **Key Innovation** | First to use appearance-based spatial attention to focus on skin regions; temporal difference as explicit motion input |
| **Params** | ~700K |
| **Memory** | ~2-4 GB |
| **Code** | [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) |
| **Contact PPG Suitability** | **HIGH** - Dual-branch with attention is useful for contact PPG to focus on well-perfused regions |
| **Recommendation** | 3/5 - Good starting point but limited temporal modeling |

**Architecture Detail**:
```
Motion (diff frames): Conv2d(3->32->32->64->64) + Tanh
Appearance (raw frames): Conv2d(3->32->32->64->64) + Tanh
Attention: 1x1 Conv -> Sigmoid -> normalize -> gate motion
Output: AvgPool -> Flatten -> FC(128) -> FC(1)
```

#### TS-CAN / MTTS-CAN (Liu et al., 2020)

| Item | Detail |
|------|--------|
| **Paper** | "Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement" (NeurIPS 2020) |
| **Year** | 2020 |
| **Architecture** | Extends DeepPhys with Temporal Shift Module (TSM) for efficient temporal modeling without 3D convolutions. Multi-task version predicts both pulse and respiration. |
| **Input** | (diff_frame + raw_frame), 36x36, 6 channels concatenated |
| **Output** | Per-frame scalar (pulse), optionally respiration |
| **Key Innovation** | TSM shifts 1/3 channels forward, 1/3 backward for zero-cost temporal modeling; achieves near-3D CNN performance with 2D CNN cost |
| **Params** | ~2.8M |
| **Memory** | ~10-15 GB |
| **Performance** | MAE=1.47 BPM, MAPE=1.56%, r=0.99 (on UBFC) |
| **Code** | [MTTS-CAN](https://github.com/xliucs/MTTS-CAN), [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) |
| **Contact PPG Suitability** | **HIGH** - Already implemented as Scheme C in our project |
| **Recommendation** | 4/5 - Proven architecture, good balance of efficiency and accuracy |

**TSM Detail**:
```
For channel tensor of shape (B, T, C, H, W):
  channels[0:C/3] shifted left (future --> current)
  channels[C/3:2C/3] shifted right (past --> current)
  channels[2C/3:C] unchanged
This provides temporal context without additional computation.
```

#### EfficientPhys (Liu et al., 2023)

| Item | Detail |
|------|--------|
| **Paper** | "EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement" (WACV 2023) |
| **Year** | 2023 |
| **Architecture** | Simplified version of MTTS-CAN. Uses TSM + attention but removes preprocessing requirements (no face detection, no normalization). Can use either transformer or CNN backbone. |
| **Input** | Raw video frames (3, H, W), typically 36x36 or 72x72 |
| **Output** | Per-frame scalar (PPG) |
| **Key Innovation** | End-to-end with no preprocessing; 33% more efficient than MTTS-CAN; works directly from raw frames |
| **Params** | ~1.5M |
| **Memory** | ~10 GB (batch=8) |
| **Code** | [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) |
| **Contact PPG Suitability** | **HIGH** - Simplified preprocessing is a big advantage; already in project as Scheme F |
| **Recommendation** | 4/5 - Strong candidate for contact PPG |

**Architecture Detail**:
```
Input BatchNorm -> 4x TSM stages with Conv2d(3->32->32->64->64)
Attention gates at 2 stages
Flatten -> FC(128) -> FC(1)
TSM operates on n_segment (default 20) frames
```

#### BigSmall (Narayanswamy et al., 2023)

| Item | Detail |
|------|--------|
| **Paper** | "BigSmall: Efficient Multi-Task Learning for Disparate Spatial and Temporal Physiological Measurements" (NeurIPS 2023) |
| **Year** | 2023 |
| **Architecture** | Dual-resolution design: "Big" branch processes high-res frames, "Small" branch with Wrapping TSM (WTSM) processes low-res. Features fused via element-wise addition. Three task heads: Action Units, BVP, Respiration. |
| **Input** | Two resolutions of video frames |
| **Output** | Multi-task: AU (12 classes), BVP (1D), Respiration (1D) |
| **Key Innovation** | Efficient multi-task across disparate temporal scales; WTSM wraps shifted channels around |
| **Params** | ~2-3M |
| **Code** | [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) |
| **Contact PPG Suitability** | MEDIUM - Multi-task overhead not needed for our single-task scenario |
| **Recommendation** | 2/5 - Interesting but over-engineered for our use case |

#### PhysNet (Yu et al., 2019)

| Item | Detail |
|------|--------|
| **Paper** | "Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks" (BMVC 2019) |
| **Year** | 2019 |
| **Architecture** | Pure 3D CNN. Encoder: Conv3d blocks progressively downsample spatially while preserving temporal. Decoder: Transposed Conv3d upsamples temporal. Final: AdaptiveAvgPool3d collapses spatial, Conv1d outputs 1D signal. |
| **Input** | (B, 3, T, 128, 128) - diff-normalized video frames |
| **Output** | (B, T) - rPPG signal |
| **Key Innovation** | First successful 3D CNN for rPPG; encoder-decoder with temporal upsampling; Negative Pearson correlation loss |
| **Params** | ~3-5M |
| **Memory** | ~20-25 GB |
| **Code** | [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox), [PhysNet](https://github.com/ZitongYu/PhysNet) |
| **Contact PPG Suitability** | **HIGH** - 3D CNN captures spatiotemporal patterns; already in project as Scheme G |
| **Recommendation** | 4/5 - Strong architecture but memory-intensive |

**Architecture Detail**:
```
Encoder:
  ConvBlock1: Conv3d(3->16, k=[1,5,5]) -> MaxPool(spatial 2x)
  ConvBlock2-3: Conv3d(16->32->64, k=3) -> MaxPool
  ConvBlock4-9: Conv3d(64->64, k=3) with spatiotemporal pooling
  Spatial: 128->64->32->16->8
  Temporal: T -> T/4

Decoder:
  TransConv3d: T/4 -> T (upsample temporal)
  AdaptiveAvgPool3d(None,1,1): collapse spatial
  Conv1d(64->1): output head
```

#### PhysFormer (Yu et al., 2022)

| Item | Detail |
|------|--------|
| **Paper** | "PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer" (CVPR 2022) |
| **Year** | 2022 |
| **Architecture** | 3D CNN stem for spatial feature extraction --> Temporal Difference Transformer (TD-Transformer) for temporal modeling --> Decoder with temporal upsampling. Uses Center-Difference Convolution (CDC) in attention for temporal sensitivity. |
| **Input** | (B, 3, 160, H, W) - 160 frames per clip |
| **Output** | (B, 160) - rPPG signal |
| **Key Innovations** | (1) Temporal Difference Convolution in Q/K projections captures motion; (2) "gra_sharp" attention scaling adapts across layers; (3) Spatiotemporal feed-forward with depthwise 3D conv |
| **Params** | ~5-8M |
| **Memory** | ~25-30 GB |
| **Performance** | State-of-the-art on VIPL-HR, PURE, UBFC datasets |
| **Code** | [PhysFormer](https://github.com/ZitongYu/PhysFormer), [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) |
| **Contact PPG Suitability** | **HIGH** - Temporal difference mechanism is perfect for detecting subtle brightness changes |
| **Recommendation** | 5/5 - Top recommendation for upgrade path; temporal difference is ideal for contact PPG |

**Architecture Detail**:
```
Stem (3D CNN):
  Conv3d(3->dim/4, k=[1,5,5]) -> MaxPool -> BN -> ReLU
  Conv3d(dim/4->dim/2, k=3) -> MaxPool -> BN -> ReLU
  Conv3d(dim/2->dim, k=3) -> MaxPool -> BN -> ReLU
  Output: (B, dim, T/4, 4, 4)

Transformer Blocks (x4-8):
  CDC_T(Q): center-difference temporal conv for queries
  CDC_T(K): center-difference temporal conv for keys
  Standard Conv(V): for values
  Multi-head attention with gra_sharp scaling
  ST Feed-Forward: FC->DepthwiseConv3d->FC

Decoder:
  Upsample T/4 -> T via ConvTranspose1d
  AdaptiveAvgPool (spatial) -> Conv1d output
```

**Temporal Difference Convolution (CDC_T)**:
```
Standard conv output: F_standard
Difference conv: F_diff = weight[:,0,:,:] + weight[:,2,:,:] (temporal kernel=3)
Combined: F = theta * F_diff + (1-theta) * F_standard
theta=0.6 by default (emphasizes difference)
```

#### PhysFormer++ (Yu et al., 2023)

| Item | Detail |
|------|--------|
| **Paper** | "PhysFormer++: Facial Video-based Physiological Measurement with SlowFast Temporal Difference Transformer" (IJCV 2023) |
| **Year** | 2023 |
| **Architecture** | Extends PhysFormer with SlowFast dual-pathway processing. Slow path captures low-frequency trends, Fast path captures high-frequency cardiac details. Cross-pathway attention for fusion. |
| **Key Innovation** | SlowFast design separates frequency components; better for both HR and HRV estimation |
| **Contact PPG Suitability** | **HIGH** |
| **Recommendation** | 5/5 - If PhysFormer works well, this is the natural upgrade |

#### RhythmFormer (Zou et al., 2024)

| Item | Detail |
|------|--------|
| **Paper** | "Extracting rPPG Signals Based on Hierarchical Temporal Periodic Transformer" (AAAI 2024) |
| **Year** | 2024 |
| **Architecture** | Fusion stem (RGB + temporal diff weighted fusion) --> Patch embedding --> Hierarchical Temporal Periodic Transformer (TPT) blocks with bilateral regional attention --> rPPG output. Multi-scale temporal processing with downsampling/upsampling. |
| **Key Innovations** | (1) Bilateral Regional Attention: efficient local-to-global attention with region routing; (2) Hierarchical multi-scale temporal decomposition; (3) Fusion stem combines RGB and temporal difference streams with learnable weights |
| **Params** | ~5-10M |
| **Code** | [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) |
| **Contact PPG Suitability** | **HIGH** - Hierarchical temporal modeling is excellent for cardiac signal extraction |
| **Recommendation** | 4/5 - Advanced architecture; consider if PhysFormer is insufficient |

#### PhysMamba (Luo et al., 2024)

| Item | Detail |
|------|--------|
| **Paper** | "PhysMamba: Efficient Remote Physiological Measurement with SlowFast Temporal Difference Mamba" |
| **Year** | 2024 |
| **Architecture** | Replaces Transformer in PhysFormer with State Space Model (Mamba). Uses bidirectional Mamba layers for long-range temporal dependencies. SlowFast dual-stream with lateral connections. CDC_T for temporal difference in both streams. |
| **Key Innovations** | (1) Mamba SSM for O(n) temporal modeling (vs O(n^2) attention); (2) Bidirectional processing for non-causal signal recovery; (3) Channel attention + SSM hybrid |
| **Params** | ~3-5M |
| **Code** | [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) |
| **Contact PPG Suitability** | **HIGH** - Efficient long-range modeling; linear complexity |
| **Recommendation** | 4/5 - Great efficiency; requires mamba-ssm library |

**SlowFast Design**:
```
Slow stream: stride-4 temporal (captures trends)
  ConvBlock4 -> MambaLayer -> ChannelAttention3D -> ConvBlock6
Fast stream: stride-2 temporal (captures pulses)
  ConvBlock5 -> MambaLayer -> ChannelAttention3D -> ConvBlock7
LateralConnection: Fast -> Slow (conv + temporal resample)
Merge at two points with stride-2 fusion
```

#### FactorizePhys (Joshi et al., 2024)

| Item | Detail |
|------|--------|
| **Paper** | "Matrix Factorization for Multidimensional Attention in Remote Physiological Sensing" |
| **Year** | 2024 |
| **Architecture** | 3D CNN feature extractor (8->12->16 channels) + Non-negative Matrix Factorization (NMF) attention module (FSAM). Uses InstanceNorm3d and Tanh activations. Supports RGB, thermal, and multi-modal inputs. |
| **Key Innovation** | Uses NMF to decompose feature maps into physically meaningful components (blood volume, motion, etc.); attention masks from NMF basis vectors |
| **Params** | ~500K-1M |
| **Code** | [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) |
| **Contact PPG Suitability** | **HIGH** - NMF decomposition could separate PPG signal from noise physically |
| **Recommendation** | 4/5 - Novel approach; lightweight; interpretable attention |

#### ContrastPhys (Sun & Yu, 2022)

| Item | Detail |
|------|--------|
| **Paper** | "Contrast-Phys: Unsupervised Video-based Remote Physiological Measurement via Spatiotemporal Contrast" (ECCV 2022) |
| **Year** | 2022 |
| **Architecture** | 3D CNN backbone + contrastive learning framework. Learns rPPG without ground-truth labels by using spatial/temporal augmentations and frequency-based contrastive loss. |
| **Key Innovation** | Self-supervised pretraining; no need for PPG labels during pretraining |
| **Contact PPG Suitability** | MEDIUM - Could be used for pretraining on unlabeled finger videos |
| **Recommendation** | 3/5 - Useful for self-supervised pretraining stage |

---

### 2.3 PPG Representation Comparison

This section is **critical** for our project. The way PPG information is represented from video determines the information bottleneck.

#### 2.3.1 Representation: 1D Signal (Channel Mean)

```
Video frames --> Spatial average over ROI --> 1D signal (T,) or (T, C)
```

| Aspect | Detail |
|--------|--------|
| **Dimension** | (T, 1) for single channel; (T, 3) for RGB |
| **Information** | Only global brightness changes retained; ALL spatial information lost |
| **Methods using this** | GREEN, CHROM, POS, ICA, PBV; Our Scheme D (RGB mean), Scheme E (green channel) |
| **Pros** | Ultra-lightweight; fast; simple |
| **Cons** | **Massive information loss**; cannot capture spatial heterogeneity of perfusion; motion artifacts harder to separate |
| **Our assessment** | Insufficient for ECG morphology reconstruction. May capture heart rate but not P/QRS/T wave details. |

#### 2.3.2 Representation: 2D STMap (Spatio-Temporal Map)

```
Video frames --> Select N spatial locations (ROIs/patches)
             --> For each location, extract temporal signal
             --> Stack to form 2D image: (N_spatial, T_temporal)
```

**STMap Variants**:

1. **ROI-based STMap**: Divide face/finger into K regions, compute mean per region over time. Size: (K, T).
2. **Pixel-line STMap**: Sample pixels along a line (e.g., horizontal line across finger). Size: (N_pixels, T).
3. **Multi-scale STMap**: Compute STMaps at multiple spatial resolutions and stack as channels. Size: (C_scales, K, T).
4. **Frequency-STMap**: Apply STFT to each spatial location's temporal signal. Size: (N_spatial, N_freq, T_windows).

| Aspect | Detail |
|--------|--------|
| **Dimension** | (N_spatial, T) or (C, N_spatial, T) |
| **Information** | Preserves spatial distribution of pulsatile signals; captures spatial heterogeneity |
| **Methods using this** | Several rPPG papers (2019-2024); dual-GAN rPPG; HR-CNN variants |
| **Pros** | **Much more information than 1D**; can use 2D CNN/ViT for processing; captures regional perfusion differences |
| **Cons** | ROI selection is a design choice; not end-to-end; fixed spatial sampling may miss important regions |
| **Our assessment** | **Strong candidate for improvement**. For contact finger PPG, we can define spatial regions across the finger image and build STMaps. |

**How to build STMap for contact finger PPG**:
```python
# For each frame t:
#   1. Divide finger ROI into grid of K patches (e.g., 8x8 = 64 patches)
#   2. Compute mean R, G, B per patch
#   3. STMap[k, t] = mean_channel(patch_k, frame_t)
# Result: (K, T) for single channel, or (3, K, T) for RGB
```

#### 2.3.3 Representation: 3D Video Volume (End-to-End)

```
Video frames --> Direct 3D/4D tensor input to model --> (T, C, H, W)
```

| Aspect | Detail |
|--------|--------|
| **Dimension** | (T, 3, H, W) typically 300x3x64x64 |
| **Information** | **Maximum information preservation**; all spatial and temporal details retained |
| **Methods using this** | PhysNet, PhysFormer, EfficientPhys, PhysMamba; Our Schemes C, F, G |
| **Pros** | No information loss; model learns optimal features; end-to-end trainable |
| **Cons** | **High memory cost**; requires large model; risk of overfitting with small datasets |
| **Our assessment** | Theoretically best but requires sufficient data and compute. With only 98 samples / 1042 windows, overfitting is a real risk. |

#### 2.3.4 Representation: Feature-Level (Intermediate Neural Features)

```
Video --> Pretrained backbone (freeze or finetune) --> Feature map (T, D, h, w)
      --> Process features for ECG prediction
```

| Aspect | Detail |
|--------|--------|
| **Dimension** | (T, D, h, w) where D = feature channels, h,w = spatial resolution |
| **Information** | Semantically rich features; preserves spatial structure at reduced resolution |
| **Methods using this** | Transfer learning approaches; two-stage fine-tuning |
| **Pros** | Benefits from pretraining; reduced input dimension; semantic features |
| **Cons** | Pretrained model may not be sensitive to PPG-relevant features; feature alignment needed |
| **Our assessment** | Promising direction: pretrain on rPPG data (PhysNet/PhysFormer), then fine-tune for ECG. |

#### 2.3.5 Representation Comparison Summary

| Representation | Info Preserved | Memory | Data Efficiency | ECG Reconstruction Potential |
|---|---|---|---|---|
| 1D (channel mean) | ~1% | Minimal | High | Low (only HR, not morphology) |
| 2D (STMap) | ~20-40% | Low | Medium | Medium (captures spatial variation) |
| 3D (full video) | 100% | High | Low | High (if enough data) |
| Feature-level | ~60-80% (semantic) | Medium | Medium-High | High (with good pretraining) |

**Recommendation for our project**: Given 98 samples, a **hybrid approach** is advisable:
1. Use STMap or feature-level representation to balance information and data efficiency
2. OR use end-to-end (3D) with strong regularization, pretraining, and data augmentation

---

## 3. Direction 2: PPG to ECG Reconstruction

### 3.1 PPG2ECG: Attentional Encoder-Decoder (Tian et al., IEEE Sensors 2020)

| Item | Detail |
|------|--------|
| **Paper** | "Reconstructing QRS Complex from PPG by Transformed Attentional Neural Networks" |
| **Year** | 2020 |
| **Architecture** | Encoder-Decoder with Conv1d. Encoder: Conv1d(1->32->64->128->256->512) with stride-2, PReLU. Decoder: mirrored ConvTranspose1d, Tanh output. Optional STN for time-offset calibration. Optional multi-head attention for QRS focus. |
| **Input** | (B, 1, 256) - PPG at 125 Hz, 2 seconds |
| **Output** | (B, 1, 256) - ECG |
| **Loss** | QRS-enhanced L1 loss (higher weight on R-peak regions) |
| **Dataset** | BIDMC (PhysioNet) |
| **Reported Metrics** | Pearson r = 0.844 |
| **Code** | [james77777778/ppg2ecg-pytorch](https://github.com/james77777778/ppg2ecg-pytorch); **already in our project as `models/ppg2ecg.py`** |
| **Pros** | Simple; proven architecture; STN handles misalignment |
| **Cons** | Short window (2s); no adversarial training; simple loss function |
| **Recommendation** | 3/5 - Good baseline; already implemented |

### 3.2 CardioGAN (Sarkar & Etemad, AAAI 2021)

| Item | Detail |
|------|--------|
| **Paper** | "CardioGAN: Attentive Generative Adversarial Network with Dual Discriminators for Synthesis of ECG from PPG" |
| **Year** | 2021 |
| **Architecture** | Generator: Attention U-Net (encoder-decoder with self-gated attention on skip connections). Discriminator_T: Time-domain discriminator (raw signal). Discriminator_F: Frequency-domain discriminator (FFT). |
| **Input** | (B, 1, 512) - PPG at 128 Hz, 4 seconds |
| **Output** | (B, 1, 512) - ECG |
| **Loss** | Adversarial (WGAN-GP) + Reconstruction (L1) + Feature matching. Dual discriminator ensures both time and frequency fidelity. |
| **Datasets** | BIDMC, CAPNO, DALIA, WESAD |
| **Code** | [pritamqu/ppg2ecg-cardiogan](https://github.com/pritamqu/ppg2ecg-cardiogan); **already in our project as `models/cardiogan.py`** |
| **Pros** | Dual discriminator covers both domains; attention on skip connections preserves detail; GAN produces sharper waveforms |
| **Cons** | GAN training instability; more complex; longer training |
| **Recommendation** | 4/5 - Significant improvement over PPG2ECG; worth running on our data |

### 3.3 P2E-WGAN (Vo et al., ACM SAC 2021)

| Item | Detail |
|------|--------|
| **Paper** | "P2E-WGAN: ECG Waveform Synthesis from PPG with Conditional Wasserstein Generative Adversarial Networks" |
| **Year** | 2021 |
| **Architecture** | End-to-end 1D convolutional generator and discriminator. Uses Wasserstein distance with gradient penalty (WGAN-GP). ECG feature indices condition the generation. |
| **Input** | PPG waveform + ECG feature indices |
| **Output** | Synthetic ECG waveform |
| **Loss** | ECG feature-based WGAN-GP loss (targets ECG-specific characteristics like P/QRS/T features) |
| **Code** | [khuongav/P2E-WGAN-ecg-ppg-reconstruction](https://github.com/khuongav/P2E-WGAN-ecg-ppg-reconstruction) (33 stars) |
| **Pros** | Feature-conditioned generation; WGAN-GP is more stable than standard GAN |
| **Cons** | Requires ECG feature detection as preprocessing |
| **Recommendation** | 3/5 - Interesting feature-conditioning approach |

### 3.4 W-Net: Dual U-Net (2024)

| Item | Detail |
|------|--------|
| **Paper** | Project/thesis work |
| **Year** | 2024 |
| **Architecture** | Two cascaded U-Nets (W-shaped). First U-Net refines PPG; second reconstructs ECG from refined PPG. |
| **Loss** | Hybrid: MAE + MSE + (1 - Pearson correlation) |
| **Key Findings** | **Data quality > data quantity**: Artifact filtering achieved Pearson r=0.7363 (vs 0.7088 baseline). Best augmentation: CutMix (r=0.7184). |
| **Code** | [AlonBanai/W-Net-Architecture](https://github.com/AlonBanaiPortfolioSite/W-Net-Architecture-for-Complex-Data-Reconstruction) |
| **Pros** | Dual-stage refinement; hybrid loss is well-motivated |
| **Cons** | Marginal improvement over baseline |
| **Recommendation** | 3/5 - The finding that data quality matters more than augmentation is important for our project |

### 3.5 Transformer-based PPG2ECG (2023-2025)

Multiple recent works have applied Transformers to PPG-to-ECG:

| Approach | Key Idea | Advantage |
|----------|----------|-----------|
| Cross-attention PPG2ECG | PPG as query, learnable ECG template as key/value | Can capture non-local dependencies between PPG and ECG features |
| Temporal Transformer | Self-attention over PPG time series | Models long-range temporal patterns |
| Hybrid CNN-Transformer | CNN encoder + Transformer decoder | Local features (CNN) + global context (Transformer) |

**Recommendation**: 4/5 - Cross-attention between PPG and ECG is a promising direction.

### 3.6 Diffusion Model-based ECG Generation (2024-2026, Emerging)

| Approach | Key Idea | Status |
|----------|----------|--------|
| Conditional Diffusion for ECG | PPG as condition, generate ECG via reverse diffusion | Very early stage; primarily for ECG synthesis, not PPG-conditioned |
| Score-based SDE for signal translation | Translate between signal domains | Theoretical interest, not yet widely validated for PPG->ECG |
| Latent Diffusion ECG | Diffusion in latent space (lighter than pixel-space) | Promising for resource-constrained settings |

**Current status**: Diffusion models for ECG generation exist (e.g., synthetic ECG for data augmentation), but **PPG-conditioned ECG reconstruction via diffusion is still nascent (2025-2026)**. This is a forward-looking direction.

**Recommendation**: 2/5 for now - Monitor for 2025-2026 developments; not mature enough for immediate implementation.

### 3.7 PPG-to-ECG Method Comparison

| Method | Type | Input Len | Pearson r | Key Advantage | Complexity |
|--------|------|-----------|-----------|---------------|------------|
| PPG2ECG (Tian 2020) | Enc-Dec | 256 (2s) | 0.844 | Simple, STN | Low |
| CardioGAN (2021) | GAN | 512 (4s) | ~0.85+ | Dual discriminator | High |
| P2E-WGAN (2021) | WGAN | varies | ~0.80 | Feature conditioning | Medium |
| W-Net (2024) | Dual UNet | varies | 0.736 | Data quality insight | Medium |
| 1D UNet (Scheme E) | UNet | 300 (10s) | TBD | Skip connections | Low |
| Transformer PPG2ECG | Transformer | varies | TBD | Long-range attention | Medium |

---

## 4. Direction 3: Video to ECG End-to-End

### 4.1 Does Direct Video-to-ECG Exist?

**Short answer: No widely published paper does direct video-to-ECG reconstruction as of early 2026.**

The existing literature is split into:
1. **Video --> PPG** (rPPG field, extensively studied)
2. **PPG --> ECG** (signal translation field, growing)
3. **Video --> Heart Rate / HRV** (simplified version, common)

**Why the gap exists**:
- ECG reconstruction from video is an extremely ill-posed problem
- ECG morphology (P wave, QRS complex, T wave) carries information about cardiac electrical activity that is not directly observable in optical signals
- PPG captures mechanical/volumetric changes, while ECG captures electrical activity
- The mapping is inherently many-to-one: many ECG waveforms can produce similar PPG
- Most researchers take the two-stage approach for interpretability

### 4.2 Our Existing End-to-End Schemes

Our project already has three end-to-end video-to-ECG models:

| Scheme | Architecture | Strategy |
|--------|-------------|----------|
| **C (MTTS-CAN)** | 2D CNN with TSM + attention --> TemporalDecoder | Implicitly extracts PPG features, then decodes to ECG |
| **F (EfficientPhys)** | 3D temporal-spatial CNN --> Global pool --> Conv1d --> Interpolate | Factorized 3D processing, then temporal decoding |
| **G (PhysNet)** | 3D CNN --> Global pool --> Temporal decoder | Classic spatiotemporal encoding |

### 4.3 Proposed Improvements for End-to-End

#### 4.3.1 PhysFormer Backbone + ECG Decoder (Recommended)

```
Video (B, 3, T, H, W)
  --> 3D CNN Stem (spatial features)
  --> Temporal Difference Transformer (temporal modeling)
  --> Feature sequence (B, T, D)
  --> ECG Temporal Decoder (Conv1d upsampling)
  --> ECG waveform (B, T_ecg)
```

**Why**: PhysFormer's temporal difference mechanism is specifically designed to detect subtle brightness changes, which is exactly what PPG in video is. The Transformer captures long-range dependencies needed for ECG wave morphology.

#### 4.3.2 Multi-Task: PPG + ECG Simultaneous Prediction

```
Video --> Shared Encoder (3D CNN or PhysFormer)
       --> Branch A: PPG prediction (auxiliary task, per-frame)
       --> Branch B: ECG decoder (main task)
```

**Why**: Predicting PPG as an auxiliary task provides a strong learning signal and acts as regularization. The shared encoder learns better features when trained for both tasks.

**Loss**: `L = L_ecg + lambda * L_ppg`

#### 4.3.3 Two-Stage with Neural Feature Bridge

```
Stage 1: PhysFormer pretrained on rPPG datasets (PURE, UBFC, VIPL-HR)
         --> Fine-tune backbone on our finger video data
         --> Extract feature sequence (B, T, D, h, w) [NOT collapsed to 1D]

Stage 2: Feature-to-ECG decoder
         --> Processes (B, T, D, h, w) features
         --> U-Net or Transformer decoder
         --> ECG waveform (B, T_ecg)
```

**Why**: Avoids the 1D information bottleneck. The feature map preserves spatial distribution of perfusion that may carry morphological information.

#### 4.3.4 Video Swin Transformer + ECG Head

| Item | Detail |
|------|--------|
| **Base Model** | Video Swin Transformer (pretrained on video understanding) |
| **Adaptation** | Replace classification head with temporal decoder |
| **Input** | (B, 3, T, H, W) |
| **Output** | (B, T_ecg) |
| **Pros** | Powerful pretrained backbone; hierarchical features; efficient shifted window attention |
| **Cons** | Very large; pretraining is on action recognition (different domain) |
| **Recommendation** | 3/5 - Worth trying if compute allows |

### 4.4 Analysis: Why End-to-End May Work for Our Task

While no published paper does video-to-ECG directly, our task has unique advantages:

1. **Contact PPG**: Much stronger signal than rPPG (finger pressed on camera)
2. **Controlled conditions**: User is relatively still during recording
3. **High SNR**: Red-saturated video with clear pulsatile signal
4. **Same subjects**: Possible to learn subject-specific mappings (with user-level split caveat)

The main challenge is: **can spatial information in the video help reconstruct ECG morphology beyond what 1D PPG provides?**

Potentially yes, because:
- Different finger regions have different perfusion depths and timing
- Spatial heterogeneity of the pulse wave may correlate with ECG morphology
- Motion artifacts have spatial patterns that can be identified and separated

---

## 5. Direction 4: Multi-Modal Fusion

### 5.1 IMU Fusion

Our project already supports IMU fusion in all schemes. Current approach: IMUEncoder (1D CNN) encodes IMU to match video temporal resolution, then concatenated.

#### 5.1.1 IMU as Motion Artifact Removal

| Strategy | Description | Complexity |
|----------|------------|------------|
| **Direct subtraction** | Use accelerometer signal to estimate motion component, subtract from PPG | Low |
| **Adaptive filtering** | LMS/RLS filter with IMU reference to cancel motion noise | Medium |
| **Neural gating** | IMU features as gating signal to suppress motion-contaminated frames | Medium |
| **Cross-attention** | IMU attends to video features; high-motion frames get low attention weight | High |

#### 5.1.2 IMU as Additional Information Source

Accelerometer contains cardiac-related information:
- **Ballistocardiography (BCG)**: Body micro-movements caused by heartbeat
- **Seismocardiography (SCG)**: Chest vibrations from cardiac activity

For phone-held-against-finger:
- Finger arterial pulsations may create detectable accelerometer signals
- Gyroscope may capture subtle rotational movements from pulse wave

#### 5.1.3 Recommended IMU Fusion Strategies

| Priority | Strategy | Implementation |
|----------|----------|----------------|
| 1st | **Attention-gated fusion** | IMU motion magnitude gates video features; high motion = low weight |
| 2nd | **Cross-attention** | IMU and video features attend to each other via cross-attention |
| 3rd | **Multi-scale fusion** | Fuse IMU at multiple encoder levels (early + middle + late) |

### 5.2 Audio/Heart Sound (PCG) Fusion

**⚠️ UPDATE (2026-02-08)**: 经验证，全部 98 个视频均包含 AAC 音轨，部分样本可听到明显心跳声。PCG 融合可行性从 2/5 上调至 **3.5/5**。

Phone microphones can capture heart sounds (S1, S2). The relationship between PCG and ECG:
- S1 (first heart sound) corresponds to mitral/tricuspid valve closure, near QRS complex
- S2 (second heart sound) corresponds to aortic/pulmonic valve closure, near T wave end
- 手指贴住摄像头时，手指本身可能充当心音传导介质

| Strategy | Description |
|----------|------------|
| **Audio preprocessing** | Bandpass 20-200 Hz, denoise, extract envelope |
| **Feature extraction** | Mel spectrogram or MFCC as features |
| **Heart sound segmentation** | S1/S2 detection → event timing features |
| **Fusion** | Late fusion (concat with video features) or cross-attention |

**已确认数据可用性**: 全部 98 个视频含 AAC 音轨，用户确认部分文件有明显心跳声。

**Challenge**: 音频质量因样本而异；需评估有多少样本的心音清晰可用。

**Recommendation**: 3.5/5 - 数据已确认可用。在视频模型验证后，应作为第二优先的融合模态。

### 5.3 Fusion Architecture Strategies

#### 5.3.1 Early Fusion
```
Concatenate raw inputs at input level:
  [video_features, imu_features, audio_features] --> Shared encoder --> ECG
```
- Simple but may overwhelm with irrelevant information
- Works when modalities have similar temporal resolution

#### 5.3.2 Middle Fusion (Recommended)
```
Video --> Video Encoder --> Video Features (B, T, D_v)
IMU   --> IMU Encoder   --> IMU Features (B, T, D_i)
                               |
                         Cross-Attention or Concat
                               |
                         --> Fusion Features (B, T, D_f)
                         --> ECG Decoder --> ECG
```
- Each modality processed by specialized encoder
- Fusion at feature level allows learned alignment
- **Cross-attention fusion** is most flexible

#### 5.3.3 Late Fusion
```
Video --> Video Model --> ECG_v prediction
IMU   --> IMU Model   --> ECG_i prediction
                           |
                      Weighted average or learned combination
                           |
                      --> Final ECG prediction
```
- Most modular; each modality can be developed independently
- May miss cross-modal interactions

#### 5.3.4 Hierarchical Fusion (Most Comprehensive)
```
Video --> Encoder_v [L1_v, L2_v, L3_v, L4_v]
IMU   --> Encoder_i [L1_i, L2_i, L3_i, L4_i]
                |       |       |       |
            Fuse_L1  Fuse_L2  Fuse_L3  Fuse_L4  (at each encoder level)
                                        |
                                  --> ECG Decoder
```
- Fusion at every level captures both low-level and high-level interactions
- Most complex but potentially most powerful

---

## 6. Direction 5: Loss Functions and Training Tricks

### 6.1 Loss Functions

#### 6.1.1 Time-Domain Losses

| Loss | Formula | Use Case |
|------|---------|----------|
| **MSE** | `mean((pred - target)^2)` | General waveform matching; sensitive to amplitude |
| **MAE / L1** | `mean(|pred - target|)` | Less sensitive to outliers; produces blurrier outputs |
| **Smooth L1 / Huber** | Quadratic for small errors, linear for large | Best of MSE and MAE |
| **Neg Pearson** | `1 - pearson_r(pred, target)` | **Crucial**: focuses on shape correlation, ignores scale/offset |

#### 6.1.2 Frequency-Domain Losses

| Loss | Formula | Use Case |
|------|---------|----------|
| **Spectral Magnitude** | `L1(|FFT(pred)|, |FFT(target)|)` | Preserves frequency content |
| **Cross-Entropy Power Spectrum** | Softmax on power spectrum, cross-entropy | PhysFormer's approach; emphasizes dominant frequency (HR) |
| **STFT Loss** | Multi-scale STFT magnitude + phase | Captures both time and frequency; used in audio synthesis |
| **Spectral Convergence** | `||S_pred - S_target||_F / ||S_target||_F` | Normalized spectral error |

#### 6.1.3 Physiological Constraint Losses

| Loss | Formula | Use Case |
|------|---------|----------|
| **QRS-Enhanced L1** | `L1 with higher weight on R-peak regions` | PPG2ECG paper; focuses on most important feature |
| **HR Consistency** | `|HR_pred - HR_target|` where HR from peak detection | Ensures physiological plausibility |
| **HRV Loss** | `MSE(IBI_pred, IBI_target)` on inter-beat intervals | Preserves beat-to-beat variability |
| **Morphology Loss** | `DTW(pred_beat, target_beat)` per-beat DTW alignment | Ensures individual beat morphology matches |

#### 6.1.4 GAN Losses

| Loss | Formula | Use Case |
|------|---------|----------|
| **WGAN-GP** | Wasserstein distance + gradient penalty | Stable GAN training; CardioGAN uses this |
| **LSGAN** | Least-squares GAN loss | More stable than original GAN |
| **Dual Discriminator** | Time-domain D + Frequency-domain D | CardioGAN: ensures both waveform and spectral fidelity |

#### 6.1.5 Recommended Loss Combination

**For our project (Composite Loss v2)**:
```python
L_total = w1 * L_mse           # Waveform amplitude matching
        + w2 * L_neg_pearson    # Shape correlation (CRITICAL)
        + w3 * L_spectral       # Frequency content preservation
        + w4 * L_stft           # Multi-scale time-frequency (NEW)
        + w5 * L_qrs_enhanced   # R-peak focus (NEW)
```

**Suggested weights**: `w1=1.0, w2=0.5, w3=0.1, w4=0.1, w5=0.2`

**PhysFormer training strategy**: Dynamic loss weighting by epoch:
- Epochs 0-10: `a=1.0 * L_pearson + b=1.0 * (L_freq + L_kl)`
- Epochs 10+: `a=0.05 * L_pearson + b=5.0 * (L_freq + L_kl)`
- Rationale: Start with shape matching, then refine frequency content

### 6.2 Data Augmentation

#### 6.2.1 Signal-Level Augmentation

| Method | Description | Effect |
|--------|------------|--------|
| **Gaussian noise** | Add N(0, sigma) noise to PPG/video | Robustness to sensor noise |
| **Temporal scaling** | Speed up/slow down by factor k | Robustness to HR variation |
| **Amplitude scaling** | Scale signal by random factor | Robustness to signal strength variation |
| **Baseline wander** | Add low-frequency sinusoidal drift | Robustness to DC drift |
| **Random cropping** | Crop window from longer segment | Position invariance |
| **CutMix** | Exchange segments between two samples | Regularization (shown effective by W-Net) |
| **MixUp** | Blend two samples linearly | Regularization (may cause instability per W-Net) |
| **Jittering** | Small random perturbations | Simulates physiological variation |

#### 6.2.2 Video-Level Augmentation

| Method | Description | Effect |
|--------|------------|--------|
| **Random brightness** | Scale pixel values | Robustness to illumination |
| **Random contrast** | Adjust contrast | Robustness to camera settings |
| **Gaussian blur** | Apply random blur | Robustness to focus variation |
| **Spatial crop/resize** | Random spatial crop | Position invariance |
| **Temporal jitter** | Shift/drop random frames | Robustness to frame timing |
| **Channel dropout** | Zero out random color channel | Robustness to channel failure |
| **Color jitter** | Random hue/saturation/value changes | Robustness to color variation |

**Caution for contact PPG**: Aggressive color/brightness augmentation may destroy the PPG signal. Use conservative ranges.

#### 6.2.3 ECG-Level Augmentation (Target Side)

| Method | Description |
|--------|------------|
| **Beat morphology variation** | Slightly warp individual beats |
| **Noise injection to labels** | Small noise as regularization |
| **Baseline wander in target** | Add/remove baseline wander |

### 6.3 Training Tricks

#### 6.3.1 Curriculum Learning

```
Phase 1 (epochs 0-20): Predict smoothed ECG (low-pass filtered)
Phase 2 (epochs 20-50): Predict full-bandwidth ECG
Phase 3 (epochs 50+): Add frequency-domain loss
```
**Rationale**: Start with easier task (overall shape) before fine details (QRS morphology).

#### 6.3.2 Self-Supervised Pretraining

```
Step 1: Pretrain encoder on unlabeled finger videos using:
  - Contrastive learning (ContrastPhys-style)
  - Masked autoencoder (reconstruct masked video patches)
  - Temporal prediction (predict next frame features)

Step 2: Fine-tune on labeled (video, ECG) pairs
```

#### 6.3.3 Transfer Learning from rPPG Models

```
Step 1: Take pretrained PhysNet/PhysFormer (trained on PURE/UBFC for rPPG)
Step 2: Replace output head with ECG temporal decoder
Step 3: Fine-tune on our contact finger video data
```

**This is a highly recommended strategy** because:
- rPPG pretraining teaches the model to detect subtle brightness changes
- Our contact PPG has much higher SNR than rPPG, so fine-tuning should be easy
- Addresses the small dataset problem

#### 6.3.4 Gradient Accumulation and AMP

Already implemented in our project. Key settings:
- 3090: batch=8, grad_accum=2, AMP=True
- A6000: batch=16-32, grad_accum=1, AMP=True

#### 6.3.5 Learning Rate Scheduling

| Strategy | Description | When to use |
|----------|------------|-------------|
| **CosineAnnealingLR** | Cosine decay | Standard choice; already in project |
| **OneCycleLR** | Warmup + cosine decay | Good for short training |
| **ReduceLROnPlateau** | Reduce on val loss plateau | Conservative approach |
| **Warmup + constant + decay** | Linear warmup for transformers | PhysFormer-style |

#### 6.3.6 Regularization

| Technique | Description |
|-----------|------------|
| **Dropout** | Standard; 0.1-0.25 for encoder, 0.5 for classifier |
| **DropPath** | RhythmFormer uses this; drops entire residual branches |
| **Weight decay** | 1e-4 typical |
| **Label smoothing** | Slightly perturb target ECG values |
| **Early stopping** | Already implemented with patience 20-30 |

---

## 7. Direction 6: Pretrained Backbone Selection

### 7.1 Why ResNet is NOT Suitable

ResNet and similar image classification backbones are **not suitable** for PPG/rPPG extraction because:

1. **Global Average Pooling (GAP)**: ResNet's final GAP layer averages all spatial features into a single vector. This destroys spatial distribution information that may carry PPG/ECG-relevant details.

2. **Designed for semantic features**: ResNet learns to detect objects, textures, edges -- high-level semantic features. PPG requires detecting **sub-pixel brightness changes** across frames, which is fundamentally different.

3. **No temporal modeling**: ResNet processes individual frames independently; no mechanism for temporal dynamics.

4. **Deep features lose local contrast**: As features pass through many layers with batch normalization, subtle local brightness variations are normalized away.

5. **ImageNet pretraining bias**: Features optimized for 1000-class object recognition are not transferable to detecting 0.1% brightness changes.

### 7.2 Suitable Backbones

#### 7.2.1 rPPG-Specific Pretrained Models (Best Choice)

| Model | Type | Pretrained On | Why Suitable |
|-------|------|--------------|--------------|
| **PhysNet** | 3D CNN | VIPL-HR, PURE, UBFC | Designed for brightness change detection; 3D temporal modeling |
| **PhysFormer** | 3D CNN + Transformer | VIPL-HR | CDC captures temporal differences; best temporal modeling |
| **EfficientPhys** | 2D CNN + TSM | Multiple rPPG datasets | Lightweight; TSM for temporal; attention for ROI |
| **FactorizePhys** | 3D CNN + NMF | Multiple | Interpretable; NMF separates signal from noise |

**Recommendation**: Use PhysFormer or PhysNet pretrained checkpoints as backbone, replace output head with ECG decoder.

#### 7.2.2 Video Understanding Models (Possible)

| Model | Type | Why Consider | Why Risky |
|-------|------|-------------|-----------|
| **Video Swin Transformer** | Hierarchical Transformer | Strong spatiotemporal features | Pretrained on action recognition (different domain) |
| **TimeSformer** | Divided attention | Efficient temporal attention | Same domain gap |
| **SlowFast Networks** | Dual-rate CNN | Multi-temporal-scale | Large; may overfit |
| **MViT (Multiscale ViT)** | Pooling attention | Efficient; multi-scale | Action recognition bias |

**Recommendation**: 3/5 - Worth trying if rPPG backbones fail, but expect domain gap.

#### 7.2.3 Architecture Properties for PPG Sensitivity

The ideal backbone for PPG extraction should have:

| Property | Why | Models That Have It |
|----------|-----|---------------------|
| **Temporal difference / CDC** | Amplifies frame-to-frame changes | PhysFormer, PhysMamba, RhythmFormer |
| **Spatial attention** | Focuses on pulsating regions | DeepPhys, EfficientPhys, FactorizePhys |
| **No global average pooling** | Preserves spatial distribution | All rPPG models (use temporal pooling only) |
| **Batch/Instance normalization** | Normalizes for brightness variations | All (but InstanceNorm preserves within-sample variation better) |
| **Skip connections** | Preserves fine-grained temporal details | UNet-based, PhysNet |
| **Multi-scale temporal** | Captures both HR (fast) and HRV (slow) | PhysFormer++, PhysMamba (SlowFast) |

#### 7.2.4 InstanceNorm vs BatchNorm for PPG

**Important design choice**: FactorizePhys uses **InstanceNorm3d** instead of BatchNorm3d.

- **BatchNorm**: Normalizes across the batch. If different samples have very different brightness (different subjects, lighting), BN helps generalization but may smooth out within-sample PPG variations.
- **InstanceNorm**: Normalizes each sample independently. Preserves the relative brightness changes within each video sequence, which IS the PPG signal.

**Recommendation for contact PPG**: Consider InstanceNorm in early layers (preserves PPG), BatchNorm in later layers (helps generalization).

---

## 8. Recommended Implementation Plan

### Batch 1: Quick Wins (1-2 weeks)

These require minimal code changes and can validate feasibility:

| Priority | Task | Expected Impact | Effort |
|----------|------|-----------------|--------|
| **1a** | Train existing Scheme E (1D UNet) with quality filter | Baseline PPG->ECG performance | Low (just run training) |
| **1b** | Train existing Scheme F (EfficientPhys) end-to-end | End-to-end baseline | Low |
| **1c** | Run PPG2ECG baseline on BIDMC dataset | Validate PPG->ECG is feasible | Low (code already exists) |
| **1d** | Train CardioGAN on BIDMC | Better PPG->ECG baseline | Low (code already exists) |
| **1e** | Add **red channel** as alternative to green channel in Scheme E | Contact PPG uses red light; may be stronger signal | Low |
| **1f** | Improve CompositeLoss with QRS-enhanced weighting | Better focus on R-peak reconstruction | Low |

### Batch 2: Architecture Upgrades (2-4 weeks)

| Priority | Task | Expected Impact | Effort |
|----------|------|-----------------|--------|
| **2a** | Implement **PhysFormer backbone** (from rPPG-Toolbox) + ECG decoder | Best temporal modeling; CDC captures subtle changes | Medium |
| **2b** | Add **multi-task loss**: PPG + ECG prediction | Regularization; better feature learning | Medium |
| **2c** | Implement **STMap representation** as alternative input | Preserves spatial info without full 3D cost | Medium |
| **2d** | Add **cross-attention IMU fusion** (replace simple concat) | Better motion artifact handling | Medium |
| **2e** | Implement **InstanceNorm** option for early layers | Better PPG signal preservation | Low |
| **2f** | Add **STFT loss** and **dynamic loss weighting** (PhysFormer-style) | Better frequency content preservation | Medium |
| **2g** | Implement **curriculum learning** (smooth ECG first, then full) | Easier optimization trajectory | Low |

### Batch 3: Advanced Methods (4-8 weeks)

| Priority | Task | Expected Impact | Effort |
|----------|------|-----------------|--------|
| **3a** | Transfer learning: pretrain PhysFormer on rPPG datasets, fine-tune on our data | Address small dataset problem | High |
| **3b** | Implement **PhysMamba** (SSM backbone) | O(n) temporal modeling; efficient | High (requires mamba-ssm) |
| **3c** | Implement **RhythmFormer** | Hierarchical temporal modeling | High |
| **3d** | Two-stage with **feature bridge** (no 1D bottleneck) | Preserve spatial information | High |
| **3e** | Implement **FactorizePhys with NMF attention** | Interpretable; physically meaningful decomposition | Medium |
| **3f** | **CardioGAN-style training** for end-to-end model | GAN sharpens ECG waveforms | High |
| **3g** | Self-supervised pretraining on unlabeled finger videos | More training signal | High |

### Future Directions (Research)

| Direction | Potential | Readiness |
|-----------|-----------|-----------|
| **Diffusion-based ECG generation** | High (sharp, diverse outputs) | Low (nascent field) |
| **Neural ODE for PPG->ECG dynamics** | Medium (models continuous dynamics) | Medium |
| **Graph Neural Networks for ECG** | Medium (models inter-lead relationships) | Low (single lead) |
| **Foundation model fine-tuning** | High (if video/biosignal foundation models emerge) | Low (not yet available) |

---

## 9. Key Design Decisions

### Decision 1: 1D PPG vs 2D STMap vs 3D End-to-End

| Factor | 1D PPG | 2D STMap | 3D End-to-End |
|--------|--------|---------|---------------|
| Information | Minimal | Moderate | Maximum |
| Memory | Very low | Low | Very high |
| Data efficiency | High | Medium | Low |
| Our data (98 samples) | **Safe choice** | **Good balance** | **Risky (overfit)** |
| ECG morphology potential | Low | Medium | High |

**Recommendation**: Start with **1D (Scheme E)** to validate feasibility, then move to **STMap** or **3D with pretraining**.

### Decision 2: Two-Stage vs End-to-End

| Factor | Two-Stage (PPG-->ECG) | End-to-End (Video-->ECG) |
|--------|----------------------|-------------------------|
| Interpretability | High (can inspect intermediate PPG) | Low |
| Error propagation | PPG errors cascade | Direct optimization |
| Training data needed | Less (can train stages separately) | More (needs paired video+ECG) |
| Flexibility | Can swap PPG method | Monolithic |
| Performance ceiling | Limited by PPG quality | Higher (preserves all info) |

**Recommendation**: **Two-stage with feature bridge** -- extract features (not 1D PPG) from video, then decode to ECG. This preserves information while being trainable in stages.

### Decision 3: Which Channel for Contact PPG

Our finger videos are red-dominant (LED transmissive mode). Channel analysis:

| Channel | Signal Strength | Physiological Basis |
|---------|----------------|---------------------|
| **Red** | **Strongest** (highest intensity in transmissive mode) | Deep tissue penetration; strong pulsatile component |
| **Green** | Moderate | Most absorbed by hemoglobin (classic PPG literature) |
| **Blue** | Weakest | Shortest penetration depth; mostly surface |
| **All RGB** | Combined | Let model learn optimal combination |

**Recommendation**: Use **all RGB channels** as input (let model learn), but also test **red channel** as a separate Scheme E variant. Green channel (current Scheme E) may not be optimal for transmissive contact PPG.

### Decision 4: Data Split Strategy

With 98 samples:
- **Random split**: Easier; validates model works at all
- **User split**: Harder but necessary; tests generalization to new subjects

**Recommendation**: Use random split for debugging (Phase 1), user split for final evaluation (Phase 2). This matches the strategy in MEMORY.md.

### Decision 5: Addressing Small Dataset

98 samples (1042 windows) is very small for deep learning. Mitigation strategies:

| Strategy | Priority | Description |
|----------|----------|-------------|
| **Quality filtering** | HIGH | Remove poor samples (already implemented) |
| **Transfer learning** | HIGH | Pretrain on rPPG or PPG-ECG public datasets |
| **Data augmentation** | HIGH | Temporal, amplitude, noise augmentation |
| **Smaller models** | HIGH | Scheme E (500K) over Scheme G (3-5M) |
| **Strong regularization** | HIGH | Dropout, weight decay, early stopping |
| **Self-supervised pretraining** | MEDIUM | Use unlabeled finger videos |
| **Collect more data** | MEDIUM | If possible, more recordings |

---

## 10. References and Open-Source Code

### 10.1 Video-to-PPG (rPPG) Papers

| # | Paper | Year | Venue | Code |
|---|-------|------|-------|------|
| 1 | **GREEN**: "Remote plethysmographic imaging using ambient light" (Verkruysse et al.) | 2008 | Optics Express | pyVHR, rPPG-Toolbox |
| 2 | **ICA**: "Advancements in noncontact multiparameter physiological measurements using a webcam" (Poh et al.) | 2011 | IEEE TBME | pyVHR, rPPG-Toolbox |
| 3 | **CHROM**: "Robust pulse rate from chrominance-based rPPG" (de Haan & Jeanne) | 2013 | IEEE TBME | pyVHR, rPPG-Toolbox |
| 4 | **PBV**: "Improved motion robustness of remote-PPG by using the blood volume pulse signature" (de Haan & van Leest) | 2014 | Physiol. Meas. | rPPG-Toolbox |
| 5 | **POS**: "Algorithmic principles of remote PPG" (Wang et al.) | 2017 | IEEE TBME | pyVHR, rPPG-Toolbox |
| 6 | **DeepPhys**: "DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks" (Chen & McDuff) | 2018 | AAAI | [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) |
| 7 | **LGI**: "Local Group Invariance for heart rate estimation from face videos in the wild" (Pilz et al.) | 2018 | CVPR-W | rPPG-Toolbox |
| 8 | **PhysNet**: "Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks" (Yu et al.) | 2019 | BMVC | [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) |
| 9 | **TS-CAN / MTTS-CAN**: "Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement" (Liu et al.) | 2020 | NeurIPS | [MTTS-CAN](https://github.com/xliucs/MTTS-CAN), rPPG-Toolbox |
| 10 | **ContrastPhys**: "Contrast-Phys: Unsupervised Video-based Remote Physiological Measurement via Spatiotemporal Contrast" (Sun & Yu) | 2022 | ECCV | [ContrastPhys](https://github.com/zhaodongsun/contrast-phys) |
| 11 | **PhysFormer**: "PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer" (Yu et al.) | 2022 | CVPR | [PhysFormer](https://github.com/ZitongYu/PhysFormer), rPPG-Toolbox |
| 12 | **EfficientPhys**: "EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement" (Liu et al.) | 2023 | WACV | [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) |
| 13 | **BigSmall**: "Efficient Multi-Task Learning for Disparate Spatial and Temporal Physiological Measurements" (Narayanswamy et al.) | 2023 | NeurIPS | rPPG-Toolbox |
| 14 | **PhysFormer++**: "PhysFormer++: Facial Video-based Physiological Measurement with SlowFast Temporal Difference Transformer" (Yu et al.) | 2023 | IJCV | [PhysFormer](https://github.com/ZitongYu/PhysFormer) |
| 15 | **OMIT / Face2PPG**: "Face2PPG: An Unsupervised Pipeline for Blood Volume Pulse Extraction From Faces" (Alvarez et al.) | 2023 | CVPR-W | rPPG-Toolbox |
| 16 | **FactorizePhys**: "Matrix Factorization for Multidimensional Attention in Remote Physiological Sensing" (Joshi et al.) | 2024 | NeurIPS | rPPG-Toolbox |
| 17 | **iBVPNet**: 3D CNN from iBVP dataset (Joshi et al.) | 2024 | - | rPPG-Toolbox |
| 18 | **PhysMamba**: "Efficient Remote Physiological Measurement with SlowFast Temporal Difference Mamba" (Luo et al.) | 2024 | - | rPPG-Toolbox |
| 19 | **RhythmFormer**: "Extracting rPPG Signals Based on Hierarchical Temporal Periodic Transformer" (Zou et al.) | 2024 | AAAI | rPPG-Toolbox |

### 10.2 PPG-to-ECG Papers

| # | Paper | Year | Venue | Code |
|---|-------|------|-------|------|
| 1 | **PPG2ECG**: "Reconstructing QRS Complex from PPG by Transformed Attentional Neural Networks" (Tian et al.) | 2020 | IEEE Sensors | [ppg2ecg-pytorch](https://github.com/james77777778/ppg2ecg-pytorch) |
| 2 | **CardioGAN**: "Attentive Generative Adversarial Network with Dual Discriminators for Synthesis of ECG from PPG" (Sarkar & Etemad) | 2021 | AAAI | [ppg2ecg-cardiogan](https://github.com/pritamqu/ppg2ecg-cardiogan) |
| 3 | **P2E-WGAN**: "ECG Waveform Synthesis from PPG with Conditional WGAN" (Vo et al.) | 2021 | ACM SAC | [P2E-WGAN](https://github.com/khuongav/P2E-WGAN-ecg-ppg-reconstruction) |
| 4 | **W-Net PPG2ECG**: Dual U-Net with hybrid loss | 2024 | Project | [W-Net](https://github.com/AlonBanaiPortfolioSite/W-Net-Architecture-for-Complex-Data-Reconstruction) |

### 10.3 Key Toolboxes and Frameworks

| Name | URL | Models | Stars |
|------|-----|--------|-------|
| **rPPG-Toolbox** | https://github.com/ubicomplab/rPPG-Toolbox | DeepPhys, PhysNet, TS-CAN, EfficientPhys, PhysFormer, BigSmall, PhysMamba, RhythmFormer, FactorizePhys, iBVPNet | 700+ |
| **pyVHR** | https://github.com/phuselab/pyVHR | GREEN, CHROM, ICA, PCA, POS, SSR, LGI, PBV, OMIT, MTTS-CAN | 300+ |

### 10.4 Public Datasets for Pretraining/Validation

| Dataset | Type | Size | Signals | Use |
|---------|------|------|---------|-----|
| **BIDMC** | Contact PPG + ECG | 53 subjects | PPG, ECG, SpO2, RR | PPG->ECG validation |
| **CAPNO** | Contact PPG + ECG | ~40 subjects | PPG, ECG, CO2 | PPG->ECG validation |
| **DALIA (PPG-DaLiA)** | Wrist PPG + ECG | 15 subjects | PPG, ECG, ACC | PPG->ECG with motion |
| **WESAD** | Wrist PPG + ECG | 15 subjects | PPG, ECG, EDA, ACC | Stress detection |
| **PURE** | Face video + PPG | 10 subjects | Video, PPG | rPPG pretraining |
| **UBFC-rPPG** | Face video + PPG | 42 subjects | Video, PPG | rPPG pretraining |
| **VIPL-HR** | Face video + PPG | 107 subjects | Video, PPG | rPPG pretraining (large) |
| **iBVP** | Finger video + BVP | - | Video, BVP | **Directly relevant** |

**Note**: The **iBVP dataset** (finger video + blood volume pulse) is **directly relevant** to our task. It provides finger contact video with ground truth BVP, which is very similar to our data modality. The iBVPNet model from rPPG-Toolbox is trained on this dataset and could be a strong starting point for transfer learning.

---

## Appendix A: Architecture Comparison at a Glance

| Model | Type | Input | Temporal | Spatial Attention | Params | Memory | Contact PPG Score |
|-------|------|-------|----------|-------------------|--------|--------|-------------------|
| DeepPhys | 2D CNN | Diff+Raw 36x36 | None (per-frame) | Appearance-gated | 700K | 2GB | 3/5 |
| MTTS-CAN | 2D CNN+TSM | Diff+Raw 36x36 | TSM (shift) | Appearance-gated | 2.8M | 15GB | 4/5 |
| EfficientPhys | 2D CNN+TSM | Raw 36-96 | TSM (shift) | Learned mask | 1.5M | 10GB | 4/5 |
| PhysNet | 3D CNN | DiffNorm 128x128 | 3D Conv | None (global pool) | 3-5M | 25GB | 4/5 |
| PhysFormer | 3D+Trans | Raw 128x128 | **TD-Trans (CDC)** | Self-attention | 5-8M | 30GB | **5/5** |
| PhysFormer++ | 3D+Trans | Raw 128x128 | **SlowFast TD** | Self-attention | 5-8M | 30GB | **5/5** |
| RhythmFormer | 3D+Trans | RGB+Diff fused | **Hierarchical TPT** | BiFormer regional | 5-10M | 25GB | 4/5 |
| PhysMamba | 3D+SSM | Raw | **SlowFast Mamba** | Channel attention | 3-5M | 20GB | 4/5 |
| FactorizePhys | 3D CNN+NMF | Raw/Thermal | 3D Conv | **NMF attention** | 500K-1M | 5GB | 4/5 |
| BigSmall | Dual 2D | Hi+Lo res | WTSM | Multi-task | 2-3M | 10GB | 2/5 |

## Appendix B: Quick Reference -- Our Existing Schemes vs Recommended Upgrades

| Current | Architecture | Limitation | Recommended Upgrade | Key Change |
|---------|-------------|------------|---------------------|------------|
| Scheme C | MTTS-CAN | No 3D conv; TSM limited temporal range | **PhysFormer backbone** | Replace TSM with TD-Transformer |
| Scheme D | 1D TCN | Only 1D signal; no spatial info | **STMap + 2D CNN** | Use STMap representation |
| Scheme E | 1D UNet | Only 1D signal; green channel only | **Red+Green+Blue channels; larger UNet** | Multi-channel + architecture scale |
| Scheme F | EfficientPhys | Basic 3D blocks | **PhysFormer-style TD blocks** | CDC temporal convolutions |
| Scheme G | PhysNet | Basic 3D CNN | **PhysFormer or PhysMamba** | Transformer/SSM temporal modeling |

---

*This report was compiled on 2026-02-07 based on literature survey covering 2008-2026.*
*Key recommendation: Start with existing Scheme E/F as baselines, then implement PhysFormer backbone (Batch 2a) as the primary upgrade path.*
