"""
Video-to-ECG Reconstruction Model.

Scheme A: ResNet-18 (64x64) + 1D Temporal CNN, video only
Scheme B: ResNet-50 (224x224) + IMU branch + 1D Temporal CNN, composite loss
Scheme C: MTTS-CAN-inspired dual-branch attention + TSM, lightweight (~1M params)
Scheme D: 1D Signal-Centric TCN, ultra-lightweight (~100K params)

Controlled entirely via config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FrameEncoder(nn.Module):
    """CNN backbone applied per-frame, outputs (B, T, D)."""

    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        if backbone == "resnet18":
            net = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            self.out_dim = 512
        elif backbone == "resnet50":
            net = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            self.out_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # -> (B, D, 1, 1)

    def forward(self, x):
        """x: (B, T, C, H, W) -> (B, T, D)"""
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feat = self.backbone(x).squeeze(-1).squeeze(-1)  # (B*T, D)
        return feat.reshape(B, T, self.out_dim)


class IMUEncoder(nn.Module):
    """1D CNN encoder for IMU data: (B, T_imu, 6) -> (B, T_v, D_imu)."""

    def __init__(self, in_channels=6, out_dim=64, target_len=300):
        super().__init__()
        self.target_len = target_len
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """x: (B, T_imu, 6) -> (B, T_v, D_imu)"""
        x = x.permute(0, 2, 1)  # (B, 6, T_imu)
        x = self.net(x)  # (B, D_imu, T_imu)
        # Resample to match video frame count
        x = F.interpolate(x, size=self.target_len, mode="linear", align_corners=False)
        return x.permute(0, 2, 1)  # (B, T_v, D_imu)


class TemporalDecoder(nn.Module):
    """
    1D CNN decoder: (B, T_v, D) -> (B, T_ecg).
    Upsamples via transposed convolutions, then interpolates to exact length.
    """

    def __init__(self, in_dim=512, channels=(256, 128, 64), upsample_factor=8,
                 target_ecg_len=2500):
        super().__init__()
        self.target_ecg_len = target_ecg_len

        n_upsample_stages = 0
        factor = upsample_factor
        while factor > 1:
            factor //= 2
            n_upsample_stages += 1

        layers = []
        prev_ch = in_dim
        for i, ch in enumerate(channels):
            if i < n_upsample_stages:
                layers.append(nn.ConvTranspose1d(prev_ch, ch, kernel_size=4,
                                                 stride=2, padding=1))
            else:
                layers.append(nn.Conv1d(prev_ch, ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout1d(0.1))
            prev_ch = ch

        self.temporal_net = nn.Sequential(*layers)
        self.head = nn.Conv1d(prev_ch, 1, kernel_size=3, padding=1)

    def forward(self, x):
        """x: (B, T_v, D) -> (B, T_ecg)"""
        x = x.permute(0, 2, 1)  # (B, D, T_v)
        x = self.temporal_net(x)
        x = self.head(x)  # (B, 1, T_up)
        if x.shape[-1] != self.target_ecg_len:
            x = F.interpolate(x, size=self.target_ecg_len, mode="linear",
                              align_corners=False)
        return x.squeeze(1)


class VideoECGModel(nn.Module):
    """
    End-to-end Video (+ optional IMU) -> ECG model.
    Scheme A: video only.  Scheme B: video + IMU fusion.
    """

    def __init__(self, backbone="resnet18", pretrained_encoder=True,
                 temporal_channels=(256, 128, 64), upsample_factor=8,
                 window_sec=10, video_fps=30, ecg_sr=250,
                 use_imu=False, imu_dim=64, imu_sr=100):
        super().__init__()
        self.use_imu = use_imu

        self.encoder = FrameEncoder(backbone=backbone, pretrained=pretrained_encoder)
        encoder_dim = self.encoder.out_dim

        decoder_in_dim = encoder_dim
        if use_imu:
            video_frames = int(window_sec * video_fps)
            self.imu_encoder = IMUEncoder(in_channels=6, out_dim=imu_dim,
                                          target_len=video_frames)
            decoder_in_dim = encoder_dim + imu_dim

        target_ecg_len = int(window_sec * ecg_sr)
        self.decoder = TemporalDecoder(
            in_dim=decoder_in_dim,
            channels=temporal_channels,
            upsample_factor=upsample_factor,
            target_ecg_len=target_ecg_len,
        )

    def forward(self, video, imu=None):
        """
        video: (B, T_v, C, H, W)
        imu:   (B, T_imu, 6) or None
        returns: (B, T_ecg)
        """
        feat = self.encoder(video)  # (B, T_v, D_enc)

        if self.use_imu and imu is not None:
            imu_feat = self.imu_encoder(imu)  # (B, T_v, D_imu)
            feat = torch.cat([feat, imu_feat], dim=-1)  # (B, T_v, D_enc+D_imu)

        return self.decoder(feat)


# ──────────────────────────────────────────────
#  Scheme C: MTTS-CAN-inspired dual-branch + TSM
# ──────────────────────────────────────────────

class TemporalShift(nn.Module):
    """Temporal Shift Module: shift 1/3 channels forward, 1/3 backward, 1/3 unchanged."""

    def __init__(self, n_segment):
        super().__init__()
        self.n_segment = n_segment

    def forward(self, x):
        """x: (B*T, C, H, W) -> (B*T, C, H, W) with temporal shift."""
        BT, C, H, W = x.shape
        T = self.n_segment
        # #region agent log
        try:
            import json, time
            rem = int(BT % T) if T else None
            payload = {
                "sessionId": "debug-session",
                "runId": "repro",
                "hypothesisId": "H1",
                "location": "models/video_ecg_model.py:TemporalShift.forward",
                "message": "tsm_view_check",
                "data": {"BT": int(BT), "C": int(C), "H": int(H), "W": int(W), "n_segment": int(T), "BT_mod_n_segment": rem},
                "timestamp": int(time.time() * 1000),
            }
            with open("/home/xinchen/ECG/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion
        B = BT // T
        x = x.view(B, T, C, H, W)
        fold = C // 3

        out = x.clone()
        # Shift left (future -> current)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, -1, :fold] = 0
        # Shift right (past -> current)
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        out[:, 0, fold:2*fold] = 0
        # Third part unchanged

        return out.view(BT, C, H, W)


class TSMConv2d(nn.Module):
    """TSM + Conv2d + BN + ReLU block."""

    def __init__(self, in_ch, out_ch, kernel_size=3, n_segment=300):
        super().__init__()
        self.tsm = TemporalShift(n_segment)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.tsm(x)
        return self.relu(self.bn(self.conv(x)))


class AttentionMask(nn.Module):
    """Spatial attention mask normalized to sum to H*W*0.5 (MTTS-CAN style)."""

    def forward(self, x):
        # x: (BT, 1, H, W) sigmoid output
        xsum = x.sum(dim=(2, 3), keepdim=True) + 1e-8
        H, W = x.shape[2], x.shape[3]
        return x / xsum * (H * W * 0.5)


class DualBranchEncoder(nn.Module):
    """
    MTTS-CAN-inspired dual-branch encoder.
    Motion branch (diff frames): TSM-Conv2d blocks
    Appearance branch (raw frames): Conv2d blocks -> spatial attention mask
    Output: (B, T, D) per-frame features.

    Input: (B, T, 6, H, W) where channels = [diff_R, diff_G, diff_B, raw_R, raw_G, raw_B]
    """

    def __init__(self, n_segment=300, channels=(32, 64), img_size=36):
        super().__init__()
        self.n_segment = n_segment

        # Motion branch (diff frames, channels 0:3)
        self.motion_blocks = nn.ModuleList()
        self.appear_blocks = nn.ModuleList()
        self.attn_convs = nn.ModuleList()
        self.attn_mask = AttentionMask()

        in_ch = 3
        for ch in channels:
            self.motion_blocks.append(nn.Sequential(
                TSMConv2d(in_ch, ch, 3, n_segment),
                TSMConv2d(ch, ch, 3, n_segment),
            ))
            self.appear_blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(True),
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(True),
            ))
            self.attn_convs.append(nn.Sequential(
                nn.Conv2d(ch, 1, 1),
                nn.Sigmoid(),
            ))
            in_ch = ch

        self.pool = nn.AvgPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)

        # Compute output dim: after len(channels) pooling stages
        final_size = img_size
        for _ in channels:
            final_size = final_size // 2
        self.out_dim = channels[-1] * final_size * final_size

    def forward(self, x):
        """x: (B, T, 6, H, W) -> (B, T, D)"""
        B, T, C, H, W = x.shape
        # #region agent log
        try:
            import json, time
            payload = {
                "sessionId": "debug-session",
                "runId": "repro",
                "hypothesisId": "H1",
                "location": "models/video_ecg_model.py:DualBranchEncoder.forward",
                "message": "dual_branch_input_shape",
                "data": {"B": int(B), "T": int(T), "C": int(C), "H": int(H), "W": int(W)},
                "timestamp": int(time.time() * 1000),
            }
            with open("/home/xinchen/ECG/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion
        diff = x[:, :, :3].reshape(B * T, 3, H, W)
        raw = x[:, :, 3:].reshape(B * T, 3, H, W)

        motion = diff
        appear = raw
        for m_block, a_block, attn_conv in zip(
            self.motion_blocks, self.appear_blocks, self.attn_convs
        ):
            motion = m_block(motion)
            appear = a_block(appear)
            mask = self.attn_mask(attn_conv(appear))
            motion = motion * mask
            motion = self.dropout(self.pool(motion))
            appear = self.dropout(self.pool(appear))

        # Flatten spatial dims
        motion = motion.view(B, T, -1)  # (B, T, D)
        return motion


class MTTSCANECGModel(nn.Module):
    """
    Scheme C: MTTS-CAN-inspired model for Video -> ECG.
    Dual-branch (diff + raw) with attention gating, TSM for temporal modeling,
    followed by TemporalDecoder for ECG waveform reconstruction.
    Optionally fuses IMU data.
    """

    def __init__(self, n_segment=300, channels=(32, 64), img_size=36,
                 temporal_channels=(128, 64, 32), upsample_factor=8,
                 target_ecg_len=2500, use_imu=False, imu_dim=64, imu_sr=100,
                 window_sec=10):
        super().__init__()
        self.use_imu = use_imu

        self.encoder = DualBranchEncoder(
            n_segment=n_segment, channels=channels, img_size=img_size
        )

        decoder_in_dim = self.encoder.out_dim
        if use_imu:
            self.imu_encoder = IMUEncoder(
                in_channels=6, out_dim=imu_dim, target_len=n_segment
            )
            decoder_in_dim += imu_dim

        self.decoder = TemporalDecoder(
            in_dim=decoder_in_dim,
            channels=temporal_channels,
            upsample_factor=upsample_factor,
            target_ecg_len=target_ecg_len,
        )

    def forward(self, video, imu=None):
        """
        video: (B, T, 6, H, W) - concatenated [diff, raw] frames
        imu: (B, T_imu, 6) or None
        returns: (B, T_ecg)
        """
        feat = self.encoder(video)  # (B, T, D)

        if self.use_imu and imu is not None:
            imu_feat = self.imu_encoder(imu)  # (B, T, D_imu)
            feat = torch.cat([feat, imu_feat], dim=-1)

        return self.decoder(feat)


# ──────────────────────────────────────────────
#  Scheme D: 1D Signal-Centric TCN
# ──────────────────────────────────────────────

class TCNBlock(nn.Module):
    """Temporal Convolutional Block with residual connection and dilation."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Residual connection
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return self.relu(x + res)


class Signal1DEncoder(nn.Module):
    """
    1D TCN encoder for PPG-like signal.
    Input: (B, C_in, T_v) where C_in is number of signal channels (e.g., 3 for RGB means)
    Output: (B, D, T_v)
    """

    def __init__(self, in_channels=3, channels=(32, 64, 128), kernel_size=7, dropout=0.1):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for i, ch in enumerate(channels):
            dilation = 2 ** i  # exponentially increasing dilation
            layers.append(TCNBlock(prev_ch, ch, kernel_size, dilation, dropout))
            prev_ch = ch
        self.net = nn.Sequential(*layers)
        self.out_dim = channels[-1]

    def forward(self, x):
        return self.net(x)


class Signal1DECGModel(nn.Module):
    """
    Scheme D: Ultra-lightweight 1D signal to ECG model.
    Input: (B, T_v, C_in) - per-frame ROI statistics (e.g., RGB channel means)
    Output: (B, T_ecg) - reconstructed ECG waveform

    Total params ~100K (without IMU), ideal for small datasets.
    Optionally fuses IMU data.
    """

    def __init__(self, in_channels=3, encoder_channels=(32, 64, 128),
                 decoder_channels=(64, 32), kernel_size=7,
                 upsample_factor=8, target_ecg_len=2500, dropout=0.1,
                 use_imu=False, imu_dim=32, n_segment=300):
        super().__init__()
        self.use_imu = use_imu
        self.n_segment = n_segment

        self.encoder = Signal1DEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        decoder_in_dim = self.encoder.out_dim
        if use_imu:
            # Lightweight IMU encoder for 1D model
            self.imu_encoder = IMUEncoder(
                in_channels=6, out_dim=imu_dim, target_len=n_segment
            )
            decoder_in_dim += imu_dim

        # Decoder with upsampling
        layers = []
        prev_ch = decoder_in_dim
        n_upsample = 0
        factor = upsample_factor
        while factor > 1:
            factor //= 2
            n_upsample += 1

        for i, ch in enumerate(decoder_channels):
            if i < n_upsample:
                layers.append(nn.ConvTranspose1d(prev_ch, ch, 4, stride=2, padding=1))
            else:
                layers.append(nn.Conv1d(prev_ch, ch, 3, padding=1))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_ch = ch

        self.decoder = nn.Sequential(*layers)
        self.head = nn.Conv1d(prev_ch, 1, 3, padding=1)
        self.target_ecg_len = target_ecg_len

    def forward(self, signal, imu=None):
        """
        signal: (B, T_v, C_in) - per-frame signal (RGB means, etc.)
        imu: (B, T_imu, 6) or None
        returns: (B, T_ecg)
        """
        x = signal.permute(0, 2, 1)  # (B, C_in, T_v)
        x = self.encoder(x)  # (B, D, T_v)

        if self.use_imu and imu is not None:
            imu_feat = self.imu_encoder(imu)  # (B, T_v, D_imu)
            imu_feat = imu_feat.permute(0, 2, 1)  # (B, D_imu, T_v)
            x = torch.cat([x, imu_feat], dim=1)  # (B, D+D_imu, T_v)

        x = self.decoder(x)  # (B, ch, T_up)
        x = self.head(x)  # (B, 1, T_up)
        if x.shape[-1] != self.target_ecg_len:
            x = F.interpolate(x, size=self.target_ecg_len, mode="linear", align_corners=False)
        return x.squeeze(1)


# ──────────────────────────────────────────────
#  Composite Loss (Scheme B/C/D)
# ──────────────────────────────────────────────

class CompositeLoss(nn.Module):
    """
    MSE + spectral magnitude loss + negative Pearson correlation loss.
    Weights controlled by alpha/beta.
    """

    def __init__(self, alpha_freq=0.1, beta_pearson=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha_freq
        self.beta = beta_pearson

    def _spectral_loss(self, pred, target):
        """L1 loss on FFT magnitudes."""
        pred_fft = torch.fft.rfft(pred, dim=-1)
        tgt_fft = torch.fft.rfft(target, dim=-1)
        return F.l1_loss(pred_fft.abs(), tgt_fft.abs())

    def _pearson_loss(self, pred, target):
        """Negative mean Pearson correlation (lower is better)."""
        pred_c = pred - pred.mean(dim=-1, keepdim=True)
        tgt_c = target - target.mean(dim=-1, keepdim=True)
        num = (pred_c * tgt_c).sum(dim=-1)
        den = torch.sqrt((pred_c ** 2).sum(dim=-1) * (tgt_c ** 2).sum(dim=-1) + 1e-8)
        r = num / den
        return 1.0 - r.mean()  # minimize -> maximize correlation

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        loss_freq = self._spectral_loss(pred, target)
        loss_pearson = self._pearson_loss(pred, target)
        return loss_mse + self.alpha * loss_freq + self.beta * loss_pearson


# ──────────────────────────────────────────────
#  Build helpers
# ──────────────────────────────────────────────

def build_model(cfg):
    """Build model from config dict."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    model_type = model_cfg.get("type", "video_ecg")
    target_ecg_len = int(data_cfg["window_seconds"] * data_cfg["ecg_sr"])
    n_segment = int(data_cfg["window_seconds"] * data_cfg["video_fps"])
    if model_type == "mtts_can" and data_cfg.get("use_diff_frames", False):
        n_segment = n_segment - 1  # diff frames yield T-1 frames per window

    # Common IMU parameters (available to all schemes)
    use_imu = data_cfg.get("use_imu", False)
    imu_dim = model_cfg.get("imu_dim", 64)
    imu_sr = data_cfg.get("imu_sr", 100)
    window_sec = data_cfg["window_seconds"]

    if model_type == "mtts_can":
        # Scheme C
        model = MTTSCANECGModel(
            n_segment=n_segment,
            channels=tuple(model_cfg["encoder_channels"]),
            img_size=data_cfg["img_height"],
            temporal_channels=tuple(model_cfg["temporal_channels"]),
            upsample_factor=model_cfg["upsample_factor"],
            target_ecg_len=target_ecg_len,
            use_imu=use_imu,
            imu_dim=imu_dim,
            imu_sr=imu_sr,
            window_sec=window_sec,
        )
    elif model_type == "signal_1d":
        # Scheme D
        model = Signal1DECGModel(
            in_channels=model_cfg.get("in_channels", 3),
            encoder_channels=tuple(model_cfg["encoder_channels"]),
            decoder_channels=tuple(model_cfg["decoder_channels"]),
            kernel_size=model_cfg.get("kernel_size", 7),
            upsample_factor=model_cfg["upsample_factor"],
            target_ecg_len=target_ecg_len,
            dropout=model_cfg.get("dropout", 0.1),
            use_imu=use_imu,
            imu_dim=imu_dim,
            n_segment=n_segment,
        )
    else:
        # Scheme A/B
        model = VideoECGModel(
            backbone=model_cfg.get("encoder", "resnet18"),
            pretrained_encoder=True,
            temporal_channels=tuple(model_cfg["temporal_channels"]),
            upsample_factor=model_cfg["upsample_factor"],
            window_sec=window_sec,
            video_fps=data_cfg["video_fps"],
            ecg_sr=data_cfg["ecg_sr"],
            use_imu=use_imu,
            imu_dim=imu_dim,
            imu_sr=imu_sr,
        )
    return model


def build_criterion(cfg):
    """Build loss function from config."""
    loss_cfg = cfg["train"].get("loss", "mse")
    if loss_cfg == "mse":
        return nn.MSELoss()
    elif loss_cfg == "composite":
        alpha = cfg["train"].get("alpha_freq", 0.1)
        beta = cfg["train"].get("beta_pearson", 0.1)
        return CompositeLoss(alpha_freq=alpha, beta_pearson=beta)
    else:
        raise ValueError(f"Unknown loss: {loss_cfg}")


if __name__ == "__main__":
    # Quick test for all schemes
    print("=== Scheme A (ResNet-18, video only) ===")
    model_a = VideoECGModel(backbone="resnet18")
    dummy_v = torch.randn(2, 300, 3, 64, 64)
    out_a = model_a(dummy_v)
    print(f"  Input: {dummy_v.shape} -> Output: {out_a.shape}")
    print(f"  Params: {sum(p.numel() for p in model_a.parameters()):,}")

    print("\n=== Scheme B (ResNet-50, video + IMU) ===")
    model_b = VideoECGModel(backbone="resnet50", use_imu=True, imu_dim=64)
    dummy_imu = torch.randn(2, 1000, 6)
    out_b = model_b(dummy_v, dummy_imu)
    print(f"  Input: video {dummy_v.shape} + imu {dummy_imu.shape} -> Output: {out_b.shape}")
    print(f"  Params: {sum(p.numel() for p in model_b.parameters()):,}")

    print("\n=== Scheme C (MTTS-CAN dual-branch + TSM) ===")
    model_c = MTTSCANECGModel(n_segment=300, channels=(32, 64), img_size=36)
    dummy_vc = torch.randn(2, 300, 6, 36, 36)
    out_c = model_c(dummy_vc)
    print(f"  Input: {dummy_vc.shape} -> Output: {out_c.shape}")
    print(f"  Params: {sum(p.numel() for p in model_c.parameters()):,}")

    print("\n=== Scheme C + IMU ===")
    model_c_imu = MTTSCANECGModel(n_segment=300, channels=(32, 64), img_size=36, use_imu=True, imu_dim=64)
    out_c_imu = model_c_imu(dummy_vc, dummy_imu)
    print(f"  Input: video {dummy_vc.shape} + imu {dummy_imu.shape} -> Output: {out_c_imu.shape}")
    print(f"  Params: {sum(p.numel() for p in model_c_imu.parameters()):,}")

    print("\n=== Scheme D (1D Signal TCN) ===")
    model_d = Signal1DECGModel(in_channels=3, encoder_channels=(32, 64, 128),
                                decoder_channels=(64, 32))
    dummy_sig = torch.randn(2, 300, 3)  # (B, T, C) - RGB means per frame
    out_d = model_d(dummy_sig)
    print(f"  Input: {dummy_sig.shape} -> Output: {out_d.shape}")
    print(f"  Params: {sum(p.numel() for p in model_d.parameters()):,}")

    print("\n=== Scheme D + IMU ===")
    model_d_imu = Signal1DECGModel(in_channels=3, encoder_channels=(32, 64, 128),
                                    decoder_channels=(64, 32), use_imu=True, imu_dim=32)
    out_d_imu = model_d_imu(dummy_sig, dummy_imu)
    print(f"  Input: signal {dummy_sig.shape} + imu {dummy_imu.shape} -> Output: {out_d_imu.shape}")
    print(f"  Params: {sum(p.numel() for p in model_d_imu.parameters()):,}")

    print("\n=== Composite Loss ===")
    loss_fn = CompositeLoss()
    loss = loss_fn(out_a, torch.randn_like(out_a))
    print(f"  Loss: {loss.item():.4f}")
