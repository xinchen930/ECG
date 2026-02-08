"""
STMap-based ECG Reconstruction Models.

Two architectures for reconstructing ECG waveforms from Spatio-Temporal Maps:

Scheme I-Direct (STMapDirectECG):
    STMap (C, N_spatial, T) -> 2D CNN Encoder -> Temporal Decoder -> ECG (T_ecg,)
    Direct regression without intermediate PPG step.

Scheme I-TwoStage (STMapTwoStageECG):
    STMap (C, N_spatial, T) -> 2D CNN -> PPG (T,) -> 1D UNet -> ECG (T_ecg,)
    Two-stage with optional multi-task PPG prediction.

Both models:
    - Accept STMap tensors of shape (B, C, N_spatial, T_video)
    - Output ECG waveforms of shape (B, T_ecg)
    - Support optional IMU fusion
    - Are lightweight (STMap eliminates the 3D video cost)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
#  Shared components
# ------------------------------------------------------------------

class IMUEncoder(nn.Module):
    """1D CNN encoder for IMU data: (B, T_imu, 6) -> (B, T_target, D_imu).

    Replicates the same architecture from video_ecg_model.py to avoid
    circular imports. Kept intentionally identical.
    """

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
        """x: (B, T_imu, 6) -> (B, T_target, D_imu)"""
        x = x.permute(0, 2, 1)                  # (B, 6, T_imu)
        x = self.net(x)                          # (B, D_imu, T_imu)
        x = F.interpolate(x, size=self.target_len,
                          mode="linear", align_corners=False)
        return x.permute(0, 2, 1)               # (B, T_target, D_imu)


class Conv2dBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU block with optional stride control.

    When spatial_stride > 1, downsamples along the spatial (height) axis
    while preserving the temporal (width) axis.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3,
                 spatial_stride=1, temporal_stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size,
            stride=(spatial_stride, temporal_stride),
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: (B, C, N_spatial, T)
        return self.dropout(self.relu(self.bn(self.conv(x))))


class ResBlock2d(nn.Module):
    """Residual block: two Conv2d + BN + ReLU with skip connection.

    Optional spatial-only downsampling (stride along height axis).
    """

    def __init__(self, in_ch, out_ch, kernel_size=3,
                 spatial_stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size,
                               stride=(spatial_stride, 1), padding=padding)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Shortcut
        if in_ch != out_ch or spatial_stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(spatial_stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # x: (B, C_in, N_spatial, T)
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)          # (B, C_out, N'/spatial_stride, T)
        return out


# ------------------------------------------------------------------
#  Scheme I-Direct: STMap -> 2D CNN -> ECG (direct regression)
# ------------------------------------------------------------------

class STMapDirectECG(nn.Module):
    """Direct ECG reconstruction from STMap without intermediate PPG.

    Architecture:
        STMap (B, C, N_spatial, T_video)
          -> 2D CNN encoder (spatial downsampling, temporal preserved)
          -> AdaptiveAvgPool to collapse spatial dim
          -> 1D temporal decoder (upsample T_video -> T_ecg)
          -> ECG (B, T_ecg)

    The 2D CNN treats the STMap like an image where:
        height = spatial patches (N_spatial)
        width  = time steps (T_video)

    Encoder progressively reduces the spatial dimension while keeping
    the temporal dimension intact, then a 1D decoder upsamples to ECG
    sampling rate.

    Parameters: ~200K-500K depending on config.
    Memory: ~2-4 GB on GPU (very lightweight compared to 3D video models).
    """

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]

        # Core dimensions
        in_channels = self._infer_in_channels(data_cfg)
        encoder_channels = model_cfg.get("encoder_channels", [32, 64, 128, 256])
        decoder_channels = model_cfg.get("decoder_channels", [128, 64])
        kernel_size = model_cfg.get("kernel_size", 3)
        dropout = model_cfg.get("dropout", 0.1)

        self.target_ecg_len = int(data_cfg["window_seconds"] * data_cfg["ecg_sr"])
        self.n_segment = int(data_cfg["window_seconds"] * data_cfg["video_fps"])

        # --- 2D CNN Encoder ---
        # Spatial stride=2 at each layer to progressively reduce N_spatial
        # Temporal stride=1 to preserve time resolution
        self.encoder = nn.ModuleList()
        ch_in = in_channels
        for i, ch_out in enumerate(encoder_channels):
            spatial_stride = 2 if i > 0 else 1  # first layer no downsample
            self.encoder.append(
                ResBlock2d(ch_in, ch_out, kernel_size=kernel_size,
                           spatial_stride=spatial_stride, dropout=dropout)
            )
            ch_in = ch_out

        # Collapse remaining spatial dimension
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, None))
        # Output: (B, encoder_channels[-1], 1, T) -> squeeze -> (B, C, T)

        # --- 1D Temporal Decoder ---
        # Upsample from T_video to T_ecg via transposed convolutions
        decoder_in = encoder_channels[-1]

        # IMU fusion
        self.use_imu = data_cfg.get("use_imu", False)
        imu_dim = model_cfg.get("imu_dim", 32)
        if self.use_imu:
            self.imu_encoder = IMUEncoder(
                in_channels=6, out_dim=imu_dim, target_len=self.n_segment
            )
            decoder_in += imu_dim

        self.decoder = self._build_temporal_decoder(
            decoder_in, decoder_channels, self.target_ecg_len, dropout
        )

    def forward(self, stmap, imu=None):
        """
        Args:
            stmap: (B, C, N_spatial, T_video) STMap tensor
            imu:   (B, T_imu, 6) optional IMU tensor

        Returns:
            ecg: (B, T_ecg) predicted ECG waveform
        """
        x = stmap  # (B, C, N_spatial, T)

        # 2D CNN encoder: progressively reduce spatial dim
        for block in self.encoder:
            x = block(x)
        # x: (B, C_enc, N_spatial', T)

        # Collapse spatial -> (B, C_enc, 1, T) -> (B, C_enc, T)
        x = self.spatial_pool(x).squeeze(2)

        # IMU fusion
        if self.use_imu and imu is not None:
            imu_feat = self.imu_encoder(imu)               # (B, T, D_imu)
            imu_feat = imu_feat.permute(0, 2, 1)           # (B, D_imu, T)
            if imu_feat.shape[-1] != x.shape[-1]:
                imu_feat = F.interpolate(imu_feat, size=x.shape[-1],
                                         mode="linear", align_corners=False)
            x = torch.cat([x, imu_feat], dim=1)            # (B, C_enc+D_imu, T)

        # Temporal decoder -> (B, T_ecg)
        x = self.decoder(x)
        return x

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_in_channels(data_cfg):
        """Infer number of input channels from data config."""
        ch_mode = data_cfg.get("stmap_channels", "rgb")
        if ch_mode in ("rgb", "all"):
            return 3
        elif ch_mode in ("red", "green"):
            return 1
        else:
            return 3

    @staticmethod
    def _build_temporal_decoder(in_ch, channels, target_len, dropout):
        """Build a 1D decoder that upsamples to target ECG length.

        Uses transposed convolutions for upsampling followed by a final
        interpolation to exactly match target_len.
        """
        layers = []
        prev = in_ch

        # How many 2x upsample stages? ECG/video ratio ~ 2500/300 ~ 8.3
        # Use 3 stages (2^3=8) and then interpolate the remainder
        n_upsample = min(len(channels), 3)

        for i, ch in enumerate(channels):
            if i < n_upsample:
                layers.append(nn.ConvTranspose1d(prev, ch, kernel_size=4,
                                                  stride=2, padding=1))
            else:
                layers.append(nn.Conv1d(prev, ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = ch

        layers.append(nn.Conv1d(prev, 1, kernel_size=3, padding=1))

        class _Decoder(nn.Module):
            def __init__(self, seq, tgt):
                super().__init__()
                self.net = nn.Sequential(*seq)
                self.target_len = tgt

            def forward(self, x):
                # x: (B, C_in, T)
                x = self.net(x)                # (B, 1, T_up)
                if x.shape[-1] != self.target_len:
                    x = F.interpolate(x, size=self.target_len,
                                      mode="linear", align_corners=False)
                return x.squeeze(1)            # (B, T_ecg)

        return _Decoder(layers, target_len)


# ------------------------------------------------------------------
#  Scheme I-TwoStage: STMap -> PPG -> ECG
# ------------------------------------------------------------------

class PPGExtractor(nn.Module):
    """Stage 1: Extract a 1D PPG signal from STMap.

    Architecture:
        STMap (B, C, N_spatial, T)
          -> 2D CNN (spatial downsampling)
          -> AdaptiveAvgPool2d((1, T))
          -> 1D Conv to single channel PPG
          -> PPG (B, T)
    """

    def __init__(self, in_channels=3, channels=(32, 64, 128),
                 kernel_size=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch_in = in_channels
        for i, ch_out in enumerate(channels):
            spatial_stride = 2 if i > 0 else 1
            self.blocks.append(
                ResBlock2d(ch_in, ch_out, kernel_size=kernel_size,
                           spatial_stride=spatial_stride, dropout=dropout)
            )
            ch_in = ch_out

        self.spatial_pool = nn.AdaptiveAvgPool2d((1, None))
        # Produce a single-channel PPG from the feature map
        self.ppg_head = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1] // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[-1] // 2, 1, kernel_size=1),
        )

    def forward(self, stmap):
        """
        Args:
            stmap: (B, C, N_spatial, T)
        Returns:
            ppg: (B, T) extracted PPG signal
            features: (B, D, T) intermediate features for optional reuse
        """
        x = stmap
        for block in self.blocks:
            x = block(x)
        # x: (B, C_last, N', T)

        x = self.spatial_pool(x).squeeze(2)   # (B, C_last, T)
        features = x                           # save for potential multi-task use

        ppg = self.ppg_head(x).squeeze(1)     # (B, T)
        return ppg, features


class ECGDecoder1DUNet(nn.Module):
    """Stage 2: Reconstruct ECG from 1D PPG via a 1D UNet.

    Architecture:
        PPG (B, 1, T_video)
          -> UNet encoder (downsample)
          -> bottleneck
          -> UNet decoder (upsample with skip connections)
          -> interpolate to T_ecg
          -> ECG (B, T_ecg)
    """

    def __init__(self, in_channels=1, base_channels=64, depth=3,
                 kernel_size=7, target_ecg_len=2500, dropout=0.1):
        super().__init__()
        self.target_ecg_len = target_ecg_len
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(self._conv_block(ch, out_ch, kernel_size, dropout))
            self.pools.append(nn.MaxPool1d(2))
            ch = out_ch

        # Bottleneck
        bottleneck_ch = base_channels * (2 ** depth)
        self.bottleneck = self._conv_block(ch, bottleneck_ch, kernel_size, dropout)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            out_ch = base_channels * (2 ** i)
            self.upconvs.append(nn.ConvTranspose1d(ch, out_ch, 4, stride=2, padding=1))
            self.decoders.append(self._conv_block(out_ch * 2, out_ch, kernel_size, dropout))
            ch = out_ch

        self.head = nn.Conv1d(base_channels, 1, kernel_size=1)

    @staticmethod
    def _conv_block(in_ch, out_ch, kernel_size, dropout):
        pad = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, T) PPG signal (single channel)
        Returns:
            ecg: (B, T_ecg) reconstructed ECG
        """
        # Encoder with skip connections
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            skip = skips[-(i + 1)]
            # Match temporal sizes
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1],
                                  mode="linear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.head(x)  # (B, 1, T)
        if x.shape[-1] != self.target_ecg_len:
            x = F.interpolate(x, size=self.target_ecg_len,
                              mode="linear", align_corners=False)
        return x.squeeze(1)  # (B, T_ecg)


class STMapTwoStageECG(nn.Module):
    """Two-stage ECG reconstruction: STMap -> PPG -> ECG.

    Stage 1: 2D CNN extracts a 1D PPG signal from the STMap.
    Stage 2: 1D UNet reconstructs ECG from the PPG.

    Optionally supports:
        - Multi-task training with PPG auxiliary loss
        - IMU fusion at the PPG/bottleneck level

    Parameters: ~300K-800K depending on config.
    Memory: ~2-5 GB (lightweight).
    """

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]

        in_channels = self._infer_in_channels(data_cfg)
        ppg_channels = model_cfg.get("ppg_extractor_channels", [32, 64, 128])
        ecg_dec_channels = model_cfg.get("ecg_decoder_channels", [64, 128, 256])
        ecg_base_ch = ecg_dec_channels[0] if ecg_dec_channels else 64
        ecg_depth = model_cfg.get("ecg_decoder_depth", 3)
        kernel_size = model_cfg.get("kernel_size", 3)
        dropout = model_cfg.get("dropout", 0.1)

        self.target_ecg_len = int(data_cfg["window_seconds"] * data_cfg["ecg_sr"])
        self.n_segment = int(data_cfg["window_seconds"] * data_cfg["video_fps"])

        self.multitask_ppg = model_cfg.get("multitask_ppg", False)
        self.ppg_loss_weight = model_cfg.get("ppg_loss_weight", 0.3)

        # Stage 1: PPG extractor
        self.ppg_extractor = PPGExtractor(
            in_channels=in_channels,
            channels=ppg_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Stage 2: ECG decoder (1D UNet)
        # Input to ECG decoder: PPG (1 channel) [+ optional IMU features]
        ecg_in_channels = 1
        self.use_imu = data_cfg.get("use_imu", False)
        imu_dim = model_cfg.get("imu_dim", 32)
        if self.use_imu:
            self.imu_encoder = IMUEncoder(
                in_channels=6, out_dim=imu_dim, target_len=self.n_segment
            )
            ecg_in_channels += imu_dim  # concatenate IMU features with PPG

        self.ecg_decoder = ECGDecoder1DUNet(
            in_channels=ecg_in_channels,
            base_channels=ecg_base_ch,
            depth=ecg_depth,
            kernel_size=7,  # larger kernel for 1D temporal
            target_ecg_len=self.target_ecg_len,
            dropout=dropout,
        )

    def forward(self, stmap, imu=None):
        """
        Args:
            stmap: (B, C, N_spatial, T_video) STMap tensor
            imu:   (B, T_imu, 6) optional IMU tensor

        Returns:
            ecg: (B, T_ecg) predicted ECG waveform

        Note:
            When multitask_ppg is enabled and the model is in training mode,
            the intermediate PPG is stored in self.last_ppg for auxiliary loss
            computation. Use get_ppg_aux_loss(ecg_target) after forward to
            retrieve the auxiliary loss.
        """
        # Stage 1: extract PPG
        ppg, features = self.ppg_extractor(stmap)   # ppg: (B, T), features: (B, D, T)

        # Store for multi-task loss
        self.last_ppg = ppg if (self.multitask_ppg and self.training) else None

        # Prepare input for ECG decoder
        x = ppg.unsqueeze(1)                         # (B, 1, T)

        # IMU fusion: concatenate with PPG signal
        if self.use_imu and imu is not None:
            imu_feat = self.imu_encoder(imu)         # (B, T, D_imu)
            imu_feat = imu_feat.permute(0, 2, 1)    # (B, D_imu, T)
            if imu_feat.shape[-1] != x.shape[-1]:
                imu_feat = F.interpolate(imu_feat, size=x.shape[-1],
                                         mode="linear", align_corners=False)
            x = torch.cat([x, imu_feat], dim=1)     # (B, 1+D_imu, T)

        # Stage 2: PPG -> ECG
        ecg = self.ecg_decoder(x)                    # (B, T_ecg)
        return ecg

    def get_ppg_aux_loss(self, ecg_target):
        """Compute auxiliary PPG loss if multi-task is enabled.

        Call this after forward() during training to get the PPG auxiliary loss.
        The loss encourages the intermediate PPG to correlate with the
        (downsampled) ECG and to be smooth.

        Args:
            ecg_target: (B, T_ecg) ground truth ECG

        Returns:
            Scalar loss tensor, or 0 if multi-task is disabled / not training.
        """
        if self.last_ppg is None:
            return torch.tensor(0.0, device=ecg_target.device)

        ppg = self.last_ppg
        # Smoothness: penalize large second derivatives
        if ppg.shape[-1] >= 3:
            d2 = ppg[:, 2:] - 2 * ppg[:, 1:-1] + ppg[:, :-2]
            loss_smooth = (d2 ** 2).mean()
        else:
            loss_smooth = torch.tensor(0.0, device=ppg.device)

        # Correlation with downsampled ECG
        ecg_down = F.interpolate(
            ecg_target.unsqueeze(1), size=ppg.shape[-1],
            mode="linear", align_corners=False
        ).squeeze(1)                                  # (B, T_video)
        ppg_c = ppg - ppg.mean(dim=-1, keepdim=True)
        ecg_c = ecg_down - ecg_down.mean(dim=-1, keepdim=True)
        num = (ppg_c * ecg_c).sum(dim=-1)
        den = torch.sqrt((ppg_c**2).sum(dim=-1) * (ecg_c**2).sum(dim=-1) + 1e-8)
        corr = (num / den).mean()
        loss_corr = 1.0 - corr

        return self.ppg_loss_weight * (loss_corr + 0.01 * loss_smooth)

    @staticmethod
    def _infer_in_channels(data_cfg):
        ch_mode = data_cfg.get("stmap_channels", "rgb")
        if ch_mode in ("rgb", "all"):
            return 3
        elif ch_mode in ("red", "green"):
            return 1
        else:
            return 3


# ------------------------------------------------------------------
#  Multi-task loss wrapper (for two-stage with PPG auxiliary)
# ------------------------------------------------------------------

class STMapMultiTaskLoss(nn.Module):
    """Combined loss for two-stage model: ECG loss + PPG auxiliary loss.

    The PPG loss encourages the intermediate PPG to be physiologically
    plausible (smooth, periodic). Since we don't have ground-truth PPG,
    we use a self-supervised proxy: negative Pearson correlation between
    PPG and (downsampled) ECG as a regularizer, plus a smoothness term.
    """

    def __init__(self, ecg_criterion, ppg_weight=0.3):
        super().__init__()
        self.ecg_criterion = ecg_criterion
        self.ppg_weight = ppg_weight

    def forward(self, pred, target, ppg=None):
        """
        Args:
            pred: (B, T_ecg) predicted ECG
            target: (B, T_ecg) ground truth ECG
            ppg: (B, T_video) predicted PPG (optional)
        """
        loss_ecg = self.ecg_criterion(pred, target)

        if ppg is not None and self.ppg_weight > 0:
            # Smoothness regularization: penalize large second derivatives
            if ppg.shape[-1] >= 3:
                d2 = ppg[:, 2:] - 2 * ppg[:, 1:-1] + ppg[:, :-2]
                loss_smooth = (d2 ** 2).mean()
            else:
                loss_smooth = torch.tensor(0.0, device=ppg.device)

            # Periodicity: PPG should correlate with downsampled ECG
            ecg_down = F.interpolate(
                target.unsqueeze(1), size=ppg.shape[-1],
                mode="linear", align_corners=False
            ).squeeze(1)                               # (B, T_video)
            ppg_c = ppg - ppg.mean(dim=-1, keepdim=True)
            ecg_c = ecg_down - ecg_down.mean(dim=-1, keepdim=True)
            num = (ppg_c * ecg_c).sum(dim=-1)
            den = torch.sqrt((ppg_c**2).sum(dim=-1) * (ecg_c**2).sum(dim=-1) + 1e-8)
            corr = (num / den).mean()
            loss_ppg = (1.0 - corr) + 0.01 * loss_smooth

            return loss_ecg + self.ppg_weight * loss_ppg

        return loss_ecg


# ------------------------------------------------------------------
#  Self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== STMap ECG Models Self-Test ===\n")

    # Minimal config for testing
    cfg_direct = {
        "model": {
            "type": "stmap_direct",
            "encoder_channels": [32, 64, 128, 256],
            "decoder_channels": [128, 64],
            "kernel_size": 3,
            "dropout": 0.1,
        },
        "data": {
            "window_seconds": 10,
            "video_fps": 30,
            "ecg_sr": 250,
            "stmap_channels": "rgb",
            "use_imu": False,
        },
    }

    cfg_twostage = {
        "model": {
            "type": "stmap_twostage",
            "ppg_extractor_channels": [32, 64, 128],
            "ecg_decoder_channels": [64, 128, 256],
            "ecg_decoder_depth": 3,
            "kernel_size": 3,
            "dropout": 0.1,
            "multitask_ppg": True,
            "ppg_loss_weight": 0.3,
        },
        "data": {
            "window_seconds": 10,
            "video_fps": 30,
            "ecg_sr": 250,
            "stmap_channels": "rgb",
            "use_imu": False,
        },
    }

    B, C, N, T = 4, 3, 64, 300  # batch, channels, spatial patches, time
    stmap = torch.randn(B, C, N, T)

    # --- Direct model ---
    print("=== STMapDirectECG ===")
    model_d = STMapDirectECG(cfg_direct)
    params_d = sum(p.numel() for p in model_d.parameters())
    ecg_d = model_d(stmap)
    print(f"  Input:  stmap {tuple(stmap.shape)}")
    print(f"  Output: ecg {tuple(ecg_d.shape)}")
    print(f"  Params: {params_d:,}")

    # --- Direct + IMU ---
    print("\n=== STMapDirectECG + IMU ===")
    cfg_direct_imu = {**cfg_direct}
    cfg_direct_imu["data"] = {**cfg_direct["data"], "use_imu": True}
    cfg_direct_imu["model"] = {**cfg_direct["model"], "imu_dim": 32}
    model_d_imu = STMapDirectECG(cfg_direct_imu)
    imu = torch.randn(B, 1000, 6)
    ecg_d_imu = model_d_imu(stmap, imu)
    print(f"  Input:  stmap {tuple(stmap.shape)} + imu {tuple(imu.shape)}")
    print(f"  Output: ecg {tuple(ecg_d_imu.shape)}")
    print(f"  Params: {sum(p.numel() for p in model_d_imu.parameters()):,}")

    # --- Two-stage model ---
    print("\n=== STMapTwoStageECG (multi-task) ===")
    model_ts = STMapTwoStageECG(cfg_twostage)
    params_ts = sum(p.numel() for p in model_ts.parameters())
    model_ts.train()
    ecg_ts = model_ts(stmap)
    ppg_ts = model_ts.last_ppg  # stored during forward when multitask_ppg=True
    print(f"  Input:  stmap {tuple(stmap.shape)}")
    print(f"  Output: ecg {tuple(ecg_ts.shape)}")
    print(f"  PPG (intermediate): {tuple(ppg_ts.shape)}")
    print(f"  Params: {params_ts:,}")

    # Test PPG auxiliary loss
    ecg_target = torch.randn(B, 2500)
    ppg_aux = model_ts.get_ppg_aux_loss(ecg_target)
    print(f"  PPG aux loss: {ppg_aux.item():.4f}")

    # Test eval mode (no PPG stored)
    model_ts.eval()
    ecg_ts_eval = model_ts(stmap)
    print(f"  Eval mode output: ecg {tuple(ecg_ts_eval.shape)}")
    assert model_ts.last_ppg is None, "PPG should not be stored in eval mode"
    print(f"  Eval mode PPG: None (correct)")

    # --- Multi-task loss wrapper (standalone usage) ---
    print("\n=== STMapMultiTaskLoss (standalone wrapper) ===")
    model_ts.train()
    ecg_ts2 = model_ts(stmap)
    ppg_ts2 = model_ts.last_ppg
    base_criterion = nn.MSELoss()
    mt_loss_fn = STMapMultiTaskLoss(base_criterion, ppg_weight=0.3)
    loss = mt_loss_fn(ecg_ts2, ecg_target, ppg_ts2)
    print(f"  Loss: {loss.item():.4f}")

    print("\nAll tests passed.")
