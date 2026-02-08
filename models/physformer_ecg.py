"""
PhysFormer-ECG: Temporal Difference Transformer for Video -> ECG Reconstruction.

Based on: "PhysFormer: Facial Video-based Physiological Measurement with
Temporal Difference Transformer" (Yu et al., CVPR 2022).

Key innovations adapted for ECG reconstruction:
  1. 3D CNN Stem -- progressively downsamples spatial dimensions while
     preserving temporal resolution.
  2. Temporal Difference (Center-Difference) Convolution in Q/K projections
     of multi-head self-attention, amplifying subtle frame-to-frame
     brightness changes caused by blood flow.
  3. Spatio-Temporal Feed-Forward with depthwise 3D convolution.
  4. ECG Temporal Decoder with ConvTranspose1d upsampling from video frame
     rate (~300 frames / 10 s) to ECG sample rate (2500 samples / 10 s).
  5. Optional IMU fusion via cross-attention.

Scheme H in the project configuration system.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ---------------------------------------------------------------------------
#  3D CNN Stem -- spatial feature extraction
# ---------------------------------------------------------------------------

class Stem3D(nn.Module):
    """
    3D CNN stem that progressively downsamples spatial dimensions while
    keeping the temporal dimension mostly intact.

    Input:  (B, C_in, T, H, W)  -- e.g. (B, 3, 300, 128, 128)
    Output: (B, dim,  T, h, w)  -- e.g. (B, 96, 300, 4, 4)

    Three stages, each: Conv3d -> BN -> ReLU -> MaxPool3d(1,2,2).
    Temporal stride is always 1 so that every video frame produces one
    token in the transformer.
    """

    def __init__(self, in_channels: int = 3, dim: int = 96):
        super().__init__()
        dim4 = dim // 4   # e.g. 24
        dim2 = dim // 2   # e.g. 48

        # Stage 1: large spatial kernel, no temporal downsampling
        self.conv1 = nn.Conv3d(in_channels, dim4,
                               kernel_size=(1, 5, 5),
                               stride=(1, 1, 1),
                               padding=(0, 2, 2), bias=False)
        self.bn1 = nn.BatchNorm3d(dim4)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # Stage 2
        self.conv2 = nn.Conv3d(dim4, dim2,
                               kernel_size=(3, 3, 3),
                               stride=(1, 1, 1),
                               padding=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(dim2)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # Stage 3
        self.conv3 = nn.Conv3d(dim2, dim,
                               kernel_size=(3, 3, 3),
                               stride=(1, 1, 1),
                               padding=(1, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(dim)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)   # spatial /2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)   # spatial /4
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)   # spatial /8
        return x  # (B, dim, T, H/8, W/8)


# ---------------------------------------------------------------------------
#  Center-Difference Temporal Convolution (CDC_T)
# ---------------------------------------------------------------------------

class CenterDiffTemporalConv(nn.Module):
    """
    Temporal center-difference convolution (CDC_T) from PhysFormer.

    Combines standard temporal convolution with a difference convolution that
    emphasises frame-to-frame intensity changes (the PPG signal in video).

    output = theta * F_diff + (1 - theta) * F_standard

    where F_diff uses the *center-difference* of the temporal conv kernel:
        w_diff[t] = w[t] - w[center]   for t != center
        w_diff[center] = 0

    Args:
        in_features:  input channel dimension
        out_features: output channel dimension
        kernel_size:  temporal kernel size (default 3)
        theta:        mixing ratio for difference vs standard (default 0.6)
    """

    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int = 3, theta: float = 0.6):
        super().__init__()
        self.theta = theta
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Standard 1-D temporal convolution (operates on the T dimension)
        # We reshape tokens to (B*h*w, C, T) to apply Conv1d.
        self.conv = nn.Conv1d(in_features, out_features,
                              kernel_size=kernel_size,
                              padding=self.padding, bias=False)

    def forward(self, x):
        """
        x: (B, N, C) where N = T * h * w  (flattened spatiotemporal tokens)
             -- OR --
           (B, T, C) if spatial dims already pooled.

        For PhysFormer we actually operate on (B*hw, C, T).
        The caller is responsible for reshaping; here we simply accept
        (BN, C, T) shaped tensors.
        """
        # x: (BN, C, T)
        F_standard = self.conv(x)  # (BN, C_out, T)

        # Build the center-difference kernel from the learned weights
        weight = self.conv.weight   # (C_out, C_in, K)
        center = self.kernel_size // 2
        # Subtract center slice from every temporal position
        w_diff = weight.clone()
        w_diff[:, :, center] = 0  # zero out center
        # For each non-center position: w_diff[t] = w[t] - w[center] already
        # But the paper defines diff kernel as w[t] - w[center], so:
        w_diff = weight - weight[:, :, center:center+1]
        w_diff[:, :, center] = 0

        F_diff = F.conv1d(x, w_diff, bias=None, padding=self.padding)

        return self.theta * F_diff + (1.0 - self.theta) * F_standard


# ---------------------------------------------------------------------------
#  Temporal Difference Multi-Head Attention
# ---------------------------------------------------------------------------

class TDMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with temporal-difference convolutions on Q and K
    (CDC_T) and standard linear projection on V.

    Includes the *gra_sharp* attention scaling from PhysFormer that
    sharpens/softens attention across layers.

    Args:
        dim:         token dimension
        heads:       number of attention heads
        theta:       CDC mixing ratio
        gra_sharp:   >1 sharpens attention, <1 softens (default 2.0)
    """

    def __init__(self, dim: int, heads: int = 4,
                 theta: float = 0.6, gra_sharp: float = 2.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.gra_sharp = gra_sharp

        # Q and K use temporal difference convolution
        self.q_cdc = CenterDiffTemporalConv(dim, dim, kernel_size=3, theta=theta)
        self.k_cdc = CenterDiffTemporalConv(dim, dim, kernel_size=3, theta=theta)
        # V uses standard linear projection
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, T: int, h: int, w: int):
        """
        x: (B, N, C)  where N = T * h * w
        T, h, w: spatiotemporal dimensions for reshaping
        returns: (B, N, C)
        """
        B, N, C = x.shape

        # --- Q and K via CDC_T ---
        # Reshape to (B*h*w, C, T) so Conv1d operates along temporal axis
        x_t = x.view(B, T, h * w, C)          # (B, T, hw, C)
        x_t = x_t.permute(0, 2, 3, 1)         # (B, hw, C, T)
        x_t = x_t.reshape(B * h * w, C, T)    # (B*hw, C, T)

        q = self.q_cdc(x_t)                    # (B*hw, C, T)
        k = self.k_cdc(x_t)                    # (B*hw, C, T)

        # Reshape back to (B, N, C)
        q = q.reshape(B, h * w, C, T).permute(0, 3, 1, 2).reshape(B, N, C)
        k = k.reshape(B, h * w, C, T).permute(0, 3, 1, 2).reshape(B, N, C)

        # --- V via linear projection ---
        v = self.v_proj(x)  # (B, N, C)

        # --- Multi-head attention ---
        # Reshape to (B, heads, N, head_dim)
        q = q.view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention with gra_sharp scaling
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn * self.gra_sharp  # sharpen / soften
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
#  Spatio-Temporal Feed-Forward Network
# ---------------------------------------------------------------------------

class STFeedForward(nn.Module):
    """
    Spatio-temporal feed-forward block from PhysFormer.

    FC -> Reshape to 3D -> DepthwiseConv3d -> GELU -> Reshape -> FC

    The depthwise 3D convolution mixes information across neighbouring
    spatiotemporal positions while keeping the model lightweight.

    Args:
        dim:       token dimension
        mlp_ratio: expansion ratio for the hidden dimension (default 4.0)
        drop:      dropout rate
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, hidden)
        # Depthwise conv3d on (hidden, T, h, w)
        self.dw_conv = nn.Conv3d(hidden, hidden, kernel_size=3, padding=1,
                                 groups=hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x, T: int, h: int, w: int):
        """
        x: (B, N, C)  where N = T*h*w
        T, h, w: spatiotemporal token grid dimensions
        returns: (B, N, C)
        """
        B, N, C = x.shape

        x = self.fc1(x)                               # (B, N, hidden)
        hidden = x.shape[-1]

        # Reshape to 5D for depthwise 3D conv
        x = x.view(B, T, h, w, hidden).permute(0, 4, 1, 2, 3)  # (B, hidden, T, h, w)
        x = self.act(self.dw_conv(x))
        x = x.permute(0, 2, 3, 4, 1).reshape(B, N, hidden)     # (B, N, hidden)

        x = self.drop(x)
        x = self.fc2(x)        # (B, N, C)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
#  Temporal Difference Transformer Block
# ---------------------------------------------------------------------------

class TDTransformerBlock(nn.Module):
    """
    One Temporal Difference Transformer block:
        LayerNorm -> TD-MHSA -> Residual -> LayerNorm -> ST-FFN -> Residual

    Args:
        dim:       token dimension
        heads:     number of attention heads
        theta:     CDC temporal difference mixing ratio
        gra_sharp: attention sharpness scaling factor
        mlp_ratio: FFN expansion ratio
        drop:      dropout rate
        attn_drop: attention dropout rate
    """

    def __init__(self, dim: int, heads: int = 4, theta: float = 0.6,
                 gra_sharp: float = 2.0, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TDMultiHeadAttention(
            dim=dim, heads=heads, theta=theta,
            gra_sharp=gra_sharp, attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = STFeedForward(dim=dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x, T: int, h: int, w: int):
        """
        x: (B, N, C)
        returns: (B, N, C)
        """
        x = x + self.attn(self.norm1(x), T, h, w)
        x = x + self.ffn(self.norm2(x), T, h, w)
        return x


# ---------------------------------------------------------------------------
#  IMU Cross-Attention Fusion
# ---------------------------------------------------------------------------

class IMUCrossAttention(nn.Module):
    """
    Cross-attention layer that fuses encoded IMU features into the video
    feature sequence.

    Video tokens attend to IMU tokens:
        Q = video features,  K = V = IMU features.

    Args:
        dim:      video token dimension
        imu_dim:  IMU encoded dimension
        heads:    number of attention heads
    """

    def __init__(self, dim: int, imu_dim: int, heads: int = 4, drop: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.norm_video = nn.LayerNorm(dim)
        self.norm_imu = nn.LayerNorm(imu_dim)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(imu_dim, dim, bias=False)
        self.v_proj = nn.Linear(imu_dim, dim, bias=False)

        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, video_feat, imu_feat):
        """
        video_feat: (B, N_v, C)
        imu_feat:   (B, N_imu, C_imu)
        returns:    (B, N_v, C)
        """
        B, Nv, C = video_feat.shape

        q = self.q_proj(self.norm_video(video_feat))   # (B, Nv, C)
        k = self.k_proj(self.norm_imu(imu_feat))       # (B, N_imu, C)
        v = self.v_proj(self.norm_imu(imu_feat))       # (B, N_imu, C)

        # Multi-head
        q = q.view(B, Nv, self.heads, self.head_dim).permute(0, 2, 1, 3)
        Ni = k.shape[1]
        k = k.view(B, Ni, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, Ni, self.heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Nv, C)
        out = self.out_proj(out)
        return video_feat + out  # residual


# ---------------------------------------------------------------------------
#  IMU Encoder (1D CNN, matches the project convention)
# ---------------------------------------------------------------------------

class IMUEncoder1D(nn.Module):
    """
    Encode IMU data (accelerometer + gyroscope) with a small 1D CNN.

    Input:  (B, T_imu, 6)
    Output: (B, T_target, imu_dim)
    """

    def __init__(self, in_channels: int = 6, imu_dim: int = 64,
                 target_len: int = 300):
        super().__init__()
        self.target_len = target_len
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, imu_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(imu_dim), nn.ReLU(inplace=True),
        )
        self.out_dim = imu_dim

    def forward(self, x):
        """x: (B, T_imu, 6) -> (B, T_target, imu_dim)"""
        x = x.permute(0, 2, 1)            # (B, 6, T_imu)
        x = self.net(x)                   # (B, imu_dim, T_imu)
        x = F.interpolate(x, size=self.target_len,
                          mode="linear", align_corners=False)
        return x.permute(0, 2, 1)         # (B, T_target, imu_dim)


# ---------------------------------------------------------------------------
#  ECG Temporal Decoder
# ---------------------------------------------------------------------------

class ECGTemporalDecoder(nn.Module):
    """
    Decode a per-frame feature sequence into the ECG waveform.

    Input:  (B, T_video, dim)   e.g. (B, 300, 96)
    Output: (B, T_ecg)          e.g. (B, 2500)

    Uses 1-D transposed convolutions for learned upsampling followed by a
    final linear interpolation to the exact target length.
    """

    def __init__(self, in_dim: int = 96, hidden_channels: tuple = (128, 64, 32),
                 target_ecg_len: int = 2500, dropout: float = 0.1):
        super().__init__()
        self.target_ecg_len = target_ecg_len

        layers = []
        prev_ch = in_dim

        # Three ConvTranspose1d stages: each upsamples x2  -> total x8
        # 300 frames -> 600 -> 1200 -> 2400, then interpolate to 2500
        for ch in hidden_channels:
            layers.extend([
                nn.ConvTranspose1d(prev_ch, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.Dropout1d(dropout),
            ])
            prev_ch = ch

        self.upsample = nn.Sequential(*layers)
        self.head = nn.Conv1d(prev_ch, 1, kernel_size=3, padding=1)

    def forward(self, x):
        """x: (B, T, C) -> (B, T_ecg)"""
        x = x.permute(0, 2, 1)        # (B, C, T)
        x = self.upsample(x)          # (B, ch_last, T*8)
        x = self.head(x)              # (B, 1, T*8)
        if x.shape[-1] != self.target_ecg_len:
            x = F.interpolate(x, size=self.target_ecg_len,
                              mode="linear", align_corners=False)
        return x.squeeze(1)           # (B, T_ecg)


# ---------------------------------------------------------------------------
#  PhysFormerECG -- full model
# ---------------------------------------------------------------------------

class PhysFormerECG(nn.Module):
    """
    PhysFormer backbone adapted for Video -> ECG reconstruction (Scheme H).

    Architecture:
        1. 3D CNN Stem: (B, 3, T, H, W) -> (B, dim, T, h, w)
        2. Flatten to tokens: (B, T*h*w, dim)
        3. N x TD-Transformer blocks
        4. Reshape back, adaptive spatial pool -> (B, T, dim)
        5. ECG Temporal Decoder -> (B, T_ecg)

    Optional:
        - IMU cross-attention fusion after transformer blocks
        - Gradient checkpointing for memory efficiency

    Args:
        cfg: project config dict (must contain 'model' and 'data' keys)
    """

    def __init__(self, cfg: dict):
        super().__init__()

        model_cfg = cfg["model"]
        data_cfg = cfg["data"]

        # Core dimensions
        self.dim = model_cfg.get("dim", 96)
        depth = model_cfg.get("depth", 4)
        heads = model_cfg.get("heads", 4)
        theta = model_cfg.get("theta", 0.6)
        mlp_ratio = model_cfg.get("mlp_ratio", 4.0)
        drop = model_cfg.get("dropout", 0.1)
        attn_drop = model_cfg.get("attn_drop", 0.0)
        in_channels = model_cfg.get("in_channels", 3)

        # Compute target sequence lengths from data config
        window_sec = data_cfg["window_seconds"]
        video_fps = data_cfg["video_fps"]
        ecg_sr = data_cfg["ecg_sr"]
        self.n_frames = int(window_sec * video_fps)      # e.g. 300
        self.target_ecg_len = int(window_sec * ecg_sr)    # e.g. 2500

        # IMU settings
        self.use_imu = data_cfg.get("use_imu", False)
        imu_dim = model_cfg.get("imu_dim", 64)
        imu_sr = data_cfg.get("imu_sr", 100)

        # Whether to use gradient checkpointing (saves VRAM at cost of speed)
        self.use_grad_ckpt = model_cfg.get("gradient_checkpointing", False)

        # --- 3D CNN Stem ---
        self.stem = Stem3D(in_channels=in_channels, dim=self.dim)

        # --- Positional embedding (learnable) ---
        # We create it lazily in forward() on the first pass so that it
        # automatically adapts to the actual spatial size after the stem.
        self.pos_embed = None  # will be initialised in forward

        # --- TD-Transformer blocks ---
        # gra_sharp schedule: linearly increase across layers
        # (shallow layers use softer attention, deeper layers sharper)
        gra_sharps = [1.0 + (2.0 - 1.0) * i / max(depth - 1, 1)
                      for i in range(depth)]

        self.blocks = nn.ModuleList([
            TDTransformerBlock(
                dim=self.dim, heads=heads, theta=theta,
                gra_sharp=gra_sharps[i], mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(self.dim)

        # --- IMU fusion ---
        if self.use_imu:
            self.imu_encoder = IMUEncoder1D(
                in_channels=6, imu_dim=imu_dim, target_len=self.n_frames,
            )
            self.imu_cross_attn = IMUCrossAttention(
                dim=self.dim, imu_dim=imu_dim, heads=heads, drop=drop,
            )

        # --- ECG Decoder ---
        decoder_channels = tuple(model_cfg.get(
            "decoder_channels", [128, 64, 32]))
        self.decoder = ECGTemporalDecoder(
            in_dim=self.dim,
            hidden_channels=decoder_channels,
            target_ecg_len=self.target_ecg_len,
            dropout=drop,
        )

        # Parameter initialisation
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear/conv layers, ones/zeros for BN/LN."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # -- helpers for gradient checkpointing --
    def _block_forward(self, block, x, T, h, w):
        """Wrapper that enables torch.utils.checkpoint for a single block."""
        return block(x, T, h, w)

    def forward(self, video, imu=None):
        """
        Args:
            video: (B, T, C, H, W)  video frames, float32, values in [0, 1]
            imu:   (B, T_imu, 6)    optional IMU data (acc_xyz + gyro_xyz)

        Returns:
            ecg:   (B, T_ecg)       reconstructed ECG waveform
        """
        B, T_v, C, H, W = video.shape

        # --- 3D CNN Stem ---
        # Permute to (B, C, T, H, W) for 3D convolutions
        x = video.permute(0, 2, 1, 3, 4)          # (B, C, T, H, W)
        x = self.stem(x)                           # (B, dim, T, h, w)

        _, D, T, h, w = x.shape

        # --- Flatten to token sequence ---
        # (B, dim, T, h, w) -> (B, T*h*w, dim)
        x = x.permute(0, 2, 3, 4, 1)              # (B, T, h, w, dim)
        x = x.reshape(B, T * h * w, D)            # (B, N, dim)

        # --- Positional embedding (lazy init) ---
        N = T * h * w
        if self.pos_embed is None or self.pos_embed.shape[1] != N:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, N, D, device=x.device, dtype=x.dtype),
                requires_grad=True,
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        x = x + self.pos_embed                    # (B, N, dim)

        # --- TD-Transformer blocks ---
        for block in self.blocks:
            if self.use_grad_ckpt and self.training:
                # gradient checkpointing to save memory
                x = grad_checkpoint(
                    self._block_forward, block, x, T, h, w,
                    use_reentrant=False,
                )
            else:
                x = block(x, T, h, w)

        x = self.norm(x)                          # (B, N, dim)

        # --- Spatial pooling: collapse h, w -> per-frame features ---
        # Reshape back to (B, T, h, w, dim)
        x = x.view(B, T, h, w, D)
        # Average pool over spatial dims -> (B, T, dim)
        x = x.mean(dim=(2, 3))                    # (B, T, dim)

        # --- Optional IMU cross-attention fusion ---
        if self.use_imu and imu is not None:
            imu_feat = self.imu_encoder(imu)       # (B, T, imu_dim)
            x = self.imu_cross_attn(x, imu_feat)  # (B, T, dim)

        # --- ECG Decoder ---
        ecg = self.decoder(x)                      # (B, T_ecg)

        return ecg


# ---------------------------------------------------------------------------
#  Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test with a minimal config
    cfg = {
        "model": {
            "type": "physformer_ecg",
            "dim": 96,
            "depth": 2,
            "heads": 4,
            "theta": 0.6,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "in_channels": 3,
            "decoder_channels": [128, 64, 32],
            "gradient_checkpointing": False,
        },
        "data": {
            "window_seconds": 10,
            "video_fps": 30,
            "ecg_sr": 250,
            "img_height": 128,
            "img_width": 128,
            "use_imu": False,
        },
    }

    model = PhysFormerECG(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"PhysFormerECG (depth=2, dim=96) params: {total_params:,}")

    # Forward pass with dummy data
    B = 2
    video = torch.randn(B, 300, 3, 128, 128)
    ecg = model(video)
    print(f"Input:  video {video.shape}")
    print(f"Output: ecg   {ecg.shape}")
    assert ecg.shape == (B, 2500), f"Expected (2, 2500), got {ecg.shape}"

    # Test with IMU
    cfg["data"]["use_imu"] = True
    cfg["model"]["imu_dim"] = 64
    cfg["data"]["imu_sr"] = 100
    model_imu = PhysFormerECG(cfg)
    imu = torch.randn(B, 1000, 6)
    ecg_imu = model_imu(video, imu)
    print(f"\nWith IMU:")
    print(f"Input:  video {video.shape}, imu {imu.shape}")
    print(f"Output: ecg   {ecg_imu.shape}")
    assert ecg_imu.shape == (B, 2500)

    # Test with smaller resolution (64x64)
    cfg["data"]["img_height"] = 64
    cfg["data"]["img_width"] = 64
    cfg["data"]["use_imu"] = False
    model_64 = PhysFormerECG(cfg)
    video_64 = torch.randn(B, 300, 3, 64, 64)
    ecg_64 = model_64(video_64)
    print(f"\n64x64 resolution:")
    print(f"Input:  video {video_64.shape}")
    print(f"Output: ecg   {ecg_64.shape}")
    assert ecg_64.shape == (B, 2500)

    print("\nAll tests passed.")
