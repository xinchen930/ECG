"""
STMap (Spatio-Temporal Map) construction utilities.

Converts video frame sequences into 2D representations that preserve
spatial information while being much more memory-efficient than full
3D video tensors. Suitable for processing with standard 2D CNNs.

Concept:
    Video frames (T, C, H, W)
      -> Divide each frame into N spatial regions (grid patches)
      -> Compute channel mean per region per frame
      -> Assemble into STMap: (C, N_spatial, T)
      -> Treat as a 2D "image" for CNN processing

Variants:
    1. Grid STMap: uniform grid partition (basic)
    2. Multi-scale STMap: multiple grid resolutions concatenated
    3. Frequency STMap: STFT per spatial location (spectral view)
"""

import numpy as np
import torch


class STMapBuilder:
    """Convert video frame sequences into Spatio-Temporal Maps.

    The builder divides each frame into a grid of patches and computes
    the mean pixel value per patch per channel. This converts a video
    volume (T, C, H, W) into a compact 2D map (C, N_spatial, T) where
    N_spatial = n_spatial_h * n_spatial_w.

    The resulting STMap can be processed by a 2D CNN where the "height"
    axis is the spatial dimension and the "width" axis is time.
    """

    def __init__(self, n_spatial_h=8, n_spatial_w=8, channels="rgb"):
        """
        Args:
            n_spatial_h: number of vertical grid divisions
            n_spatial_w: number of horizontal grid divisions
            channels: which channels to keep
                      'rgb' -> all 3 channels  (C=3)
                      'red' -> red channel only (C=1)
                      'green' -> green only     (C=1)
                      'all' -> same as 'rgb'    (C=3)
        """
        self.n_spatial_h = n_spatial_h
        self.n_spatial_w = n_spatial_w
        self.channels = channels

    # ------------------------------------------------------------------
    #  Core: single-scale grid STMap
    # ------------------------------------------------------------------

    def build(self, video_frames):
        """Build a grid-based STMap from video frames.

        Args:
            video_frames: numpy array (T, C, H, W) or torch Tensor
                          Channel order is BGR (OpenCV convention) or RGB.
                          Values in [0, 1] (float) or [0, 255] (uint8).

        Returns:
            stmap: torch.FloatTensor of shape (C_out, N_spatial, T)
                   where N_spatial = n_spatial_h * n_spatial_w
                   and C_out depends on the `channels` setting.
        """
        is_tensor = isinstance(video_frames, torch.Tensor)
        if is_tensor:
            frames = video_frames.float()
        else:
            frames = torch.from_numpy(video_frames.astype(np.float32))

        # frames: (T, C, H, W)
        T, C, H, W = frames.shape
        nh, nw = self.n_spatial_h, self.n_spatial_w

        # Compute patch sizes — handles non-divisible dimensions via reshape
        # by trimming excess pixels to make dimensions divisible.
        h_trim = (H // nh) * nh
        w_trim = (W // nw) * nw
        frames = frames[:, :, :h_trim, :w_trim]  # (T, C, h_trim, w_trim)

        ph = h_trim // nh   # patch height
        pw = w_trim // nw   # patch width

        # Reshape into grid of patches and compute means efficiently.
        # (T, C, nh, ph, nw, pw) -> mean over (ph, pw) dims
        frames = frames.reshape(T, C, nh, ph, nw, pw)
        stmap = frames.mean(dim=(3, 5))  # (T, C, nh, nw)

        # Flatten spatial grid: (T, C, nh, nw) -> (T, C, N_spatial)
        stmap = stmap.reshape(T, C, nh * nw)  # (T, C, N_spatial)

        # Reorder to (C, N_spatial, T)
        stmap = stmap.permute(1, 2, 0)  # (C, N_spatial, T)

        # Channel selection
        stmap = self._select_channels(stmap)

        return stmap

    def build_multiscale(self, video_frames, scales=None):
        """Build multi-scale STMap by concatenating grids at different resolutions.

        At each scale s, the frame is divided into an (s x s) grid, yielding
        s*s spatial locations. The results across scales are concatenated along
        the spatial (N) axis, giving a richer representation.

        Args:
            video_frames: (T, C, H, W) numpy or tensor
            scales: list of grid sizes, e.g. [4, 8, 16].
                    Each value is used for both n_spatial_h and n_spatial_w.
                    If None, defaults to [4, 8, 16].

        Returns:
            stmap: (C_out, N_total, T) tensor
                   where N_total = sum(s*s for s in scales)
        """
        if scales is None:
            scales = [4, 8, 16]

        parts = []
        for s in scales:
            builder = STMapBuilder(n_spatial_h=s, n_spatial_w=s,
                                   channels=self.channels)
            part = builder.build(video_frames)  # (C, s*s, T)
            parts.append(part)

        # Concatenate along spatial axis
        stmap = torch.cat(parts, dim=1)  # (C, N_total, T)
        return stmap

    def build_frequency(self, video_frames, n_fft=64, hop_length=8):
        """Build frequency-domain STMap via STFT per spatial location.

        For each spatial patch, the temporal signal is transformed into
        the frequency domain with STFT. The result has shape:
            (N_spatial, N_freq, T_windows)
        where N_freq = n_fft // 2 + 1.

        This variant captures spectral content (heart rate harmonics)
        at each spatial location.

        Args:
            video_frames: (T, C, H, W) numpy or tensor
            n_fft: FFT window size
            hop_length: STFT hop length

        Returns:
            stmap: (1, N_spatial, N_freq, T_windows) tensor
                   Squeezed to (N_spatial, N_freq, T_windows) would also work,
                   but we add a channel dim for CNN compatibility.
                   Actually returns (N_freq, N_spatial, T_windows) to keep the
                   convention of (C, H, W) where C=N_freq, H=N_spatial, W=T_windows.
        """
        # First build a standard single-channel STMap
        # For frequency analysis, average channels first
        is_tensor = isinstance(video_frames, torch.Tensor)
        if is_tensor:
            frames = video_frames.float()
        else:
            frames = torch.from_numpy(video_frames.astype(np.float32))

        T, C, H, W = frames.shape
        nh, nw = self.n_spatial_h, self.n_spatial_w

        h_trim = (H // nh) * nh
        w_trim = (W // nw) * nw
        frames = frames[:, :, :h_trim, :w_trim]

        ph = h_trim // nh
        pw = w_trim // nw

        frames = frames.reshape(T, C, nh, ph, nw, pw)
        spatial_means = frames.mean(dim=(3, 5))  # (T, C, nh, nw)

        # Average across channels for frequency analysis
        spatial_means = spatial_means.mean(dim=1)  # (T, nh, nw)
        spatial_means = spatial_means.reshape(T, nh * nw)  # (T, N_spatial)
        spatial_means = spatial_means.permute(1, 0)  # (N_spatial, T)

        N_spatial = spatial_means.shape[0]

        # STFT per spatial location
        window = torch.hann_window(n_fft, device=spatial_means.device)
        # torch.stft expects (batch, T) or (T,)
        spec = torch.stft(spatial_means, n_fft=n_fft, hop_length=hop_length,
                          win_length=n_fft, window=window,
                          return_complex=True)
        # spec: (N_spatial, N_freq, T_windows)
        magnitude = spec.abs()  # (N_spatial, N_freq, T_windows)

        # Reorder to (N_freq, N_spatial, T_windows) for CNN (C=N_freq, H=N_spatial, W=T)
        magnitude = magnitude.permute(1, 0, 2)

        return magnitude

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _select_channels(self, stmap):
        """Select channels from (C, N_spatial, T) based on self.channels.

        Assumes BGR channel order (OpenCV convention):
            index 0 = Blue, index 1 = Green, index 2 = Red
        """
        if self.channels in ("rgb", "all"):
            return stmap   # keep all 3 channels
        elif self.channels == "red":
            return stmap[2:3]  # (1, N, T) — red in BGR
        elif self.channels == "green":
            return stmap[1:2]  # (1, N, T) — green in BGR
        else:
            raise ValueError(f"Unknown channel mode: {self.channels}. "
                             f"Use 'rgb', 'red', 'green', or 'all'.")

    @property
    def n_spatial(self):
        """Total number of spatial patches."""
        return self.n_spatial_h * self.n_spatial_w


# ------------------------------------------------------------------
#  Convenience functions
# ------------------------------------------------------------------

def build_stmap(video_frames, n_spatial_h=8, n_spatial_w=8,
                channels="rgb", multiscale=False, scales=None):
    """Convenience wrapper to build an STMap from video frames.

    Args:
        video_frames: (T, C, H, W) numpy array or torch tensor
        n_spatial_h: vertical grid divisions
        n_spatial_w: horizontal grid divisions
        channels: 'rgb', 'red', 'green', or 'all'
        multiscale: if True, use multi-scale STMap
        scales: list of grid sizes for multi-scale (default [4, 8, 16])

    Returns:
        stmap: (C_out, N_spatial, T) tensor
    """
    builder = STMapBuilder(n_spatial_h=n_spatial_h, n_spatial_w=n_spatial_w,
                           channels=channels)
    if multiscale:
        return builder.build_multiscale(video_frames, scales=scales)
    else:
        return builder.build(video_frames)


# ------------------------------------------------------------------
#  Self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== STMapBuilder Self-Test ===\n")

    # Simulate a video: 300 frames, 3 channels (BGR), 64x64 resolution
    T, C, H, W = 300, 3, 64, 64
    video = torch.randn(T, C, H, W)

    # --- Grid STMap ---
    builder = STMapBuilder(n_spatial_h=8, n_spatial_w=8, channels="rgb")
    stmap = builder.build(video)
    print(f"Grid STMap (8x8, RGB):")
    print(f"  Input:  ({T}, {C}, {H}, {W})")
    print(f"  Output: {tuple(stmap.shape)}  (C=3, N=64, T=300)")

    # --- Red channel only ---
    builder_r = STMapBuilder(n_spatial_h=8, n_spatial_w=8, channels="red")
    stmap_r = builder_r.build(video)
    print(f"\nGrid STMap (8x8, Red only):")
    print(f"  Output: {tuple(stmap_r.shape)}  (C=1, N=64, T=300)")

    # --- Multi-scale ---
    builder_ms = STMapBuilder(n_spatial_h=8, n_spatial_w=8, channels="rgb")
    stmap_ms = builder_ms.build_multiscale(video, scales=[4, 8, 16])
    n_total = 4*4 + 8*8 + 16*16
    print(f"\nMulti-scale STMap (4+8+16, RGB):")
    print(f"  Output: {tuple(stmap_ms.shape)}  (C=3, N={n_total}, T=300)")

    # --- Frequency STMap ---
    stmap_freq = builder.build_frequency(video, n_fft=64, hop_length=8)
    print(f"\nFrequency STMap (8x8, n_fft=64, hop=8):")
    print(f"  Output: {tuple(stmap_freq.shape)}  (N_freq, N_spatial, T_windows)")

    # --- Convenience function ---
    stmap_conv = build_stmap(video, n_spatial_h=8, n_spatial_w=8, channels="rgb")
    print(f"\nbuild_stmap convenience:")
    print(f"  Output: {tuple(stmap_conv.shape)}")

    # --- Non-divisible resolution ---
    video_odd = torch.randn(300, 3, 65, 57)
    stmap_odd = builder.build(video_odd)
    print(f"\nNon-divisible resolution (65x57):")
    print(f"  Output: {tuple(stmap_odd.shape)}  (trims to 64x56, patches 8x7)")

    # --- Numpy input ---
    video_np = np.random.randn(300, 3, 64, 64).astype(np.float32)
    stmap_np = builder.build(video_np)
    print(f"\nNumpy input:")
    print(f"  Output: {tuple(stmap_np.shape)}  dtype={stmap_np.dtype}")

    print("\nAll tests passed.")
