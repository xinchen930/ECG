"""
CardioGAN - Attentive GAN with Dual Discriminators for PPG to ECG Synthesis

Paper: "CardioGAN: Attentive Generative Adversarial Network with Dual Discriminators
        for Synthesis of ECG from PPG" (AAAI 2021)
Authors: Pritam Sarkar, Ali Etemad
Source: https://github.com/pritamqu/ppg2ecg-cardiogan

Architecture:
- Generator: Attention U-Net (encoder-decoder with self-gated attention on skip connections)
- Discriminator_T: Time-domain discriminator (operates on raw signal)
- Discriminator_F: Frequency-domain discriminator (operates on FFT)

Input: [batch, 1, 512] PPG signal @ 128 Hz (4 seconds), normalized to [-1, 1]
Output: [batch, 1, 512] ECG signal

Key Differences from PPG2ECG:
- GAN-based (adversarial training)
- Self-gated attention on skip connections (vs multi-head attention)
- Dual discriminators (time + frequency domain)
- 128 Hz sampling rate, 512 samples (vs 125 Hz, 256 samples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Building Blocks
# ============================================================================

class ConvBlock(nn.Module):
    """
    Convolutional block: Conv1d -> BatchNorm -> LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvTransposeBlock(nn.Module):
    """
    Transposed convolutional block: ConvTranspose1d -> BatchNorm -> LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AttentionGate(nn.Module):
    """
    Self-gated attention mechanism for skip connections.

    Learns to weight features from encoder based on decoder context.

    gate = sigmoid(W_g * g + W_x * x + b)
    output = x * gate

    where g is decoder features, x is encoder features (skip connection)
    """
    def __init__(self, gate_channels, skip_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = skip_channels // 2

        # Gate signal from decoder
        self.W_g = nn.Conv1d(gate_channels, inter_channels, kernel_size=1, bias=True)
        # Skip connection features
        self.W_x = nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=True)
        # Attention coefficient
        self.psi = nn.Conv1d(inter_channels, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        """
        Args:
            g: Gate signal from decoder [batch, gate_channels, T_g]
            x: Skip connection from encoder [batch, skip_channels, T_x]

        Returns:
            Attention-weighted skip connection [batch, skip_channels, T_x]
        """
        # Interpolate g to match x's temporal dimension if needed
        if g.shape[2] != x.shape[2]:
            g = F.interpolate(g, size=x.shape[2], mode='linear', align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi


# ============================================================================
# Generator (Attention U-Net)
# ============================================================================

class Generator(nn.Module):
    """
    Attention U-Net Generator for PPG → ECG synthesis.

    Architecture:
        Encoder: 5 downsampling blocks (512 → 256 → 128 → 64 → 32 → 16)
        Decoder: 5 upsampling blocks with attention-gated skip connections
        Output: Tanh activation for [-1, 1] range

    Channels: 1 → 64 → 128 → 256 → 512 → 1024 (bottleneck)
    """
    def __init__(self, input_channels=1, base_filters=64):
        super().__init__()
        self.input_channels = input_channels
        bf = base_filters

        # Encoder
        self.enc1 = ConvBlock(input_channels, bf, kernel_size=7, stride=2, padding=3)      # 512 → 256
        self.enc2 = ConvBlock(bf, bf*2, kernel_size=5, stride=2, padding=2)               # 256 → 128
        self.enc3 = ConvBlock(bf*2, bf*4, kernel_size=5, stride=2, padding=2)             # 128 → 64
        self.enc4 = ConvBlock(bf*4, bf*8, kernel_size=3, stride=2, padding=1)             # 64 → 32
        self.enc5 = ConvBlock(bf*8, bf*16, kernel_size=3, stride=2, padding=1)            # 32 → 16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bf*16, bf*16, kernel_size=3, padding=1),
            nn.BatchNorm1d(bf*16),
            nn.LeakyReLU(0.2),
        )

        # Decoder with attention gates
        self.dec5 = ConvTransposeBlock(bf*16, bf*8, kernel_size=3, stride=2, padding=1, output_padding=1)  # 16 → 32
        self.attn5 = AttentionGate(bf*8, bf*8)

        self.dec4 = ConvTransposeBlock(bf*16, bf*4, kernel_size=3, stride=2, padding=1, output_padding=1)  # 32 → 64
        self.attn4 = AttentionGate(bf*4, bf*4)

        self.dec3 = ConvTransposeBlock(bf*8, bf*2, kernel_size=5, stride=2, padding=2, output_padding=1)   # 64 → 128
        self.attn3 = AttentionGate(bf*2, bf*2)

        self.dec2 = ConvTransposeBlock(bf*4, bf, kernel_size=5, stride=2, padding=2, output_padding=1)     # 128 → 256
        self.attn2 = AttentionGate(bf, bf)

        self.dec1 = ConvTransposeBlock(bf*2, bf, kernel_size=7, stride=2, padding=3, output_padding=1)     # 256 → 512
        self.attn1 = AttentionGate(bf, input_channels)  # Note: skip from input

        # Final output
        self.final = nn.Sequential(
            nn.Conv1d(bf + input_channels, input_channels, kernel_size=7, padding=3),
            nn.Tanh(),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Gaussian (stddev=0.02) as in original."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Input PPG [batch, 1, 512]

        Returns:
            Generated ECG [batch, 1, 512]
        """
        # Store input for final skip connection
        x_input = x

        # Encoder
        e1 = self.enc1(x)       # [B, 64, 256]
        e2 = self.enc2(e1)      # [B, 128, 128]
        e3 = self.enc3(e2)      # [B, 256, 64]
        e4 = self.enc4(e3)      # [B, 512, 32]
        e5 = self.enc5(e4)      # [B, 1024, 16]

        # Bottleneck
        b = self.bottleneck(e5)  # [B, 1024, 16]

        # Decoder with attention-gated skip connections
        d5 = self.dec5(b)                       # [B, 512, 32]
        a5 = self.attn5(d5, e4)                 # Attention on e4
        d5 = torch.cat([d5, a5], dim=1)         # [B, 1024, 32]

        d4 = self.dec4(d5)                      # [B, 256, 64]
        a4 = self.attn4(d4, e3)
        d4 = torch.cat([d4, a4], dim=1)         # [B, 512, 64]

        d3 = self.dec3(d4)                      # [B, 128, 128]
        a3 = self.attn3(d3, e2)
        d3 = torch.cat([d3, a3], dim=1)         # [B, 256, 128]

        d2 = self.dec2(d3)                      # [B, 64, 256]
        a2 = self.attn2(d2, e1)
        d2 = torch.cat([d2, a2], dim=1)         # [B, 128, 256]

        d1 = self.dec1(d2)                      # [B, 64, 512]
        a1 = self.attn1(d1, x_input)
        d1 = torch.cat([d1, a1], dim=1)         # [B, 65, 512]

        # Final output
        out = self.final(d1)                    # [B, 1, 512]

        return out


# ============================================================================
# Discriminators
# ============================================================================

class DiscriminatorTime(nn.Module):
    """
    Time-domain discriminator.

    Operates on raw signal waveform to verify temporal characteristics.

    Architecture: Conv blocks → Flatten → FC → Sigmoid
    """
    def __init__(self, input_channels=1, base_filters=64):
        super().__init__()
        bf = base_filters

        self.model = nn.Sequential(
            # Input: [B, 1, 512]
            nn.Conv1d(input_channels, bf, kernel_size=7, stride=2, padding=3),     # 256
            nn.LeakyReLU(0.2),

            nn.Conv1d(bf, bf*2, kernel_size=5, stride=2, padding=2),               # 128
            nn.BatchNorm1d(bf*2),
            nn.LeakyReLU(0.2),

            nn.Conv1d(bf*2, bf*4, kernel_size=5, stride=2, padding=2),             # 64
            nn.BatchNorm1d(bf*4),
            nn.LeakyReLU(0.2),

            nn.Conv1d(bf*4, bf*8, kernel_size=3, stride=2, padding=1),             # 32
            nn.BatchNorm1d(bf*8),
            nn.LeakyReLU(0.2),

            nn.Conv1d(bf*8, bf*16, kernel_size=3, stride=2, padding=1),            # 16
            nn.BatchNorm1d(bf*16),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(bf*16 * 16, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Signal [batch, 1, 512]

        Returns:
            Validity score [batch, 1]
        """
        return self.model(x)


class DiscriminatorFreq(nn.Module):
    """
    Frequency-domain discriminator.

    Operates on FFT magnitude spectrum to verify spectral characteristics.

    Architecture: FFT → Conv blocks → FC → Sigmoid
    """
    def __init__(self, input_channels=1, base_filters=64):
        super().__init__()
        bf = base_filters

        # FFT will give us 257 frequency bins for 512 samples (n_fft/2 + 1)
        freq_bins = 257

        self.model = nn.Sequential(
            # Input: [B, 1, 257] (magnitude spectrum)
            nn.Conv1d(input_channels, bf, kernel_size=7, stride=2, padding=3),     # 129
            nn.LeakyReLU(0.2),

            nn.Conv1d(bf, bf*2, kernel_size=5, stride=2, padding=2),               # 65
            nn.BatchNorm1d(bf*2),
            nn.LeakyReLU(0.2),

            nn.Conv1d(bf*2, bf*4, kernel_size=5, stride=2, padding=2),             # 33
            nn.BatchNorm1d(bf*4),
            nn.LeakyReLU(0.2),

            nn.Conv1d(bf*4, bf*8, kernel_size=3, stride=2, padding=1),             # 17
            nn.BatchNorm1d(bf*8),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(bf*8 * 17, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_fft(self, x):
        """Compute magnitude spectrum."""
        # x: [batch, 1, 512]
        x_squeezed = x.squeeze(1)  # [batch, 512]
        fft = torch.fft.rfft(x_squeezed, dim=-1)  # [batch, 257]
        magnitude = torch.abs(fft)  # [batch, 257]
        return magnitude.unsqueeze(1)  # [batch, 1, 257]

    def forward(self, x):
        """
        Args:
            x: Signal [batch, 1, 512]

        Returns:
            Validity score [batch, 1]
        """
        x_freq = self.compute_fft(x)
        return self.model(x_freq)


# ============================================================================
# Loss Functions
# ============================================================================

class CardioGANLoss(nn.Module):
    """
    Combined loss for CardioGAN training.

    L_total = α * L_adv + β * L_cycle + λ * L_recon

    Default weights from paper:
        α = 3.0 (adversarial)
        β = 1.0 (cycle consistency)
        λ = 30.0 (reconstruction)
    """
    def __init__(self, alpha=3.0, beta=1.0, lambda_recon=30.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_recon = lambda_recon
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    def adversarial_loss(self, pred, target_is_real):
        """Binary cross-entropy adversarial loss."""
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.bce(pred, target)

    def reconstruction_loss(self, pred, target):
        """L1 reconstruction loss."""
        return self.l1(pred, target)

    def generator_loss(self, d_t_fake, d_f_fake, pred_ecg, real_ecg):
        """
        Generator loss = adversarial (fool both discriminators) + reconstruction.

        Args:
            d_t_fake: Time discriminator output for generated ECG
            d_f_fake: Frequency discriminator output for generated ECG
            pred_ecg: Generated ECG
            real_ecg: Ground truth ECG

        Returns:
            Total generator loss
        """
        # Adversarial: fool discriminators (target = real)
        loss_adv_t = self.adversarial_loss(d_t_fake, target_is_real=True)
        loss_adv_f = self.adversarial_loss(d_f_fake, target_is_real=True)
        loss_adv = loss_adv_t + loss_adv_f

        # Reconstruction
        loss_recon = self.reconstruction_loss(pred_ecg, real_ecg)

        total = self.alpha * loss_adv + self.lambda_recon * loss_recon
        return total, {
            'adv_t': loss_adv_t.item(),
            'adv_f': loss_adv_f.item(),
            'recon': loss_recon.item(),
        }

    def discriminator_loss(self, d_real, d_fake):
        """
        Discriminator loss = real should be 1, fake should be 0.

        Args:
            d_real: Discriminator output for real signal
            d_fake: Discriminator output for fake signal

        Returns:
            Discriminator loss
        """
        loss_real = self.adversarial_loss(d_real, target_is_real=True)
        loss_fake = self.adversarial_loss(d_fake, target_is_real=False)
        return (loss_real + loss_fake) * 0.5


# ============================================================================
# Full Model
# ============================================================================

class CardioGAN(nn.Module):
    """
    Complete CardioGAN model with Generator and Dual Discriminators.

    Usage:
        model = CardioGAN()
        optimizer_G, optimizer_D = model.get_optimizers()

        # Training step
        loss_D, loss_G = model.train_step(ppg, ecg, optimizer_G, optimizer_D)

        # Inference
        pred_ecg = model.generate(ppg)
    """
    def __init__(self, base_filters=64, lr=0.0002, beta1=0.5, beta2=0.999,
                 alpha=3.0, beta=1.0, lambda_recon=30.0):
        super().__init__()
        self.generator = Generator(input_channels=1, base_filters=base_filters)
        self.discriminator_t = DiscriminatorTime(input_channels=1, base_filters=base_filters)
        self.discriminator_f = DiscriminatorFreq(input_channels=1, base_filters=base_filters)

        self.loss_fn = CardioGANLoss(alpha=alpha, beta=beta, lambda_recon=lambda_recon)

        # Store hyperparameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def generate(self, ppg):
        """Generate ECG from PPG."""
        return self.generator(ppg)

    def forward(self, ppg):
        """Forward pass (generation only)."""
        return {'output': self.generate(ppg)}

    def get_optimizers(self):
        """Get separate optimizers for generator and discriminators."""
        optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )
        optimizer_D = torch.optim.Adam(
            list(self.discriminator_t.parameters()) + list(self.discriminator_f.parameters()),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )
        return optimizer_G, optimizer_D


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing CardioGAN model...")

    # Create model
    model = CardioGAN()

    # Count parameters
    n_gen = count_parameters(model.generator)
    n_disc_t = count_parameters(model.discriminator_t)
    n_disc_f = count_parameters(model.discriminator_f)
    print(f"Generator parameters: {n_gen:,}")
    print(f"Discriminator_T parameters: {n_disc_t:,}")
    print(f"Discriminator_F parameters: {n_disc_f:,}")
    print(f"Total parameters: {n_gen + n_disc_t + n_disc_f:,}")

    # Test forward pass
    ppg = torch.randn(4, 1, 512)  # [batch=4, channels=1, seq_len=512]
    ecg = torch.randn(4, 1, 512)

    print(f"\nInput PPG shape: {ppg.shape}")

    # Generator
    pred_ecg = model.generate(ppg)
    print(f"Generated ECG shape: {pred_ecg.shape}")

    # Discriminators
    d_t_real = model.discriminator_t(ecg)
    d_t_fake = model.discriminator_t(pred_ecg)
    print(f"Discriminator_T output shape: {d_t_real.shape}")

    d_f_real = model.discriminator_f(ecg)
    d_f_fake = model.discriminator_f(pred_ecg)
    print(f"Discriminator_F output shape: {d_f_real.shape}")

    # Loss computation
    loss_G, loss_details = model.loss_fn.generator_loss(d_t_fake, d_f_fake, pred_ecg, ecg)
    loss_D_t = model.loss_fn.discriminator_loss(d_t_real, d_t_fake)
    loss_D_f = model.loss_fn.discriminator_loss(d_f_real, d_f_fake)

    print(f"\nGenerator loss: {loss_G.item():.4f}")
    print(f"  Adversarial_T: {loss_details['adv_t']:.4f}")
    print(f"  Adversarial_F: {loss_details['adv_f']:.4f}")
    print(f"  Reconstruction: {loss_details['recon']:.4f}")
    print(f"Discriminator_T loss: {loss_D_t.item():.4f}")
    print(f"Discriminator_F loss: {loss_D_f.item():.4f}")

    print("\nAll tests passed!")
