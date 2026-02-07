"""
PPG2ECG Model - Faithful port from james77777778/ppg2ecg-pytorch

Paper: "Reconstructing QRS Complex from PPG by Transformed Attentional Neural Networks"
       IEEE Sensors Journal, 2020

Architecture:
- Encoder: Conv1d (1→32→64→128→256→512) with stride-2 and PReLU
- Decoder: ConvTranspose1d mirror of encoder, Tanh output
- Optional: Spatial Transformer Network (STN) for offset calibration
- Optional: Attention mechanism for QRS focus

Input: [batch, 1, 256] PPG signal (normalized to [-1, 1])
Output: [batch, 1, 256] ECG signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Flatten layer for FC input."""
    def forward(self, x):
        return x.view(x.size(0), -1)


class GaussianNoise(nn.Module):
    """Add Gaussian noise during training for regularization."""
    def __init__(self, sigma=0.1):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma * x.detach()
            return x + noise
        return x


class PPG2ECG(nn.Module):
    """
    Main PPG to ECG conversion model.

    Args:
        input_size: Length of input sequence (256 for BIDMC at 125Hz)
        use_stn: Enable Spatial Transformer Network for offset calibration
        use_attention: Enable attention mechanism for QRS focus
    """
    def __init__(self, input_size=256, use_stn=False, use_attention=False):
        super(PPG2ECG, self).__init__()
        self.input_size = input_size
        self.use_stn = use_stn
        self.use_attention = use_attention

        # Encoder: progressively downsample and increase channels
        # 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),    # 256 -> 128
            nn.PReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),   # 128 -> 64
            nn.PReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # 64 -> 32
            nn.PReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1), # 32 -> 16
            nn.PReLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1), # 16 -> 8
            nn.PReLU(),
        )

        # Decoder: progressively upsample and decrease channels
        # 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8 -> 16
            nn.PReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.PReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),   # 32 -> 64
            nn.PReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),    # 64 -> 128
            nn.PReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),     # 128 -> 256
            nn.Tanh(),
        )

        # Optional: Spatial Transformer Network for offset calibration
        if use_stn:
            self.stn_localization = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3),
                nn.MaxPool1d(2),
                nn.ReLU(),
                nn.Conv1d(8, 10, kernel_size=5, stride=2, padding=2),
                nn.MaxPool1d(2),
                nn.ReLU(),
            )
            # Calculate flattened size for FC layer
            # input_size=256 -> 128 -> 64 -> 32 -> 16 -> 8
            stn_size = input_size // 32  # 8 for input_size=256
            self.stn_fc = nn.Sequential(
                Flatten(),
                nn.Linear(10 * stn_size, 32),
                nn.ReLU(),
                nn.Linear(32, 4),  # 4 params for 1D affine: [scale_x, 0, shift_x, 0, scale_y, shift_y] -> simplified to 4
            )
            # Initialize to identity transform
            self.stn_fc[3].weight.data.zero_()
            self.stn_fc[3].bias.data.copy_(torch.tensor([1, 0, 0, 1], dtype=torch.float))

        # Optional: Attention mechanism
        if use_attention:
            # Attention on encoder output (512 channels, 8 time steps)
            self.attention = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.Softmax(dim=-1),
            )

    def stn(self, x):
        """Apply Spatial Transformer Network."""
        # x: [batch, 1, 256]
        xs = self.stn_localization(x)  # [batch, 10, 8]
        theta = self.stn_fc(xs)  # [batch, 4]

        # Convert to 2D affine matrix for grid_sample
        # We treat 1D signal as 2D with height=1
        theta = theta.view(-1, 2, 2)  # [batch, 2, 2]

        # Expand to [batch, 2, 3] affine matrix
        batch_size = x.size(0)
        affine = torch.zeros(batch_size, 2, 3, device=x.device, dtype=x.dtype)
        affine[:, 0, 0] = theta[:, 0, 0]  # scale_x
        affine[:, 0, 2] = theta[:, 0, 1]  # shift_x
        affine[:, 1, 1] = theta[:, 1, 0]  # scale_y
        affine[:, 1, 2] = theta[:, 1, 1]  # shift_y

        # Create grid and sample
        # Treat as [batch, 1, 1, 256] -> [batch, 1, 1, 256]
        x_2d = x.unsqueeze(2)  # [batch, 1, 1, 256]
        grid = F.affine_grid(affine, x_2d.size(), align_corners=False)
        x_transformed = F.grid_sample(x_2d, grid, align_corners=False)
        return x_transformed.squeeze(2)  # [batch, 1, 256]

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input PPG signal [batch, 1, 256]

        Returns:
            dict with 'output' (ECG) and optionally 'output_stn' (STN intermediate)
        """
        result = {}

        # Apply STN if enabled
        if self.use_stn:
            x_stn = self.stn(x)
            result['output_stn'] = x_stn
            x = x_stn

        # Encode
        features = self.encoder(x)  # [batch, 512, 8]

        # Apply attention if enabled
        if self.use_attention:
            # features: [batch, 512, 8]
            # Compute attention over time dimension
            attn_input = features.permute(0, 2, 1)  # [batch, 8, 512]
            attn_weights = self.attention(attn_input)  # [batch, 8, 512]
            attn_weights = attn_weights.permute(0, 2, 1)  # [batch, 512, 8]
            features = features * attn_weights

        # Decode
        output = self.decoder(features)  # [batch, 1, 256]
        result['output'] = output

        return result


class PPG2ECG_LSTM(nn.Module):
    """
    LSTM baseline model for PPG to ECG conversion.

    Args:
        input_size: Length of input sequence
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
    """
    def __init__(self, input_size=256, hidden_size=200, num_layers=2, dropout=0.1):
        super(PPG2ECG_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Bidirectional LSTM encoder
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Bidirectional LSTM decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input PPG signal [batch, 1, input_size]

        Returns:
            dict with 'output' (ECG)
        """
        # x: [batch, 1, input_size] -> [batch, input_size, 1]
        x = x.permute(0, 2, 1)

        # Encode
        enc_out, _ = self.encoder(x)  # [batch, input_size, hidden*2]

        # Decode
        dec_out, _ = self.decoder(enc_out)  # [batch, input_size, hidden*2]

        # Project to output
        output = self.fc(dec_out)  # [batch, input_size, 1]
        output = self.tanh(output)

        # Reshape: [batch, input_size, 1] -> [batch, 1, input_size]
        output = output.permute(0, 2, 1)

        return {'output': output}


# ============================================================================
# Loss Functions
# ============================================================================

class QRSLoss(nn.Module):
    """
    QRS Complex-Enhanced Loss.

    Applies higher weight to R-peak regions to improve QRS reconstruction.

    Loss = L1(pred * (1 + beta * exp_rpeaks), target * (1 + beta * exp_rpeaks))

    Args:
        beta: Weight multiplier for R-peak regions (default: 5)
    """
    def __init__(self, beta=5):
        super(QRSLoss, self).__init__()
        self.beta = beta

    def forward(self, pred, target, exp_rpeaks):
        """
        Compute QRS-weighted L1 loss.

        Args:
            pred: Predicted ECG [batch, 1, seq_len]
            target: Ground truth ECG [batch, 1, seq_len]
            exp_rpeaks: Gaussian-expanded R-peak locations [batch, 1, seq_len]
                       Values in [0, 1], peaks at R-peak locations

        Returns:
            Weighted L1 loss
        """
        weight = 1 + self.beta * exp_rpeaks
        return F.l1_loss(pred * weight, target * weight)


class CombinedLoss(nn.Module):
    """
    Combined loss for training: L1 + optional QRS weighting.

    When exp_rpeaks is provided, uses QRS-enhanced loss.
    Otherwise, falls back to standard L1 loss.
    """
    def __init__(self, beta=5, use_qrs_loss=True):
        super(CombinedLoss, self).__init__()
        self.use_qrs_loss = use_qrs_loss
        self.qrs_loss = QRSLoss(beta=beta)

    def forward(self, pred, target, exp_rpeaks=None):
        """
        Compute loss.

        Args:
            pred: Predicted ECG [batch, 1, seq_len]
            target: Ground truth ECG [batch, 1, seq_len]
            exp_rpeaks: Optional Gaussian-expanded R-peak locations

        Returns:
            Loss value
        """
        if self.use_qrs_loss and exp_rpeaks is not None:
            return self.qrs_loss(pred, target, exp_rpeaks)
        else:
            return F.l1_loss(pred, target)


# ============================================================================
# Utility Functions
# ============================================================================

def create_model(config):
    """
    Factory function to create model from config.

    Args:
        config: Dict with model configuration

    Returns:
        Model instance
    """
    model_type = config.get('type', 'ppg2ecg')
    input_size = config.get('input_size', 256)

    if model_type == 'ppg2ecg':
        return PPG2ECG(
            input_size=input_size,
            use_stn=config.get('use_stn', True),
            use_attention=config.get('use_attention', True),
        )
    elif model_type == 'lstm':
        return PPG2ECG_LSTM(
            input_size=input_size,
            hidden_size=config.get('hidden_size', 200),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing PPG2ECG model...")

    # Test main model
    model = PPG2ECG(input_size=256, use_stn=True, use_attention=True)
    print(f"PPG2ECG parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 1, 256)  # [batch=4, channels=1, seq_len=256]
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output['output'].shape}")
    if 'output_stn' in output:
        print(f"STN output shape: {output['output_stn'].shape}")

    # Test LSTM baseline
    print("\nTesting PPG2ECG_LSTM model...")
    model_lstm = PPG2ECG_LSTM(input_size=256)
    print(f"PPG2ECG_LSTM parameters: {count_parameters(model_lstm):,}")

    output_lstm = model_lstm(x)
    print(f"LSTM output shape: {output_lstm['output'].shape}")

    # Test loss
    print("\nTesting QRSLoss...")
    loss_fn = CombinedLoss(beta=5, use_qrs_loss=True)
    target = torch.randn(4, 1, 256)
    exp_rpeaks = torch.zeros(4, 1, 256)
    exp_rpeaks[:, :, 128] = 1.0  # Simulated R-peak at center

    loss = loss_fn(output['output'], target, exp_rpeaks)
    print(f"Loss value: {loss.item():.4f}")

    print("\nAll tests passed!")
