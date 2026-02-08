"""
PPG2ECG Model - Faithful port from james77777778/ppg2ecg-pytorch

Paper: "Reconstructing QRS Complex from PPG by Transformed Attentional Neural Networks"
       IEEE Sensors Journal, 2020

Architecture (matching reference exactly):
- Encoder: Conv1d (1→32→64→128→256→512) with kernel_size=31, stride=[2,1,2,1,2], PReLU
- Decoder: ConvTranspose1d mirror of encoder, Tanh output
- Optional: Spatial Transformer Network (STN) with affine + scale/bias
- Optional: Attention mechanism on input signal (before encoder)

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
        self.noise = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        if self.sigma != 0:
            scale = self.sigma * x.detach()
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class PPG2ECG(nn.Module):
    """
    Main PPG to ECG conversion model.

    Matches the reference implementation exactly:
    - kernel_size=31 for all conv layers
    - stride pattern: [2, 1, 2, 1, 2] (encoder output is 32 time steps, not 8)
    - STN with stride=1 conv, affine transform + scale/bias
    - Attention on input signal (before encoder), not on features

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

        # Encoder-Decoder as a single sequential (matching reference)
        # Encoder: stride pattern [2, 1, 2, 1, 2]
        # 256 -> 128 -> 128 -> 64 -> 64 -> 32
        self.main = nn.Sequential(
            # encoder
            nn.Conv1d(1, 32, kernel_size=31, stride=2, padding=15),     # 256 -> 128
            nn.PReLU(32),
            nn.Conv1d(32, 64, kernel_size=31, stride=1, padding=15),    # 128 -> 128
            nn.PReLU(64),
            nn.Conv1d(64, 128, kernel_size=31, stride=2, padding=15),   # 128 -> 64
            nn.PReLU(128),
            nn.Conv1d(128, 256, kernel_size=31, stride=1, padding=15),  # 64 -> 64
            nn.PReLU(256),
            nn.Conv1d(256, 512, kernel_size=31, stride=2, padding=15),  # 64 -> 32
            nn.PReLU(512),
            # decoder
            nn.ConvTranspose1d(512, 256, kernel_size=31, stride=2, padding=15, output_padding=1),  # 32 -> 64
            nn.PReLU(256),
            nn.ConvTranspose1d(256, 128, kernel_size=31, stride=1, padding=15),  # 64 -> 64
            nn.PReLU(128),
            nn.ConvTranspose1d(128, 64, kernel_size=31, stride=2, padding=15, output_padding=1),   # 64 -> 128
            nn.PReLU(64),
            nn.ConvTranspose1d(64, 32, kernel_size=31, stride=1, padding=15),    # 128 -> 128
            nn.PReLU(32),
            nn.ConvTranspose1d(32, 1, kernel_size=31, stride=2, padding=15, output_padding=1),     # 128 -> 256
            nn.Tanh(),
        )

        # Optional: Spatial Transformer Network (matching reference exactly)
        if use_stn:
            self.restriction = torch.tensor(
                [1, 0, 0, 0], dtype=torch.float, requires_grad=False)
            self.register_buffer('restriction_const', self.restriction)

            self.stn_conv = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, stride=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(in_channels=8, out_channels=10, kernel_size=5, stride=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
            )
            # Dynamically compute flattened size
            n_stn_conv = self._get_stn_conv_out(input_size)
            self.stn_fc = nn.Sequential(
                Flatten(),
                nn.Linear(n_stn_conv, 32),
                nn.ReLU(True),
                nn.Linear(32, 4)
            )
            # Initialize to identity: [1, 0, 1, 0]
            self.stn_fc[3].weight.data.zero_()
            self.stn_fc[3].bias.data = torch.FloatTensor([1, 0, 1, 0])

        # Optional: Attention on input signal (before encoder)
        if use_attention:
            self.attn = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(),
                nn.Linear(input_size, input_size)
            )
            self.attn_len = input_size

    def _get_stn_conv_out(self, input_size):
        """Dynamically compute STN conv output size."""
        x = torch.zeros(1, 1, input_size)
        out = self.stn_conv(x)
        return out.data.view(1, -1).size(1)

    def stn(self, x):
        """Apply Spatial Transformer Network (matching reference exactly)."""
        xs = self.stn_conv(x)
        theta = self.stn_fc(xs)  # [batch, 4]

        # First 2 params: affine transform
        theta1 = theta[:, :2]
        # Pad with restriction constant [1, 0, 0, 0] to form [1, 0, theta1[0], 0, theta1[1], 0]
        # -> reshaped to [2, 3] affine matrix
        theta1 = torch.cat(
            (self.restriction_const.repeat(theta1.size(0), 1), theta1), 1)
        theta1 = theta1.view(-1, 2, 3)

        # 1D -> 2D for grid operations
        x = x.unsqueeze(-1)  # [batch, 1, seq_len, 1]
        grid = F.affine_grid(theta1, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, padding_mode='border', align_corners=False)

        # Last 2 params: scale and bias
        thetaw = theta[:, 2].contiguous().view(x.size(0), 1, 1, 1)
        thetab = theta[:, 3].contiguous().view(x.size(0), 1, 1, 1)
        x = torch.mul(x, thetaw)
        x = torch.add(x, thetab)

        # 2D -> 1D
        x = x.squeeze(-1)  # [batch, 1, seq_len]
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input PPG signal [batch, 1, 256]

        Returns:
            dict with 'output' (ECG) and 'output_stn' (STN intermediate)
        """
        x1 = x
        # Apply STN if enabled
        if self.use_stn:
            x2 = self.stn(x1)
        else:
            x2 = x

        # Apply attention on input signal (before encoder)
        if self.use_attention:
            attn_weights = F.softmax(self.attn(x2), dim=2) * self.attn_len
            x3 = x2 * attn_weights
        else:
            x3 = x2

        # Main encoder-decoder
        x4 = self.main(x3)

        return {'output': x4, 'output_stn': x2}


class PPG2ECG_LSTM(nn.Module):
    """
    LSTM baseline model for PPG to ECG conversion (matching reference).

    Args:
        input_size: Length of input sequence
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
    """
    def __init__(self, input_size=256, hidden_size=200, num_layers=2, dropout=0.1):
        super(PPG2ECG_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=True)

        self.decoder = nn.LSTM(
            hidden_size * 2, input_size, num_layers,
            batch_first=True, bidirectional=True)

        self.linear = nn.Linear(input_size * 2, input_size)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input PPG signal [batch, 1, input_size]

        Returns:
            dict with 'output' (ECG)
        """
        x = x.view(x.size(0), -1, self.input_size)
        encoded_output, _ = self.encoder(x, None)
        encoded_output = nn.ReLU()(encoded_output)
        decoded_output, _ = self.decoder(encoded_output, None)
        decoded_output = self.linear(decoded_output)
        decoded_output = decoded_output.view(x.size(0), -1, self.input_size)
        decoded_output = decoded_output.view(decoded_output.size(0), 1, -1)
        decoded_output = nn.Tanh()(decoded_output)
        return {'output': decoded_output}


# ============================================================================
# Loss Functions
# ============================================================================

class QRSLoss(nn.Module):
    """
    QRS Complex-Enhanced Loss (matching reference exactly).

    Loss = L1(pred * (1 + beta * exp_rpeaks), target * (1 + beta * exp_rpeaks))
    """
    def __init__(self, beta=5):
        super(QRSLoss, self).__init__()
        self.beta = beta

    def forward(self, pred, target, exp_rpeaks):
        weight = 1 + self.beta * exp_rpeaks
        return F.l1_loss(pred * weight, target * weight)


class CombinedLoss(nn.Module):
    """Combined loss: QRS-weighted L1 or plain L1."""
    def __init__(self, beta=5, use_qrs_loss=True):
        super(CombinedLoss, self).__init__()
        self.use_qrs_loss = use_qrs_loss
        self.qrs_loss = QRSLoss(beta=beta)

    def forward(self, pred, target, exp_rpeaks=None):
        if self.use_qrs_loss and exp_rpeaks is not None:
            return self.qrs_loss(pred, target, exp_rpeaks)
        else:
            return F.l1_loss(pred, target)


# ============================================================================
# Utility Functions
# ============================================================================

def create_model(config):
    """Factory function to create model from config."""
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
    x = torch.randn(4, 1, 256)
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
    exp_rpeaks[:, :, 128] = 1.0

    loss = loss_fn(output['output'], target, exp_rpeaks)
    print(f"Loss value: {loss.item():.4f}")

    print("\nAll tests passed!")
