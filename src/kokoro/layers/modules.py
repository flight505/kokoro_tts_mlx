"""Kokoro neural network modules: text encoder, prosody predictor, etc."""

import mlx.core as mx
import mlx.nn as nn

from kokoro.layers.lstm import LSTM


class LinearNorm(nn.Module):
    """Linear layer wrapper."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on style embedding."""

    def __init__(self, style_dim: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        """
        Args:
            x: Input tensor (batch, channels, seq_len) or (batch, seq_len, channels)
            s: Style embedding (batch, style_dim)
        """
        # Get affine parameters from style
        h = self.fc(s)
        gamma, beta = mx.split(h, 2, axis=-1)

        # Handle different input shapes
        if x.ndim == 3:
            # Assume (batch, seq_len, channels) or (batch, channels, seq_len)
            if x.shape[-1] == self.channels:
                # (batch, seq_len, channels)
                mean = mx.mean(x, axis=-1, keepdims=True)
                var = mx.var(x, axis=-1, keepdims=True)
                x_norm = (x - mean) / mx.sqrt(var + self.eps)
                return gamma[:, None, :] * x_norm + beta[:, None, :]
            else:
                # (batch, channels, seq_len)
                mean = mx.mean(x, axis=1, keepdims=True)
                var = mx.var(x, axis=1, keepdims=True)
                x_norm = (x - mean) / mx.sqrt(var + self.eps)
                return gamma[:, :, None] * x_norm + beta[:, :, None]
        else:
            mean = mx.mean(x, axis=-1, keepdims=True)
            var = mx.var(x, axis=-1, keepdims=True)
            x_norm = (x - mean) / mx.sqrt(var + self.eps)
            return gamma * x_norm + beta


class Conv1dWithWeightNorm(nn.Module):
    """1D Convolution with weight normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Weight normalization parameters
        self.weight_v = mx.random.normal(
            (out_channels, in_channels, kernel_size)
        ) * 0.02
        self.weight_g = mx.ones((out_channels, 1, 1))

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input tensor (batch, in_channels, seq_len)
        """
        # Compute normalized weight
        norm = mx.sqrt(mx.sum(self.weight_v ** 2, axis=(1, 2), keepdims=True) + 1e-8)
        weight = self.weight_g * self.weight_v / norm

        # Apply padding
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (self.padding, self.padding)])

        # Transpose for MLX conv: (batch, seq_len, in_channels)
        x = x.transpose(0, 2, 1)

        # Transpose weight: (out_channels, in_channels, kernel) -> (kernel, in_channels, out_channels)
        weight_t = weight.transpose(2, 1, 0)

        # 1D convolution
        out = mx.conv1d(x, weight_t, stride=self.stride, padding=0)

        # Transpose back: (batch, seq_len, out_channels) -> (batch, out_channels, seq_len)
        out = out.transpose(0, 2, 1)

        if self.bias is not None:
            out = out + self.bias[:, None]

        return out


class TextEncoder(nn.Module):
    """
    Text encoder for Kokoro.

    Converts token embeddings through convolutions and bidirectional LSTM.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        depth: int,
        n_symbols: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        # Convolutional layers
        self.convs = []
        self.norms = []
        padding = (kernel_size - 1) // 2

        for _ in range(depth):
            self.convs.append(
                Conv1dWithWeightNorm(
                    channels, channels, kernel_size, padding=padding
                )
            )
            self.norms.append(nn.LayerNorm(channels))

        # Bidirectional LSTM
        self.lstm = LSTM(
            channels, channels // 2, num_layers=1, bidirectional=True
        )

    def __call__(
        self,
        x: mx.array,
        input_lengths: mx.array,
        text_mask: mx.array,
    ) -> mx.array:
        """
        Args:
            x: Token ids (batch, seq_len)
            input_lengths: Lengths of each sequence
            text_mask: Boolean mask for padding

        Returns:
            Encoded text (batch, channels, seq_len)
        """
        # Embed tokens
        x = self.embedding(x)  # (batch, seq_len, channels)
        x = x.transpose(0, 2, 1)  # (batch, channels, seq_len)

        # Apply convolutions with residual connections
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x)
            x = x.transpose(0, 2, 1)  # (batch, seq_len, channels)
            x = norm(x)
            x = x.transpose(0, 2, 1)  # (batch, channels, seq_len)
            x = nn.relu(x)
            x = x + residual

        # Apply mask
        mask = (~text_mask).astype(x.dtype)[:, None, :]
        x = x * mask

        # LSTM
        x = x.transpose(0, 2, 1)  # (batch, seq_len, channels)
        x, _ = self.lstm(x)
        x = x.transpose(0, 2, 1)  # (batch, channels, seq_len)

        return x


class DurationEncoder(nn.Module):
    """Encodes duration information with style conditioning."""

    def __init__(
        self,
        style_dim: int,
        d_hid: int,
        nlayers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstms = []
        self.norms = []
        self.dropout = dropout

        for _ in range(nlayers):
            self.lstms.append(
                LSTM(d_hid + style_dim, d_hid // 2, bidirectional=True)
            )
            self.norms.append(AdaLayerNorm(style_dim, d_hid))

    def __call__(
        self,
        x: mx.array,
        style: mx.array,
        input_lengths: mx.array,
        text_mask: mx.array,
    ) -> mx.array:
        """
        Args:
            x: Input features (batch, channels, seq_len)
            style: Style embedding (batch, style_dim)
            input_lengths: Sequence lengths
            text_mask: Padding mask
        """
        # Expand style for concatenation
        style_expanded = style[:, :, None].broadcast_to(
            (style.shape[0], style.shape[1], x.shape[2])
        )

        for lstm, norm in zip(self.lstms, self.norms):
            # Concatenate with style
            x_cat = mx.concatenate([x, style_expanded], axis=1)
            x_cat = x_cat.transpose(0, 2, 1)  # (batch, seq_len, channels + style)

            # LSTM
            out, _ = lstm(x_cat)
            out = out.transpose(0, 2, 1)  # (batch, channels, seq_len)

            # Apply adaptive normalization
            x = norm(out, style) + x

        return x


class ProsodyPredictor(nn.Module):
    """
    Predicts prosody: duration, F0 (pitch), and noise.
    """

    def __init__(
        self,
        style_dim: int,
        d_hid: int,
        nlayers: int,
        max_dur: int = 50,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_dur = max_dur

        # Duration encoder
        self.text_encoder = DurationEncoder(
            style_dim=style_dim,
            d_hid=d_hid,
            nlayers=nlayers,
            dropout=dropout,
        )

        # Duration prediction LSTM
        self.lstm = LSTM(d_hid, d_hid // 2, bidirectional=True)

        # Duration projection
        self.duration_proj = nn.Linear(d_hid, max_dur)

        # F0 and Noise prediction networks
        self.F0_proj = Conv1dWithWeightNorm(d_hid, 1, kernel_size=1)
        self.N_proj = Conv1dWithWeightNorm(d_hid, 1, kernel_size=1)

        # F0 network
        self.F0_layers = []
        for _ in range(nlayers):
            self.F0_layers.append(
                (LSTM(d_hid + style_dim, d_hid // 2, bidirectional=True),
                 AdaLayerNorm(style_dim, d_hid))
            )

        # Noise network
        self.N_layers = []
        for _ in range(nlayers):
            self.N_layers.append(
                (LSTM(d_hid + style_dim, d_hid // 2, bidirectional=True),
                 AdaLayerNorm(style_dim, d_hid))
            )

    def F0Ntrain(
        self,
        x: mx.array,
        style: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Predict F0 and noise from encoded features.

        Args:
            x: Encoded features (batch, channels, seq_len)
            style: Style embedding (batch, style_dim)

        Returns:
            F0: Fundamental frequency (batch, 1, seq_len)
            N: Noise component (batch, 1, seq_len)
        """
        # F0 prediction
        f0 = x
        style_expanded = style[:, :, None].broadcast_to(
            (style.shape[0], style.shape[1], x.shape[2])
        )

        for lstm, norm in self.F0_layers:
            f0_cat = mx.concatenate([f0, style_expanded], axis=1)
            f0_cat = f0_cat.transpose(0, 2, 1)
            out, _ = lstm(f0_cat)
            out = out.transpose(0, 2, 1)
            f0 = norm(out, style) + f0

        F0 = self.F0_proj(f0)

        # Noise prediction
        n = x
        for lstm, norm in self.N_layers:
            n_cat = mx.concatenate([n, style_expanded], axis=1)
            n_cat = n_cat.transpose(0, 2, 1)
            out, _ = lstm(n_cat)
            out = out.transpose(0, 2, 1)
            n = norm(out, style) + n

        N = self.N_proj(n)

        return F0, N
