"""ISTFTNet vocoder for audio generation."""

import math

import mlx.core as mx
import mlx.nn as nn


def snake_activation(x: mx.array, alpha: mx.array) -> mx.array:
    """Snake activation: x + (1/alpha) * sin^2(alpha * x)."""
    return x + (1.0 / alpha) * mx.sin(alpha * x) ** 2


class Snake(nn.Module):
    """Learnable Snake activation."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, channels, 1))

    def __call__(self, x: mx.array) -> mx.array:
        return snake_activation(x, self.alpha)


class AdaIN1d(nn.Module):
    """Adaptive Instance Normalization for 1D signals."""

    def __init__(self, style_dim: int, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.fc = nn.Linear(style_dim, num_features * 2)

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        """
        Args:
            x: Input (batch, channels, seq_len)
            s: Style (batch, style_dim)
        """
        h = self.fc(s)
        gamma, beta = mx.split(h, 2, axis=-1)

        # Instance norm
        mean = mx.mean(x, axis=2, keepdims=True)
        var = mx.var(x, axis=2, keepdims=True)
        x_norm = (x - mean) / mx.sqrt(var + self.eps)

        return gamma[:, :, None] * x_norm + beta[:, :, None]


class Conv1dWeightNorm(nn.Module):
    """1D Conv with weight normalization."""

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
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight_v = mx.random.normal(
            (out_channels, in_channels, kernel_size)
        ) * 0.02
        self.weight_g = mx.ones((out_channels, 1, 1))
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        # Normalize weight
        norm = mx.sqrt(mx.sum(self.weight_v ** 2, axis=(1, 2), keepdims=True) + 1e-8)
        weight = self.weight_g * self.weight_v / norm

        # Pad input
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (self.padding, self.padding)])

        # Conv1d: input (B, C, L) -> (B, L, C)
        x = x.transpose(0, 2, 1)
        weight_t = weight.transpose(2, 1, 0)
        out = mx.conv1d(x, weight_t, stride=self.stride)
        out = out.transpose(0, 2, 1)

        if self.bias is not None:
            out = out + self.bias[:, None]

        return out


class ConvTranspose1d(nn.Module):
    """1D Transposed convolution for upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # Weight: (out_channels, in_channels, kernel_size)
        self.weight = mx.random.normal(
            (out_channels, in_channels, kernel_size)
        ) * 0.02
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """Transposed conv via upsampling + conv."""
        batch, channels, length = x.shape

        # Upsample by inserting zeros
        if self.stride > 1:
            upsampled = mx.zeros((batch, channels, length * self.stride))
            upsampled = upsampled.at[:, :, ::self.stride].set(x)
            x = upsampled

        # Apply convolution
        pad = self.kernel_size - 1 - self.padding
        if pad > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (pad, pad)])

        x = x.transpose(0, 2, 1)
        weight_t = self.weight.transpose(2, 1, 0)
        out = mx.conv1d(x, weight_t, stride=1)
        out = out.transpose(0, 2, 1)

        out = out + self.bias[:, None]

        return out


class AdaINResBlock(nn.Module):
    """Residual block with adaptive instance normalization."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        style_dim: int,
        dilations: tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()

        self.convs = []
        self.norms = []
        self.activations = []

        for dilation in dilations:
            padding = (kernel_size - 1) * dilation // 2
            self.convs.append(
                Conv1dWeightNorm(channels, channels, kernel_size, padding=padding)
            )
            self.norms.append(AdaIN1d(style_dim, channels))
            self.activations.append(Snake(channels))

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        for conv, norm, act in zip(self.convs, self.norms, self.activations):
            residual = x
            x = act(x)
            x = conv(x)
            x = norm(x, s)
            x = x + residual
        return x


class SineGen(nn.Module):
    """Sine wave generator for harmonic synthesis."""

    def __init__(
        self,
        sample_rate: int = 24000,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    def __call__(self, f0: mx.array) -> mx.array:
        """
        Generate sine waves from F0.

        Args:
            f0: Fundamental frequency (batch, 1, time)

        Returns:
            Sine waves (batch, harmonic_num + 1, time)
        """
        batch, _, length = f0.shape

        # Generate harmonics
        harmonics = mx.arange(1, self.harmonic_num + 2, dtype=f0.dtype)[None, :, None]
        freqs = f0 * harmonics  # (batch, harmonics, time)

        # Phase accumulation
        phase_inc = freqs / self.sample_rate
        phase = mx.cumsum(phase_inc, axis=2) * 2 * math.pi

        # Generate sine waves
        sine = mx.sin(phase) * self.sine_amp

        # Add noise for unvoiced regions
        voiced = (f0 > self.voiced_threshold).astype(f0.dtype)
        noise = mx.random.normal(sine.shape) * self.noise_std
        sine = sine * voiced + noise * (1 - voiced)

        return sine


class SourceModuleHnNSF(nn.Module):
    """Harmonic + Noise Source module."""

    def __init__(
        self,
        sample_rate: int = 24000,
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
    ):
        super().__init__()
        self.sine_gen = SineGen(
            sample_rate=sample_rate,
            harmonic_num=harmonic_num,
            sine_amp=sine_amp,
            noise_std=noise_std,
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def __call__(self, f0: mx.array) -> tuple[mx.array, mx.array]:
        """
        Generate harmonic and noise sources.

        Args:
            f0: Fundamental frequency (batch, 1, time)

        Returns:
            har_source: Harmonic source (batch, 1, time)
            noi_source: Noise source (batch, 1, time)
        """
        sine = self.sine_gen(f0)  # (batch, harmonics, time)
        sine = sine.transpose(0, 2, 1)  # (batch, time, harmonics)
        har = self.l_linear(sine)  # (batch, time, 1)
        har = har.transpose(0, 2, 1)  # (batch, 1, time)

        noi = mx.random.normal(har.shape) * 0.003

        return har, noi


class MLXSTFT:
    """Short-Time Fourier Transform utilities."""

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Hann window
        self.window = 0.5 - 0.5 * mx.cos(
            2 * math.pi * mx.arange(win_length) / win_length
        )

    def stft(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """
        Compute STFT.

        Args:
            x: Audio signal (batch, time)

        Returns:
            magnitude, phase
        """
        # This is a simplified STFT - in production you'd want a more complete implementation
        batch, time = x.shape

        # Pad signal
        pad_amount = self.n_fft // 2
        x = mx.pad(x, [(0, 0), (pad_amount, pad_amount)])

        # Frame the signal
        n_frames = (x.shape[1] - self.n_fft) // self.hop_length + 1
        frames = []
        for i in range(n_frames):
            start = i * self.hop_length
            frame = x[:, start:start + self.n_fft] * self.window
            frames.append(frame)

        frames = mx.stack(frames, axis=1)  # (batch, frames, n_fft)

        # FFT
        spec = mx.fft.rfft(frames)

        # Magnitude and phase
        magnitude = mx.abs(spec)
        phase = mx.arctan2(spec.imag, spec.real)

        return magnitude, phase

    def istft(self, magnitude: mx.array, phase: mx.array) -> mx.array:
        """
        Inverse STFT.

        Args:
            magnitude: (batch, frames, freq_bins)
            phase: (batch, frames, freq_bins)

        Returns:
            Reconstructed audio (batch, time)
        """
        # Reconstruct complex spectrum
        real = magnitude * mx.cos(phase)
        imag = magnitude * mx.sin(phase)
        spec = real + 1j * imag

        # Inverse FFT
        frames = mx.fft.irfft(spec, n=self.n_fft)  # (batch, frames, n_fft)

        # Overlap-add
        batch, n_frames, _ = frames.shape
        output_length = (n_frames - 1) * self.hop_length + self.n_fft

        output = mx.zeros((batch, output_length))
        window_sum = mx.zeros((1, output_length))

        for i in range(n_frames):
            start = i * self.hop_length
            output = output.at[:, start:start + self.n_fft].add(
                frames[:, i, :] * self.window
            )
            window_sum = window_sum.at[:, start:start + self.n_fft].add(
                self.window ** 2
            )

        # Normalize by window sum
        output = output / (window_sum + 1e-8)

        # Remove padding
        pad_amount = self.n_fft // 2
        output = output[:, pad_amount:-pad_amount]

        return output


class Generator(nn.Module):
    """Main generator for waveform synthesis."""

    def __init__(
        self,
        initial_channel: int,
        style_dim: int,
        resblock_kernel_sizes: tuple[int, ...] | list[int] = (3, 7, 11),
        upsample_rates: tuple[int, ...] | list[int] = (10, 6),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: tuple[int, ...] | list[int] = (20, 12),
        gen_istft_n_fft: int = 20,
        gen_istft_hop_size: int = 5,
        sample_rate: int = 24000,
    ):
        super().__init__()

        self.num_upsamples = len(upsample_rates)
        self.sample_rate = sample_rate

        # Initial convolution
        self.conv_pre = Conv1dWeightNorm(
            initial_channel, upsample_initial_channel, kernel_size=7, padding=3
        )

        # Upsampling layers
        self.ups = []
        self.noise_convs = []
        ch = upsample_initial_channel

        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                ConvTranspose1d(
                    ch, ch // 2, kernel_size=kernel, stride=rate, padding=(kernel - rate) // 2
                )
            )
            self.noise_convs.append(Conv1dWeightNorm(1, ch // 2, kernel_size=1))
            ch = ch // 2

        # Residual blocks
        self.resblocks = []
        for i in range(self.num_upsamples):
            ch_i = upsample_initial_channel // (2 ** (i + 1))
            for kernel in resblock_kernel_sizes:
                self.resblocks.append(
                    AdaINResBlock(ch_i, kernel, style_dim)
                )

        # Source module
        self.m_source = SourceModuleHnNSF(sample_rate=sample_rate)

        # Output convolutions for STFT
        n_fft_half = gen_istft_n_fft // 2 + 1
        self.conv_post = Conv1dWeightNorm(ch, n_fft_half, kernel_size=7, padding=3)
        self.conv_post_phase = Conv1dWeightNorm(ch, n_fft_half, kernel_size=7, padding=3)

        self.stft = MLXSTFT(
            n_fft=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft,
        )

    def __call__(
        self,
        x: mx.array,
        f0: mx.array,
        n: mx.array,
        s: mx.array,
    ) -> mx.array:
        """
        Generate waveform.

        Args:
            x: Input features (batch, channels, time)
            f0: F0 contour (batch, 1, time)
            n: Noise (batch, 1, time)
            s: Style embedding (batch, style_dim)

        Returns:
            Audio waveform (batch, time)
        """
        # Generate source signals
        har_source, noi_source = self.m_source(f0)

        x = self.conv_pre(x)

        for i, (up, noise_conv) in enumerate(zip(self.ups, self.noise_convs)):
            x = nn.leaky_relu(x, negative_slope=0.1)
            x = up(x)

            # Add noise source
            if i < len(self.noise_convs):
                # Upsample noise source to match
                noi_up = mx.repeat(noi_source, 2 ** (i + 1), axis=2)
                noi_up = noi_up[:, :, :x.shape[2]]
                x = x + noise_conv(noi_up)

            # Apply residual blocks
            n_blocks = len(self.resblocks) // self.num_upsamples
            for j in range(n_blocks):
                x = self.resblocks[i * n_blocks + j](x, s)

        x = nn.leaky_relu(x, negative_slope=0.1)

        # Predict magnitude and phase
        magnitude = mx.exp(self.conv_post(x))
        phase = self.conv_post_phase(x)

        # ISTFT synthesis
        magnitude = magnitude.transpose(0, 2, 1)  # (batch, time, freq)
        phase = phase.transpose(0, 2, 1)
        audio = self.stft.istft(magnitude, phase)

        return audio


class ISTFTNetDecoder(nn.Module):
    """
    Full ISTFTNet decoder that wraps the generator.
    """

    def __init__(
        self,
        dim_in: int,
        style_dim: int,
        dim_out: int,
        resblock_kernel_sizes: tuple[int, ...] | list[int] = (3, 7, 11),
        resblock_dilation_sizes: list | None = None,  # Accepted but simplified in this impl
        upsample_rates: tuple[int, ...] | list[int] = (10, 6),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: tuple[int, ...] | list[int] = (20, 12),
        gen_istft_n_fft: int = 20,
        gen_istft_hop_size: int = 5,
        sample_rate: int = 24000,
    ):
        super().__init__()

        # Encode input features
        self.encode = Conv1dWeightNorm(dim_in, upsample_initial_channel, kernel_size=3, padding=1)

        # AdaIN for style conditioning
        self.adain_layers = []
        ch = upsample_initial_channel
        for _ in range(3):
            self.adain_layers.append(AdaINResBlock(ch, 3, style_dim))

        # Generator
        self.generator = Generator(
            initial_channel=upsample_initial_channel,
            style_dim=style_dim,
            resblock_kernel_sizes=resblock_kernel_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gen_istft_n_fft=gen_istft_n_fft,
            gen_istft_hop_size=gen_istft_hop_size,
            sample_rate=sample_rate,
        )

    def __call__(
        self,
        asr: mx.array,
        f0: mx.array,
        n: mx.array,
        s: mx.array,
    ) -> tuple[mx.array]:
        """
        Decode to audio.

        Args:
            asr: ASR features (batch, channels, time)
            f0: F0 contour (batch, 1, time)
            n: Noise component (batch, 1, time)
            s: Style embedding (batch, style_dim)

        Returns:
            Tuple of (audio,) for compatibility
        """
        x = self.encode(asr)

        for adain in self.adain_layers:
            x = adain(x, s)

        audio = self.generator(x, f0, n, s)

        return (audio,)

    def sanitize(self, key: str, value: mx.array) -> mx.array:
        """Sanitize weights from PyTorch format."""
        # Handle weight transpositions if needed
        if "weight_v" in key and value.ndim == 3:
            # Check if shape needs transposition
            pass
        return value
