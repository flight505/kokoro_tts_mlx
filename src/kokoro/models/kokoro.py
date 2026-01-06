"""Kokoro TTS model implementation."""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from kokoro.layers.attention import AlbertConfig, CustomAlbert
from kokoro.layers.istftnet import ISTFTNetDecoder
from kokoro.layers.modules import ProsodyPredictor, TextEncoder


@dataclass
class KokoroConfig:
    """Configuration for Kokoro TTS model."""

    # Model dimensions
    n_token: int = 178
    hidden_dim: int = 512
    style_dim: int = 256
    n_mels: int = 80
    n_layer: int = 3
    max_dur: int = 50
    dropout: float = 0.0

    # Text encoder
    text_encoder_kernel_size: int = 5

    # ALBERT (plbert) config
    plbert: dict | None = None

    # ISTFTNet config
    istftnet: dict | None = None

    # Vocab for phoneme->id mapping
    vocab: dict[str, int] | None = None

    # Audio
    sample_rate: int = 24000

    # Misc
    dim_in: int = 512
    max_conv_dim: int = 512
    multispeaker: bool = False

    def __post_init__(self):
        # Default ALBERT config
        if self.plbert is None:
            self.plbert = {
                "vocab_size": self.n_token,
                "embedding_size": 128,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_hidden_groups": 1,
                "num_attention_heads": 12,
                "intermediate_size": 2048,
                "inner_group_num": 1,
                "hidden_act": "gelu_new",
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
                "max_position_embeddings": 512,
                "type_vocab_size": 2,
                "layer_norm_eps": 1e-12,
            }

        # Default ISTFTNet config
        if self.istftnet is None:
            self.istftnet = {
                "resblock_kernel_sizes": [3, 7, 11],
                "upsample_rates": [8, 8],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16, 16],
                "gen_istft_n_fft": 16,
                "gen_istft_hop_size": 4,
            }


def sanitize_lstm_weights(key: str, value: mx.array) -> dict[str, mx.array]:
    """Convert PyTorch LSTM weight keys to MLX LSTM weight keys."""
    base_key = key.rsplit(".", 1)[0]

    weight_map = {
        "weight_ih_l0_reverse": "backward_cells.0.Wx.weight",
        "weight_hh_l0_reverse": "backward_cells.0.Wh.weight",
        "bias_ih_l0_reverse": "backward_cells.0.Wx.bias",
        "bias_hh_l0_reverse": "backward_cells.0.Wh.bias",
        "weight_ih_l0": "forward_cells.0.Wx.weight",
        "weight_hh_l0": "forward_cells.0.Wh.weight",
        "bias_ih_l0": "forward_cells.0.Wx.bias",
        "bias_hh_l0": "forward_cells.0.Wh.bias",
    }

    for suffix, new_suffix in weight_map.items():
        if key.endswith(suffix):
            # Handle LSTM weight reshaping
            if "weight" in suffix:
                # PyTorch LSTM has shape (4*hidden, input) or (4*hidden, hidden)
                # MLX Linear expects (out, in)
                pass
            return {f"{base_key}.{new_suffix}": value}

    return {key: value}


class KokoroTTS(nn.Module):
    """
    Kokoro Text-to-Speech model.

    A lightweight 82M parameter TTS model based on StyleTTS2 architecture
    with ISTFTNet vocoder.
    """

    REPO_ID = "mlx-community/Kokoro-82M-bf16"

    def __init__(self, config: KokoroConfig):
        super().__init__()
        self.config = config
        self.vocab = config.vocab or {}

        # ALBERT for duration prediction
        # Merge vocab_size from n_token into plbert config
        plbert_config = {**config.plbert} if config.plbert else {}
        plbert_config.setdefault("vocab_size", config.n_token)
        albert_config = AlbertConfig(**plbert_config)
        self.bert = CustomAlbert(albert_config)

        # Project ALBERT output to hidden dim
        self.bert_encoder = nn.Linear(albert_config.hidden_size, config.hidden_dim)
        self.context_length = albert_config.max_position_embeddings

        # Prosody predictor (duration, F0, noise)
        self.predictor = ProsodyPredictor(
            style_dim=config.style_dim,
            d_hid=config.hidden_dim,
            nlayers=config.n_layer,
            max_dur=config.max_dur,
            dropout=config.dropout,
        )

        # Text encoder
        self.text_encoder = TextEncoder(
            channels=config.hidden_dim,
            kernel_size=config.text_encoder_kernel_size,
            depth=config.n_layer,
            n_symbols=config.n_token,
        )

        # ISTFTNet decoder
        self.decoder = ISTFTNetDecoder(
            dim_in=config.hidden_dim,
            style_dim=config.style_dim,
            dim_out=config.n_mels,
            sample_rate=config.sample_rate,
            **config.istftnet,
        )

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def __call__(
        self,
        phonemes: str,
        voice: mx.array,
        speed: float = 1.0,
    ) -> tuple[mx.array, mx.array]:
        """
        Generate audio from phonemes.

        Args:
            phonemes: String of phonemes (space-separated or concatenated)
            voice: Voice embedding tensor (1, 256) containing style information
            speed: Speech speed multiplier (1.0 = normal)

        Returns:
            audio: Generated audio waveform (samples,)
            durations: Predicted durations per phoneme
        """
        # Convert phonemes to token ids
        input_ids = []
        for p in phonemes:
            if p in self.vocab:
                input_ids.append(self.vocab[p])

        if not input_ids:
            raise ValueError("No valid phonemes found in input")

        # Add boundary tokens
        input_ids = [0] + input_ids + [0]

        assert len(input_ids) <= self.context_length, (
            f"Input length {len(input_ids)} exceeds context length {self.context_length}"
        )

        # Convert to arrays
        input_ids = mx.array([[*input_ids]])
        input_lengths = mx.array([input_ids.shape[-1]])

        # Create attention mask
        max_len = int(input_lengths.max())
        text_mask = mx.arange(max_len)[None, :] >= input_lengths[:, None]

        # ALBERT encoding for duration
        bert_dur, _ = self.bert(input_ids, attention_mask=(~text_mask).astype(mx.int32))
        d_en = self.bert_encoder(bert_dur).transpose(0, 2, 1)

        # Split voice embedding into style components
        # voice shape: (1, 256) - first 128 for decoder, last 128 for prosody
        s = voice[:, 128:]  # Style for prosody

        # Duration encoding
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        # Duration prediction
        x, _ = self.predictor.lstm(d.transpose(0, 2, 1))
        x = x.transpose(0, 2, 1)
        duration = self.predictor.duration_proj(x.transpose(0, 2, 1))
        duration = mx.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = mx.clip(mx.round(duration), a_min=1, a_max=None).astype(mx.int32)[0]

        # Build alignment matrix
        indices = mx.concatenate([
            mx.repeat(mx.array(i), int(n.item()))
            for i, n in enumerate(pred_dur)
        ])
        pred_aln_trg = mx.zeros((input_ids.shape[1], indices.shape[0]))
        for i, idx in enumerate(indices):
            pred_aln_trg = pred_aln_trg.at[int(idx.item()), i].set(1.0)
        pred_aln_trg = pred_aln_trg[None, :, :]

        # Align features to predicted durations
        en = d @ pred_aln_trg

        # Predict F0 and noise
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        # Text encoding
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

        # Decode to audio
        audio = self.decoder(asr, F0_pred, N_pred, voice[:, :128])[0]

        mx.eval(audio, pred_dur)

        return audio[0], pred_dur

    def sanitize_weights(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize weights from PyTorch/safetensors format."""
        sanitized = {}

        for key, value in weights.items():
            # Skip position_ids
            if "position_ids" in key:
                continue

            # Handle LSTM weights
            if any(suffix in key for suffix in [
                "weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"
            ]):
                sanitized.update(sanitize_lstm_weights(key, value))
                continue

            # Handle layer norm gamma/beta -> weight/bias
            if key.endswith(".gamma"):
                new_key = key.rsplit(".", 1)[0] + ".weight"
                sanitized[new_key] = value
                continue
            if key.endswith(".beta"):
                new_key = key.rsplit(".", 1)[0] + ".bias"
                sanitized[new_key] = value
                continue

            # Handle weight_v transpositions for convolutions
            if "weight_v" in key and value.ndim == 3:
                # MLX conv expects (kernel, in, out) for conv1d
                # PyTorch has (out, in, kernel)
                sanitized[key] = value
                continue

            sanitized[key] = value

        return sanitized


@dataclass
class GenerationResult:
    """Result from TTS generation."""

    audio: mx.array
    sample_rate: int
    duration_seconds: float
    phonemes: str | None = None
    processing_time: float = 0.0
