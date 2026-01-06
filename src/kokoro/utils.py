"""Utilities for model loading and management."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download

from kokoro.models.kokoro import KokoroConfig, KokoroTTS


def from_pretrained(
    repo_id: str = "mlx-community/Kokoro-82M-bf16",
    dtype: mx.Dtype = mx.bfloat16,
) -> KokoroTTS:
    """
    Load a pretrained Kokoro TTS model from Hugging Face.

    Args:
        repo_id: Hugging Face repository ID
        dtype: Data type for model weights

    Returns:
        Loaded KokoroTTS model
    """
    # Download model files
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

    # Try different weight file names
    weight_filenames = [
        "kokoro-v1_0.safetensors",
        "model.safetensors",
        "weights.safetensors",
    ]

    weights_path = None
    for weight_file in weight_filenames:
        try:
            weights_path = hf_hub_download(repo_id=repo_id, filename=weight_file)
            break
        except Exception:
            continue

    if weights_path is None:
        raise FileNotFoundError(
            f"Could not find weight file in {repo_id}. "
            f"Tried: {weight_filenames}"
        )

    # Load config
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Create config object
    config = KokoroConfig(
        n_token=config_dict.get("n_token", 178),
        hidden_dim=config_dict.get("hidden_dim", 512),
        style_dim=config_dict.get("style_dim", 256),
        n_mels=config_dict.get("n_mels", 80),
        n_layer=config_dict.get("n_layer", 3),
        max_dur=config_dict.get("max_dur", 50),
        dropout=config_dict.get("dropout", 0.0),
        text_encoder_kernel_size=config_dict.get("text_encoder_kernel_size", 5),
        plbert=config_dict.get("plbert"),
        istftnet=config_dict.get("istftnet"),
        vocab=config_dict.get("vocab"),
        sample_rate=config_dict.get("sample_rate", 24000),
        dim_in=config_dict.get("dim_in", 512),
        max_conv_dim=config_dict.get("max_conv_dim", 512),
        multispeaker=config_dict.get("multispeaker", False),
    )

    # Create model
    model = KokoroTTS(config)

    # Load weights
    from safetensors import safe_open

    weights = {}
    with safe_open(weights_path, framework="numpy") as f:
        for key in f.keys():
            weights[key] = mx.array(f.get_tensor(key), dtype=dtype)

    # Sanitize weights
    weights = model.sanitize_weights(weights)

    # Load into model
    model.load_weights(list(weights.items()))

    return model


def from_config(config: KokoroConfig | dict) -> KokoroTTS:
    """
    Create a Kokoro TTS model from config (without loading weights).

    Args:
        config: Model configuration

    Returns:
        Uninitialized KokoroTTS model
    """
    if isinstance(config, dict):
        config = KokoroConfig(**config)

    return KokoroTTS(config)


def save_audio(
    audio: mx.array,
    path: str | Path,
    sample_rate: int = 24000,
) -> None:
    """
    Save audio array to file.

    Args:
        audio: Audio tensor (samples,) or (channels, samples)
        path: Output file path
        sample_rate: Sample rate in Hz
    """
    import numpy as np
    import soundfile as sf

    # Convert to numpy
    audio_np = np.array(audio)

    # Ensure 1D or 2D
    if audio_np.ndim == 1:
        pass  # OK
    elif audio_np.ndim == 2:
        # (channels, samples) -> (samples, channels) for soundfile
        audio_np = audio_np.T
    else:
        raise ValueError(f"Unexpected audio shape: {audio_np.shape}")

    # Normalize if needed
    max_val = np.abs(audio_np).max()
    if max_val > 1.0:
        audio_np = audio_np / max_val

    # Save
    sf.write(str(path), audio_np, sample_rate)
