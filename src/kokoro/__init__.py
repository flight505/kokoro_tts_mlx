"""Kokoro TTS - Text-to-Speech for Apple Silicon using MLX."""

from kokoro.models.kokoro import KokoroTTS, KokoroConfig
from kokoro.utils import from_pretrained

__version__ = "0.1.0"
__all__ = [
    "KokoroTTS",
    "KokoroConfig",
    "from_pretrained",
]
