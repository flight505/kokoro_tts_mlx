"""Clean wrapper around mlx-audio's Kokoro TTS implementation.

Provides three API levels:
- generate(): Zero overhead, direct passthrough to mlx-audio
- stream(): Low overhead, yields TTSResult per segment
- synthesize(): Convenient, returns single TTSResult with concatenated audio
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Literal

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_audio.tts.models.kokoro import Model


@dataclass(slots=True)
class TTSResult:
    """Result from TTS generation."""

    audio: mx.array
    sample_rate: int
    text: str
    phonemes: str | None = None
    duration_seconds: float = 0.0


class KokoroTTS:
    """
    Kokoro Text-to-Speech wrapper.

    Provides multiple API levels for different performance needs:

    - generate(): Zero overhead passthrough to mlx-audio
    - stream(): Low overhead streaming with TTSResult metadata
    - synthesize(): Convenient single-call with concatenated audio

    Example (zero overhead):
        >>> tts = KokoroTTS()
        >>> for result in tts.generate("Hello"):
        ...     audio = result.audio  # Raw mlx-audio GenerationResult

    Example (convenient):
        >>> tts = KokoroTTS()
        >>> result = tts.synthesize("Hello, world!")
        >>> tts.save(result.audio, "output.wav")
    """

    DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"

    __slots__ = ("model_id", "default_voice", "lang", "_model", "_sample_rate")

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        voice: str = "af_heart",
        lang: Literal["a", "b", "j", "z"] = "a",
    ):
        """
        Initialize Kokoro TTS.

        Args:
            model_id: HuggingFace model ID (supports bf16, 8bit, 4bit variants)
            voice: Default voice (e.g., "af_heart", "am_adam")
            lang: Language code
                - 'a': American English
                - 'b': British English
                - 'j': Japanese
                - 'z': Mandarin Chinese
        """
        self.model_id = model_id
        self.default_voice = voice
        self.lang = lang
        self._model: "Model | None" = None
        self._sample_rate: int = 24000  # Default, updated on load

    def _ensure_loaded(self) -> "Model":
        """Lazily load the model. Returns the model for chaining."""
        if self._model is None:
            import json

            import dacite
            from huggingface_hub import hf_hub_download
            from mlx_audio.tts.models.kokoro import Model, ModelConfig

            config_path = hf_hub_download(self.model_id, "config.json")
            with open(config_path) as f:
                config_dict = json.load(f)

            model_config = dacite.from_dict(ModelConfig, config_dict)
            self._model = Model(model_config, repo_id=self.model_id)
            self._sample_rate = self._model.sample_rate

        return self._model

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        self._ensure_loaded()
        return self._sample_rate

    @property
    def model(self) -> "Model":
        """Direct access to the underlying mlx-audio Model."""
        return self._ensure_loaded()

    def generate(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ):
        """
        Zero-overhead generation - direct passthrough to mlx-audio.

        Use this when maximum performance is required. Returns the raw
        mlx-audio GenerationResult objects with full metrics (RTF, timing, etc).

        Args:
            text: Text to synthesize
            voice: Voice name (uses default if None)
            speed: Speech speed multiplier (0.5-2.0)

        Yields:
            GenerationResult from mlx-audio with:
                - audio: mx.array of audio samples
                - real_time_factor: inference_time / audio_duration
                - processing_time_seconds: time for this segment
                - sample_rate: audio sample rate
                - And more metrics...

        Example:
            >>> tts = KokoroTTS()
            >>> for result in tts.generate("Hello world"):
            ...     print(f"RTF: {result.real_time_factor:.2f}x")
            ...     audio = result.audio
        """
        model = self._ensure_loaded()
        return model.generate(
            text,
            voice=voice or self.default_voice,
            speed=speed,
            lang_code=self.lang,
        )

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """
        Synthesize speech from text (convenient API).

        Collects all audio segments and returns a single TTSResult.
        For streaming or zero-overhead, use generate() or stream() instead.

        Args:
            text: Text to synthesize
            voice: Voice name (uses default if None)
            speed: Speech speed multiplier (0.5-2.0)

        Returns:
            TTSResult with concatenated audio array and metadata
        """
        model = self._ensure_loaded()
        sr = self._sample_rate  # Cache locally
        voice = voice or self.default_voice

        # Collect audio segments
        segments = []
        for result in model.generate(text, voice=voice, speed=speed, lang_code=self.lang):
            if result.audio is not None:
                segments.append(result.audio)

        if not segments:
            raise ValueError("No audio generated")

        # Single segment: no concatenation needed
        if len(segments) == 1:
            audio = segments[0]
        else:
            audio = mx.concatenate(segments, axis=-1)

        return TTSResult(
            audio=audio,
            sample_rate=sr,
            text=text,
            phonemes=None,
            duration_seconds=audio.shape[-1] / sr,
        )

    def stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> Generator[TTSResult, None, None]:
        """
        Stream audio generation segment by segment.

        Lower overhead than synthesize() - yields each segment as it's generated.
        For zero overhead, use generate() instead.

        Args:
            text: Text to synthesize
            voice: Voice name
            speed: Speech speed multiplier

        Yields:
            TTSResult for each segment
        """
        model = self._ensure_loaded()
        sr = self._sample_rate  # Cache locally
        voice = voice or self.default_voice

        for result in model.generate(text, voice=voice, speed=speed, lang_code=self.lang):
            if result.audio is not None:
                audio = result.audio
                yield TTSResult(
                    audio=audio,
                    sample_rate=sr,
                    text="",
                    phonemes=None,
                    duration_seconds=audio.shape[-1] / sr,
                )

    def save(
        self,
        audio: mx.array,
        path: str | Path,
        sample_rate: int | None = None,
    ) -> None:
        """
        Save audio to file.

        Args:
            audio: Audio array
            path: Output file path
            sample_rate: Sample rate (uses model's if None)
        """
        import numpy as np
        import soundfile as sf

        sr = sample_rate or self._sample_rate
        audio_np = np.array(audio)

        # Normalize if needed
        max_val = np.abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val

        sf.write(str(path), audio_np, sr)

    @staticmethod
    def list_voices() -> dict[str, list[str]]:
        """
        List available voices by category.

        Returns:
            Dict mapping category to list of voice names
        """
        return {
            "american_female": [
                "af_heart", "af_bella", "af_nova", "af_sarah",
                "af_nicole", "af_sky", "af_alloy", "af_aoede",
                "af_jessica", "af_kore", "af_river", "af_sage",
            ],
            "american_male": [
                "am_adam", "am_echo", "am_eric", "am_fenrir",
                "am_liam", "am_michael", "am_onyx", "am_puck",
            ],
            "british_female": ["bf_emma", "bf_isabella", "bf_alice", "bf_lily"],
            "british_male": ["bm_george", "bm_lewis", "bm_daniel", "bm_fable"],
            "japanese_female": ["jf_alpha", "jf_gongitsune"],
            "japanese_male": ["jm_kumo"],
            "chinese_female": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxuan"],
            "chinese_male": ["zm_yunjian", "zm_yunxi", "zm_yunyang"],
        }


# Convenience function
def synthesize(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    output: str | Path | None = None,
) -> TTSResult:
    """
    Quick synthesis function.

    Args:
        text: Text to synthesize
        voice: Voice name
        speed: Speech speed
        output: Optional output file path

    Returns:
        TTSResult with audio
    """
    tts = KokoroTTS(voice=voice)
    result = tts.synthesize(text, speed=speed)

    if output:
        tts.save(result.audio, output)

    return result
