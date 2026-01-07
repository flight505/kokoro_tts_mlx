"""Clean wrapper around mlx-audio's Kokoro TTS implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Literal

import mlx.core as mx


@dataclass
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

    A simple interface to mlx-audio's Kokoro implementation
    optimized for Apple Silicon.

    Example:
        >>> tts = KokoroTTS()
        >>> result = tts.synthesize("Hello, world!")
        >>> tts.save(result.audio, "output.wav")
    """

    DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        voice: str = "af_heart",
        lang: Literal["a", "b", "j", "z"] = "a",
    ):
        """
        Initialize Kokoro TTS.

        Args:
            model_id: HuggingFace model ID
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
        self._model = None
        self._pipeline = None

    def _ensure_loaded(self):
        """Lazily load the model."""
        if self._model is None:
            import json
            import dacite
            from huggingface_hub import hf_hub_download
            from mlx_audio.tts.models.kokoro import Model, ModelConfig

            # Download and load config
            config_path = hf_hub_download(self.model_id, "config.json")
            with open(config_path) as f:
                config_dict = json.load(f)

            # Create model from config
            model_config = dacite.from_dict(ModelConfig, config_dict)
            self._model = Model(model_config, repo_id=self.model_id)

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        self._ensure_loaded()
        return self._model.sample_rate

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Voice name (uses default if None)
            speed: Speech speed multiplier (0.5-2.0)

        Returns:
            TTSResult with audio array and metadata
        """
        self._ensure_loaded()

        voice = voice or self.default_voice

        # Collect all audio segments using model.generate()
        all_audio = []

        for result in self._model.generate(
            text, voice=voice, speed=speed, lang_code=self.lang
        ):
            if result.audio is not None:
                all_audio.append(result.audio)

        if not all_audio:
            raise ValueError("No audio generated")

        # Concatenate audio segments
        if len(all_audio) == 1:
            combined_audio = all_audio[0]
        else:
            combined_audio = mx.concatenate(all_audio, axis=-1)

        # Flatten if needed
        if combined_audio.ndim > 1:
            combined_audio = combined_audio.flatten()

        duration = len(combined_audio) / self.sample_rate

        return TTSResult(
            audio=combined_audio,
            sample_rate=self.sample_rate,
            text=text,
            phonemes=None,
            duration_seconds=duration,
        )

    def stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> Generator[TTSResult, None, None]:
        """
        Stream audio generation segment by segment.

        Useful for real-time playback of long texts.

        Args:
            text: Text to synthesize
            voice: Voice name
            speed: Speech speed multiplier

        Yields:
            TTSResult for each segment
        """
        self._ensure_loaded()

        voice = voice or self.default_voice

        for result in self._model.generate(
            text, voice=voice, speed=speed, lang_code=self.lang
        ):
            if result.audio is not None:
                audio = result.audio
                if audio.ndim > 1:
                    audio = audio.flatten()

                yield TTSResult(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    text=getattr(result, "graphemes", "") or "",
                    phonemes=getattr(result, "phonemes", None),
                    duration_seconds=len(audio) / self.sample_rate,
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

        sample_rate = sample_rate or self.sample_rate
        audio_np = np.array(audio)

        # Normalize
        max_val = np.abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val

        sf.write(str(path), audio_np, sample_rate)

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
