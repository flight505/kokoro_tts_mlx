"""Basic import tests for Kokoro TTS MLX."""

import pytest


def test_import_kokoro():
    """Test that main package can be imported."""
    import kokoro

    assert hasattr(kokoro, "__version__")
    assert hasattr(kokoro, "KokoroTTS")
    assert hasattr(kokoro, "TTSResult")
    assert hasattr(kokoro, "synthesize")


def test_import_tts_module():
    """Test that tts module can be imported."""
    from kokoro.tts import KokoroTTS, TTSResult, synthesize

    assert KokoroTTS is not None
    assert TTSResult is not None
    assert synthesize is not None


def test_list_voices():
    """Test voice listing."""
    from kokoro import KokoroTTS

    voices = KokoroTTS.list_voices()

    assert isinstance(voices, dict)
    assert "american_female" in voices
    assert "american_male" in voices
    assert "af_heart" in voices["american_female"]
    assert "am_adam" in voices["american_male"]


def test_tts_result_dataclass():
    """Test TTSResult dataclass."""
    import mlx.core as mx
    from kokoro import TTSResult

    result = TTSResult(
        audio=mx.zeros((100,)),
        sample_rate=24000,
        text="test",
        phonemes=None,
        duration_seconds=0.1,
    )

    assert result.sample_rate == 24000
    assert result.text == "test"
    assert result.duration_seconds == 0.1
