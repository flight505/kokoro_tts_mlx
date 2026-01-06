"""Basic import tests for Kokoro TTS MLX."""

import pytest


def test_import_kokoro():
    """Test that main package can be imported."""
    import kokoro

    assert hasattr(kokoro, "__version__")
    assert hasattr(kokoro, "KokoroTTS")
    assert hasattr(kokoro, "KokoroConfig")
    assert hasattr(kokoro, "from_pretrained")


def test_import_layers():
    """Test that layer modules can be imported."""
    from kokoro.layers.lstm import LSTM, LSTMCell
    from kokoro.layers.attention import AlbertConfig, CustomAlbert, AlbertAttention
    from kokoro.layers.modules import TextEncoder, ProsodyPredictor
    from kokoro.layers.istftnet import ISTFTNetDecoder, Generator


def test_import_processing():
    """Test that processing modules can be imported."""
    from kokoro.processing.text import TextProcessor, phonemize
    from kokoro.processing.voice import VoiceManager, load_voice, VOICES


def test_import_utils():
    """Test that utils can be imported."""
    from kokoro.utils import from_pretrained, from_config, save_audio


def test_lstm_basic():
    """Test basic LSTM functionality."""
    import mlx.core as mx
    from kokoro.layers.lstm import LSTM

    lstm = LSTM(input_size=32, hidden_size=16, bidirectional=True)

    # Create dummy input
    x = mx.random.normal((2, 10, 32))  # batch=2, seq_len=10, features=32

    # Forward pass
    output, (h_n, c_n) = lstm(x)

    # Check output shape: (batch, seq_len, hidden_size * 2) for bidirectional
    assert output.shape == (2, 10, 32), f"Expected (2, 10, 32), got {output.shape}"


def test_albert_basic():
    """Test basic ALBERT functionality."""
    import mlx.core as mx
    from kokoro.layers.attention import AlbertConfig, CustomAlbert

    config = AlbertConfig(
        vocab_size=100,
        embedding_size=64,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=64,
    )

    model = CustomAlbert(config)

    # Create dummy input
    input_ids = mx.array([[1, 2, 3, 4, 5]])

    # Forward pass
    sequence_output, pooled_output = model(input_ids)

    assert sequence_output.shape == (1, 5, 128)
    assert pooled_output.shape == (1, 128)


def test_voice_list():
    """Test voice listing."""
    from kokoro.processing.voice import VoiceManager, VOICES

    manager = VoiceManager()

    # Check voices exist
    assert len(VOICES) > 0

    # Check list function
    all_voices = manager.list_voices()
    assert "af_heart" in all_voices

    # Check language filtering
    us_voices = manager.list_voices(lang="en-us")
    assert all(v.startswith(("af_", "am_")) for v in us_voices)
