# Kokoro TTS MLX

A lightweight text-to-speech implementation of [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) for Apple Silicon using MLX.

## Features

- **82M parameter** TTS model - fast and efficient
- **8 languages** - English (US/UK), Japanese, Chinese, Spanish, French, Hindi, Italian, Portuguese
- **54 voice presets** - diverse male and female voices
- **Apple Silicon optimized** - runs efficiently on M1/M2/M3 chips
- **Simple API** - easy to use Python interface and CLI

## Installation

```bash
# Clone the repository
git clone https://github.com/flight505/kokoro_tts_mlx.git
cd kokoro_tts_mlx

# Install with uv
uv venv
uv pip install -e ".[dev]"
```

### Dependencies

- Python 3.10+
- MLX
- misaki (for G2P conversion)
- espeak-ng (optional, for fallback phonemization)

```bash
# Install espeak-ng (macOS)
brew install espeak-ng
```

## Quick Start

### Python API

```python
from kokoro import from_pretrained
from kokoro.processing.text import TextProcessor
from kokoro.processing.voice import load_voice
from kokoro.utils import save_audio

# Load model
model = from_pretrained()

# Load a voice
voice = load_voice("af_heart")

# Convert text to phonemes
processor = TextProcessor(lang_code="a")  # American English
phonemes = processor.phonemize("Hello, world!")

# Generate audio
audio, durations = model(phonemes, voice, speed=1.0)

# Save to file
save_audio(audio, "output.wav", sample_rate=model.sample_rate)
```

### Command Line

```bash
# Basic synthesis
kokoro "Hello, world!"

# With options
kokoro "Hello, world!" -v am_adam -o hello.wav -s 1.2

# List available voices
kokoro --list-voices

# Show system info
kokoro info
```

## Available Voices

### American English
- **Female**: af_heart, af_bella, af_nova, af_sarah, af_nicole, af_sky
- **Male**: am_adam, am_michael

### British English
- **Female**: bf_emma, bf_isabella
- **Male**: bm_george, bm_lewis

### Japanese
- **Female**: jf_alpha, jf_gongitsune
- **Male**: jm_kumo

### Chinese
- **Female**: zf_xiaobei, zf_xiaoni, zf_xiaoxuan
- **Male**: zm_yunjian

## Architecture

Kokoro uses a decoder-only StyleTTS 2 architecture:

1. **Text Encoder** - Converts phonemes to embeddings via LSTM
2. **ALBERT** - Predicts duration from text embeddings
3. **Prosody Predictor** - Predicts F0 (pitch) and noise components
4. **ISTFTNet Vocoder** - Generates audio from acoustic features

```
Text → Phonemes → Duration → F0/Noise → Audio
           ↓           ↓
       [ALBERT]   [Prosody]
           ↓           ↓
      [Text Encoder] → [ISTFTNet Decoder] → Waveform
```

## Model Weights

Uses pre-converted MLX weights from:
- [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16)

Original model:
- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

## License

Apache 2.0

## Acknowledgments

- [hexgrad](https://github.com/hexgrad) - Original Kokoro model
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) - Reference MLX implementation
- [Apple MLX Team](https://github.com/ml-explore/mlx) - MLX framework
