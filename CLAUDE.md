# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Kokoro TTS MLX is a text-to-speech implementation of the Kokoro-82M model for Apple Silicon using MLX. It provides both a Python API and CLI for speech synthesis.

## Development Setup

```bash
# Create virtual environment
uv venv

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

## Architecture

### Package Structure
```
src/kokoro/
├── models/           # Model implementations
│   └── kokoro.py     # Main KokoroTTS model
├── layers/           # Neural network components
│   ├── lstm.py       # Custom LSTM (MLX doesn't have built-in)
│   ├── attention.py  # ALBERT attention layers
│   ├── modules.py    # Text encoder, prosody predictor
│   └── istftnet.py   # ISTFTNet vocoder
├── processing/       # Text and audio processing
│   ├── text.py       # G2P conversion (misaki)
│   └── voice.py      # Voice loading from HuggingFace
├── cli.py            # Typer CLI
└── utils.py          # Model loading utilities
```

### Key Components

1. **KokoroTTS** (`models/kokoro.py`)
   - Main model class
   - Forward pass: phonemes + voice → audio
   - Weight sanitization for PyTorch→MLX conversion

2. **LSTM** (`layers/lstm.py`)
   - Custom bidirectional LSTM implementation
   - MLX doesn't have built-in LSTM, so we implement it

3. **ALBERT** (`layers/attention.py`)
   - Used for duration prediction
   - Parameter sharing for efficiency

4. **ISTFTNet** (`layers/istftnet.py`)
   - Vocoder for audio generation
   - Uses harmonic+noise source modeling

5. **TextProcessor** (`processing/text.py`)
   - G2P via misaki library
   - Fallback to espeak-ng

## Commands

```bash
# Run CLI
kokoro "Hello world" -v af_heart -o output.wav

# Run tests
pytest tests/ -v

# Type checking
mypy src/kokoro

# Linting
ruff check src/
ruff format src/
```

## Model Weights

Default: `mlx-community/Kokoro-82M-bf16`

The model config contains:
- `vocab`: Phoneme→ID mapping
- `plbert`: ALBERT configuration
- `istftnet`: Vocoder configuration

## Important Notes

1. **LSTM Weight Conversion**: PyTorch LSTM weights need sanitization for MLX
2. **Voice Embeddings**: 256-dim vectors (128 for decoder style, 128 for prosody)
3. **G2P Dependencies**: Requires misaki[en] for English, misaki[ja] for Japanese, etc.
4. **Sample Rate**: 24000 Hz (hardcoded in model)

## Testing

Tests are in `tests/`. Run with:
```bash
pytest tests/ -v
```

## Dependencies

Core:
- mlx
- huggingface-hub
- safetensors
- soundfile
- misaki

Optional:
- espeak-ng (system package for fallback G2P)
