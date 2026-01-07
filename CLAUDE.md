# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Kokoro TTS MLX is a clean wrapper around mlx-audio's Kokoro TTS implementation, providing a simple API for text-to-speech synthesis on Apple Silicon.

## Architecture

This project is a **thin wrapper** around mlx-audio, not a reimplementation:

```
kokoro_tts_mlx/
├── src/kokoro/
│   ├── __init__.py   # Public API exports
│   ├── tts.py        # KokoroTTS wrapper class
│   └── cli.py        # Typer CLI
└── tests/
```

The heavy lifting is done by `mlx-audio`. This package provides:
- Clean, simple API (`KokoroTTS` class)
- Convenience functions (`synthesize()`)
- CLI interface

## Development Setup

```bash
uv venv
uv pip install -e ".[dev]"
```

## Key Components

### `KokoroTTS` class (tts.py)

Main wrapper class that:
- Lazily loads mlx-audio's Model and KokoroPipeline
- Provides `synthesize()` for one-shot generation
- Provides `stream()` for segment-by-segment generation
- Handles audio saving via soundfile

### CLI (cli.py)

Typer-based CLI with commands:
- `kokoro "text"` - Synthesize speech
- `kokoro voices` - List voices
- `kokoro info` - Show system info

## Commands

```bash
# Run tests
pytest tests/ -v

# Synthesize speech
kokoro "Hello world" -o output.wav

# List voices
kokoro --list-voices
```

## Dependencies

- `mlx-audio` - Backend TTS implementation
- `soundfile` - Audio file I/O
- `typer` - CLI framework

## Design Decisions

1. **Wrapper approach**: Rather than reimplementing 2000+ lines of complex audio processing code, we use mlx-audio as the backend. This ensures compatibility with pretrained weights and benefits from upstream improvements.

2. **Lazy loading**: Model is only loaded when first needed, keeping imports fast.

3. **Streaming support**: The `stream()` method allows processing long texts segment by segment, useful for real-time applications.
