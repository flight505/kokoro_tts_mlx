# Kokoro TTS MLX

A clean, simple wrapper for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech on Apple Silicon.

Built on top of [mlx-audio](https://github.com/Blaizzy/mlx-audio), this package provides a streamlined API for speech synthesis with **three performance levels** to match your needs.

## Features

- **Three API levels** - Zero-overhead `generate()`, streaming `stream()`, or convenient `synthesize()`
- **82M parameters** - Compact and efficient
- **50+ voices** - American/British English, Japanese, Chinese
- **~20-40x real-time** - Generates 1 minute of audio in ~2-3 seconds
- **Apple Silicon optimized** - Runs efficiently on M1/M2/M3/M4

## System Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Chip** | Apple M1 | Apple M1 Pro or better |
| **RAM** | 8 GB | 16 GB+ |
| **macOS** | 13.0 (Ventura) | 14.0+ (Sonoma) |
| **Disk** | ~500 MB | ~1 GB (with cache) |

### Resource Usage

| Metric | Value |
|--------|-------|
| **Model Size** | 339 MB (bf16) / 297 MB (4-bit) |
| **Memory During Inference** | ~1-2 GB |
| **First Load Time** | 2-5 seconds |
| **Subsequent Loads** | <1 second (cached) |

### Performance by Device (approximate)

| Device | Real-Time Factor | 1 min audio generates in |
|--------|------------------|--------------------------|
| **M1** | ~0.05x | ~3 seconds |
| **M1 Pro/Max** | ~0.03-0.04x | ~2 seconds |
| **M2** | ~0.04x | ~2.5 seconds |
| **M2 Pro/Max/Ultra** | ~0.02-0.03x | ~1.5 seconds |
| **M3** | ~0.03x | ~2 seconds |
| **M3 Pro/Max** | ~0.02x | ~1.2 seconds |

> **Note**: Real-Time Factor (RTF) < 1.0 means faster than real-time. RTF of 0.03 = 33x faster than real-time.

### Can My Mac Run This?

- **M1 MacBook Air (8GB)**: ✅ Yes - Works well for short-to-medium texts
- **M1 Pro/Max MacBook Pro**: ✅ Yes - Excellent performance
- **M2/M3 MacBook Air**: ✅ Yes - Great performance
- **M2/M3 Pro/Max**: ✅ Yes - Best performance
- **Intel Mac**: ❌ No - Requires Apple Silicon

## Installation

```bash
# Clone the repository
git clone https://github.com/flight505/kokoro_tts_mlx.git
cd kokoro_tts_mlx

# Install with uv (recommended)
uv venv
uv pip install -e .

# Or with pip
pip install -e .
```

### System Dependencies

```bash
# Required for text-to-phoneme conversion
brew install espeak-ng
```

## Quick Start

```python
from kokoro import KokoroTTS

tts = KokoroTTS()
result = tts.synthesize("Hello, world!")
tts.save(result.audio, "hello.wav")
```

## API Levels: Choosing the Right One

This package provides **three API levels** with different performance/convenience tradeoffs:

### 1. `generate()` - Zero Overhead (Production)

Direct passthrough to mlx-audio. Use this for **maximum performance** in production.

```python
tts = KokoroTTS()

for result in tts.generate("Your text here"):
    audio = result.audio
    # Full mlx-audio metrics available:
    print(f"RTF: {result.real_time_factor:.3f}x")
    print(f"Processing time: {result.processing_time_seconds:.2f}s")
```

**Best for**: Production APIs, high-volume processing, when you need performance metrics.

### 2. `stream()` - Low Overhead (Real-time)

Yields audio segments as they're generated. Good for **real-time playback**.

```python
tts = KokoroTTS()

for segment in tts.stream("Long text to process..."):
    # Play or process each segment immediately
    play_audio(segment.audio, segment.sample_rate)
```

**Best for**: Real-time applications, playing audio while generating, progress feedback.

### 3. `synthesize()` - Convenient (Scripts)

Returns all audio concatenated. Most **convenient** but has overhead on long texts.

```python
tts = KokoroTTS()

result = tts.synthesize("Hello, world!")
tts.save(result.audio, "output.wav")
print(f"Duration: {result.duration_seconds:.1f}s")
```

**Best for**: Quick scripts, short texts, prototyping, when convenience > performance.

### Performance Comparison

| Text Length | `generate()` | `stream()` | `synthesize()` |
|-------------|--------------|------------|----------------|
| Short (30 chars) | baseline | ~0% overhead | ~0% overhead |
| Medium (160 chars) | baseline | ~1% overhead | ~9% overhead |
| Long (500+ chars) | baseline | ~12% overhead | ~20% overhead |

**Why the overhead?**
- `stream()`: Creates `TTSResult` wrapper for each segment
- `synthesize()`: Collects all segments + concatenates audio arrays

For short texts, use whichever is most convenient. For long texts in production, use `generate()`.

## Command Line

```bash
# Basic synthesis
kokoro synthesize "Hello, world!"

# With options
kokoro synthesize "Hello, world!" -v am_adam -o hello.wav -s 1.2

# List available voices
kokoro voices

# Show system info
kokoro info
```

## Available Voices

### American English
- **Female**: af_heart, af_bella, af_nova, af_sarah, af_nicole, af_sky, af_alloy, af_aoede, af_jessica, af_kore, af_river, af_sage
- **Male**: am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck

### British English
- **Female**: bf_emma, bf_isabella, bf_alice, bf_lily
- **Male**: bm_george, bm_lewis, bm_daniel, bm_fable

### Japanese
- **Female**: jf_alpha, jf_gongitsune
- **Male**: jm_kumo

### Chinese
- **Female**: zf_xiaobei, zf_xiaoni, zf_xiaoxuan
- **Male**: zm_yunjian, zm_yunxi, zm_yunyang

## Model Variants

All variants are available on HuggingFace and perform **nearly identically** on Apple Silicon:

| Model | Size | Notes |
|-------|------|-------|
| `mlx-community/Kokoro-82M-bf16` | 339 MB | Default, full precision |
| `mlx-community/Kokoro-82M-8bit` | 303 MB | 8-bit quantized |
| `mlx-community/Kokoro-82M-4bit` | 297 MB | 4-bit quantized, smallest |

```python
# Use a specific variant
tts = KokoroTTS(model_id="mlx-community/Kokoro-82M-4bit")
```

> **Note**: Quantization saves disk space but doesn't improve inference speed on Apple Silicon. The bf16 model is recommended unless disk space is constrained.

## API Reference

### `KokoroTTS`

```python
class KokoroTTS:
    def __init__(
        self,
        model_id: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "af_heart",
        lang: str = "a",  # a=American, b=British, j=Japanese, z=Chinese
    ): ...

    def generate(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> Generator[GenerationResult, None, None]:
        """Zero-overhead generation. Returns raw mlx-audio results."""
        ...

    def stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> Generator[TTSResult, None, None]:
        """Stream segments as TTSResult objects."""
        ...

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Synthesize and return concatenated audio."""
        ...

    def save(
        self,
        audio: mx.array,
        path: str | Path,
        sample_rate: int | None = None,
    ) -> None:
        """Save audio to WAV file."""
        ...

    @property
    def model(self) -> Model:
        """Direct access to underlying mlx-audio model."""
        ...

    @staticmethod
    def list_voices() -> dict[str, list[str]]: ...
```

### `TTSResult`

```python
@dataclass
class TTSResult:
    audio: mx.array          # Audio waveform
    sample_rate: int         # Sample rate (24000 Hz)
    text: str                # Input text
    phonemes: str | None     # Generated phonemes (if available)
    duration_seconds: float  # Audio duration
```

### `synthesize()` (convenience function)

```python
from kokoro import synthesize

# One-liner synthesis
result = synthesize("Hello!", voice="af_heart", output="hello.wav")
```

## Benchmarking

Run the included benchmark to test performance on your system:

```bash
uv run python benchmarks/benchmark.py
```

## Troubleshooting

### "No module named 'espeak'"
```bash
brew install espeak-ng
```

### Slow first generation
The first generation downloads voice data (~50MB per voice). Subsequent generations are fast.

### Out of memory on 8GB Mac
Try shorter text segments or use the 4-bit quantized model:
```python
tts = KokoroTTS(model_id="mlx-community/Kokoro-82M-4bit")
```

## License

Apache 2.0

## Acknowledgments

- [hexgrad](https://github.com/hexgrad) - Original Kokoro model
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) - MLX implementation
- [Apple MLX Team](https://github.com/ml-explore/mlx) - MLX framework
