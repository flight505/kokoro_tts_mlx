# Kokoro TTS MLX

A clean, simple wrapper for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech on Apple Silicon.

Built on top of [mlx-audio](https://github.com/Blaizzy/mlx-audio), this package provides a streamlined API for speech synthesis.

## Features

- **Simple API** - One class, one function, done
- **82M parameters** - Fast and efficient
- **50+ voices** - American/British English, Japanese, Chinese
- **Apple Silicon optimized** - Runs efficiently on M1/M2/M3/M4

## Installation

```bash
# Clone the repository
git clone https://github.com/flight505/kokoro_tts_mlx.git
cd kokoro_tts_mlx

# Install with uv
uv venv
uv pip install -e .
```

## Quick Start

### Python API

```python
from kokoro import KokoroTTS, synthesize

# Quick synthesis
result = synthesize("Hello, world!", voice="af_heart", output="hello.wav")

# Or use the class for more control
tts = KokoroTTS(voice="am_adam", lang="a")
result = tts.synthesize("Welcome to Kokoro TTS!", speed=1.0)
tts.save(result.audio, "welcome.wav")

print(f"Generated {result.duration_seconds:.2f}s of audio")
```

### Streaming (for long texts)

```python
tts = KokoroTTS()

for segment in tts.stream("This is a long text that will be processed segment by segment..."):
    # Process each segment as it's generated
    print(f"Segment: {segment.duration_seconds:.2f}s")
    # Play or save segment.audio
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

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult: ...

    def stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> Generator[TTSResult, None, None]: ...

    def save(
        self,
        audio: mx.array,
        path: str | Path,
        sample_rate: int | None = None,
    ) -> None: ...

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
    phonemes: str | None     # Generated phonemes
    duration_seconds: float  # Audio duration
```

### `synthesize()` (convenience function)

```python
def synthesize(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    output: str | Path | None = None,
) -> TTSResult: ...
```

## License

Apache 2.0

## Acknowledgments

- [hexgrad](https://github.com/hexgrad) - Original Kokoro model
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) - MLX implementation
- [Apple MLX Team](https://github.com/ml-explore/mlx) - MLX framework
