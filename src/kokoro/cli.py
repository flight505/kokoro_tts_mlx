"""Command-line interface for Kokoro TTS."""

import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import typer

app = typer.Typer(
    name="kokoro",
    help="Kokoro TTS - Text-to-Speech for Apple Silicon",
    add_completion=False,
)


@app.command()
def synthesize(
    text: str = typer.Argument(..., help="Text to synthesize"),
    output: Path = typer.Option(
        Path("output.wav"),
        "-o", "--output",
        help="Output audio file path",
    ),
    voice: str = typer.Option(
        "af_heart",
        "-v", "--voice",
        help="Voice name (e.g., af_heart, am_adam, bf_emma)",
    ),
    speed: float = typer.Option(
        1.0,
        "-s", "--speed",
        help="Speech speed multiplier (0.5-2.0)",
    ),
    lang: str = typer.Option(
        "a",
        "-l", "--lang",
        help="Language code (a=American, b=British, j=Japanese, z=Chinese)",
    ),
    model: str = typer.Option(
        "mlx-community/Kokoro-82M-bf16",
        "-m", "--model",
        help="Hugging Face model ID",
    ),
    list_voices: bool = typer.Option(
        False,
        "--list-voices",
        help="List available voices and exit",
    ),
):
    """
    Synthesize speech from text.

    Examples:
        kokoro "Hello, world!"
        kokoro "Hello" -v am_adam -o hello.wav
        kokoro --list-voices
    """
    from kokoro.processing.voice import VoiceManager, VOICES

    # List voices mode
    if list_voices:
        typer.echo("Available voices:\n")

        categories = {
            "American English (Female)": ["af_"],
            "American English (Male)": ["am_"],
            "British English (Female)": ["bf_"],
            "British English (Male)": ["bm_"],
            "Japanese (Female)": ["jf_"],
            "Japanese (Male)": ["jm_"],
            "Chinese (Female)": ["zf_"],
            "Chinese (Male)": ["zm_"],
        }

        for cat_name, prefixes in categories.items():
            voices = [v for v in VOICES.keys() if any(v.startswith(p) for p in prefixes)]
            if voices:
                typer.echo(f"  {cat_name}:")
                for v in sorted(voices):
                    typer.echo(f"    - {v}")
                typer.echo("")

        return

    # Validate speed
    if not 0.5 <= speed <= 2.0:
        typer.echo("Error: Speed must be between 0.5 and 2.0", err=True)
        raise typer.Exit(1)

    # Load model
    typer.echo(f"Loading model from {model}...")
    start = time.time()

    from kokoro.utils import from_pretrained, save_audio
    from kokoro.processing.text import TextProcessor

    try:
        tts_model = from_pretrained(model)
    except Exception as e:
        typer.echo(f"Error loading model: {e}", err=True)
        raise typer.Exit(1)

    load_time = time.time() - start
    typer.echo(f"Model loaded in {load_time:.2f}s")

    # Load voice
    typer.echo(f"Loading voice: {voice}")
    voice_manager = VoiceManager(repo_id=model)

    try:
        voice_embedding = voice_manager.load_voice(voice)
    except Exception as e:
        typer.echo(f"Error loading voice: {e}", err=True)
        raise typer.Exit(1)

    # Text processing
    typer.echo("Processing text...")
    text_processor = TextProcessor(lang_code=lang, vocab=tts_model.vocab)

    # Convert to phonemes
    phonemes = text_processor.phonemize(text)
    typer.echo(f"Phonemes: {phonemes[:100]}{'...' if len(phonemes) > 100 else ''}")

    # Generate audio
    typer.echo("Generating audio...")
    start = time.time()

    try:
        audio, durations = tts_model(phonemes, voice_embedding, speed=speed)
    except Exception as e:
        typer.echo(f"Error during synthesis: {e}", err=True)
        raise typer.Exit(1)

    gen_time = time.time() - start
    audio_duration = len(audio) / tts_model.sample_rate

    typer.echo(f"Generated {audio_duration:.2f}s of audio in {gen_time:.2f}s")
    typer.echo(f"Real-time factor: {gen_time / audio_duration:.2f}x")

    # Save audio
    save_audio(audio, output, sample_rate=tts_model.sample_rate)
    typer.echo(f"Saved to: {output}")


@app.command()
def voices():
    """List all available voices."""
    from kokoro.processing.voice import VOICES, VOICE_CATEGORIES

    typer.echo("Available voices:\n")

    for category, voice_list in VOICE_CATEGORIES.items():
        category_name = category.replace("_", " ").title()
        typer.echo(f"{category_name}:")
        for v in voice_list:
            typer.echo(f"  - {v}")
        typer.echo("")


@app.command()
def info():
    """Show model and system information."""
    import sys

    typer.echo("Kokoro TTS MLX\n")
    typer.echo(f"Python: {sys.version}")
    typer.echo(f"MLX version: {mx.__version__}")

    # Check GPU
    try:
        typer.echo(f"MLX default device: {mx.default_device()}")
    except Exception:
        pass

    # Check dependencies
    typer.echo("\nDependencies:")

    deps = ["mlx", "huggingface_hub", "safetensors", "soundfile", "misaki"]
    for dep in deps:
        try:
            mod = __import__(dep.replace("-", "_"))
            version = getattr(mod, "__version__", "installed")
            typer.echo(f"  {dep}: {version}")
        except ImportError:
            typer.echo(f"  {dep}: NOT INSTALLED")


if __name__ == "__main__":
    app()
