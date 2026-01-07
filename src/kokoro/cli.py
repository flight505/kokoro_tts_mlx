"""Command-line interface for Kokoro TTS."""

import time
from pathlib import Path

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
    from kokoro import KokoroTTS

    # List voices mode
    if list_voices:
        voices = KokoroTTS.list_voices()
        typer.echo("Available voices:\n")
        for category, voice_list in voices.items():
            category_name = category.replace("_", " ").title()
            typer.echo(f"{category_name}:")
            for v in voice_list:
                typer.echo(f"  - {v}")
            typer.echo("")
        return

    # Validate speed
    if not 0.5 <= speed <= 2.0:
        typer.echo("Error: Speed must be between 0.5 and 2.0", err=True)
        raise typer.Exit(1)

    # Initialize TTS
    typer.echo(f"Initializing Kokoro TTS with voice: {voice}")
    start = time.time()

    try:
        tts = KokoroTTS(voice=voice, lang=lang)
    except Exception as e:
        typer.echo(f"Error initializing TTS: {e}", err=True)
        raise typer.Exit(1)

    # Generate audio
    typer.echo(f"Generating speech...")

    try:
        result = tts.synthesize(text, speed=speed)
    except Exception as e:
        typer.echo(f"Error during synthesis: {e}", err=True)
        raise typer.Exit(1)

    gen_time = time.time() - start

    typer.echo(f"Generated {result.duration_seconds:.2f}s of audio in {gen_time:.2f}s")
    typer.echo(f"Real-time factor: {gen_time / result.duration_seconds:.2f}x")

    # Save audio
    tts.save(result.audio, output)
    typer.echo(f"Saved to: {output}")


@app.command()
def voices():
    """List all available voices."""
    from kokoro import KokoroTTS

    voices = KokoroTTS.list_voices()
    typer.echo("Available voices:\n")
    for category, voice_list in voices.items():
        category_name = category.replace("_", " ").title()
        typer.echo(f"{category_name}:")
        for v in voice_list:
            typer.echo(f"  - {v}")
        typer.echo("")


@app.command()
def info():
    """Show system information."""
    import sys
    import mlx.core as mx

    typer.echo("Kokoro TTS MLX\n")
    typer.echo(f"Python: {sys.version}")
    typer.echo(f"MLX version: {mx.__version__}")

    try:
        typer.echo(f"MLX default device: {mx.default_device()}")
    except Exception:
        pass

    # Check mlx-audio
    typer.echo("\nBackend:")
    try:
        import mlx_audio
        typer.echo(f"  mlx-audio: {getattr(mlx_audio, '__version__', 'installed')}")
    except ImportError:
        typer.echo("  mlx-audio: NOT INSTALLED")


if __name__ == "__main__":
    app()
