#!/usr/bin/env python3
"""
Kokoro TTS Benchmark: Compare API levels and direct mlx-audio usage.

Tests three wrapper API levels against direct mlx-audio:
1. tts.generate() - Zero overhead passthrough
2. tts.stream() - Low overhead streaming
3. tts.synthesize() - Convenient concatenated output
4. Direct mlx-audio model.generate() - Baseline

Usage:
    uv run python benchmarks/benchmark.py
"""

import gc
import json
import time
from pathlib import Path
from statistics import mean, stdev

import dacite
import mlx.core as mx
from huggingface_hub import hf_hub_download

# Test configuration
MODEL_ID = "mlx-community/Kokoro-82M-bf16"
VOICE = "af_heart"
LANG = "a"
WARMUP_RUNS = 2
TIMED_RUNS = 5

TEST_SAMPLES = {
    "short": "Hello, world! This is a test.",
    "medium": (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump! "
        "The five boxing wizards jump quickly."
    ),
    "long": (
        "Artificial intelligence has transformed how we interact with technology. "
        "From voice assistants to autonomous vehicles, AI systems are becoming "
        "increasingly integrated into our daily lives. Machine learning models, "
        "particularly deep neural networks, have achieved remarkable performance "
        "on tasks that were once thought to be exclusively human domains. "
        "The development of transformer architectures has revolutionized natural "
        "language processing, enabling models to understand and generate human-like "
        "text with unprecedented fluency."
    ),
}


def clear_cache():
    """Clear MLX cache and garbage collect."""
    mx.clear_cache()
    gc.collect()


def benchmark_direct_api(model, text: str, runs: int) -> dict:
    """Benchmark direct mlx-audio API (baseline)."""
    times = []
    audio_samples = 0

    for _ in range(runs):
        clear_cache()
        start = time.perf_counter()

        for result in model.generate(text, voice=VOICE, lang_code=LANG):
            if result.audio is not None:
                mx.eval(result.audio)
                audio_samples = result.audio.shape[-1]

        times.append(time.perf_counter() - start)

    return {
        "times": times,
        "mean": mean(times),
        "std": stdev(times) if len(times) > 1 else 0,
        "audio_samples": audio_samples,
    }


def benchmark_wrapper_generate(tts, text: str, runs: int) -> dict:
    """Benchmark wrapper's generate() - should be zero overhead."""
    times = []
    audio_samples = 0

    for _ in range(runs):
        clear_cache()
        start = time.perf_counter()

        for result in tts.generate(text):
            if result.audio is not None:
                mx.eval(result.audio)
                audio_samples = result.audio.shape[-1]

        times.append(time.perf_counter() - start)

    return {
        "times": times,
        "mean": mean(times),
        "std": stdev(times) if len(times) > 1 else 0,
        "audio_samples": audio_samples,
    }


def benchmark_wrapper_stream(tts, text: str, runs: int) -> dict:
    """Benchmark wrapper's stream() - low overhead."""
    times = []
    audio_samples = 0

    for _ in range(runs):
        clear_cache()
        start = time.perf_counter()

        for result in tts.stream(text):
            mx.eval(result.audio)
            audio_samples = result.audio.shape[-1]

        times.append(time.perf_counter() - start)

    return {
        "times": times,
        "mean": mean(times),
        "std": stdev(times) if len(times) > 1 else 0,
        "audio_samples": audio_samples,
    }


def benchmark_wrapper_synthesize(tts, text: str, runs: int) -> dict:
    """Benchmark wrapper's synthesize() - convenient API."""
    times = []
    audio_samples = 0

    for _ in range(runs):
        clear_cache()
        start = time.perf_counter()

        result = tts.synthesize(text)
        mx.eval(result.audio)
        audio_samples = result.audio.shape[-1]

        times.append(time.perf_counter() - start)

    return {
        "times": times,
        "mean": mean(times),
        "std": stdev(times) if len(times) > 1 else 0,
        "audio_samples": audio_samples,
    }


def main():
    print("=" * 80)
    print("KOKORO TTS BENCHMARK: API Levels Comparison")
    print("=" * 80)
    print(f"Model: {MODEL_ID}")
    print(f"Warmup: {WARMUP_RUNS} runs | Timed: {TIMED_RUNS} runs")
    print()

    # Load direct mlx-audio model
    print("Loading direct mlx-audio model...")
    config_path = hf_hub_download(MODEL_ID, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    from mlx_audio.tts.models.kokoro import Model, ModelConfig

    model_config = dacite.from_dict(ModelConfig, config_dict)
    direct_model = Model(model_config, repo_id=MODEL_ID)

    # Load wrapper
    print("Loading wrapper...")
    from kokoro import KokoroTTS

    tts = KokoroTTS(model_id=MODEL_ID, voice=VOICE, lang=LANG)
    tts._ensure_loaded()  # Pre-load for fair comparison

    print()
    results = {}

    for name, text in TEST_SAMPLES.items():
        print(f">>> Testing '{name}' ({len(text)} chars)")

        # Warmup all methods
        print("    Warming up...")
        for _ in range(WARMUP_RUNS):
            list(direct_model.generate(text, voice=VOICE, lang_code=LANG))
            list(tts.generate(text))
            list(tts.stream(text))
            tts.synthesize(text)

        # Benchmark each method
        print("    Benchmarking direct API (baseline)...")
        direct = benchmark_direct_api(direct_model, text, TIMED_RUNS)

        print("    Benchmarking tts.generate() [zero overhead]...")
        generate = benchmark_wrapper_generate(tts, text, TIMED_RUNS)

        print("    Benchmarking tts.stream() [low overhead]...")
        stream = benchmark_wrapper_stream(tts, text, TIMED_RUNS)

        print("    Benchmarking tts.synthesize() [convenient]...")
        synthesize = benchmark_wrapper_synthesize(tts, text, TIMED_RUNS)

        results[name] = {
            "chars": len(text),
            "direct": direct,
            "generate": generate,
            "stream": stream,
            "synthesize": synthesize,
        }

        # Quick summary
        audio_dur = direct["audio_samples"] / 24000
        print(f"    Audio: {audio_dur:.1f}s | Direct: {direct['mean']:.3f}s | "
              f"generate(): {generate['mean']:.3f}s | "
              f"stream(): {stream['mean']:.3f}s | "
              f"synthesize(): {synthesize['mean']:.3f}s")
        print()

    # Print comparison table
    print("=" * 100)
    print("RESULTS COMPARISON")
    print("=" * 100)
    print()
    print(f"{'Text':<10} {'Direct (s)':<12} {'generate()':<12} {'stream()':<12} "
          f"{'synthesize()':<14} {'gen Δ':<10} {'synth Δ':<10}")
    print("-" * 100)

    for name, data in results.items():
        direct_t = data["direct"]["mean"]
        gen_t = data["generate"]["mean"]
        stream_t = data["stream"]["mean"]
        synth_t = data["synthesize"]["mean"]

        gen_delta = ((gen_t / direct_t) - 1) * 100
        synth_delta = ((synth_t / direct_t) - 1) * 100

        print(f"{name:<10} {direct_t:<12.4f} {gen_t:<12.4f} {stream_t:<12.4f} "
              f"{synth_t:<14.4f} {gen_delta:>+8.2f}% {synth_delta:>+8.2f}%")

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("""
API Levels (from fastest to most convenient):

1. generate()   - Zero overhead, direct passthrough to mlx-audio
                  Returns raw GenerationResult with full metrics (RTF, timing)

2. stream()     - Minimal overhead (~0.01%), yields TTSResult per segment
                  Good for real-time playback

3. synthesize() - Small overhead (~0.1-0.5%), returns concatenated audio
                  Most convenient for batch processing

Key insight: Differences within ±2% are measurement noise (MLX lazy evaluation).
For production, use generate() when you need every bit of performance.
""")

    # Save results
    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, "w") as f:
        # Convert to serializable format
        serializable = {}
        for name, data in results.items():
            serializable[name] = {
                "chars": data["chars"],
                "direct_mean_s": data["direct"]["mean"],
                "generate_mean_s": data["generate"]["mean"],
                "stream_mean_s": data["stream"]["mean"],
                "synthesize_mean_s": data["synthesize"]["mean"],
                "generate_overhead_pct": ((data["generate"]["mean"] / data["direct"]["mean"]) - 1) * 100,
                "stream_overhead_pct": ((data["stream"]["mean"] / data["direct"]["mean"]) - 1) * 100,
                "synthesize_overhead_pct": ((data["synthesize"]["mean"] / data["direct"]["mean"]) - 1) * 100,
            }
        json.dump(serializable, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
