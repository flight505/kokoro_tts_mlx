"""Voice loading and management for Kokoro TTS."""

from pathlib import Path
from typing import Literal

import mlx.core as mx
from huggingface_hub import hf_hub_download


# Available voice presets by language
VOICES = {
    # American English
    "af_heart": "voices/af_heart.safetensors",
    "af_bella": "voices/af_bella.safetensors",
    "af_nova": "voices/af_nova.safetensors",
    "af_sarah": "voices/af_sarah.safetensors",
    "af_nicole": "voices/af_nicole.safetensors",
    "af_sky": "voices/af_sky.safetensors",
    "am_adam": "voices/am_adam.safetensors",
    "am_michael": "voices/am_michael.safetensors",

    # British English
    "bf_emma": "voices/bf_emma.safetensors",
    "bf_isabella": "voices/bf_isabella.safetensors",
    "bm_george": "voices/bm_george.safetensors",
    "bm_lewis": "voices/bm_lewis.safetensors",

    # Japanese
    "jf_alpha": "voices/jf_alpha.safetensors",
    "jf_gongitsune": "voices/jf_gongitsune.safetensors",
    "jm_kumo": "voices/jm_kumo.safetensors",

    # Chinese
    "zf_xiaobei": "voices/zf_xiaobei.safetensors",
    "zf_xiaoni": "voices/zf_xiaoni.safetensors",
    "zf_xiaoxuan": "voices/zf_xiaoxuan.safetensors",
    "zm_yunjian": "voices/zm_yunjian.safetensors",
}

# Voice categories
VOICE_CATEGORIES = {
    "american_female": ["af_heart", "af_bella", "af_nova", "af_sarah", "af_nicole", "af_sky"],
    "american_male": ["am_adam", "am_michael"],
    "british_female": ["bf_emma", "bf_isabella"],
    "british_male": ["bm_george", "bm_lewis"],
    "japanese_female": ["jf_alpha", "jf_gongitsune"],
    "japanese_male": ["jm_kumo"],
    "chinese_female": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxuan"],
    "chinese_male": ["zm_yunjian"],
}


class VoiceManager:
    """Manages voice loading and caching for Kokoro TTS."""

    def __init__(self, repo_id: str = "mlx-community/Kokoro-82M-bf16"):
        """
        Initialize voice manager.

        Args:
            repo_id: Hugging Face repository ID
        """
        self.repo_id = repo_id
        self._cache: dict[str, mx.array] = {}

    def list_voices(
        self,
        lang: Literal["all", "en-us", "en-gb", "ja", "zh"] = "all"
    ) -> list[str]:
        """
        List available voices.

        Args:
            lang: Filter by language

        Returns:
            List of voice names
        """
        if lang == "all":
            return list(VOICES.keys())

        prefix_map = {
            "en-us": ["af_", "am_"],
            "en-gb": ["bf_", "bm_"],
            "ja": ["jf_", "jm_"],
            "zh": ["zf_", "zm_"],
        }

        prefixes = prefix_map.get(lang, [])
        return [v for v in VOICES.keys() if any(v.startswith(p) for p in prefixes)]

    def load_voice(
        self,
        voice_name: str,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> mx.array:
        """
        Load a voice embedding.

        Args:
            voice_name: Name of the voice (e.g., "af_heart")
            dtype: Data type for the voice tensor

        Returns:
            Voice embedding tensor (1, 256)
        """
        # Check cache
        cache_key = f"{voice_name}_{dtype}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get voice file path
        if voice_name not in VOICES:
            available = ", ".join(sorted(VOICES.keys())[:10])
            raise ValueError(
                f"Unknown voice: {voice_name}. Available: {available}..."
            )

        voice_path = VOICES[voice_name]

        # Download from HuggingFace
        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=voice_path,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download voice {voice_name}: {e}")

        # Load safetensors
        from safetensors import safe_open

        with safe_open(local_path, framework="numpy") as f:
            # Voice files typically have a single tensor
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"Empty voice file: {voice_name}")

            # Load the voice tensor
            voice_data = f.get_tensor(keys[0])

        # Convert to MLX
        voice = mx.array(voice_data, dtype=dtype)

        # Ensure shape is (1, 256)
        if voice.ndim == 1:
            voice = voice[None, :]

        # Cache
        self._cache[cache_key] = voice

        return voice

    def blend_voices(
        self,
        voice_names: list[str],
        weights: list[float] | None = None,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> mx.array:
        """
        Blend multiple voices together.

        Args:
            voice_names: List of voice names to blend
            weights: Optional weights for each voice (defaults to equal)
            dtype: Data type for the voice tensor

        Returns:
            Blended voice embedding tensor (1, 256)
        """
        if weights is None:
            weights = [1.0 / len(voice_names)] * len(voice_names)

        if len(weights) != len(voice_names):
            raise ValueError("Number of weights must match number of voices")

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        # Load and blend
        blended = None
        for name, weight in zip(voice_names, weights):
            voice = self.load_voice(name, dtype=dtype)
            if blended is None:
                blended = voice * weight
            else:
                blended = blended + voice * weight

        return blended

    def clear_cache(self):
        """Clear the voice cache."""
        self._cache.clear()


def load_voice(
    voice_name: str,
    repo_id: str = "mlx-community/Kokoro-82M-bf16",
    dtype: mx.Dtype = mx.bfloat16,
) -> mx.array:
    """
    Convenience function to load a voice.

    Args:
        voice_name: Name of the voice (e.g., "af_heart")
        repo_id: Hugging Face repository ID
        dtype: Data type for the voice tensor

    Returns:
        Voice embedding tensor (1, 256)
    """
    manager = VoiceManager(repo_id=repo_id)
    return manager.load_voice(voice_name, dtype=dtype)
