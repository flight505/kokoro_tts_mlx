"""Text processing and G2P (grapheme-to-phoneme) conversion."""

import re
from typing import Generator


# Default phoneme vocabulary (subset - full vocab loaded from model config)
DEFAULT_VOCAB = {
    " ": 1,
    "!": 2,
    '"': 3,
    "#": 4,
    "$": 5,
    "%": 6,
    "&": 7,
    "'": 8,
    "(": 9,
    ")": 10,
    "*": 11,
    "+": 12,
    ",": 13,
    "-": 14,
    ".": 15,
    "/": 16,
    "0": 17,
    "1": 18,
    "2": 19,
    "3": 20,
    "4": 21,
    "5": 22,
    "6": 23,
    "7": 24,
    "8": 25,
    "9": 26,
    ":": 27,
    ";": 28,
}


class TextProcessor:
    """
    Text processor for Kokoro TTS.

    Handles text normalization and G2P conversion using misaki.
    """

    def __init__(self, lang_code: str = "a", vocab: dict[str, int] | None = None):
        """
        Initialize text processor.

        Args:
            lang_code: Language code
                - 'a': American English
                - 'b': British English
                - 'j': Japanese
                - 'z': Mandarin Chinese
            vocab: Phoneme vocabulary mapping
        """
        self.lang_code = lang_code
        self.vocab = vocab or DEFAULT_VOCAB
        self._g2p = None
        self._espeak_fallback = None

    @property
    def g2p(self):
        """Lazy-load G2P converter."""
        if self._g2p is None:
            self._init_g2p()
        return self._g2p

    def _init_g2p(self):
        """Initialize G2P backend based on language."""
        try:
            if self.lang_code in ("a", "b"):
                # English - use misaki
                from misaki import en

                self._g2p = en.G2P(british=(self.lang_code == "b"))
            elif self.lang_code == "j":
                # Japanese
                from misaki import ja

                self._g2p = ja.G2P()
            elif self.lang_code == "z":
                # Mandarin
                from misaki import zh

                self._g2p = zh.G2P()
            else:
                # Fallback to espeak
                self._init_espeak()
        except ImportError as e:
            print(f"Warning: Could not import misaki for lang={self.lang_code}: {e}")
            print("Falling back to espeak-ng")
            self._init_espeak()

    def _init_espeak(self):
        """Initialize espeak-ng fallback."""
        try:
            from phonemizer import phonemize
            from phonemizer.backend.espeak.wrapper import EspeakWrapper

            self._espeak_fallback = True
        except ImportError:
            print("Warning: phonemizer not installed. Text-to-phoneme conversion unavailable.")
            self._espeak_fallback = False

    def phonemize(self, text: str) -> str:
        """
        Convert text to phonemes.

        Args:
            text: Input text

        Returns:
            Phoneme string
        """
        if self._g2p is not None:
            try:
                # misaki returns tuple (phonemes, tokens)
                result = self._g2p(text)
                if isinstance(result, tuple):
                    return result[0]
                return result
            except Exception as e:
                print(f"G2P error: {e}, falling back to espeak")

        if self._espeak_fallback:
            from phonemizer import phonemize

            lang_map = {
                "a": "en-us",
                "b": "en-gb",
                "j": "ja",
                "z": "zh",
            }
            lang = lang_map.get(self.lang_code, "en-us")

            return phonemize(
                text,
                language=lang,
                backend="espeak",
                strip=True,
                preserve_punctuation=True,
            )

        # Last resort: return text as-is
        return text

    def tokenize(self, phonemes: str) -> list[int]:
        """
        Convert phoneme string to token IDs.

        Args:
            phonemes: Phoneme string

        Returns:
            List of token IDs
        """
        tokens = []
        for p in phonemes:
            if p in self.vocab:
                tokens.append(self.vocab[p])
        return tokens

    def chunk_text(
        self,
        text: str,
        max_chars: int = 500,
        split_pattern: str = r"\n+",
    ) -> Generator[str, None, None]:
        """
        Split text into chunks for processing.

        Args:
            text: Input text
            max_chars: Maximum characters per chunk
            split_pattern: Regex pattern for splitting

        Yields:
            Text chunks
        """
        # First split by pattern (usually newlines)
        parts = re.split(split_pattern, text)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(part) <= max_chars:
                yield part
            else:
                # Split long parts at sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", part)
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= max_chars:
                        current_chunk += (" " if current_chunk else "") + sentence
                    else:
                        if current_chunk:
                            yield current_chunk
                        current_chunk = sentence

                if current_chunk:
                    yield current_chunk


def phonemize(text: str, lang_code: str = "a") -> str:
    """
    Convenience function for text-to-phoneme conversion.

    Args:
        text: Input text
        lang_code: Language code ('a'=American English, 'b'=British, 'j'=Japanese, 'z'=Chinese)

    Returns:
        Phoneme string
    """
    processor = TextProcessor(lang_code=lang_code)
    return processor.phonemize(text)
