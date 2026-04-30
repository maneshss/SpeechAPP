"""Score phrase difficulty so the app can pick easier / harder phrases."""

from __future__ import annotations

import re
from dataclasses import dataclass

try:
    import pronouncing
except ImportError:  # pragma: no cover
    pronouncing = None

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover
    def zipf_frequency(word: str, lang: str = "en") -> float:  # fallback
        return 4.0  # neutral guess


# Phoneme clusters that tend to trip up early readers
_HARD_CLUSTERS = ("str", "thr", "spl", "scr", "shr", "spr", "th", "ch", "sh")


def _syllables(word: str) -> int:
    """Estimate syllables for a word using the CMU dict, with a vowel-group fallback."""
    word = word.lower()
    if pronouncing is not None:
        phones = pronouncing.phones_for_word(word)
        if phones:
            return pronouncing.syllable_count(phones[0])
    # Fallback heuristic: count vowel groups
    groups = re.findall(r"[aeiouy]+", word)
    return max(1, len(groups))


def _cluster_penalty(word: str) -> float:
    w = word.lower()
    return sum(1 for c in _HARD_CLUSTERS if c in w)


@dataclass
class DifficultyBreakdown:
    score: float
    avg_syllables: float
    avg_rarity: float
    cluster_penalty: float


def score_phrase(phrase: str) -> DifficultyBreakdown:
    """Return a difficulty score (~1 = easy, ~5 = hard) plus its components.

    Components:
      - avg syllables per word
      - rarity (inverse of zipf frequency; common words ~6, rare ~2)
      - phoneme cluster penalty (per word)
    """
    words = [w for w in re.findall(r"[a-zA-Z']+", phrase) if w]
    if not words:
        return DifficultyBreakdown(1.0, 0.0, 0.0, 0.0)

    syllables = [_syllables(w) for w in words]
    rarities = [max(0.0, 7.0 - zipf_frequency(w.lower(), "en")) for w in words]
    clusters = [_cluster_penalty(w) for w in words]

    avg_syl = sum(syllables) / len(words)
    avg_rare = sum(rarities) / len(words)
    avg_cluster = sum(clusters) / len(words)

    # Weighted blend; tuned so a phrase of common monosyllables ~ 1
    raw = 0.5 * avg_syl + 0.8 * avg_rare + 0.7 * avg_cluster
    # Squash into roughly 1..5
    score = max(1.0, min(5.0, 0.6 * raw))

    return DifficultyBreakdown(score, avg_syl, avg_rare, avg_cluster)


def classify_level(phrase: str) -> int:
    """Bucket a phrase into integer difficulty 1..5."""
    return int(round(score_phrase(phrase).score))
