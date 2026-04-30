"""Compare what the child said against the target phrase / word.

Three-tier strategy, in order of strictness:
  1. Exact (after normalization)
  2. Fuzzy string match (handles small ASR errors)
  3. Phonetic match (handles mispronunciations like "wabbit" vs "rabbit")

Returns a structured result so the UI can show *why* a comparison passed/failed.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import List, Optional

from rapidfuzz import fuzz

try:
    from metaphone import doublemetaphone
except ImportError:  # pragma: no cover - optional dependency
    doublemetaphone = None


# ---------- normalization ----------

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def tokens(text: str) -> List[str]:
    return normalize(text).split()


# ---------- phonetic ----------

def phonetic_key(word: str) -> str:
    """Get a phonetic encoding for a single word.

    Uses Double Metaphone if available, else falls back to a simple
    consonant-skeleton encoding.
    """
    word = normalize(word)
    if not word:
        return ""
    if doublemetaphone is not None:
        primary, _secondary = doublemetaphone(word)
        return primary or word
    # Fallback: drop vowels (very crude, but better than nothing)
    return re.sub(r"[aeiou]", "", word)


def phonetic_match(a: str, b: str) -> bool:
    """Compare single words by phonetic encoding."""
    return phonetic_key(a) == phonetic_key(b) and phonetic_key(a) != ""


# ---------- result type ----------

@dataclass
class CompareResult:
    passed: bool
    method: str          # "exact" | "fuzzy" | "phonetic" | "none"
    score: float         # 0..100 fuzzy score
    target: str
    spoken: str
    # Per-word breakdown (only filled in for phrase comparisons)
    word_results: Optional[List["CompareResult"]] = None

    def __bool__(self) -> bool:  # allow `if result:`
        return self.passed


# ---------- public API ----------

def compare_word(
    target: str,
    spoken: str,
    *,
    fuzzy_threshold: int = 85,
) -> CompareResult:
    """Compare a single spoken word against the target."""
    t = normalize(target)
    s = normalize(spoken)

    if not s:
        return CompareResult(False, "none", 0.0, t, s)

    if t == s:
        return CompareResult(True, "exact", 100.0, t, s)

    score = fuzz.ratio(t, s)
    if score >= fuzzy_threshold:
        return CompareResult(True, "fuzzy", score, t, s)

    if phonetic_match(t, s):
        return CompareResult(True, "phonetic", score, t, s)

    return CompareResult(False, "none", score, t, s)


def compare_phrase(
    target: str,
    spoken: str,
    *,
    fuzzy_threshold: int = 85,
) -> CompareResult:
    """Compare a multi-word phrase. Also reports per-word matches."""
    t_norm = normalize(target)
    s_norm = normalize(spoken)

    # Whole-phrase fast path
    if t_norm == s_norm:
        return CompareResult(True, "exact", 100.0, t_norm, s_norm)

    overall_score = fuzz.token_sort_ratio(t_norm, s_norm)

    # Per-word breakdown (positionally aligned)
    t_tokens = t_norm.split()
    s_tokens = s_norm.split()
    word_results: List[CompareResult] = []
    for i, t_word in enumerate(t_tokens):
        s_word = s_tokens[i] if i < len(s_tokens) else ""
        word_results.append(compare_word(t_word, s_word, fuzzy_threshold=fuzzy_threshold))

    all_words_pass = all(w.passed for w in word_results) and len(s_tokens) >= len(t_tokens)

    if all_words_pass:
        method = "exact" if all(w.method == "exact" for w in word_results) else "fuzzy"
        return CompareResult(True, method, overall_score, t_norm, s_norm, word_results)

    if overall_score >= fuzzy_threshold:
        return CompareResult(True, "fuzzy", overall_score, t_norm, s_norm, word_results)

    return CompareResult(False, "none", overall_score, t_norm, s_norm, word_results)
