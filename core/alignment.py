"""Word-level sequence alignment using Needleman-Wunsch global alignment.

Each word-level deviation is classified as:
  correct          – exact or phonetically near-identical (Jaro-Winkler ≥ 0.85)
  mispronunciation – phonetically similar (JW 0.50–0.85)
  substitution     – phonetically dissimilar replacement
  omission         – target word was skipped
  insertion        – extra spoken word not in target
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import re

try:
    import pronouncing  # type: ignore
except ImportError:
    pronouncing = None

try:
    from rapidfuzz.distance import JaroWinkler as _JW

    def _jaro_winkler(a: str, b: str) -> float:
        return _JW.normalized_similarity(a, b)

except Exception:
    from rapidfuzz import fuzz as _fuzz

    def _jaro_winkler(a: str, b: str) -> float:
        return _fuzz.ratio(a, b) / 100.0


_CORRECT_THRESHOLD = 0.85    # JW phoneme sim: counts as correct
_MISPRON_THRESHOLD = 0.50    # JW phoneme sim: mispronunciation
_GAP_PENALTY = -1.0


# ---------------------------------------------------------------------------
# Phoneme utilities
# ---------------------------------------------------------------------------

def _phoneme_str(word: str) -> str:
    """Return compact lowercase phoneme string for a word (stress markers stripped).

    Falls back to the word itself when not found in the CMU dictionary.
    Always returns lowercase so in-dict and out-of-dict words compare consistently.
    """
    word = word.lower()
    if pronouncing is not None:
        phones_list = pronouncing.phones_for_word(word)
        if phones_list:
            return re.sub(r"\d", "", phones_list[0]).replace(" ", "").lower()
    return word


def phoneme_jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler similarity on CMU phoneme strings (0–1)."""
    return _jaro_winkler(_phoneme_str(a), _phoneme_str(b))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class WordAlignment:
    target: str
    spoken: str
    error_type: str   # "correct" | "mispronunciation" | "substitution" | "omission" | "insertion"
    phoneme_sim: float


@dataclass
class AlignmentResult:
    alignments: List[WordAlignment] = field(default_factory=list)

    @property
    def target_alignments(self) -> List[WordAlignment]:
        """Alignments that correspond to target words (excludes insertions)."""
        return [a for a in self.alignments if a.error_type != "insertion"]

    @property
    def phoneme_accuracy(self) -> float:
        """Fraction of target words produced correctly (exact or near-phonetic)."""
        ta = self.target_alignments
        if not ta:
            return 0.0
        correct = sum(1 for a in ta if a.error_type == "correct")
        return correct / len(ta)

    @property
    def error_breakdown(self) -> dict:
        from collections import Counter
        return dict(Counter(a.error_type for a in self.alignments))

    @property
    def struggling_words(self) -> List[str]:
        """Target words the child got wrong."""
        return [
            a.target for a in self.target_alignments
            if a.error_type in ("substitution", "omission", "mispronunciation")
        ]


# ---------------------------------------------------------------------------
# Needleman-Wunsch global alignment
# ---------------------------------------------------------------------------

def _word_sim(tw: str, sw: str) -> float:
    """Alignment score between two words."""
    if tw.lower() == sw.lower():
        return 2.0
    sim = phoneme_jaro_winkler(tw, sw)
    if sim >= _CORRECT_THRESHOLD:
        return 1.5   # phonetic near-match → reward but not full score
    if sim >= _MISPRON_THRESHOLD:
        return 0.0   # mispronunciation → neutral
    return -1.0      # substitution → penalty


def align_words(target_words: List[str], spoken_words: List[str]) -> AlignmentResult:
    """Global Needleman-Wunsch alignment at word level.

    Returns an AlignmentResult with one WordAlignment per aligned pair.
    """
    m, n = len(target_words), len(spoken_words)

    # Build DP table + traceback arrows
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    tb = [[""] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i * _GAP_PENALTY
        tb[i][0] = "U"  # up → omission (target present, spoken gap)
    for j in range(n + 1):
        dp[0][j] = j * _GAP_PENALTY
        tb[0][j] = "L"  # left → insertion (spoken present, target gap)
    tb[0][0] = ""

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = dp[i - 1][j - 1] + _word_sim(target_words[i - 1], spoken_words[j - 1])
            up   = dp[i - 1][j] + _GAP_PENALTY
            left = dp[i][j - 1] + _GAP_PENALTY
            best = max(diag, up, left)
            dp[i][j] = best
            if best == diag:
                tb[i][j] = "D"
            elif best == up:
                tb[i][j] = "U"
            else:
                tb[i][j] = "L"

    # Traceback
    aligned_t: List[str | None] = []
    aligned_s: List[str | None] = []
    i, j = m, n
    while i > 0 or j > 0:
        direction = tb[i][j]
        if direction == "D":
            aligned_t.append(target_words[i - 1])
            aligned_s.append(spoken_words[j - 1])
            i -= 1
            j -= 1
        elif direction == "U":
            aligned_t.append(target_words[i - 1])
            aligned_s.append(None)
            i -= 1
        else:
            aligned_t.append(None)
            aligned_s.append(spoken_words[j - 1])
            j -= 1

    aligned_t.reverse()
    aligned_s.reverse()

    # Classify each aligned pair
    result = AlignmentResult()
    for tw, sw in zip(aligned_t, aligned_s):
        if tw is None:
            result.alignments.append(WordAlignment("", sw or "", "insertion", 0.0))
        elif sw is None:
            result.alignments.append(WordAlignment(tw, "", "omission", 0.0))
        else:
            sim = phoneme_jaro_winkler(tw, sw)
            if tw.lower() == sw.lower() or sim >= _CORRECT_THRESHOLD:
                error_type = "correct"
            elif sim >= _MISPRON_THRESHOLD:
                error_type = "mispronunciation"
            else:
                error_type = "substitution"
            result.alignments.append(WordAlignment(tw, sw, error_type, sim))

    return result
