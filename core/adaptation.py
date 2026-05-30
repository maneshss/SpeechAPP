"""Generate phonetically similar word alternatives for struggling words.

When a child struggles with a target word (e.g. "cat"), the system finds
phonetically proximate alternatives from the corpus word pool (e.g. "bat",
"mat", "cap") to use as scaffold practice targets.

Similarity score = Jaro-Winkler on CMU phoneme strings (same metric as
the mispronunciation detector in alignment.py).
"""

from __future__ import annotations

from typing import List

from .alignment import phoneme_jaro_winkler

_MIN_SIM = 0.40   # minimum similarity to be considered a valid alternative


def get_alternatives(
    target_word: str,
    word_pool: List[str],
    top_k: int = 3,
) -> List[str]:
    """Return up to top_k phonetically similar words from word_pool.

    Args:
        target_word: The word the child is struggling with.
        word_pool:   All unique words available in the corpus.
        top_k:       Maximum number of alternatives to return.

    Returns:
        List of alternative words sorted by decreasing phonetic similarity,
        filtered to those with similarity ≥ _MIN_SIM.
    """
    target = target_word.lower()
    scored: List[tuple[float, str]] = []

    for w in word_pool:
        w_lower = w.lower()
        if w_lower == target:
            continue
        sim = phoneme_jaro_winkler(target, w_lower)
        if sim >= _MIN_SIM:
            scored.append((sim, w_lower))

    scored.sort(key=lambda x: -x[0])
    return [w for _, w in scored[:top_k]]
