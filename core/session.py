"""ReadingSession state machine.

Keeps learning logic separate from Streamlit so it can be unit-tested
and ported to other UIs.

Flow:
    SHOW_PHRASE  → (child reads) →
        pass   → next phrase (RL may step level up)
        fail   → WORD_PRACTICE (word-by-word scaffold)
            each word: pass → advance | fail × N → mark struggling, advance
            after all words → RETRY_PHRASE
                pass → next phrase
                fail → EASIER_PHRASE (RL may step level down)
"""

from __future__ import annotations

import csv
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Deque, Dict, List, Optional

from .alignment import AlignmentResult, align_words
from .compare import CompareResult, compare_phrase, compare_word
from .corpus import LEVEL_LABELS, all_words, load_corpus

try:
    from .rl_engine import RLEngine
except Exception:
    RLEngine = None  # type: ignore

try:
    from .adaptation import get_alternatives
except Exception:
    get_alternatives = None  # type: ignore


class State(str, Enum):
    SHOW_PHRASE = "show_phrase"
    SPLIT_TO_WORDS = "split_to_words"
    WORD_PRACTICE = "word_practice"
    RETRY_PHRASE = "retry_phrase"
    DONE = "done"


@dataclass
class AttemptLog:
    state: State
    target: str
    spoken: str
    passed: bool
    method: str
    score: float
    level: int
    phoneme_accuracy: float = 0.0
    error_breakdown: dict = field(default_factory=dict)


@dataclass
class ReadingSession:
    phrases_by_level: Dict[int, List[str]]
    level: int = 1
    max_level: int = 3
    min_level: int = 1
    consecutive_passes_to_level_up: int = 2
    consecutive_fails_to_level_down: int = 1
    max_word_attempts: int = 3

    # runtime state
    state: State = State.SHOW_PHRASE
    current_phrase: str = ""
    current_word_index: int = 0
    current_word_attempts: int = 0
    consecutive_passes: int = 0
    consecutive_fails: int = 0
    history: List[AttemptLog] = field(default_factory=list)
    _struggling_words: List[str] = field(default_factory=list)
    _last_alignment: Optional[AlignmentResult] = field(default=None, repr=False)
    _word_pool: List[str] = field(default_factory=list)
    _rl: Optional[object] = field(default=None, repr=False)
    _recent_accuracies: Deque[float] = field(
        default_factory=lambda: deque(maxlen=5), repr=False
    )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_corpus(cls, xlsx_path: str | Path | None = None, **kwargs) -> "ReadingSession":
        """Create a session from the Excel corpus (primary entry point)."""
        corpus = load_corpus(xlsx_path)
        word_pool = all_words(corpus)
        rl = RLEngine() if RLEngine is not None else None
        session = cls(
            phrases_by_level=corpus,
            max_level=3,
            _word_pool=word_pool,
            _rl=rl,
            **kwargs,
        )
        session.pick_new_phrase()
        return session

    @classmethod
    def from_csv(cls, csv_path: str | Path, **kwargs) -> "ReadingSession":
        """Legacy loader: read phrases.csv with a 'difficulty' column."""
        phrases_by_level: Dict[int, List[str]] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lvl = int(row["difficulty"])
                phrases_by_level.setdefault(lvl, []).append(row["phrase"].strip())
        session = cls(phrases_by_level=phrases_by_level, **kwargs)
        session.pick_new_phrase()
        return session

    # ------------------------------------------------------------------
    # Phrase / word selection
    # ------------------------------------------------------------------

    def pick_new_phrase(self) -> str:
        """Choose a new phrase at the current level (with fallback)."""
        for lvl in (self.level, self.level - 1, self.level + 1):
            if lvl in self.phrases_by_level and self.phrases_by_level[lvl]:
                pool = [p for p in self.phrases_by_level[lvl] if p != self.current_phrase]
                if pool:
                    self.current_phrase = random.choice(pool)
                    self.state = State.SHOW_PHRASE
                    self.current_word_index = 0
                    self.current_word_attempts = 0
                    return self.current_phrase
        self.state = State.DONE
        return ""

    @property
    def current_word(self) -> str:
        words = self.current_phrase.split()
        if 0 <= self.current_word_index < len(words):
            return words[self.current_word_index]
        return ""

    @property
    def level_label(self) -> str:
        return LEVEL_LABELS.get(self.level, str(self.level))

    @property
    def last_alignment(self) -> Optional[AlignmentResult]:
        return self._last_alignment

    @property
    def recent_phoneme_accuracy(self) -> float:
        if not self._recent_accuracies:
            return 0.5
        return sum(self._recent_accuracies) / len(self._recent_accuracies)

    def word_alternatives(self, word: str) -> List[str]:
        """Return phonetically similar alternatives for a struggling word."""
        if get_alternatives is None or not self._word_pool:
            return []
        return get_alternatives(word, self._word_pool, top_k=3)

    # ------------------------------------------------------------------
    # Submission handlers
    # ------------------------------------------------------------------

    def submit_phrase_attempt(self, spoken: str) -> CompareResult:
        result = compare_phrase(self.current_phrase, spoken)

        # Alignment-based diagnostics (independent of pass/fail decision)
        target_words = self.current_phrase.split()
        spoken_words = spoken.split() if spoken.strip() else []
        alignment = align_words(target_words, spoken_words)
        self._last_alignment = alignment
        phon_acc = alignment.phoneme_accuracy
        self._recent_accuracies.append(phon_acc)

        log_state = (
            State.SHOW_PHRASE if self.state == State.SHOW_PHRASE else State.RETRY_PHRASE
        )
        self._log(
            log_state,
            self.current_phrase,
            spoken,
            result,
            phon_acc,
            alignment.error_breakdown,
        )

        if result.passed:
            self._on_phrase_pass()
        else:
            if self.state == State.RETRY_PHRASE:
                self._on_phrase_fail_after_retry()
            else:
                self.state = State.WORD_PRACTICE
                self.current_word_index = 0
                self.current_word_attempts = 0
                self._struggling_words = []

        return result

    def submit_word_attempt(self, spoken: str) -> CompareResult:
        target_word = self.current_word
        result = compare_word(target_word, spoken)
        self._log(State.WORD_PRACTICE, target_word, spoken, result)
        self.current_word_attempts += 1

        if result.passed:
            self._advance_word()
        elif self.current_word_attempts >= self.max_word_attempts:
            self._struggling_words.append(target_word)
            self._advance_word()

        return result

    # ------------------------------------------------------------------
    # Internal transitions
    # ------------------------------------------------------------------

    def _advance_word(self) -> None:
        self.current_word_index += 1
        self.current_word_attempts = 0
        if self.current_word_index >= len(self.current_phrase.split()):
            self.state = State.RETRY_PHRASE

    def _on_phrase_pass(self) -> None:
        self.consecutive_fails = 0
        if self._rl is not None:
            acc = self.recent_phoneme_accuracy
            new_tier = self._rl.select_tier(self.level, acc)
            self._rl.update(new_tier, acc)
            self.level = max(self.min_level, min(self.max_level, new_tier))
            self.consecutive_passes = 0
        else:
            self.consecutive_passes += 1
            if self.consecutive_passes >= self.consecutive_passes_to_level_up:
                self.level = min(self.max_level, self.level + 1)
                self.consecutive_passes = 0
        self.pick_new_phrase()

    def _on_phrase_fail_after_retry(self) -> None:
        self.consecutive_passes = 0
        if self._rl is not None:
            acc = self.recent_phoneme_accuracy
            new_tier = self._rl.select_tier(self.level, acc)
            self._rl.update(new_tier, acc)
            self.level = max(self.min_level, min(self.max_level, new_tier))
            self.consecutive_fails = 0
        else:
            self.consecutive_fails += 1
            if self.consecutive_fails >= self.consecutive_fails_to_level_down:
                self.level = max(self.min_level, self.level - 1)
                self.consecutive_fails = 0
        self.pick_new_phrase()

    # ------------------------------------------------------------------
    # Logging / export
    # ------------------------------------------------------------------

    def _log(
        self,
        state: State,
        target: str,
        spoken: str,
        result: CompareResult,
        phoneme_accuracy: float = 0.0,
        error_breakdown: dict | None = None,
    ) -> None:
        self.history.append(
            AttemptLog(
                state=state,
                target=target,
                spoken=spoken,
                passed=result.passed,
                method=result.method,
                score=result.score,
                level=self.level,
                phoneme_accuracy=phoneme_accuracy,
                error_breakdown=error_breakdown or {},
            )
        )

    def export_history(self) -> List[Dict]:
        return [
            {**a.__dict__, "state": a.state.value}
            for a in self.history
        ]
