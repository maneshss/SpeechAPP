"""ReadingSession state machine.

Keeps the learning logic separate from Streamlit so it can be unit-tested
and ported to other UIs later.

Flow:
    SHOW_PHRASE -> (child reads) -> EVALUATE
        pass -> next phrase (level can step up)
        fail -> SPLIT_TO_WORDS
            for each word: WORD_PRACTICE
                pass -> next word
                fail (after N tries) -> next word anyway, mark struggling
            after all words -> RETRY_PHRASE
                pass -> next phrase
                fail -> EASIER_PHRASE (drop a level)
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List

from .compare import CompareResult, compare_phrase, compare_word


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


@dataclass
class ReadingSession:
    phrases_by_level: Dict[int, List[str]]
    level: int = 1
    max_level: int = 5
    min_level: int = 1
    consecutive_passes_to_level_up: int = 2
    consecutive_fails_to_level_down: int = 1
    max_word_attempts: int = 3

    # ---- runtime state ----
    state: State = State.SHOW_PHRASE
    current_phrase: str = ""
    current_word_index: int = 0
    current_word_attempts: int = 0
    consecutive_passes: int = 0
    consecutive_fails: int = 0
    history: List[AttemptLog] = field(default_factory=list)
    _struggling_words: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, csv_path: str | Path, **kwargs) -> "ReadingSession":
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
        """Choose a new phrase at the current level (falling back to nearby levels)."""
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

    # ------------------------------------------------------------------
    # Submission handlers (called by the UI when audio is transcribed)
    # ------------------------------------------------------------------
    def submit_phrase_attempt(self, spoken: str) -> CompareResult:
        result = compare_phrase(self.current_phrase, spoken)
        self._log(State.SHOW_PHRASE if self.state == State.SHOW_PHRASE else State.RETRY_PHRASE,
                  self.current_phrase, spoken, result)

        if result.passed:
            self._on_phrase_pass()
        else:
            if self.state == State.RETRY_PHRASE:
                # Failed even after word-by-word practice -> drop a level, new phrase
                self._on_phrase_fail_after_retry()
            else:
                # First failure on a new phrase -> split into words
                self.state = State.SPLIT_TO_WORDS
                self.current_word_index = 0
                self.current_word_attempts = 0
                self._struggling_words = []
                self.state = State.WORD_PRACTICE
        return result

    def submit_word_attempt(self, spoken: str) -> CompareResult:
        target_word = self.current_word
        result = compare_word(target_word, spoken)
        self._log(State.WORD_PRACTICE, target_word, spoken, result)

        self.current_word_attempts += 1

        if result.passed:
            self._advance_word()
        elif self.current_word_attempts >= self.max_word_attempts:
            # Move on but remember the word was hard
            self._struggling_words.append(target_word)
            self._advance_word()
        # else: stay on this word, let the UI prompt them again

        return result

    # ------------------------------------------------------------------
    # Internal transitions
    # ------------------------------------------------------------------
    def _advance_word(self) -> None:
        self.current_word_index += 1
        self.current_word_attempts = 0
        if self.current_word_index >= len(self.current_phrase.split()):
            # Finished all words, retry the whole phrase
            self.state = State.RETRY_PHRASE

    def _on_phrase_pass(self) -> None:
        self.consecutive_passes += 1
        self.consecutive_fails = 0
        if self.consecutive_passes >= self.consecutive_passes_to_level_up:
            self.level = min(self.max_level, self.level + 1)
            self.consecutive_passes = 0
        self.pick_new_phrase()

    def _on_phrase_fail_after_retry(self) -> None:
        self.consecutive_fails += 1
        self.consecutive_passes = 0
        if self.consecutive_fails >= self.consecutive_fails_to_level_down:
            self.level = max(self.min_level, self.level - 1)
            self.consecutive_fails = 0
        self.pick_new_phrase()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, state: State, target: str, spoken: str, result: CompareResult) -> None:
        self.history.append(
            AttemptLog(
                state=state,
                target=target,
                spoken=spoken,
                passed=result.passed,
                method=result.method,
                score=result.score,
                level=self.level,
            )
        )

    def export_history(self) -> List[Dict]:
        return [a.__dict__ | {"state": a.state.value} for a in self.history]
