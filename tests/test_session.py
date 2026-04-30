"""Smoke tests for ReadingSession state transitions.

Run with:
    python -m pytest tests/
or  python tests/test_session.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.session import ReadingSession, State


def make_session() -> ReadingSession:
    return ReadingSession(
        phrases_by_level={1: ["the big dog"], 2: ["the green frog"]},
        level=1,
        consecutive_passes_to_level_up=2,
        consecutive_fails_to_level_down=1,
        max_word_attempts=2,
    )


def test_phrase_pass_advances_and_levels_up():
    rs = make_session()
    rs.pick_new_phrase()
    rs.submit_phrase_attempt("the big dog")
    assert rs.consecutive_passes == 1
    rs.submit_phrase_attempt(rs.current_phrase)  # whatever was picked
    # After 2 consecutive passes, level should go up
    assert rs.level == 2


def test_phrase_fail_splits_into_words():
    rs = make_session()
    rs.pick_new_phrase()
    rs.submit_phrase_attempt("xyz totally wrong")
    assert rs.state == State.WORD_PRACTICE
    assert rs.current_word == "the"


def test_word_loop_then_retry_phrase():
    rs = make_session()
    rs.pick_new_phrase()
    target = rs.current_phrase  # e.g. "the big dog"
    rs.submit_phrase_attempt("zzz")
    assert rs.state == State.WORD_PRACTICE

    for word in target.split():
        rs.submit_word_attempt(word)

    assert rs.state == State.RETRY_PHRASE


def test_retry_fail_drops_level():
    rs = make_session()
    rs.level = 2
    rs.pick_new_phrase()
    rs.submit_phrase_attempt("nope")
    for word in rs.current_phrase.split():
        # Fail every word until max attempts hit, so the loop advances
        rs.submit_word_attempt("zzz")
        rs.submit_word_attempt("zzz")
    assert rs.state == State.RETRY_PHRASE
    rs.submit_phrase_attempt("still nope")
    # Level should drop after retry failure
    assert rs.level == 1


if __name__ == "__main__":
    test_phrase_pass_advances_and_levels_up()
    test_phrase_fail_splits_into_words()
    test_word_loop_then_retry_phrase()
    test_retry_fail_drops_level()
    print("All tests passed.")
