"""Streamlit UI for SpeakUp! — Adaptive NLP-Driven Assistive Reading System.

Run with:
    streamlit run app.py

Notes:
    - First run downloads the Whisper model (a few hundred MB for base.en).
    - Requires ffmpeg installed at the OS level.
    - Corpus is loaded from data/Corpus word.xlsx (falls back to phrases.csv).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from core.session import ReadingSession, State
from core.corpus import LEVEL_LABELS

LOG_PATH = Path(__file__).parent / "debug_events.log"

# ---------------------------------------------------------------------------
# Optional dependencies — fail gracefully
# ---------------------------------------------------------------------------
try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

try:
    from core.stt import transcribe_bytes
except ImportError:
    transcribe_bytes = None


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def write_debug_event(event: dict) -> None:
    try:
        record = {"ts": datetime.utcnow().isoformat() + "Z", **event}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="SpeakUp!", page_icon="🗣️", layout="centered")


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------

def get_session() -> ReadingSession:
    if "rs" not in st.session_state:
        st.session_state.rs = ReadingSession.from_corpus()
    return st.session_state.rs


def reset_session() -> None:
    st.session_state.pop("rs", None)
    st.session_state.pop("last_audio", None)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def record_audio(key: str) -> bytes | None:
    counter_name = "_mic_counter"
    if counter_name not in st.session_state:
        st.session_state[counter_name] = 0
    st.session_state[counter_name] += 1
    unique_key = f"{key}_{st.session_state[counter_name]}"

    if mic_recorder is None:
        st.warning(
            "`streamlit-mic-recorder` is not installed. "
            "Run `pip install streamlit-mic-recorder` to enable recording."
        )
        return None

    audio = mic_recorder(
        start_prompt="🎤 Tap to record",
        stop_prompt="⏹ Stop",
        just_once=True,
        use_container_width=True,
        key=unique_key,
    )
    if audio and "bytes" in audio:
        st.session_state["_last_audio_len"] = len(audio.get("bytes", b""))
        return audio["bytes"]
    return None


def transcribe(audio_bytes: bytes) -> str:
    if transcribe_bytes is None:
        st.error("Whisper is not installed. Run `pip install openai-whisper`.")
        return ""
    with st.spinner("Listening…"):
        return transcribe_bytes(audio_bytes)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

_TIER_COLOUR = {1: "#4CAF50", 2: "#FF9800", 3: "#F44336"}
_TIER_EMOJI  = {1: "🟢", 2: "🟡", 3: "🔴"}


def _tier_badge(level: int) -> str:
    label = LEVEL_LABELS.get(level, str(level))
    colour = _TIER_COLOUR.get(level, "#888")
    return (
        f"<span style='background:{colour};color:white;"
        f"padding:3px 10px;border-radius:12px;font-size:14px;'>"
        f"{_TIER_EMOJI.get(level,'')} {label}</span>"
    )


def render_header(rs: ReadingSession) -> None:
    col1, col2, col3 = st.columns([2, 1, 1])
    col1.markdown("### 🗣️ SpeakUp!")
    col2.markdown(_tier_badge(rs.level), unsafe_allow_html=True)
    col3.metric("Streak", rs.consecutive_passes)


def _handle_spoken(spoken: str, rs: ReadingSession, mode: str) -> None:
    """Process a transcribed utterance in phrase or retry mode."""
    st.write(f"_I heard:_ **{spoken}**")
    result = rs.submit_phrase_attempt(spoken)

    alignment = rs.last_alignment
    if alignment:
        _render_alignment(alignment)

    if result.passed:
        st.success(f"Great job! ({result.method}, score {result.score:.0f})")
    else:
        st.warning("Let's try one word at a time.")
    st.rerun()


def _render_alignment(alignment) -> None:
    """Show a colour-coded word-by-word alignment breakdown."""
    _colour = {
        "correct":          "#4CAF50",
        "mispronunciation": "#FF9800",
        "substitution":     "#F44336",
        "omission":         "#9E9E9E",
        "insertion":        "#2196F3",
    }
    parts = []
    for wa in alignment.alignments:
        c = _colour.get(wa.error_type, "#888")
        label = wa.target if wa.target else f"[+{wa.spoken}]"
        tooltip = wa.error_type
        if wa.error_type not in ("correct", "omission", "insertion"):
            tooltip += f" (heard: '{wa.spoken}', sim {wa.phoneme_sim:.2f})"
        parts.append(
            f"<span title='{tooltip}' style='color:{c};font-weight:bold;"
            f"margin:0 3px;font-size:18px;'>{label}</span>"
        )
    st.markdown(" ".join(parts), unsafe_allow_html=True)

    bd = alignment.error_breakdown
    if bd:
        cols = st.columns(len(bd))
        for col, (k, v) in zip(cols, bd.items()):
            col.metric(k.capitalize(), v)
    st.caption(f"Phoneme accuracy: {alignment.phoneme_accuracy:.0%}")


def render_phrase_screen(rs: ReadingSession) -> None:
    st.markdown(
        f"<h1 style='text-align:center;font-size:64px;margin:40px 0;'>"
        f"{rs.current_phrase}</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Read the sentence out loud, then tap the mic.")

    audio = record_audio(key=f"phrase_{rs.current_phrase}")
    if audio:
        st.write("DEBUG: recorded bytes:", len(audio))
        try:
            spoken = transcribe(audio)
        except Exception as e:
            st.error("Transcription error")
            st.exception(e)
            return
        _handle_spoken(spoken, rs, "phrase")

    # Simulation helper
    with st.expander("Simulate (debug / no mic)"):
        sim_text = st.text_input("Type what you'd say:", key="sim_phrase")
        if st.button("Submit simulation", key="sim_phrase_btn") and sim_text:
            _handle_spoken(sim_text, rs, "phrase")


def render_word_screen(rs: ReadingSession) -> None:
    word = rs.current_word
    words = rs.current_phrase.split()
    total = len(words)

    st.progress(
        rs.current_word_index / max(1, total),
        text=f"Word {rs.current_word_index + 1} of {total}",
    )
    st.markdown(
        f"<h1 style='text-align:center;font-size:96px;margin:40px 0;'>{word}</h1>",
        unsafe_allow_html=True,
    )
    st.caption(f"Phrase: *{rs.current_phrase}* — say just this word.")

    # Word alternatives for practice
    alts = rs.word_alternatives(word)
    if alts:
        st.info(f"🔄 Similar-sounding words to practise: **{', '.join(alts)}**")

    audio = record_audio(
        key=f"word_{word}_{rs.current_word_index}_{rs.current_word_attempts}"
    )
    if audio:
        st.write("DEBUG: recorded bytes:", len(audio))
        try:
            spoken = transcribe(audio)
            st.write(f"_I heard:_ **{spoken}**")
        except Exception as e:
            st.error("Transcription error")
            st.exception(e)
            return
        _process_word(spoken, rs)

    # Simulation helper
    with st.expander("Simulate (debug / no mic)"):
        sim_w = st.text_input("Type the word:", key=f"sim_word_{rs.current_word_index}")
        if st.button("Submit simulation", key=f"sim_word_btn_{rs.current_word_index}") and sim_w:
            _process_word(sim_w, rs)


def _process_word(spoken: str, rs: ReadingSession) -> None:
    result = rs.submit_word_attempt(spoken)
    st.write(f"_I heard:_ **{spoken}**")
    if result.passed:
        st.success(f"Yes! ({result.method})")
    else:
        remaining = rs.max_word_attempts - rs.current_word_attempts
        if remaining > 0:
            st.warning(f"Try again — {remaining} attempt(s) left.")
        else:
            st.info("That's okay, let's keep going.")
    st.rerun()


def render_retry_screen(rs: ReadingSession) -> None:
    st.info("🔁 Now try the whole sentence again!")
    render_phrase_screen(rs)


def render_history(rs: ReadingSession) -> None:
    with st.expander("Session history"):
        if not rs.history:
            st.write("No attempts yet.")
            return
        rows = rs.export_history()
        # Flatten error_breakdown dict into string for display
        for r in rows:
            r["error_breakdown"] = str(r.get("error_breakdown", {}))
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download as CSV",
            data=df.to_csv(index=False).encode(),
            file_name="speakup_session.csv",
            mime="text/csv",
        )


def render_debug(rs: ReadingSession) -> None:
    with st.expander("Debug: session internals", expanded=False):
        st.write("state:", rs.state.value)
        st.write("level:", rs.level, "→", rs.level_label)
        st.write("current_phrase:", rs.current_phrase)
        st.write("current_word_index:", rs.current_word_index)
        st.write("current_word_attempts:", rs.current_word_attempts)
        st.write("recent_phoneme_accuracy:", f"{rs.recent_phoneme_accuracy:.0%}")
        st.write("consecutive_passes:", rs.consecutive_passes)
        st.write("consecutive_fails:", rs.consecutive_fails)
        st.write("struggling_words:", rs._struggling_words)
        st.write("mic_recorder_available:", mic_recorder is not None)
        st.write("last_audio_len:", st.session_state.get("_last_audio_len"))

        if rs._rl is not None:
            st.write("RL Q-table (tier × accuracy_bucket × action):")
            import numpy as np
            q = rs._rl.q_values
            for t_idx in range(q.shape[0]):
                tier_name = LEVEL_LABELS.get(t_idx + 1, str(t_idx + 1))
                st.write(f"  {tier_name}: easier={q[t_idx,1,0]:.3f} same={q[t_idx,1,1]:.3f} harder={q[t_idx,1,2]:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

rs = get_session()

with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("Restart session"):
        reset_session()
        st.rerun()
    st.markdown(f"**Level:** {_tier_badge(rs.level)}", unsafe_allow_html=True)
    st.write("**State:**", rs.state.value)
    total_sentences = sum(len(v) for v in rs.phrases_by_level.values())
    st.write("**Corpus size:**", total_sentences, "sentences")
    st.write("**Phoneme accuracy (recent):**", f"{rs.recent_phoneme_accuracy:.0%}")
    st.markdown("---")
    st.caption(
        "**Tiers:** 🟢 Beginner · 🟡 Intermediate · 🔴 Advanced\n\n"
        "The RL engine automatically adjusts difficulty based on your performance."
    )

render_header(rs)

if rs.state == State.DONE:
    st.balloons()
    st.success("🎉 All done — great reading today!")
elif rs.state == State.SHOW_PHRASE:
    render_phrase_screen(rs)
elif rs.state == State.WORD_PRACTICE:
    render_word_screen(rs)
elif rs.state == State.RETRY_PHRASE:
    render_retry_screen(rs)
else:
    st.write("Loading…")

render_history(rs)
render_debug(rs)
