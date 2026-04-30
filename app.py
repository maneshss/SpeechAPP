"""Streamlit UI for the adaptive reading practice app.

Run with:
    streamlit run app.py

Notes:
    - First run downloads the Whisper model (a few hundred MB for base.en).
    - Requires ffmpeg installed at the OS level.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core.session import ReadingSession, State

# Optional imports — fail gracefully so the UI still loads if a dep is missing.
try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

try:
    from core.stt import transcribe_bytes
except ImportError:
    transcribe_bytes = None


DATA_PATH = Path(__file__).parent / "data" / "phrases.csv"

st.set_page_config(page_title="Reading Buddy", page_icon=":book:", layout="centered")


# ---------- session bootstrap ----------
def get_session() -> ReadingSession:
    if "rs" not in st.session_state:
        st.session_state.rs = ReadingSession.from_csv(DATA_PATH)
    return st.session_state.rs


def reset_session() -> None:
    st.session_state.pop("rs", None)
    st.session_state.pop("last_audio", None)


# ---------- audio capture ----------
def record_audio(key: str) -> bytes | None:
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
        key=key,
    )
    if audio and "bytes" in audio:
        return audio["bytes"]
    return None


def transcribe(audio_bytes: bytes) -> str:
    if transcribe_bytes is None:
        st.error("Whisper isn't installed. Run `pip install openai-whisper`.")
        return ""
    with st.spinner("Listening..."):
        return transcribe_bytes(audio_bytes)


# ---------- UI ----------
def render_header(rs: ReadingSession) -> None:
    col1, col2, col3 = st.columns([2, 1, 1])
    col1.markdown(f"### Reading Buddy")
    col2.metric("Level", rs.level)
    col3.metric("Streak", rs.consecutive_passes)


def render_phrase_screen(rs: ReadingSession) -> None:
    st.markdown(
        f"<h1 style='text-align:center; font-size:64px; margin:40px 0;'>"
        f"{rs.current_phrase}</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Read the phrase out loud, then tap the mic.")
    audio = record_audio(key=f"phrase_{rs.current_phrase}")
    if audio:
        spoken = transcribe(audio)
        st.write(f"_I heard:_ **{spoken}**")
        result = rs.submit_phrase_attempt(spoken)
        if result.passed:
            st.success(f"Great job! ({result.method}, {result.score:.0f})")
        else:
            st.warning("Let's try one word at a time.")
        st.rerun()


def render_word_screen(rs: ReadingSession) -> None:
    word = rs.current_word
    total = len(rs.current_phrase.split())
    st.progress((rs.current_word_index) / max(1, total),
                text=f"Word {rs.current_word_index + 1} of {total}")
    st.markdown(
        f"<h1 style='text-align:center; font-size:96px; margin:40px 0;'>"
        f"{word}</h1>",
        unsafe_allow_html=True,
    )
    st.caption(f"Phrase: *{rs.current_phrase}* — say just this word.")
    audio = record_audio(key=f"word_{word}_{rs.current_word_index}_{rs.current_word_attempts}")
    if audio:
        spoken = transcribe(audio)
        st.write(f"_I heard:_ **{spoken}**")
        result = rs.submit_word_attempt(spoken)
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
    st.info("Now try the whole phrase again!")
    render_phrase_screen(rs)


def render_history(rs: ReadingSession) -> None:
    with st.expander("Session history"):
        if not rs.history:
            st.write("No attempts yet.")
            return
        df = pd.DataFrame(rs.export_history())
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download as CSV",
            data=df.to_csv(index=False).encode(),
            file_name="reading_session.csv",
            mime="text/csv",
        )


# ---------- main ----------
rs = get_session()

with st.sidebar:
    st.header("Settings")
    if st.button("Restart session"):
        reset_session()
        st.rerun()
    st.write("**Level**:", rs.level)
    st.write("**State**:", rs.state.value)
    st.write("**Phrases loaded**:",
             sum(len(v) for v in rs.phrases_by_level.values()))

render_header(rs)

if rs.state == State.DONE:
    st.balloons()
    st.success("All done — great reading today!")
elif rs.state in (State.SHOW_PHRASE,):
    render_phrase_screen(rs)
elif rs.state == State.WORD_PRACTICE:
    render_word_screen(rs)
elif rs.state == State.RETRY_PHRASE:
    render_retry_screen(rs)
else:
    st.write("Loading...")

render_history(rs)
