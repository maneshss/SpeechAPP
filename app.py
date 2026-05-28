"""Streamlit UI for SpeakUp! — Adaptive NLP-Driven Assistive Reading System.

Run with:
    streamlit run app.py

Notes:
    - First run downloads the Whisper model (a few hundred MB for base.en).
    - Requires ffmpeg at the OS level (brew install ffmpeg / apt install ffmpeg).
    - Corpus loaded from data/Corpus word.xlsx (falls back to phrases.csv).

Screen flow (phrase):
    LISTEN  →  RECORD  →  CHECKING  →  RESULT  →  (next phrase)

Screen flow (word-by-word):
    LISTEN  →  RECORD  →  RESULT  →  (next word)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import html as _html

import pandas as pd
import streamlit as st
import streamlit.components.v1 as _components

from core.session import ReadingSession, State
from core.corpus import LEVEL_LABELS

LOG_PATH = Path(__file__).parent / "debug_events.log"

# ---------------------------------------------------------------------------
# Optional dependencies
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
    # Keep only keys that are not phase/result/audio caches
    keep = {"rs"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]
    st.session_state.pop("rs", None)


# ---------------------------------------------------------------------------
# TTS  — Web Speech API (browser-native, no network, no autoplay restriction)
# ---------------------------------------------------------------------------

def _speak(text: str, *, auto: bool = True, button_label: str = "🔊 Play again") -> None:
    """Embed a Web Speech API speaker inside the page.

    auto=True  → speaks immediately when the component loads (once per phrase,
                  tracked via session state so reruns don't re-trigger it).
    Always renders a button so the user can replay manually.
    """
    safe = _html.escape(text, quote=True)
    auto_js = "speakNow();" if auto else ""
    _components.html(
        f"""
        <div style="text-align:center;font-family:sans-serif;padding:4px 0;">
          <button id="sp-btn" onclick="speakNow()"
              style="font-size:15px;padding:8px 24px;border-radius:8px;
                     cursor:pointer;background:#1565C0;color:#fff;border:none;
                     margin-bottom:6px;">
            {button_label}
          </button>
          <div id="sp-status" style="color:#888;font-size:13px;min-height:18px;"></div>
        </div>
        <script>
          const TTS_TEXT = "{safe}";
          function speakNow() {{
            const btn  = document.getElementById('sp-btn');
            const stat = document.getElementById('sp-status');
            if (!window.speechSynthesis) {{
              stat.textContent = '⚠ Text-to-speech not supported in this browser.';
              return;
            }}
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(TTS_TEXT);
            u.rate = 0.85;
            u.lang = 'en-US';
            u.onstart = () => {{ stat.textContent = '🔊 Speaking…'; btn.disabled = true; }};
            u.onend   = () => {{ stat.textContent = '✅ Done — click Ready when you are.'; btn.disabled = false; }};
            u.onerror = (e) => {{ stat.textContent = '⚠ Playback error: ' + e.error; btn.disabled = false; }};
            window.speechSynthesis.speak(u);
          }}
          {auto_js}
        </script>
        """,
        height=80,
    )


@st.cache_data(show_spinner=False)
def _gtts_bytes(text: str) -> bytes | None:
    """Offline fallback: return MP3 bytes via gTTS, or None on failure."""
    try:
        from gtts import gTTS  # type: ignore
        import io
        buf = io.BytesIO()
        gTTS(text=text, lang="en", slow=True).write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Microphone recorder wrapper
# ---------------------------------------------------------------------------

def _record_audio(key: str) -> bytes | None:
    """Show the mic recorder widget; return audio bytes when recording ends."""
    if mic_recorder is None:
        return None

    # Append a counter so each rerun gets a fresh widget instance
    if "_mic_ctr" not in st.session_state:
        st.session_state["_mic_ctr"] = 0
    st.session_state["_mic_ctr"] += 1
    widget_key = f"{key}_{st.session_state['_mic_ctr']}"

    audio = mic_recorder(
        start_prompt="🎤  Tap to start recording",
        stop_prompt="⏹  Tap to stop",
        just_once=True,
        use_container_width=True,
        key=widget_key,
    )
    if audio and "bytes" in audio:
        return audio["bytes"]
    return None


def _do_transcribe(audio_bytes: bytes) -> str:
    """Transcribe audio bytes; raises on failure."""
    if transcribe_bytes is None:
        raise RuntimeError("Whisper is not installed. Run: pip install openai-whisper")
    return transcribe_bytes(audio_bytes)


# ---------------------------------------------------------------------------
# Shared UI widgets
# ---------------------------------------------------------------------------

_TIER_COLOUR = {1: "#4CAF50", 2: "#FF9800", 3: "#F44336"}
_TIER_EMOJI  = {1: "🟢", 2: "🟡", 3: "🔴"}


def _tier_badge(level: int) -> str:
    label = LEVEL_LABELS.get(level, str(level))
    colour = _TIER_COLOUR.get(level, "#888")
    return (
        f"<span style='background:{colour};color:white;"
        f"padding:4px 12px;border-radius:12px;font-size:14px;font-weight:bold;'>"
        f"{_TIER_EMOJI.get(level, '')} {label}</span>"
    )


def _step_indicator(current: int) -> None:
    """Render a 3-step progress strip: Listen → Record → Result."""
    labels = ["🎧 Listen", "🎤 Record", "✅ Result"]
    cols = st.columns(3)
    for i, (col, label) in enumerate(zip(cols, labels)):
        step = i + 1
        if step < current:
            col.markdown(
                f"<div style='text-align:center;color:#aaa;font-size:13px;'>"
                f"<s>{label}</s></div>",
                unsafe_allow_html=True,
            )
        elif step == current:
            col.markdown(
                f"<div style='text-align:center;font-weight:bold;font-size:14px;"
                f"color:#1565C0;border-bottom:3px solid #1565C0;padding-bottom:4px;'>"
                f"{label}</div>",
                unsafe_allow_html=True,
            )
        else:
            col.markdown(
                f"<div style='text-align:center;color:#bbb;font-size:13px;'>"
                f"{label}</div>",
                unsafe_allow_html=True,
            )
    st.write("")   # small gap


def _render_alignment(alignment) -> None:
    """Colour-coded word-level alignment breakdown."""
    _colour = {
        "correct":          "#2E7D32",
        "mispronunciation": "#E65100",
        "substitution":     "#C62828",
        "omission":         "#757575",
        "insertion":        "#1565C0",
    }
    parts = []
    for wa in alignment.alignments:
        c = _colour.get(wa.error_type, "#555")
        label = wa.target if wa.target else f"[+{wa.spoken}]"
        tooltip = wa.error_type
        if wa.error_type not in ("correct", "omission", "insertion"):
            tooltip += f" — heard '{wa.spoken}' (sim {wa.phoneme_sim:.2f})"
        parts.append(
            f"<span title='{tooltip}' style='color:{c};font-weight:bold;"
            f"margin:0 4px;font-size:22px;'>{label}</span>"
        )

    st.markdown(
        "<div style='text-align:center;margin:12px 0;'>" + " ".join(parts) + "</div>",
        unsafe_allow_html=True,
    )

    bd = alignment.error_breakdown
    if bd:
        cols = st.columns(len(bd))
        for col, (k, v) in zip(cols, bd.items()):
            col.metric(k.capitalize(), v)
    st.caption(f"Phoneme accuracy: {alignment.phoneme_accuracy:.0%}")


def render_header(rs: ReadingSession) -> None:
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.markdown("### 🗣️ SpeakUp!")
    c2.markdown(_tier_badge(rs.level), unsafe_allow_html=True)
    c3.metric("Streak 🔥", rs.consecutive_passes)


# ---------------------------------------------------------------------------
# Phrase screen  (phases: listen → record → checking → result)
# ---------------------------------------------------------------------------

def render_phrase_screen(rs: ReadingSession, is_retry: bool = False) -> None:
    phrase = rs.current_phrase

    # Per-phrase session state keys
    pk   = f"_pp_{phrase}"    # phase key
    ak   = f"_pa_{phrase}"    # raw audio bytes
    sk   = f"_ps_{phrase}"    # simulated text
    rk   = f"_pr_{phrase}"    # saved result dict

    phase = st.session_state.get(pk, "listen")

    # ---- Large sentence display (always visible) ----
    st.markdown(
        f"<h1 style='text-align:center;font-size:58px;line-height:1.2;"
        f"margin:24px 0;'>{phrase}</h1>",
        unsafe_allow_html=True,
    )

    if is_retry:
        st.info("🔁 You've practised each word. Now try the whole sentence again!")

    st.markdown("---")

    # ===========================================================
    # Phase 1: LISTEN
    # ===========================================================
    if phase == "listen":
        _step_indicator(1)
        st.markdown(
            "<h3 style='text-align:center;'>🎧 Listen to the sentence</h3>",
            unsafe_allow_html=True,
        )

        # Auto-speak only on the first render of this phrase's listen phase
        auto_key = f"_auto_{phrase}"
        auto = auto_key not in st.session_state
        if auto:
            st.session_state[auto_key] = True
        _speak(phrase, auto=auto, button_label="🔊 Play again")

        # gTTS fallback audio player (no autoplay — avoids browser Error state)
        with st.expander("🎵 Use audio file instead (if browser speech fails)"):
            mp3 = _gtts_bytes(phrase)
            if mp3:
                st.audio(mp3, format="audio/mp3", autoplay=False)
            else:
                st.caption("Audio file unavailable (requires internet + gTTS).")

        st.markdown("")
        if st.button(
            "✅ I heard it — I'm ready to speak!",
            use_container_width=True,
            type="primary",
        ):
            st.session_state[pk] = "record"
            st.rerun()

    # ===========================================================
    # Phase 2: RECORD
    # ===========================================================
    elif phase == "record":
        _step_indicator(2)

        st.markdown(
            "<h3 style='text-align:center;color:#1565C0;'>"
            "🎤 Now say the sentence out loud!</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center;color:#555;font-size:16px;'>"
            "Tap the button below to start recording, then tap again to stop.</p>",
            unsafe_allow_html=True,
        )

        if mic_recorder is None:
            st.error(
                "🎙️ Microphone not available. "
                "Install it with: `pip install streamlit-mic-recorder`, "
                "or use the text box below."
            )
        else:
            audio = _record_audio(key=f"phrase_rec_{phrase}")
            if audio:
                # Recording received → move to checking
                st.session_state[ak] = audio
                st.session_state[pk] = "checking"
                st.rerun()

        # Replay button
        with st.expander("🔊 Hear the sentence again"):
            _speak(phrase, auto=False)

        # Text simulation fallback
        with st.expander("⌨️ No mic? Type your answer here"):
            sim = st.text_input(
                "Type what you would say:",
                key=f"sim_input_{phrase}",
                placeholder="e.g. The cat sat",
            )
            if st.button("Submit typed answer", key=f"sim_btn_{phrase}") and sim:
                st.session_state[sk] = sim
                st.session_state[pk] = "checking"
                st.rerun()

    # ===========================================================
    # Phase 3: CHECKING  (transcribe + submit — runs only once)
    # ===========================================================
    elif phase == "checking":
        _step_indicator(3)

        # Guard: if result is already saved, skip to result
        if rk in st.session_state:
            st.session_state[pk] = "result"
            st.rerun()
            return

        sim_text  = st.session_state.pop(sk, None)
        audio_buf = st.session_state.get(ak)

        st.markdown(
            "<h3 style='text-align:center;'>⏳ Checking your answer…</h3>",
            unsafe_allow_html=True,
        )
        st.progress(0.6, text="Processing your recording, please wait…")

        if sim_text:
            spoken = sim_text
        elif audio_buf:
            try:
                with st.spinner("🔍 Transcribing your voice…"):
                    spoken = _do_transcribe(audio_buf)
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
                if st.button("⬅️ Try recording again"):
                    st.session_state.pop(ak, None)
                    st.session_state[pk] = "record"
                    st.rerun()
                return
        else:
            st.error("No recording found. Please try again.")
            if st.button("⬅️ Go back"):
                st.session_state[pk] = "record"
                st.rerun()
            return

        # Submit to session and cache result
        result    = rs.submit_phrase_attempt(spoken)
        alignment = rs.last_alignment

        st.session_state[rk] = {
            "spoken":    spoken,
            "passed":    result.passed,
            "method":    result.method,
            "score":     result.score,
            "alignment": alignment,
        }
        st.session_state.pop(ak, None)
        st.session_state[pk] = "result"
        st.rerun()

    # ===========================================================
    # Phase 4: RESULT
    # ===========================================================
    elif phase == "result":
        saved = st.session_state.get(rk)
        if not saved:
            # Fallback if state is inconsistent
            st.session_state[pk] = "listen"
            st.rerun()
            return

        _step_indicator(3)

        st.markdown(
            f"<p style='text-align:center;font-size:18px;color:#555;'>"
            f"You said: <strong>{saved['spoken']}</strong></p>",
            unsafe_allow_html=True,
        )

        if saved["passed"]:
            st.success(
                f"🎉 **Well done!**  ({saved['method']} match, "
                f"score {saved['score']:.0f}/100)"
            )
        else:
            st.warning("⚠️ Not quite — let's practise each word now.")

        alignment = saved.get("alignment")
        if alignment and alignment.alignments:
            _render_alignment(alignment)

        st.markdown("")
        if st.button("Continue ➡️", use_container_width=True, type="primary"):
            # Clean up all phase keys for this phrase
            for key in [pk, rk, ak, sk]:
                st.session_state.pop(key, None)
            st.rerun()


# ---------------------------------------------------------------------------
# Word screen  (phases: listen → record → result)
# ---------------------------------------------------------------------------

def render_word_screen(rs: ReadingSession) -> None:
    word  = rs.current_word
    words = rs.current_phrase.split()
    total = len(words)

    # Progress bar
    st.progress(
        rs.current_word_index / max(1, total),
        text=f"Word {rs.current_word_index + 1} of {total}",
    )
    st.caption(f"Full sentence: *{rs.current_phrase}*")

    # Per-word session state keys (word index prevents cross-word collisions)
    wid = f"{rs.current_word_index}_{rs.current_word_attempts}"
    pk  = f"_wp_{wid}"     # phase
    ak  = f"_wa_{wid}"     # audio
    sk  = f"_ws_{wid}"     # sim text
    rk  = f"_wr_{wid}"     # result

    phase = st.session_state.get(pk, "listen")

    # ---- Large word display ----
    st.markdown(
        f"<h1 style='text-align:center;font-size:96px;font-weight:900;"
        f"margin:16px 0;'>{word}</h1>",
        unsafe_allow_html=True,
    )

    # Word alternatives
    alts = rs.word_alternatives(word)
    if alts:
        st.info(f"🔄 Similar-sounding practice words: **{', '.join(alts)}**")

    st.markdown("---")

    # ===========================================================
    # Phase 1: LISTEN
    # ===========================================================
    if phase == "listen":
        _step_indicator(1)

        st.markdown(
            "<h4 style='text-align:center;'>🎧 Listen to the word</h4>",
            unsafe_allow_html=True,
        )

        auto_key = f"_auto_w_{wid}"
        auto = auto_key not in st.session_state
        if auto:
            st.session_state[auto_key] = True
        _speak(word, auto=auto, button_label="🔊 Play again")

        st.markdown("")
        if st.button(
            "✅ I'm ready to say it!",
            key=f"wp_ready_{wid}",
            use_container_width=True,
            type="primary",
        ):
            st.session_state[pk] = "record"
            st.rerun()

    # ===========================================================
    # Phase 2: RECORD
    # ===========================================================
    elif phase == "record":
        _step_indicator(2)

        remaining = rs.max_word_attempts - rs.current_word_attempts
        st.markdown(
            f"<h4 style='text-align:center;color:#1565C0;'>"
            f"🎤 Say the word — {remaining} attempt(s) left</h4>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center;color:#555;'>Tap to record, tap again to stop.</p>",
            unsafe_allow_html=True,
        )

        if mic_recorder is None:
            st.error("Microphone not available. Use the text box below.")
        else:
            audio = _record_audio(key=f"word_rec_{wid}")
            if audio:
                st.session_state[ak] = audio
                st.session_state[pk] = "result"
                st.rerun()

        with st.expander("🔊 Hear the word again"):
            _speak(word, auto=False)

        with st.expander("⌨️ No mic? Type the word"):
            sim = st.text_input("", key=f"ws_input_{wid}", placeholder=f"Type '{word}'")
            if st.button("Submit", key=f"ws_btn_{wid}") and sim:
                st.session_state[sk] = sim
                st.session_state[pk] = "result"
                st.rerun()

    # ===========================================================
    # Phase 3: RESULT  (transcribe inline, show feedback, advance)
    # ===========================================================
    elif phase == "result":
        _step_indicator(3)

        # Avoid re-processing on subsequent reruns
        if rk not in st.session_state:
            sim_text  = st.session_state.pop(sk, None)
            audio_buf = st.session_state.pop(ak, None)

            with st.spinner("🔍 Checking your word…"):
                if sim_text:
                    spoken = sim_text
                elif audio_buf:
                    try:
                        spoken = _do_transcribe(audio_buf)
                    except Exception as exc:
                        st.error(f"Transcription failed: {exc}")
                        if st.button("⬅️ Try again", key=f"wr_retry_{wid}"):
                            st.session_state[pk] = "record"
                            st.rerun()
                        return
                else:
                    st.error("No recording captured. Try again.")
                    if st.button("⬅️ Try again", key=f"wr_retry2_{wid}"):
                        st.session_state[pk] = "record"
                        st.rerun()
                    return

            result = rs.submit_word_attempt(spoken)
            st.session_state[rk] = {
                "spoken": spoken,
                "passed": result.passed,
                "method": result.method,
            }

        saved = st.session_state[rk]

        st.markdown(
            f"<p style='text-align:center;font-size:18px;color:#555;'>"
            f"You said: <strong>{saved['spoken']}</strong></p>",
            unsafe_allow_html=True,
        )

        remaining_after = rs.max_word_attempts - rs.current_word_attempts
        if saved["passed"]:
            st.success(f"✅ Correct! ({saved['method']})")
        elif remaining_after > 0:
            st.warning(f"⚠️ Not quite — you have **{remaining_after}** attempt(s) left.")
        else:
            st.info("That's okay — let's keep going!")

        if st.button("Next word ➡️", key=f"wr_next_{wid}", use_container_width=True, type="primary"):
            for key in [pk, rk, ak, sk]:
                st.session_state.pop(key, None)
            st.rerun()


# ---------------------------------------------------------------------------
# History & debug panels
# ---------------------------------------------------------------------------

def render_history(rs: ReadingSession) -> None:
    with st.expander("📋 Session history"):
        if not rs.history:
            st.write("No attempts yet.")
            return
        rows = rs.export_history()
        for r in rows:
            r["error_breakdown"] = str(r.get("error_breakdown", {}))
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False).encode(),
            file_name="speakup_session.csv",
            mime="text/csv",
        )


def render_debug(rs: ReadingSession) -> None:
    with st.expander("🛠 Debug: session internals", expanded=False):
        st.write("state:", rs.state.value)
        st.write("level:", rs.level, "→", rs.level_label)
        st.write("current_phrase:", rs.current_phrase)
        st.write("word_index/attempts:", rs.current_word_index, "/", rs.current_word_attempts)
        st.write("recent_phoneme_accuracy:", f"{rs.recent_phoneme_accuracy:.0%}")
        st.write("consecutive_passes / fails:", rs.consecutive_passes, "/", rs.consecutive_fails)
        st.write("struggling_words:", rs._struggling_words)
        st.write("mic_available:", mic_recorder is not None)

        if rs._rl is not None:
            st.write("RL Q-table (tier × acc_bucket=medium × action):")
            q = rs._rl.q_values
            for t_idx, tier_name in enumerate(["Beginner", "Intermediate", "Advanced"]):
                st.write(
                    f"  {tier_name}: "
                    f"easier={q[t_idx,1,0]:.3f} "
                    f"same={q[t_idx,1,1]:.3f} "
                    f"harder={q[t_idx,1,2]:.3f}"
                )


def write_debug_event(event: dict) -> None:
    try:
        record = {"ts": datetime.utcnow().isoformat() + "Z", **event}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

rs = get_session()

with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🔄 Restart session", use_container_width=True):
        reset_session()
        st.rerun()
    st.markdown(f"**Level:** {_tier_badge(rs.level)}", unsafe_allow_html=True)
    st.write("**State:**", rs.state.value)
    st.write(
        "**Corpus:**",
        sum(len(v) for v in rs.phrases_by_level.values()),
        "sentences",
    )
    st.write("**Phoneme accuracy:**", f"{rs.recent_phoneme_accuracy:.0%}")
    st.markdown("---")
    st.caption(
        "🟢 Beginner · 🟡 Intermediate · 🔴 Advanced\n\n"
        "The RL engine adjusts difficulty based on phoneme accuracy."
    )

render_header(rs)

if rs.state == State.DONE:
    st.balloons()
    st.success("🎉 All done — great reading today!")
elif rs.state == State.SHOW_PHRASE:
    render_phrase_screen(rs, is_retry=False)
elif rs.state == State.WORD_PRACTICE:
    render_word_screen(rs)
elif rs.state == State.RETRY_PHRASE:
    render_phrase_screen(rs, is_retry=True)
else:
    st.write("Loading…")

render_history(rs)
render_debug(rs)
