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
st.set_page_config(page_title="SpeakUp!", page_icon="🦜", layout="centered")


# ---------------------------------------------------------------------------
# Global child-friendly CSS
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

        /* ── Apply Nunito ONLY to text nodes — NOT to icon spans ──
           Streamlit's expander arrow uses a <span> with font-family:
           'Material Icons'; overriding it renders "keyboard_arrow_right"
           as literal text. We target explicit text containers only.      */
        html, body,
        .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2,
        .stMarkdown h3, .stMarkdown h4, .stMarkdown h5,
        .stMarkdown a,
        div.stButton > button,
        .stTextInput input,
        .stSelectbox select,
        [data-testid="stText"],
        [data-testid="stCaptionContainer"] p,
        [data-baseweb="notification"] p,
        [data-baseweb="notification"] div {
            font-family: 'Nunito', sans-serif !important;
        }

        /* ── Hide Streamlit chrome (toolbar / footer / main-menu) ──
           These are not needed in a children's reading app and the
           toolbar overlaps the top of our custom header.             */
        header[data-testid="stHeader"]   { display: none !important; }
        footer                           { display: none !important; }
        #MainMenu                        { visibility: hidden !important; }

        /* Soft pastel gradient background */
        .stApp {
            background: linear-gradient(160deg, #e8f4fd 0%, #fff8e1 55%, #fce4ec 100%);
            min-height: 100vh;
        }
        .block-container {
            padding-top: 1.6rem !important;
            max-width: 760px !important;
        }

        /* ── Primary buttons ── */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 50px !important;
            font-family: 'Nunito', sans-serif !important;
            font-size: 20px !important;
            font-weight: 900 !important;
            padding: 14px 28px !important;
            box-shadow: 0 6px 18px rgba(118,75,162,0.40) !important;
            transition: transform .15s, box-shadow .15s !important;
            letter-spacing: .4px !important;
        }
        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-3px) scale(1.04) !important;
            box-shadow: 0 12px 28px rgba(118,75,162,0.50) !important;
        }
        div.stButton > button[kind="primary"]:active {
            transform: translateY(1px) scale(.97) !important;
        }

        /* ── Secondary / default buttons ── */
        div.stButton > button:not([kind="primary"]) {
            border-radius: 50px !important;
            font-family: 'Nunito', sans-serif !important;
            font-size: 16px !important;
            font-weight: 800 !important;
            border: 2.5px solid #667eea !important;
            color: #5c35ae !important;
            background: #fff !important;
            transition: all .15s !important;
        }
        div.stButton > button:not([kind="primary"]):hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: #fff !important;
            border-color: transparent !important;
            transform: translateY(-1px) !important;
        }

        /* ── Metric cards ── */
        [data-testid="stMetric"] {
            background: #fff;
            border-radius: 18px;
            padding: 10px 16px;
            box-shadow: 0 3px 14px rgba(0,0,0,.10);
        }

        /* ── Progress bar — rainbow ── */
        [data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%) !important;
            border-radius: 10px !important;
        }

        /* ── Alert / notification boxes ── */
        [data-baseweb="notification"] {
            border-radius: 18px !important;
            font-size: 17px !important;
        }

        /* ── Expander — clean card style, arrow icon preserved ── */
        [data-testid="stExpander"] {
            background: #fff !important;
            border-radius: 16px !important;
            border: 1.5px solid #e0e0e0 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,.06) !important;
            margin-bottom: 6px !important;
        }
        [data-testid="stExpander"] summary {
            font-family: 'Nunito', sans-serif !important;
            font-weight: 700 !important;
            font-size: 15px !important;
            padding: 10px 16px !important;
        }

        /* ── Sidebar — deep indigo ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a237e 0%, #311b92 100%) !important;
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label {
            color: #c5cae9 !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #fff !important;
        }
        [data-testid="stSidebar"] div.stButton > button {
            background: rgba(255,255,255,.15) !important;
            color: #fff !important;
            border-color: rgba(255,255,255,.45) !important;
        }
        [data-testid="stSidebar"] div.stButton > button:hover {
            background: rgba(255,255,255,.28) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------

def get_session() -> ReadingSession:
    if "rs" not in st.session_state:
        st.session_state.rs = ReadingSession.from_corpus()
    return st.session_state.rs


def reset_session() -> None:
    keep = {"rs"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]
    st.session_state.pop("rs", None)


# ---------------------------------------------------------------------------
# TTS  — Web Speech API (browser-native, no network, no autoplay restriction)
# ---------------------------------------------------------------------------

def _tts_lang() -> str:
    return st.session_state.get("tts_lang", "en-IN")


def _speak_with_highlight(text: str, *, auto: bool = True, font_size: int = 38) -> None:
    """Web Speech API player with per-word highlighting and a child-friendly skin.

    auto=True  → auto-speaks on first render (DOMContentLoaded + 400 ms delay).
    Language follows st.session_state["tts_lang"] (default: en-IN).
    Chrome workarounds: resume() before cancel(), 120 ms delay before speak().
    """
    lang      = _tts_lang()
    words     = text.split()
    safe_text = _html.escape(text, quote=True)

    # Each word gets a pastel colour that cycles
    word_colours = ["#ffe0b2", "#e1f5fe", "#f3e5f5", "#e8f5e9", "#fff9c4", "#fce4ec"]
    spans = " ".join(
        f'<span id="w{i}" class="word" style="background:{word_colours[i % len(word_colours)]};">'
        f'{_html.escape(w)}</span>'
        for i, w in enumerate(words)
    )

    auto_js = (
        "document.addEventListener('DOMContentLoaded', function() {"
        " setTimeout(speakNow, 400); });"
    ) if auto else ""
    # Base covers: outer padding + word row + Play button + status line.
    # Extra rows add 65 px each (≈ font_size * 1.65 line-height).
    height = 240 + max(0, (len(words) - 4) // 3) * 65

    _components.html(
        f"""
        <div style="font-family:'Nunito',sans-serif;text-align:center;
                    background:white;border-radius:20px;
                    padding:16px 12px 20px;
                    box-shadow:0 4px 18px rgba(0,0,0,.10);">
          <div id="words-row"
               style="font-size:{font_size}px;font-weight:900;line-height:1.65;
                      margin-bottom:14px;letter-spacing:.5px;">
            {spans}
          </div>
          <button id="sp-btn" onclick="speakNow()"
              style="font-size:15px;padding:9px 26px;border-radius:50px;
                     background:linear-gradient(135deg,#667eea,#764ba2);
                     color:#fff;border:none;cursor:pointer;font-weight:800;
                     box-shadow:0 4px 14px rgba(118,75,162,.40);
                     font-family:'Nunito',sans-serif;
                     transition:transform .12s;">
            🔊 Play again
          </button>
          <div id="sp-status"
               style="color:#7c4dff;font-size:14px;margin-top:8px;font-weight:700;">
          </div>
        </div>
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@900&display=swap');
          .word {{
            display: inline-block;
            padding: 3px 7px;
            border-radius: 8px;
            transition: background .08s, transform .12s, box-shadow .12s, color .08s;
            cursor: default;
          }}
          .word.active {{
            background: #FFD700 !important;
            color: #000 !important;
            transform: scale(1.22) translateY(-3px);
            box-shadow: 0 4px 12px rgba(255,193,7,.55);
          }}
          #sp-btn:hover {{ transform: translateY(-2px) scale(1.05); }}
          #sp-btn:active {{ transform: scale(.97); }}
        </style>
        <script>
          const TTS_TEXT = "{safe_text}";
          const WORDS    = TTS_TEXT.split(' ');

          function clearHL() {{
            WORDS.forEach((_, i) => {{
              const el = document.getElementById('w' + i);
              if (el) el.classList.remove('active');
            }});
          }}

          function highlightWord(charIndex) {{
            clearHL();
            let count = 0;
            for (let i = 0; i < WORDS.length; i++) {{
              if (charIndex >= count && charIndex < count + WORDS[i].length) {{
                const el = document.getElementById('w' + i);
                if (el) el.classList.add('active');
                break;
              }}
              count += WORDS[i].length + 1;
            }}
          }}

          function speakNow() {{
            const btn  = document.getElementById('sp-btn');
            const stat = document.getElementById('sp-status');
            if (!window.speechSynthesis) {{
              if (stat) stat.textContent = '⚠ TTS not supported in this browser.';
              return;
            }}
            if (window.speechSynthesis.paused) window.speechSynthesis.resume();
            window.speechSynthesis.cancel();

            const u = new SpeechSynthesisUtterance(TTS_TEXT);
            u.rate = 0.85;
            u.lang = '{lang}';
            u.onstart = () => {{
              if (stat) stat.textContent = '🎵 Reading…';
              if (btn)  btn.disabled = true;
            }};
            u.onend = () => {{
              clearHL();
              if (stat) stat.textContent = '✅ Done! Click the button below when ready.';
              if (btn)  btn.disabled = false;
            }};
            u.onerror = (e) => {{
              clearHL();
              if (stat) stat.textContent = '⚠ ' + e.error;
              if (btn)  btn.disabled = false;
            }};
            u.onboundary = (ev) => {{
              if (ev.name === 'word') highlightWord(ev.charIndex);
            }};
            // Chrome: speak() dropped immediately after cancel() — wait 120 ms
            setTimeout(() => window.speechSynthesis.speak(u), 120);
          }}
          {auto_js}
        </script>
        """,
        height=height,
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
    audio = mic_recorder(
        start_prompt="🎤  Tap to start recording",
        stop_prompt="⏹  Tap to stop",
        just_once=True,
        use_container_width=True,
        key=key,
    )
    if audio and "bytes" in audio:
        return audio["bytes"]
    return None


def _do_transcribe(audio_bytes: bytes) -> str:
    """Transcribe audio bytes via Groq API (cloud) or local Whisper."""
    if transcribe_bytes is None:
        raise RuntimeError(
            "No STT backend available. "
            "Set GROQ_API_KEY in Streamlit secrets to enable voice."
        )
    groq_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    return transcribe_bytes(audio_bytes, groq_api_key=groq_key)


# ---------------------------------------------------------------------------
# Shared UI widgets
# ---------------------------------------------------------------------------

_TIER_COLOUR  = {1: ("#4CAF50", "#A5D6A7"), 2: ("#FF9800", "#FFCC80"), 3: ("#F44336", "#EF9A9A")}
_TIER_EMOJI   = {1: "🌱", 2: "🌟", 3: "🚀"}


def _tier_badge(level: int) -> str:
    label = LEVEL_LABELS.get(level, str(level))
    c1, _ = _TIER_COLOUR.get(level, ("#888", "#bbb"))
    return (
        f"<span style='background:{c1};color:white;"
        f"padding:5px 14px;border-radius:50px;font-size:14px;font-weight:800;"
        f"font-family:Nunito,sans-serif;'>"
        f"{_TIER_EMOJI.get(level, '')} {label}</span>"
    )


def _step_indicator(current: int) -> None:
    """Colorful circle step indicator: Listen → Record → Result."""
    items = [("🎧", "Listen"), ("🎤", "Record"), ("🏆", "Result")]
    cols  = st.columns(3)
    for i, (col, (icon, label)) in enumerate(zip(cols, items)):
        step = i + 1
        if step < current:
            col.markdown(
                f"""<div style='text-align:center;'>
                  <div style='display:inline-flex;align-items:center;justify-content:center;
                    width:44px;height:44px;border-radius:50%;
                    background:#c8e6c9;font-size:20px;margin-bottom:4px;'>✅</div>
                  <div style='font-size:12px;color:#43a047;font-weight:800;
                              font-family:Nunito,sans-serif;'>{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        elif step == current:
            col.markdown(
                f"""<div style='text-align:center;'>
                  <div style='display:inline-flex;align-items:center;justify-content:center;
                    width:50px;height:50px;border-radius:50%;
                    background:linear-gradient(135deg,#667eea,#764ba2);
                    font-size:22px;margin-bottom:4px;
                    box-shadow:0 4px 14px rgba(118,75,162,.50);'>
                    {icon}
                  </div>
                  <div style='font-size:13px;color:#5c35ae;font-weight:900;
                              font-family:Nunito,sans-serif;'>{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            col.markdown(
                f"""<div style='text-align:center;'>
                  <div style='display:inline-flex;align-items:center;justify-content:center;
                    width:44px;height:44px;border-radius:50%;
                    background:#ede7f6;font-size:20px;margin-bottom:4px;opacity:.45;'>{icon}</div>
                  <div style='font-size:12px;color:#9e9e9e;font-weight:700;
                              font-family:Nunito,sans-serif;'>{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    st.markdown("<div style='margin-bottom:18px'></div>", unsafe_allow_html=True)


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
        c     = _colour.get(wa.error_type, "#555")
        label = wa.target if wa.target else f"[+{wa.spoken}]"
        tooltip = wa.error_type
        if wa.error_type not in ("correct", "omission", "insertion"):
            tooltip += f" — heard '{wa.spoken}' (sim {wa.phoneme_sim:.2f})"
        parts.append(
            f"<span title='{tooltip}' style='color:{c};font-weight:800;"
            f"margin:0 5px;font-size:22px;font-family:Nunito,sans-serif;'>{label}</span>"
        )

    st.markdown(
        "<div style='text-align:center;margin:12px 0;background:#fff;"
        "border-radius:16px;padding:12px;box-shadow:0 2px 10px rgba(0,0,0,.08);'>"
        + " ".join(parts) + "</div>",
        unsafe_allow_html=True,
    )

    bd = alignment.error_breakdown
    if bd:
        cols = st.columns(len(bd))
        for col, (k, v) in zip(cols, bd.items()):
            col.metric(k.capitalize(), v)
    st.caption(f"Phoneme accuracy: {alignment.phoneme_accuracy:.0%}")


def render_header(rs: ReadingSession) -> None:
    """Full-width gradient banner with mascot, level badge, and star streak."""
    label     = LEVEL_LABELS.get(rs.level, str(rs.level))
    c1, c2    = _TIER_COLOUR.get(rs.level, ("#888", "#bbb"))
    emoji     = _TIER_EMOJI.get(rs.level, "")
    stars     = "⭐" * min(rs.consecutive_passes, 5) if rs.consecutive_passes else "—"

    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
             border-radius:24px;padding:18px 28px 16px;margin-bottom:22px;
             box-shadow:0 8px 24px rgba(118,75,162,0.38);'>
          <div style='display:flex;align-items:center;justify-content:space-between;
                      flex-wrap:wrap;gap:10px;'>
            <div style='display:flex;align-items:center;gap:10px;'>
              <span style='font-size:42px;'>🦜</span>
              <span style='font-size:30px;font-weight:900;color:#fff;
                           font-family:Nunito,sans-serif;letter-spacing:1px;'>SpeakUp!</span>
            </div>
            <div style='display:flex;gap:10px;align-items:center;'>
              <div style='background:rgba(255,255,255,.18);border-radius:16px;
                          padding:7px 18px;text-align:center;'>
                <div style='font-size:11px;color:rgba(255,255,255,.75);font-weight:800;
                            font-family:Nunito,sans-serif;letter-spacing:.5px;'>LEVEL</div>
                <div style='font-size:15px;color:#fff;font-weight:900;
                            font-family:Nunito,sans-serif;
                            background:linear-gradient(135deg,{c1},{c2});
                            border-radius:8px;padding:2px 10px;margin-top:3px;display:inline-block;'>
                  {emoji} {label}
                </div>
              </div>
              <div style='background:rgba(255,255,255,.18);border-radius:16px;
                          padding:7px 18px;text-align:center;'>
                <div style='font-size:11px;color:rgba(255,255,255,.75);font-weight:800;
                            font-family:Nunito,sans-serif;letter-spacing:.5px;'>STREAK</div>
                <div style='font-size:22px;margin-top:2px;'>{stars}</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Phrase screen  (phases: listen → record → checking → result)
# ---------------------------------------------------------------------------

def render_phrase_screen(rs: ReadingSession, is_retry: bool = False) -> None:
    phrase = rs.current_phrase

    pk  = f"_pp_{phrase}"
    ak  = f"_pa_{phrase}"
    sk  = f"_ps_{phrase}"
    rk  = f"_pr_{phrase}"

    phase = st.session_state.get(pk, "listen")

    # ── Sentence card ────────────────────────────────────────────
    border_colour = "#ff7043" if is_retry else "#667eea"
    st.markdown(
        f"""
        <div style='background:#fff;border-radius:24px;padding:28px 32px;
             box-shadow:0 6px 24px rgba(0,0,0,.10);margin-bottom:18px;
             border-left:7px solid {border_colour};text-align:center;'>
          <div style='font-size:46px;font-weight:900;color:#1a237e;line-height:1.35;
                      font-family:Nunito,sans-serif;'>
            {phrase}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_retry:
        st.info("🔁 Great practice! Now try the whole sentence again!")

    # ── Phase 1: LISTEN ──────────────────────────────────────────
    if phase == "listen":
        _step_indicator(1)
        st.markdown(
            "<h3 style='text-align:center;color:#5c35ae;font-family:Nunito,sans-serif;'>"
            "🎧 Listen carefully to the sentence!</h3>",
            unsafe_allow_html=True,
        )

        auto = pk not in st.session_state
        if auto:
            st.session_state[pk] = "listen"
        _speak_with_highlight(phrase, auto=auto, font_size=38)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button(
            "✅  I heard it — I'm ready to speak!",
            use_container_width=True,
            type="primary",
        ):
            st.session_state[pk] = "record"
            st.rerun()

    # ── Phase 2: RECORD ──────────────────────────────────────────
    elif phase == "record":
        _step_indicator(2)

        st.markdown(
            """<div style='text-align:center;background:linear-gradient(135deg,#e3f2fd,#f3e5f5);
               border-radius:18px;padding:18px 20px;margin-bottom:14px;'>
               <div style='font-size:26px;font-weight:900;color:#1565c0;
                           font-family:Nunito,sans-serif;'>
                 🎤 Now say the sentence out loud!
               </div>
               <div style='font-size:15px;color:#5c35ae;font-weight:700;margin-top:6px;
                           font-family:Nunito,sans-serif;'>
                 Tap the button below to start, then tap again to stop.
               </div>
             </div>""",
            unsafe_allow_html=True,
        )

        if mic_recorder is None:
            st.error(
                "🎙️ Microphone not available. "
                "Install: `pip install streamlit-mic-recorder`, or type below."
            )
        else:
            audio = _record_audio(key=f"phrase_rec_{phrase}")
            if audio:
                st.session_state[ak] = audio
                st.session_state[pk] = "checking"
                st.rerun()

        with st.expander("🔊 Hear the sentence again"):
            _speak_with_highlight(phrase, auto=False, font_size=32)

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

    # ── Phase 3: CHECKING ────────────────────────────────────────
    elif phase == "checking":
        _step_indicator(3)

        if rk in st.session_state:
            st.session_state[pk] = "result"
            st.rerun()
            return

        sim_text  = st.session_state.pop(sk, None)
        audio_buf = st.session_state.get(ak)

        st.markdown(
            "<h3 style='text-align:center;color:#7c4dff;font-family:Nunito,sans-serif;'>"
            "⏳ Checking your answer…</h3>",
            unsafe_allow_html=True,
        )
        st.progress(0.6, text="Processing your recording, please wait…")

        if sim_text:
            spoken = sim_text
        elif audio_buf and transcribe_bytes is not None:
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
            # Whisper not available (cloud deployment) — ask the child to type
            st.info(
                "🎤 Voice recording captured! "
                "Voice transcription is not available in this deployment — "
                "please type what you said below."
            )
            typed = st.text_input(
                "What did you say?",
                key=f"cloud_fallback_{phrase}",
                placeholder=f'e.g. "{phrase}"',
            )
            if st.button("Submit ✅", key=f"cloud_fallback_btn_{phrase}") and typed:
                spoken = typed
            else:
                return

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

    # ── Phase 4: RESULT ──────────────────────────────────────────
    elif phase == "result":
        saved = st.session_state.get(rk)
        if not saved:
            st.session_state[pk] = "listen"
            st.rerun()
            return

        _step_indicator(3)

        st.markdown(
            f"<div style='text-align:center;font-size:18px;color:#555;"
            f"font-family:Nunito,sans-serif;margin-bottom:10px;'>"
            f"You said: <strong style='color:#1a237e;'>{saved['spoken']}</strong></div>",
            unsafe_allow_html=True,
        )

        if saved["passed"]:
            _components.html(
                """<div style="font-family:'Nunito',sans-serif;text-align:center;
                              padding:16px;background:linear-gradient(135deg,#e8f5e9,#f1f8e9);
                              border-radius:20px;border:3px solid #66bb6a;">
                     <div style="font-size:52px;">🎉</div>
                     <div style="font-size:24px;font-weight:900;color:#2e7d32;margin-top:6px;">
                       Well done! Amazing reading!
                     </div>
                     <div style="font-size:36px;margin-top:4px;">⭐ ⭐ ⭐</div>
                   </div>""",
                height=140,
            )
        else:
            st.markdown(
                """<div style='background:linear-gradient(135deg,#fff3e0,#fce4ec);
                   border-radius:20px;padding:16px 20px;border:3px solid #ffb74d;
                   text-align:center;font-family:Nunito,sans-serif;margin-bottom:8px;'>
                   <div style='font-size:36px;'>🤔</div>
                   <div style='font-size:20px;font-weight:800;color:#e65100;'>
                     Not quite — let's practise each word!
                   </div>
                 </div>""",
                unsafe_allow_html=True,
            )

        alignment = saved.get("alignment")
        if alignment and alignment.alignments:
            _render_alignment(alignment)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Continue ➡️", use_container_width=True, type="primary"):
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

    # ── Dot progress strip ───────────────────────────────────────
    dots = " ".join(
        "🟣" if i < rs.current_word_index
        else ("🔵" if i == rs.current_word_index else "⚪")
        for i in range(total)
    )
    st.markdown(
        f"<div style='text-align:center;font-size:22px;margin:6px 0 2px;'>{dots}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Word {rs.current_word_index + 1} of {total}  ·  "
        f"Full sentence: *{rs.current_phrase}*"
    )

    wid = f"{rs.current_word_index}_{rs.current_word_attempts}"
    pk  = f"_wp_{wid}"
    ak  = f"_wa_{wid}"
    sk  = f"_ws_{wid}"
    rk  = f"_wr_{wid}"

    phase = st.session_state.get(pk, "listen")

    # ── Big word bubble ──────────────────────────────────────────
    st.markdown(
        f"""
        <div style='text-align:center;margin:16px 0 10px;'>
          <div style='display:inline-block;
               background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
               border-radius:28px;padding:18px 52px;
               box-shadow:0 8px 28px rgba(118,75,162,.45);'>
            <span style='font-size:82px;font-weight:900;color:white;
                         font-family:Nunito,sans-serif;letter-spacing:3px;'>
              {word}
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    alts = rs.word_alternatives(word)
    if alts:
        st.info(f"🔄 Similar-sounding words: **{', '.join(alts)}**")

    st.markdown("<hr style='border:none;border-top:2px dashed #c5cae9;margin:10px 0;'>",
                unsafe_allow_html=True)

    # ── Phase 1: LISTEN ──────────────────────────────────────────
    if phase == "listen":
        _step_indicator(1)

        st.markdown(
            "<h4 style='text-align:center;color:#5c35ae;font-family:Nunito,sans-serif;'>"
            "🎧 Listen to the word!</h4>",
            unsafe_allow_html=True,
        )

        auto = pk not in st.session_state
        if auto:
            st.session_state[pk] = "listen"
        _speak_with_highlight(word, auto=auto, font_size=52)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button(
            "✅  I'm ready to say it!",
            key=f"wp_ready_{wid}",
            use_container_width=True,
            type="primary",
        ):
            st.session_state[pk] = "record"
            st.rerun()

    # ── Phase 2: RECORD ──────────────────────────────────────────
    elif phase == "record":
        _step_indicator(2)

        remaining = rs.max_word_attempts - rs.current_word_attempts
        st.markdown(
            f"""<div style='text-align:center;background:linear-gradient(135deg,#e3f2fd,#f3e5f5);
               border-radius:18px;padding:14px 16px;margin-bottom:12px;'>
               <div style='font-size:22px;font-weight:900;color:#1565c0;
                           font-family:Nunito,sans-serif;'>
                 🎤 Say the word — {remaining} attempt(s) left!
               </div>
               <div style='font-size:14px;color:#5c35ae;font-weight:700;margin-top:5px;
                           font-family:Nunito,sans-serif;'>
                 Tap to record, tap again to stop.
               </div>
             </div>""",
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
            _speak_with_highlight(word, auto=False, font_size=44)

        with st.expander("⌨️ No mic? Type the word"):
            sim = st.text_input("", key=f"ws_input_{wid}", placeholder=f"Type '{word}'")
            if st.button("Submit", key=f"ws_btn_{wid}") and sim:
                st.session_state[sk] = sim
                st.session_state[pk] = "result"
                st.rerun()

    # ── Phase 3: RESULT ──────────────────────────────────────────
    elif phase == "result":
        _step_indicator(3)

        if rk not in st.session_state:
            sim_text  = st.session_state.pop(sk, None)
            audio_buf = st.session_state.pop(ak, None)

            with st.spinner("🔍 Checking your word…"):
                if sim_text:
                    spoken = sim_text
                elif audio_buf and transcribe_bytes is not None:
                    try:
                        spoken = _do_transcribe(audio_buf)
                    except Exception as exc:
                        st.error(f"Transcription failed: {exc}")
                        if st.button("⬅️ Try again", key=f"wr_retry_{wid}"):
                            st.session_state[pk] = "record"
                            st.rerun()
                        return
                else:
                    # Whisper not available — ask child to type the word
                    st.info(
                        "🎤 Recording captured! Please type the word you said."
                    )
                    typed = st.text_input(
                        "Type the word:",
                        key=f"cloud_word_fallback_{wid}",
                        placeholder=f'e.g. "{word}"',
                    )
                    if st.button("Submit ✅", key=f"cloud_word_btn_{wid}") and typed:
                        spoken = typed
                    else:
                        return

            result = rs.submit_word_attempt(spoken)
            st.session_state[rk] = {
                "spoken": spoken,
                "passed": result.passed,
                "method": result.method,
            }

        saved = st.session_state[rk]

        st.markdown(
            f"<div style='text-align:center;font-size:18px;color:#555;"
            f"font-family:Nunito,sans-serif;margin-bottom:10px;'>"
            f"You said: <strong style='color:#1a237e;'>{saved['spoken']}</strong></div>",
            unsafe_allow_html=True,
        )

        remaining_after = rs.max_word_attempts - rs.current_word_attempts
        if saved["passed"]:
            _components.html(
                """<div style="font-family:'Nunito',sans-serif;text-align:center;
                              padding:12px;background:linear-gradient(135deg,#e8f5e9,#f1f8e9);
                              border-radius:18px;border:3px solid #66bb6a;">
                     <div style="font-size:40px;">🌟</div>
                     <div style="font-size:20px;font-weight:900;color:#2e7d32;">
                       Correct! Great job!
                     </div>
                   </div>""",
                height=110,
            )
        elif remaining_after > 0:
            st.warning(f"⚠️ Not quite — you have **{remaining_after}** attempt(s) left. Try again!")
        else:
            st.info("That's okay — keep going, you're doing great! 💪")

        if st.button(
            "Next word ➡️",
            key=f"wr_next_{wid}",
            use_container_width=True,
            type="primary",
        ):
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
_inject_css()

with st.sidebar:
    st.markdown(
        "<div style='text-align:center;font-size:42px;margin-bottom:4px;'>🦜</div>"
        "<h2 style='text-align:center;font-family:Nunito,sans-serif;"
        "color:#fff;margin:0 0 16px;'>SpeakUp!</h2>",
        unsafe_allow_html=True,
    )
    if st.button("🔄 Restart session", use_container_width=True):
        reset_session()
        st.rerun()

    st.markdown("<div style='margin:10px 0 4px;color:#c5cae9;font-size:12px;"
                "font-weight:800;letter-spacing:.5px;'>LEVEL</div>",
                unsafe_allow_html=True)
    st.markdown(_tier_badge(rs.level), unsafe_allow_html=True)

    st.markdown("<div style='margin:12px 0 4px;color:#c5cae9;font-size:12px;"
                "font-weight:800;letter-spacing:.5px;'>PHONEME ACCURACY</div>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:22px;font-weight:900;color:#fff;'>"
        f"{rs.recent_phoneme_accuracy:.0%}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    _LANG_OPTIONS = {
        "🇮🇳 Indian English":   "en-IN",
        "🇺🇸 American English": "en-US",
        "🇬🇧 British English":  "en-GB",
    }
    chosen_label = st.selectbox(
        "🗣️ Pronunciation",
        list(_LANG_OPTIONS.keys()),
        index=0,
    )
    st.session_state["tts_lang"] = _LANG_OPTIONS[chosen_label]
    st.markdown("---")
    st.caption(
        "🌱 Beginner · 🌟 Intermediate · 🚀 Advanced\n\n"
        "The RL engine adapts difficulty based on phoneme accuracy."
    )
    st.markdown("---")
    # ── Teacher View (hidden from child) ──────────────────────────
    teacher_mode = st.toggle("👩‍🏫 Teacher view", value=False)

if teacher_mode:
    with st.sidebar:
        render_history(rs)
        render_debug(rs)

render_header(rs)

if rs.state == State.DONE:
    st.balloons()
    _components.html(
        """<div style="font-family:'Nunito',sans-serif;text-align:center;
                      padding:24px;background:linear-gradient(135deg,#fffde7,#fce4ec);
                      border-radius:24px;border:4px solid #ffd54f;">
             <div style="font-size:64px;">🏆</div>
             <div style="font-size:28px;font-weight:900;color:#f57f17;">
               All done — fantastic reading today!
             </div>
             <div style="font-size:40px;margin-top:8px;">⭐ ⭐ ⭐ ⭐ ⭐</div>
           </div>""",
        height=180,
    )
elif rs.state == State.SHOW_PHRASE:
    render_phrase_screen(rs, is_retry=False)
elif rs.state == State.WORD_PRACTICE:
    render_word_screen(rs)
elif rs.state == State.RETRY_PHRASE:
    render_phrase_screen(rs, is_retry=True)
else:
    st.write("Loading…")

