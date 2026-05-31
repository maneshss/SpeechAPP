"""Speech-to-text via Groq Whisper API (cloud) or local openai-whisper (local).

Priority:
  1. Groq API  — lightweight client, free tier, no local GPU/RAM needed.
                  Requires GROQ_API_KEY in Streamlit secrets or environment.
  2. Local Whisper — openai-whisper package + ffmpeg installed locally.

Importing this module raises ImportError if neither backend is available,
so app.py can set transcribe_bytes = None and fall back to typed input.
"""

from __future__ import annotations

import os
import tempfile

# ── Try Groq (cloud, lightweight) first ─────────────────────────────────────
try:
    from groq import Groq as _GroqClient
    _GROQ_AVAILABLE = True
except ImportError:
    _GroqClient = None          # type: ignore
    _GROQ_AVAILABLE = False

# ── Try local Whisper second ─────────────────────────────────────────────────
try:
    import whisper as _whisper  # type: ignore
    _LOCAL_AVAILABLE = True
except ImportError:
    _whisper = None             # type: ignore
    _LOCAL_AVAILABLE = False

if not _GROQ_AVAILABLE and not _LOCAL_AVAILABLE:
    raise ImportError(
        "No STT backend found. "
        "Install groq (`pip install groq`) and set GROQ_API_KEY, "
        "or install openai-whisper (`pip install openai-whisper`)."
    )


# ── Public API ───────────────────────────────────────────────────────────────

def transcribe_bytes(audio_bytes: bytes, groq_api_key: str = "") -> str:
    """Transcribe raw audio bytes to text.

    Tries Groq first (if API key is available), then falls back to local
    Whisper.
    """
    if not audio_bytes:
        return ""

    # Write to a temp WAV file (both backends need a file path / file object)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        if _GROQ_AVAILABLE:
            key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
            if key:
                return _groq_transcribe(tmp_path, key)

        if _LOCAL_AVAILABLE:
            return _local_transcribe(tmp_path)

        raise RuntimeError(
            "No STT backend configured. "
            "Set GROQ_API_KEY in Streamlit secrets to enable voice."
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ── Groq backend ─────────────────────────────────────────────────────────────

def _groq_transcribe(audio_path: str, api_key: str) -> str:
    client = _GroqClient(api_key=api_key)
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=("recording.wav", f),
            model="whisper-large-v3-turbo",
            language="en",
        )
    return (result.text or "").strip()


# ── Local Whisper backend ────────────────────────────────────────────────────

from functools import lru_cache  # noqa: E402


@lru_cache(maxsize=1)
def _load_local_model(model_name: str = "base.en"):
    import shutil
    if shutil.which("ffmpeg") is None:
        for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
            if os.path.isdir(prefix):
                os.environ["PATH"] = prefix + os.pathsep + os.environ.get("PATH", "")
                if shutil.which("ffmpeg"):
                    break
    return _whisper.load_model(model_name)


def _local_transcribe(audio_path: str, model_name: str = "base.en") -> str:
    model  = _load_local_model(model_name)
    result = model.transcribe(audio_path, language="en", fp16=False)
    return (result.get("text") or "").strip()
