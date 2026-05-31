"""Speech-to-text wrapper around OpenAI Whisper.

Whisper is loaded lazily on first call so importing this module stays cheap
(useful while iterating on the UI without paying the model-load cost).

If openai-whisper is not installed this module raises ImportError at import
time so that callers can catch it with a simple try/except ImportError block
and gracefully fall back to text input.
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
import shutil

# Fail fast at import time if whisper is not installed.
# This lets app.py's  `try: from core.stt import …  except ImportError`
# correctly set transcribe_bytes = None instead of discovering the problem
# only when the function is first called.
try:
    import whisper as _whisper_check  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "openai-whisper is not installed — "
        "run `pip install openai-whisper` to enable voice transcription."
    ) from _exc

# Some environments (GUI-launched apps, services, or shells started before Homebrew
# was installed) don't have Homebrew's bin on PATH. Whisper invokes the `ffmpeg`
# binary via subprocess; if it's not on PATH the call will raise FileNotFoundError.
# Try to make ffmpeg discoverable by prepending common Homebrew prefixes if needed.
if shutil.which("ffmpeg") is None:
    for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
        if os.path.isdir(prefix):
            os.environ["PATH"] = prefix + os.pathsep + os.environ.get("PATH", "")
            if shutil.which("ffmpeg"):
                break


@lru_cache(maxsize=1)
def _load_model(model_name: str = "base.en"):
    """Load and cache a Whisper model.

    Recommended sizes for kids' speech:
      - "base.en"   -> fast, decent quality (good default)
      - "small.en"  -> noticeably more accurate; ~3x slower
      - "medium.en" -> best accuracy, slow on CPU
    """
    import whisper  # imported lazily

    return whisper.load_model(model_name)


def transcribe_bytes(audio_bytes: bytes, model_name: str = "base.en") -> str:
    """Transcribe raw audio bytes (e.g. a WAV blob from the mic recorder)."""
    if not audio_bytes:
        return ""

    # Whisper wants a path, so dump bytes to a temp file.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        return transcribe_path(tmp_path, model_name=model_name)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def transcribe_path(audio_path: str, model_name: str = "base.en") -> str:
    """Transcribe an audio file from disk."""
    model = _load_model(model_name)
    result = model.transcribe(audio_path, language="en", fp16=False)
    return (result.get("text") or "").strip()
