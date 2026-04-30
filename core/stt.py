"""Speech-to-text wrapper around OpenAI Whisper.

Whisper is loaded lazily on first call so importing this module stays cheap
(useful while iterating on the UI without paying the model-load cost).
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache


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
