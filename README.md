# Reading Buddy

Adaptive reading-practice app for kids. Shows a 3-word phrase, listens to the child read it, and adapts based on how they did.

## Flow

```
SHOW_PHRASE ── child reads ──► EVALUATE
                                ├─ pass → next phrase (may level up)
                                └─ fail → SPLIT_TO_WORDS
                                            └─ practice each word
                                                  └─ RETRY_PHRASE
                                                        ├─ pass → next phrase
                                                        └─ fail → easier level, new phrase
```

## Setup

```bash
# 1. Install ffmpeg (Whisper needs it)
brew install ffmpeg          # macOS
# sudo apt-get install ffmpeg  # Ubuntu

# 2. Create a venv and install Python deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

First launch will download the Whisper `base.en` model (~150 MB). Switch to `small.en` in `core/stt.py` if accuracy on a young child's voice isn't good enough.

## Project layout

```
SpeechAPP/
├── app.py                  Streamlit UI
├── core/
│   ├── session.py          ReadingSession state machine
│   ├── compare.py          exact / fuzzy / phonetic comparison
│   ├── difficulty.py       phrase difficulty scoring
│   └── stt.py              Whisper wrapper (lazy-loaded)
├── data/
│   └── phrases.csv         corpus, tagged by difficulty 1..5
├── tests/
│   └── test_session.py     smoke tests for the state machine
└── requirements.txt
```

## Where to extend next

- **Better corpus**: replace `data/phrases.csv` with phrases relevant to your child. Use `core.difficulty.classify_level(phrase)` to auto-tag new phrases.
- **TTS demo button**: add a "hear it first" button using `gTTS` so the child can hear the phrase before reading.
- **Persistent progress**: write `rs.export_history()` to SQLite at the end of each session to track improvement over time.
- **Tune comparison thresholds**: the fuzzy threshold is 85 in `core/compare.py`. Lower it (e.g. 75) if you want to be more lenient with mispronunciations.
- **Try other ASR**: swap `core/stt.py` for `vosk` (offline, lighter) or Google Cloud Speech if Whisper struggles with your child's voice.

## Development

Quick steps to get a reproducible development environment and run tests locally:

```bash
# Create and activate a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate

# Install runtime deps for development. The full requirements.txt includes
# optional heavy packages (Whisper / torch). For fast test runs use the
# CI requirements file which contains just what's needed for tests.
pip install -r requirements-ci.txt

# Run the test suite
python -m pytest -q
```

If you prefer to install everything (for running the app with Whisper),
run `pip install -r requirements.txt` instead. On macOS you must also
install `ffmpeg` (homebrew) before running Whisper.
