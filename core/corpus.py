"""Load sentence corpus from data/Corpus word.xlsx.

Three difficulty tiers:
  1 = Beginner
  2 = Intermediate
  3 = Advanced
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

_DATA_DIR = Path(__file__).parent.parent / "data"
_LEVEL_MAP = {"beginner": 1, "intermediate": 2, "advanced": 3}

LEVEL_LABELS: Dict[int, str] = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}


def load_corpus(xlsx_path: str | Path | None = None) -> Dict[int, List[str]]:
    """Return {tier: [sentences]} from the Excel corpus spreadsheet."""
    if xlsx_path is None:
        xlsx_path = _DATA_DIR / "Corpus word.xlsx"
    xlsx_path = Path(xlsx_path)

    try:
        import openpyxl  # type: ignore

        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb.active
        result: Dict[int, List[str]] = {1: [], 2: [], 3: []}
        first_row = True
        for row in ws.iter_rows(values_only=True):
            if first_row:
                first_row = False
                continue
            if not row or row[1] is None or row[2] is None:
                continue
            sentence = str(row[1]).strip()
            raw_level = str(row[2]).strip().lower()
            level = _LEVEL_MAP.get(raw_level, 1)
            result[level].append(sentence)
        return {k: v for k, v in result.items() if v}
    except Exception:
        return _csv_fallback()


def _csv_fallback() -> Dict[int, List[str]]:
    """Fall back to phrases.csv when openpyxl is unavailable."""
    import csv

    csv_path = _DATA_DIR / "phrases.csv"
    result: Dict[int, List[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = int(row.get("difficulty", 1))
            tier = 1 if d <= 2 else (2 if d <= 4 else 3)
            result.setdefault(tier, []).append(row["phrase"].strip())
    return result


def all_words(corpus: Dict[int, List[str]]) -> List[str]:
    """Return a deduplicated list of all unique words across the corpus."""
    words: set[str] = set()
    for sentences in corpus.values():
        for s in sentences:
            for w in s.lower().split():
                words.add(w.strip(".,!?"))
    return sorted(words)
