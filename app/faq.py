from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


@lru_cache(maxsize=1)
def load_faq(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(question): str(answer) for question, answer in data.items()}


def lookup_faq(question: str, faq_data: Dict[str, str]) -> str | None:
    if not question or not faq_data:
        return None

    normalized_question = _normalize(question)
    for stored_question, answer in faq_data.items():
        if normalized_question == _normalize(stored_question):
            return answer

    for stored_question, answer in faq_data.items():
        if normalized_question in _normalize(stored_question):
            return answer
    return None

