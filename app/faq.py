from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict


FAQ_QUERY_CHAR_LIMIT = 70


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def truncate_faq_query(question: str) -> str:
    if not question:
        return ""
    return question[:FAQ_QUERY_CHAR_LIMIT]


@lru_cache(maxsize=1)
def load_faq(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(question): str(answer) for question, answer in data.items()}


def lookup_faq_exact(question: str, faq_data: Dict[str, str]) -> str | None:
    if not question or not faq_data:
        return None
    normalized_question = _normalize(truncate_faq_query(question))
    if not normalized_question:
        return None
    for stored_question, answer in faq_data.items():
        if normalized_question == _normalize(stored_question):
            return answer
    return None


def lookup_faq(question: str, faq_data: Dict[str, str]) -> str | None:
    if not question or not faq_data:
        return None

    exact = lookup_faq_exact(question, faq_data)
    if exact:
        return exact

    normalized_question = _normalize(truncate_faq_query(question))
    if not normalized_question:
        return None

    for stored_question, answer in faq_data.items():
        stored = _normalize(stored_question)
        if normalized_question in stored:
            return answer

    for stored_question, answer in faq_data.items():
        stored = _normalize(stored_question)
        if stored and stored in normalized_question:
            return answer
    return None
