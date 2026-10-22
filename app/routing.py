from __future__ import annotations

import re
from typing import Dict, Literal, Tuple

from .faq import lookup_faq, lookup_faq_exact, truncate_faq_query


UPPER_SYMBOL_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")
MENU_INTEGER_PATTERN = re.compile(r"^\s*\d{1,3}\.?\s*$")
FILINGS_ACTION_PATTERN = re.compile(
    r"(look\s+up\s+(?:sec\s+)?filings?|(?:look\s+up\s+)?sec\s+filings?)",
    re.IGNORECASE,
)
DOLLAR_TICKER_PATTERN = re.compile(r"\$([A-Za-z]{1,5})\b")
CIK_TOKEN_PATTERN = re.compile(r"\b(?:cik[:\s]*)?(\d{10})\b", re.IGNORECASE)
FORM_TYPE_PATTERN = re.compile(
    r"\b(?:10-?K|10-?Q|8-?K|20-?F|6-?K|S-1|S-3|13F|13D|13G|DEF14A)\b",
    re.IGNORECASE,
)
QUESTION_STARTERS = {
    "HOW",
    "WHAT",
    "WHEN",
    "WHY",
    "WHO",
    "WHERE",
    "CAN",
    "DO",
    "DOES",
    "DID",
    "IS",
    "ARE",
    "PLEASE",
    "USE",
    "QUERY",
    "COULD",
    "WOULD",
    "SHOULD",
    "MAY",
}
SYMBOL_STOPWORDS = {
    "A", "ABOUT", "ALL", "ALSO", "AM", "AN", "AND", "ANSWER", "ANY", "ARE",
    "AS", "ASK", "AT", "BE", "BEEN", "BEING", "BUT", "BY", "CAN", "CHAT",
    "COULD", "DID", "DO", "DOES", "FAQ", "FILE", "FOR", "FORM", "FROM",
    "GET", "GO", "HAD", "HAS", "HAVE", "HE", "HELP", "HI", "HOW", "HTTP",
    "HTTPS", "HTML", "IF", "IN", "INC", "INTO", "IS", "IT", "ITS", "JSON",
    "JUST", "LOOK", "LTD", "MAY", "ME", "MIGHT", "MORE", "MUST", "MY",
    "NEED", "NEW", "NO", "NOT", "OF", "OK", "OLD", "ON", "OR", "OUR",
    "OUT", "OVER", "PER", "PLEASE", "QUERY", "SEC", "SHOULD", "SHOW", "SO",
    "TELL", "THAN", "THAT", "THE", "THEM", "THEN", "THEY", "THIS", "TO",
    "TOOL", "TYPE", "UP", "US", "USE", "VIA", "WANT", "WAS", "WE", "WERE",
    "WHAT", "WHEN", "WHERE", "WHICH", "WHO", "WHY", "WILL", "WITH", "WOULD",
    "YES", "YOU", "YOUR", "CORP",
}

PreLlmRoute = Literal["faq", "cik", "continue"]


def normalize_symbol(
    candidate: str | None, *, allow_stopwords: bool = False
) -> str | None:
    if not candidate:
        return None
    symbol = candidate.strip().upper()
    if not symbol.isalpha() or not 1 <= len(symbol) <= 5:
        return None
    if not allow_stopwords and symbol in SYMBOL_STOPWORDS:
        return None
    return symbol


def is_menu_integer(text: str) -> bool:
    return bool(text and MENU_INTEGER_PATTERN.match(text))


def is_filings_action(text: str) -> bool:
    return bool(text and FILINGS_ACTION_PATTERN.search(text))


def looks_like_question(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    first_token = stripped.split()[0]
    first = re.sub(r"[^A-Za-z]", "", first_token).upper()
    return first in QUESTION_STARTERS


def _is_bare_symbol_message(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    letters_only = re.sub(r"[^A-Za-z]", "", stripped)
    if not letters_only:
        return False
    compact = stripped.replace(" ", "")
    return letters_only.upper() == compact.upper()


def has_explicit_ticker(text: str) -> bool:
    if not text:
        return False
    if DOLLAR_TICKER_PATTERN.search(text):
        return True
    if _is_bare_symbol_message(text):
        letters_only = re.sub(r"[^A-Za-z]", "", text)
        return bool(normalize_symbol(letters_only, allow_stopwords=True))
    for match in UPPER_SYMBOL_PATTERN.findall(text):
        if normalize_symbol(match, allow_stopwords=False):
            return True
    return False


def is_company_lookup_query(text: str) -> bool:
    if not text or not text.strip():
        return False
    if has_explicit_ticker(text):
        return True
    if CIK_TOKEN_PATTERN.search(text):
        return True
    if FORM_TYPE_PATTERN.search(text):
        return True
    return False


def extract_symbol_from_text(text: str) -> str | None:
    if not text:
        return None
    if is_menu_integer(text):
        return None

    stripped = text.strip()
    letters_only = re.sub(r"[^A-Za-z]", "", stripped)
    if _is_bare_symbol_message(stripped):
        whole = normalize_symbol(letters_only, allow_stopwords=True)
        if whole:
            return whole

    dollar = DOLLAR_TICKER_PATTERN.search(stripped)
    if dollar:
        return normalize_symbol(dollar.group(1), allow_stopwords=True)

    skip_heuristic = looks_like_question(stripped) or (
        is_filings_action(stripped) and not has_explicit_ticker(stripped)
    )
    for match in UPPER_SYMBOL_PATTERN.findall(text):
        symbol = normalize_symbol(match, allow_stopwords=False)
        if symbol:
            return symbol

    if skip_heuristic:
        return None
    return None


def resolve_pre_llm_route(
    prompt: str,
    faq_data: Dict[str, str],
    *,
    has_symbol_context: bool = False,
) -> Tuple[PreLlmRoute, str | None]:
    """Classify a prompt before CIK lookup or the LangGraph loop.

    Exact FAQ matches win. Otherwise company/ticker/CIK queries go to CIK
    lookup. Remaining prompts try bidirectional FAQ matching and return that
    answer without calling the model.
    """
    truncated = truncate_faq_query(prompt)
    exact = lookup_faq_exact(truncated, faq_data)
    if exact:
        return "faq", exact
    if is_company_lookup_query(truncated):
        return "cik", None
    if (
        is_filings_action(truncated)
        and has_symbol_context
        and (has_explicit_ticker(truncated) or not looks_like_question(truncated))
    ):
        return "cik", None
    partial = lookup_faq(truncated, faq_data)
    if partial:
        return "faq", partial
    return "continue", None
