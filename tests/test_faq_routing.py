from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.faq import (
    FAQ_QUERY_CHAR_LIMIT,
    load_faq,
    lookup_faq,
    lookup_faq_exact,
    truncate_faq_query,
)
from app.routing import (
    extract_symbol_from_text,
    is_company_lookup_query,
    resolve_pre_llm_route,
)


FAQ_PATH = Path(__file__).resolve().parents[1] / "data" / "faq.json"
ADDED_FAQ_QUESTIONS = [
    "How do I submit a ticket?",
    "How do I notify Slack?",
    "How do I start an SEC inquiry?",
    "How do I download a reference?",
    "How do I chat about filings?",
    "What ticker format should I use?",
    "How do I pick a company?",
    "What is a CIK number?",
]


def _faq_data() -> dict[str, str]:
    load_faq.cache_clear()
    return load_faq(FAQ_PATH)


def test_extract_symbol_ignores_how_use_query_retries() -> None:
    assert extract_symbol_from_text("How do I look up SEC filings?") is None
    assert (
        extract_symbol_from_text(
            "Use the query_faq tool to answer: How do I look up SEC filings?"
        )
        is None
    )
    assert (
        extract_symbol_from_text(
            "Query the query_faq tool to answer: How do I look up SEC filings?"
        )
        is None
    )


def test_extract_symbol_keeps_real_tickers() -> None:
    assert extract_symbol_from_text("AAPL") == "AAPL"
    assert extract_symbol_from_text("HOW") == "HOW"
    assert extract_symbol_from_text("$msft") == "MSFT"
    assert extract_symbol_from_text("How do I look up AAPL 10-K filings?") == "AAPL"
    assert extract_symbol_from_text("Apple 10-K") is None


def test_how_to_filings_question_is_not_company_lookup() -> None:
    assert not is_company_lookup_query("How do I look up SEC filings?")
    assert is_company_lookup_query("AAPL")
    assert is_company_lookup_query("Apple 10-K")
    assert is_company_lookup_query("How do I look up AAPL 10-K filings?")


def test_exact_faq_wins_for_sec_how_to() -> None:
    faq_data = _faq_data()
    route, answer = resolve_pre_llm_route(
        "How do I look up SEC filings?", faq_data
    )
    assert route == "faq"
    assert "Company Form" in (answer or "")


def test_retry_prompts_route_to_faq_not_cik() -> None:
    faq_data = _faq_data()
    for prompt in (
        "Use the query_faq tool to answer: How do I look up SEC filings?",
        "Query the query_faq tool to answer: How do I look up SEC filings?",
        "Use the query_faq tool to answer: [How do I look up SEC filings?]",
    ):
        route, answer = resolve_pre_llm_route(
            prompt, faq_data, has_symbol_context=True
        )
        assert route == "faq", prompt
        assert "Company Form" in (answer or ""), prompt


def test_company_form_and_ticker_go_to_cik() -> None:
    faq_data = _faq_data()
    route, answer = resolve_pre_llm_route("Apple 10-K", faq_data)
    assert route == "cik"
    assert answer is None

    route, answer = resolve_pre_llm_route(
        "How do I look up AAPL 10-K filings?", faq_data
    )
    assert route == "cik"
    assert answer is None


def test_filings_action_with_session_symbol_goes_to_cik() -> None:
    faq_data = _faq_data()
    route, answer = resolve_pre_llm_route(
        "look up filings", faq_data, has_symbol_context=True
    )
    assert route == "cik"
    assert answer is None


def test_bidirectional_faq_and_truncation() -> None:
    faq_data = {
        "How do I look up SEC filings?": "Ask for Company Form.",
    }
    wrapped = "Please query_faq now: How do I look up SEC filings?"
    assert lookup_faq_exact(wrapped, faq_data) is None
    assert lookup_faq(wrapped, faq_data) == "Ask for Company Form."
    assert len(truncate_faq_query("x" * 100)) == FAQ_QUERY_CHAR_LIMIT


def test_added_faq_pairs_exist_within_word_limits() -> None:
    faq_data = _faq_data()
    for question in ADDED_FAQ_QUESTIONS:
        assert question in faq_data
        assert len(question.split()) <= 8, question
        assert len(faq_data[question].split()) <= 40, question


def test_invoke_returns_faq_for_how_to_and_retries() -> None:
    pytest.importorskip("bedrock_agentcore")
    from app.app import invoke
    from app.memory import SessionMemory

    context = SimpleNamespace(
        session_id="faq-route-test",
        actor_id="tester",
        identity={"actor_id": "tester"},
        memory=SessionMemory(),
    )
    expected = "Company Form"
    for prompt in (
        "How do I look up SEC filings?",
        "Use the query_faq tool to answer: How do I look up SEC filings?",
        "Query the query_faq tool to answer: How do I look up SEC filings?",
    ):
        result = invoke({"prompt": prompt}, context=context)
        assert expected in result["response"]
        assert "Unable to resolve a CIK" not in result["response"]
