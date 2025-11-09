from __future__ import annotations

import logging
import re
import uuid
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Dict, Iterable

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain_core.messages import HumanMessage, SystemMessage

from .config import load_settings
from .graph import build_graph
from .observability import begin_subsegment, configure_observability, end_subsegment
from .sec import (
    SecLookupError,
    format_cik_matches,
    format_filings as format_sec_filings,
    get_recent_filings,
    list_ciks_for_symbol,
)

try:
    from botocore.exceptions import ClientError
except Exception:  # noqa: BLE001
    ClientError = None

INTRO_MESSAGE = (
    "Welcome to the AgentCore proof-of-concept assistant. I help operations and compliance "
    "teams retrieve FAQ answers, review SEC filings, and capture inquiry details. "
    "To get started, share the company name or a 3-4 letter stock ticker symbol you're focused on."
)

BASE_SYSTEM_PROMPT = (
    "You are the AgentCore proof-of-concept assistant supporting operations and compliance teams. "
    "Provide clear, tactical guidance, use tools when appropriate, and keep responses concise."
)

TICKER_REMINDER_PROMPT = (
    "If the user hasn't supplied a specific company or a 3-4 letter stock ticker yet, "
    "help them provide one when necessary to proceed."
)

_UPPER_SYMBOL_PATTERN = re.compile(r"\b([A-Z]{3,4})\b")
_ALPHA_SYMBOL_PATTERN = re.compile(r"\b([A-Za-z]{3,4})\b")

logger = logging.getLogger(__name__)

app = BedrockAgentCoreApp()
_initial_settings = load_settings()
configure_observability(asdict(_initial_settings))
graph = build_graph(_initial_settings)
_known_threads: set[str] = set()
_thread_states: dict[str, Dict[str, Any]] = {}
_CONFIRM_PATTERN = re.compile(r"\b(yes|y|confirm|ok|okay|proceed)\b", re.IGNORECASE)
_CANCEL_PATTERN = re.compile(r"\b(cancel|stop|abort|no)\b", re.IGNORECASE)


def _get_thread_state(thread_id: str) -> Dict[str, Any]:
    state = _thread_states.setdefault(
        thread_id,
        {
            "symbol": None,
            "candidates": [],
            "selected_cik": None,
            "filings_shared": False,
            "company_search": None,
        },
    )
    return state


def _reset_thread_state(state: Dict[str, Any]) -> None:
    state.update(
        {
            "symbol": None,
            "candidates": [],
            "selected_cik": None,
            "filings_shared": False,
            "company_search": None,
        }
    )


def _extract_cik_choice(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"\b\d{7,10}\b", text)
    if not match:
        return None
    return match.group(0).zfill(10)


def _extract_prompt(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""
    if "prompt" in payload:
        return str(payload["prompt"])
    if "messages" in payload and payload["messages"]:
        last_message = payload["messages"][-1]
        return str(last_message.get("content", ""))
    return ""


def _normalize_symbol(candidate: str | None) -> str | None:
    if not candidate:
        return None
    symbol = candidate.strip().upper()
    if 3 <= len(symbol) <= 4 and symbol.isalpha():
        return symbol
    return None


def _extract_symbol_from_text(text: str) -> str | None:
    if not text:
        return None

    for pattern in (_UPPER_SYMBOL_PATTERN, _ALPHA_SYMBOL_PATTERN):
        for match in pattern.findall(text):
            symbol = _normalize_symbol(match)
            if symbol:
                return symbol

    stripped = re.sub(r"[^A-Za-z]", "", text)
    return _normalize_symbol(stripped)


def _collect_system_messages(symbol: str | None) -> Iterable[SystemMessage]:
    yield SystemMessage(content=BASE_SYSTEM_PROMPT)
    if symbol:
        yield SystemMessage(
            content=(
                f"The user provided the stock ticker '{symbol}'. "
                "Use this ticker when researching filings or crafting guidance."
            )
        )
    else:
        yield SystemMessage(content=TICKER_REMINDER_PROMPT)


def _coalesce_attribute(candidate: Any, *attribute_names: str) -> str | None:
    if candidate is None:
        return None

    if isinstance(candidate, dict):
        for name in attribute_names:
            value = candidate.get(name)
            if value:
                return str(value)

    for name in attribute_names:
        if hasattr(candidate, name):
            value = getattr(candidate, name)
            if value:
                return str(value)
    return None


def _resolve_actor_from_context(context: Any, fallback: str) -> str:
    actor_id = fallback
    identity = getattr(context, "identity", None) if context else None

    identity_actor = _coalesce_attribute(
        identity,
        "actor_id",
        "actorId",
        "user_id",
        "userId",
    )
    if identity_actor:
        actor_id = identity_actor

    context_actor = _coalesce_attribute(
        context,
        "actor_id",
        "actorId",
        "user_id",
        "userId",
    )
    if context_actor:
        actor_id = context_actor

    return actor_id


@app.entrypoint
def invoke(payload, context):
    settings = load_settings()

    actor_id = _resolve_actor_from_context(context, settings.actor_id)

    session_id = getattr(context, "session_id", None) if context else None
    thread_id = session_id or str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
            "actor_id": actor_id,
        }
    }

    prompt = _extract_prompt(payload or {})
    if not prompt:
        return {"response": INTRO_MESSAGE}

    symbol: str | None = None
    if isinstance(payload, dict):
        raw_symbol = payload.get("symbol")
        if isinstance(raw_symbol, str):
            symbol = _normalize_symbol(raw_symbol)
    if not symbol:
        symbol = _extract_symbol_from_text(prompt)

    thread_state = _get_thread_state(thread_id)
    if symbol and symbol != thread_state["symbol"]:
        _reset_thread_state(thread_state)
        thread_state["symbol"] = symbol
        thread_state["company_search"] = symbol

        if not settings.enable_network_tools:
            return {
                "response": (
                    "SEC lookups are disabled. Set ENABLE_NETWORK_TOOLS=true to retrieve CIK data."
                )
            }

        try:
            candidates = list_ciks_for_symbol(symbol, timeout=settings.http_timeout)
        except (ValueError, SecLookupError) as exc:
            _reset_thread_state(thread_state)
            return {
                "response": f"Unable to resolve a CIK for '{symbol}': {exc}"
            }

        thread_state["candidates"] = candidates
        return {"response": format_cik_matches(candidates)}

    if (
        thread_state["symbol"]
        and thread_state["candidates"]
        and not thread_state["selected_cik"]
    ):
        if _CANCEL_PATTERN.search(prompt):
            _reset_thread_state(thread_state)
            return {
                "response": "Cancelled CIK selection. Provide a new ticker symbol to continue."
            }

        selected = _extract_cik_choice(prompt)
        if not selected and _CONFIRM_PATTERN.search(prompt):
            if len(thread_state["candidates"]) == 1:
                selected = thread_state["candidates"][0].cik
            else:
                return {
                    "response": (
                        "Multiple CIKs matched. Please reply with the 10-digit CIK you want to use."
                    )
                }

        if not selected:
            return {
                "response": (
                    "Reply with the 10-digit CIK you want to use or type 'cancel' to start over."
                )
            }

        for candidate in thread_state["candidates"]:
            if candidate.cik == selected:
                thread_state["selected_cik"] = selected
                break
        else:
            return {
                "response": (
                    f"CIK {selected} is not in the current list. Choose one of the suggested CIKs."
                )
            }

        if not settings.enable_network_tools:
            return {
                "response": (
                    "SEC lookups are disabled. Set ENABLE_NETWORK_TOOLS=true to retrieve filings."
                )
            }

        try:
            filings = get_recent_filings(
                thread_state["selected_cik"],
                timeout=settings.http_timeout,
            )
        except SecLookupError as exc:
            _reset_thread_state(thread_state)
            return {
                "response": f"Unable to retrieve filings for CIK {selected}: {exc}"
            }

        thread_state["filings_shared"] = True
        formatted = format_sec_filings(
            filings,
            None,
            thread_state["company_search"] or thread_state["symbol"],
            thread_state["selected_cik"],
        )
        return {"response": formatted}

    if not symbol and thread_state.get("selected_cik"):
        symbol = thread_state["symbol"]

    include_system_messages = thread_id not in _known_threads
    if include_system_messages:
        _known_threads.add(thread_id)
    messages = list(_collect_system_messages(symbol)) if include_system_messages else []
    messages.append(HumanMessage(content=prompt))

    input_messages = {"messages": messages}
    subsegment = begin_subsegment("agent-graph")
    try:
        response = graph.invoke(input_messages, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent invocation failed", exc_info=exc)
        if ClientError and isinstance(exc, ClientError):
            error_code = exc.response.get("Error", {}).get("Code")
        else:
            error_code = None

        if error_code == "AccessDeniedException" or (
            error_code is None and "AccessDeniedException" in str(exc)
        ):
            return {
                "response": (
                    "I can't access the configured Bedrock model yet. "
                    "Please confirm your AWS credentials allow bedrock:InvokeModel for the selected "
                    "model or switch to credentials with Bedrock access, then try again."
                )
            }
        return {
            "response": (
                "The agent hit an unexpected runtime error while contacting Bedrock. "
                f"Details: {exc}"
            )
        }
    finally:
        end_subsegment(subsegment)

    final_message = response["messages"][-1].content
    return {"response": final_message}


def run_local_cli() -> None:
    print("AgentCore POC CLI. Type 'exit' to quit.")
    settings = load_settings()
    session_id = str(uuid.uuid4())
    context = SimpleNamespace(
        session_id=session_id,
        actor_id=settings.actor_id,
        identity={"actor_id": settings.actor_id},
    )
    intro = invoke({}, context=context)
    print(f"Agent: {intro['response']}")
    last_symbol: str | None = None
    while True:
        user_input = input("Company or symbol: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        payload: Dict[str, Any] = {"prompt": user_input}
        detected_symbol = _extract_symbol_from_text(user_input)
        if detected_symbol:
            last_symbol = detected_symbol
            payload["symbol"] = detected_symbol
        elif last_symbol:
            payload["symbol"] = last_symbol
        result = invoke(payload, context=context)
        print(f"Agent: {result['response']}")


if __name__ == "__main__":
    run_local_cli()

