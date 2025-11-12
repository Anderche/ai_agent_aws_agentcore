from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast
from urllib.parse import quote_plus

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain_core.messages import HumanMessage, SystemMessage

from .config import load_settings
from .filing_chat import (
    FilingChatError,
    FilingChatSession,
    answer_filing_question,
    prepare_filing_chat,
)
from .graph import build_graph, get_memory_store
from .memory import SessionMemory
from .observability import begin_subsegment, configure_observability, end_subsegment
from .rag_pipeline import (
    EmbeddingPipelineError,
    RagQueryError,
    VECTORSTORE_DIR,
    query_vectorstore,
    run_embedding_pipeline,
)
from .sec import (
    CikMatch,
    FilingMenuEntry,
    SecDownloadError,
    SecLookupError,
    SecRateLimitError,
    SecFiling,
    download_filing_to_directory,
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


def _format_possessive(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        return "this company's"
    if cleaned[-1].lower() == "s":
        return f"{cleaned}'"
    return f"{cleaned}'s"


def _format_sec_rate_limit_message(company_reference: str | None) -> str:
    display = (company_reference or "this company").strip() or "this company"
    possessive = _format_possessive(display)
    search_query = quote_plus(display)
    edgar_link = f"https://www.sec.gov/edgar/search/#/q={search_query}"
    return (
        "I apologize for the continued difficulty. It appears there might be an issue with accessing the SEC database at the moment. "
        'The error suggests that there are too many requests being made to the SEC website (status 429 typically means "Too Many Requests").\n\n'
        "Since I'm unable to retrieve the SEC filings directly at this time, I can offer you some alternative options:\n\n"
        "1. We can try again later when the SEC database might be more accessible.\n"
        f"2. I can provide you with a direct link to the SEC's EDGAR database where you can search for {possessive} filings yourself.\n"
        f"   {edgar_link}\n"
        "3. We can submit a support ticket to report this issue and request assistance.\n\n"
        f"Which option would you prefer? Or is there another way I can assist you regarding {display}?"
    )


_UPPER_SYMBOL_PATTERN = re.compile(r"\b([A-Z]{3,4})\b")
_ALPHA_SYMBOL_PATTERN = re.compile(r"\b([A-Za-z]{3,4})\b")
_CHAT_FILING_PATTERN = re.compile(
    r"\bchat\b.*\bfiling", re.IGNORECASE
)
_EXIT_FILING_CHAT_PATTERN = re.compile(
    r"\b(exit|quit|stop)\b.*\bfiling\b", re.IGNORECASE
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorstoreSummary:
    path: Path
    label: str
    form: str | None
    filing_date: str | None
    cik: str | None
    source_url: str | None

    def describe(self) -> str:
        details: list[str] = []
        if self.form:
            details.append(f"Form {self.form}")
        if self.filing_date:
            details.append(f"filed {self.filing_date}")
        if self.cik:
            details.append(f"CIK {self.cik}")
        details_text = ", ".join(details) if details else "Metadata not available"
        return f"{self.label} — {details_text}"


def _load_vectorstore_summary(path: Path) -> VectorstoreSummary | None:
    metadata: dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                metadata = record.get("metadata") or {}
                break
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to read vector store metadata from %s: %s", path, exc)
        return None

    label = metadata.get("document_name") or metadata.get("title")
    if not label:
        label = path.stem.replace("_vectorstore", "")

    return VectorstoreSummary(
        path=path,
        label=label,
        form=metadata.get("form"),
        filing_date=metadata.get("filing_date"),
        cik=metadata.get("cik"),
        source_url=metadata.get("source_url"),
    )


def _discover_vectorstore_summaries() -> list[VectorstoreSummary]:
    if not VECTORSTORE_DIR.exists():
        return []
    summaries: list[VectorstoreSummary] = []
    for path in sorted(VECTORSTORE_DIR.glob("*_vectorstore.jsonl")):
        summary = _load_vectorstore_summary(path)
        if summary:
            summaries.append(summary)
    return summaries


def _print_vectorstore_overview(summaries: list[VectorstoreSummary]) -> None:
    logger.info("Embedded filings available locally:")
    for index, summary in enumerate(summaries, start=1):
        logger.info("  %d. %s", index, summary.describe())
        if summary.source_url:
            logger.info("     Source: %s", summary.source_url)
    logger.info(
        "Options: [1] Query embedded filing   [2] Search SEC EDGAR   Type 'exit' to quit."
    )


def _vectorstore_menu(
    summaries: list[VectorstoreSummary],
    settings,
) -> str:
    if not summaries:
        return "sec"

    while True:
        _print_vectorstore_overview(summaries)
        choice = input("Select option (1, 2, or 'exit'): ").strip().lower()
        if choice in {"exit", "quit"}:
            return "exit"
        if choice in {"2", "sec", "search"}:
            return "sec"
        if choice in {"1", "query"}:
            continue_session = _run_vectorstore_query_session(summaries, settings)
            if not continue_session:
                return "exit"
        else:
            logger.info("Please enter '1', '2', or 'exit'.")


def _run_vectorstore_query_session(
    summaries: list[VectorstoreSummary],
    settings,
) -> bool:
    if not summaries:
        logger.info("No embedded filings found to query.")
        return True

    while True:
        selection = input(
            "Select a filing number to query "
            "(or type 'menu' to return, 'exit' to quit): "
        ).strip().lower()

        if selection in {"menu"}:
            return True
        if selection in {"exit", "quit"}:
            return False
        if selection in {"list", "ls"}:
            _print_vectorstore_overview(summaries)
            continue

        if selection.isdigit():
            index = int(selection)
            if 1 <= index <= len(summaries):
                result = _run_vectorstore_question_loop(summaries[index - 1], settings)
                if result == "exit":
                    return False
                if result == "menu":
                    return True
                # result == "select" means pick another filing
                continue

        logger.info(
            "Enter a number between 1 and %s, 'menu' to return, or 'exit' to quit.",
            len(summaries),
        )


def _run_vectorstore_question_loop(
    summary: VectorstoreSummary,
    settings,
) -> str:
    logger.info(
        "Querying `%s`. Ask a question, or type 'back' for other filings, "
        "'menu' for main options, 'exit' to quit.",
        summary.label,
    )

    while True:
        question = input("Vectorstore question: ").strip()
        lowered = question.lower()

        if lowered in {"exit", "quit"}:
            return "exit"
        if lowered == "menu":
            return "menu"
        if lowered in {"back", "change"}:
            return "select"
        if not question:
            logger.info("Please enter a question or one of the commands above.")
            continue

        try:
            answer = query_vectorstore(summary.path, question, settings=settings)
        except RagQueryError as exc:
            logger.warning("Vectorstore error: %s", exc)
            continue
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected vectorstore error: %s", exc)
            continue
        logger.info("Vectorstore: %s", answer)

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
            "filings": [],
            "awaiting_filing_selection": False,
            "filing_chat_session": None,
            "filings_display": [],
            "filing_menu_entries": [],
            "latest_download_path": None,
            "latest_vectorstore_path": None,
            "rag_active": False,
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
            "filings": [],
            "awaiting_filing_selection": False,
            "filing_chat_session": None,
            "filings_display": [],
            "filing_menu_entries": [],
            "latest_download_path": None,
            "latest_vectorstore_path": None,
            "rag_active": False,
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


def _extract_memory_snippet(record: Any) -> str | None:
    if record is None:
        return None

    data = record
    if isinstance(record, dict):
        if record.get("value") is not None:
            data = record["value"]
        elif record.get("record") is not None:
            data = record["record"]
    else:
        value_attr = getattr(record, "value", None)
        if value_attr is not None:
            data = value_attr

    if isinstance(data, dict):
        for key in ("summary", "content", "message", "text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in data.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
    elif isinstance(data, str):
        stripped = data.strip()
        if stripped:
            return stripped
    return None


@app.entrypoint
def invoke(payload, context):
    settings = load_settings()

    session_memory: SessionMemory | None = getattr(context, "memory", None) if context else None
    if session_memory is None:
        session_memory = SessionMemory()
        if context is not None:
            setattr(context, "memory", session_memory)

    actor_id = _resolve_actor_from_context(context, settings.actor_id)

    session_id = getattr(context, "session_id", None) if context else None
    thread_id = session_id or str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
            "actor_id": actor_id,
        }
    }
    memory_store = get_memory_store()
    memory_namespace = (actor_id, thread_id) if memory_store else None

    prompt = _extract_prompt(payload or {})
    if session_memory:
        session_memory.record_goal(prompt)
    if not prompt:
        return {"response": INTRO_MESSAGE}

    thread_state = _get_thread_state(thread_id)

    vectorstore_path = thread_state.get("latest_vectorstore_path")
    rag_active = thread_state.get("rag_active")

    if rag_active and vectorstore_path:
        if _CANCEL_PATTERN.search(prompt):
            thread_state["rag_active"] = False
            thread_state["latest_vectorstore_path"] = None
            return {
                "response": (
                    "Exited the filing Q&A workflow. Provide a ticker or new request to continue."
                )
            }
        try:
            answer = query_vectorstore(
                Path(vectorstore_path),
                prompt,
                settings=settings,
            )
            if session_memory:
                session_memory.record_tool_event(
                    "query_vectorstore",
                    "success",
                    f"path={vectorstore_path}",
                )
        except RagQueryError as exc:
            thread_state["rag_active"] = False
            if session_memory:
                session_memory.record_failure("query_vectorstore", str(exc))
            return {
                "response": (
                    f"Unable to answer using the embedded filing: {exc}. "
                    "You can request a new filing to analyze."
                )
            }
        except Exception as exc:  # noqa: BLE001
            thread_state["rag_active"] = False
            if session_memory:
                session_memory.record_failure("query_vectorstore", str(exc))
            return {
                "response": (
                    f"The filing Q&A workflow hit an unexpected error: {exc}. "
                    "Please re-run the filing analysis if needed."
                )
            }
        return {"response": answer}

    symbol: str | None = None
    if isinstance(payload, dict):
        raw_symbol = payload.get("symbol")
        if isinstance(raw_symbol, str):
            symbol = _normalize_symbol(raw_symbol)
    if not symbol:
        symbol = _extract_symbol_from_text(prompt)

    if session_memory and symbol:
        session_memory.set_primary_symbol(symbol)

    retrieved_long_term: list[str] = []
    if memory_store and memory_namespace and prompt:
        try:
            search_results = memory_store.search(memory_namespace, query=prompt, limit=5)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AgentCore Memory search failed: %s", exc)
            search_results = []
        for item in search_results:
            snippet = _extract_memory_snippet(item)
            if snippet:
                retrieved_long_term.append(snippet)
        retrieved_long_term = retrieved_long_term[:3]

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
            if session_memory:
                session_memory.record_failure("list_ciks_for_symbol", str(exc))
            return {
                "response": f"Unable to resolve a CIK for '{symbol}': {exc}"
            }

        thread_state["candidates"] = candidates
        if session_memory:
            session_memory.record_tool_event(
                "list_ciks_for_symbol",
                "success",
                f"symbol={symbol} matches={len(candidates)}",
            )
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
        if not selected:
            selected = _extract_cik_index(prompt, thread_state["candidates"])
        if not selected and _CONFIRM_PATTERN.search(prompt):
            if len(thread_state["candidates"]) == 1:
                selected = thread_state["candidates"][0].cik
            else:
                return {
                    "response": (
                        "Multiple CIKs matched. Reply with the numbered option or the 10-digit CIK you want to use."
                    )
                }

        if not selected:
            return {
                "response": (
                    "Reply with the numbered option or the 10-digit CIK you want to use, or type 'cancel' to start over."
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
            if session_memory:
                session_memory.record_failure("get_recent_filings", str(exc))
            return {
                "response": f"Unable to retrieve filings for CIK {selected}: {exc}"
            }

        if session_memory:
            session_memory.record_tool_event(
                "get_recent_filings",
                "success",
                f"cik={thread_state['selected_cik']} count={len(filings)}",
            )
        thread_state["filings_shared"] = True
        thread_state["filings"] = filings
        thread_state["awaiting_filing_selection"] = True
        company_context = thread_state["company_search"] or thread_state["symbol"] or ""
        formatted = _format_and_store_filings(
            filings,
            thread_state,
            company_context,
            thread_state["selected_cik"],
            form_type=None,
            include_summary=True,
        )
        instruction = _selection_instruction(thread_state)
        return {
            "response": (
                f"{formatted}\n\n{instruction}"
            )
        }

    filings_cache: List[SecFiling] = thread_state.get("filings", [])

    if thread_state.get("filing_chat_session"):
        if _EXIT_FILING_CHAT_PATTERN.search(prompt) or prompt.strip().lower() in {
            "exit",
            "quit",
            "cancel",
            "done",
        }:
            thread_state["filing_chat_session"] = None
            return {
                "response": (
                    "Closed the filing chat session. Let me know if you need anything else."
                )
            }
        if not settings.enable_network_tools:
            return {
                "response": (
                    "Filing chat requires network access. "
                    "Set ENABLE_NETWORK_TOOLS=true to continue."
                )
            }
        session: FilingChatSession = thread_state["filing_chat_session"]
        try:
            answer = answer_filing_question(
                session,
                prompt,
                settings=settings,
            )
        except FilingChatError as exc:
            thread_state["filing_chat_session"] = None
            if session_memory:
                session_memory.record_failure("answer_filing_question", str(exc))
            return {
                "response": (
                    f"Unable to continue the filing chat because {exc}. "
                    "Start a new filing chat when ready."
                )
            }
        if session_memory:
            session_memory.record_tool_event(
                "answer_filing_question",
                "success",
                f"filing={session.filing.form if session.filing else 'unknown'}",
            )
        return {"response": answer}

    if thread_state.get("awaiting_filing_selection"):
        filings_cache, retrieval_error = _ensure_recent_filings(settings, thread_state)
        if retrieval_error:
            thread_state["awaiting_filing_selection"] = False
            return {"response": f"Unable to retrieve filings: {retrieval_error}"}

        if _CANCEL_PATTERN.search(prompt):
            thread_state["awaiting_filing_selection"] = False
            return {
                "response": (
                    "Cancelled filing chat selection. "
                    "Ask again if you want to chat with a filing."
                )
            }

        menu_entries: List[FilingMenuEntry] = thread_state.get("filing_menu_entries", [])
        display_filings: List[SecFiling] = thread_state.get("filings_display") or filings_cache
        company_context = thread_state["company_search"] or thread_state["symbol"] or ""
        lowered_prompt = prompt.lower()

        if "by type" in lowered_prompt:
            form_filter = _extract_form_filter(prompt, filings_cache)
            if not form_filter:
                available_forms = ", ".join(sorted({filing.form for filing in filings_cache}))
                return {
                    "response": (
                        "Specify the filing form when requesting 'by type'. "
                        f"Available forms: {available_forms or 'none'}."
                    )
                }
            filtered = [filing for filing in filings_cache if filing.form == form_filter]
            if not filtered:
                return {"response": f"No {form_filter} filings found in the recent history."}
            formatted = _format_and_store_filings(
                filtered,
                thread_state,
                company_context,
                thread_state["selected_cik"],
                form_type=form_filter,
                include_summary=False,
            )
            thread_state["awaiting_filing_selection"] = True
            instruction = _selection_instruction(thread_state)
            return {"response": f"{formatted}\n\n{instruction}"}

        if "by year" in lowered_prompt:
            year_filter = _extract_year_filter(prompt, filings_cache)
            if not year_filter and "unknown" in lowered_prompt:
                year_filter = "Unknown"
            if not year_filter:
                available_years = ", ".join(
                    sorted(
                        {filing.date[:4] for filing in filings_cache if filing.date and len(filing.date) >= 4}
                        | ({"Unknown"} if any(len(filing.date or "") < 4 for filing in filings_cache) else set())
                    )
                )
                return {
                    "response": (
                        "Specify the filing year when requesting 'by year'. "
                        f"Available years: {available_years or 'none'}."
                    )
                }
            if year_filter == "Unknown":
                filtered = [
                    filing
                    for filing in filings_cache
                    if not (filing.date and len(filing.date) >= 4)
                ]
            else:
                filtered = [
                    filing
                    for filing in filings_cache
                    if filing.date and filing.date.startswith(year_filter)
                ]
            if not filtered:
                return {"response": f"No filings found for the year {year_filter}."}
            formatted = _format_and_store_filings(
                filtered,
                thread_state,
                company_context,
                thread_state["selected_cik"],
                form_type=None,
                include_summary=False,
            )
            thread_state["awaiting_filing_selection"] = True
            instruction = _selection_instruction(thread_state)
            return {"response": f"{formatted}\n\n{instruction}"}

        max_menu_index = menu_entries[-1].index if menu_entries else len(display_filings)
        selection_index = _extract_menu_index(prompt)
        entry: FilingMenuEntry | None = (
            _find_menu_entry(menu_entries, selection_index)
            if selection_index is not None
            else None
        )
        if (
            entry is None
            and selection_index is not None
            and not menu_entries
            and 1 <= selection_index <= len(display_filings)
        ):
            entry = FilingMenuEntry(
                index=selection_index,
                kind="filing",
                filing=display_filings[selection_index - 1],
            )

        if entry:
            if entry.kind == "filing":
                if not settings.enable_network_tools:
                    thread_state["awaiting_filing_selection"] = False
                    return {
                        "response": (
                            "Downloading filings requires network access. "
                            "Set ENABLE_NETWORK_TOOLS=true to continue."
                        )
                    }
                selected_filing = entry.filing
                if selected_filing is None:
                    thread_state["awaiting_filing_selection"] = False
                    return {
                        "response": "The selected filing is no longer available. Start a new filing lookup."
                    }
                if "chat" in lowered_prompt:
                    try:
                        session = prepare_filing_chat(
                            selected_filing,
                            cik=thread_state["selected_cik"],
                            settings=settings,
                        )
                    except FilingChatError as exc:
                        thread_state["awaiting_filing_selection"] = False
                        if session_memory:
                            session_memory.record_failure("prepare_filing_chat", str(exc))
                        return {"response": f"Unable to prepare filing chat: {exc}"}

                    if session_memory:
                        session_memory.record_tool_event(
                            "prepare_filing_chat",
                            "success",
                            f"form={selected_filing.form} date={selected_filing.date}",
                        )
                    thread_state["filing_chat_session"] = session
                    thread_state["awaiting_filing_selection"] = False
                    return {
                        "response": (
                            f"Ready to chat about filing #{entry.index}: {selected_filing.form} "
                            f"from {selected_filing.date}. Ask a question about this filing "
                            "or type 'exit filing chat' when finished."
                        )
                    }

                try:
                    download_path = download_filing_to_directory(
                        selected_filing,
                        timeout=settings.http_timeout,
                        max_size_bytes=int(
                            settings.sec_max_filing_download_mb * 1024 * 1024
                        ),
                    )
                except SecDownloadError as exc:
                    if session_memory:
                        session_memory.record_failure("download_filing_to_directory", str(exc))
                    return {"response": f"Unable to download the filing: {exc}"}

                thread_state["latest_download_path"] = str(download_path)
                if session_memory:
                    session_memory.record_tool_event(
                        "download_filing_to_directory",
                        "success",
                        f"path={download_path}",
                    )

                try:
                    artifacts = run_embedding_pipeline(
                        download_path,
                        metadata={
                            "cik": thread_state["selected_cik"],
                            "form": selected_filing.form,
                            "filing_date": selected_filing.date,
                            "source_url": selected_filing.url,
                        },
                        settings=settings,
                    )
                    thread_state["latest_vectorstore_path"] = str(artifacts.output_path)
                    thread_state["rag_active"] = True
                    thread_state["awaiting_filing_selection"] = False
                    if session_memory:
                        session_memory.record_tool_event(
                            "run_embedding_pipeline",
                            "success",
                            f"output={artifacts.output_path}",
                        )
                    return {"response": "embedding complete. ask query"}
                except EmbeddingPipelineError as exc:
                    thread_state["awaiting_filing_selection"] = True
                    if session_memory:
                        session_memory.record_failure("run_embedding_pipeline", str(exc))
                    return {
                        "response": (
                            f"Downloaded filing #{entry.index} ({selected_filing.form} "
                            f"from {selected_filing.date}) to {download_path}, "
                            f"but embedding failed: {exc}"
                        )
                    }
                except Exception as exc:  # noqa: BLE001
                    thread_state["awaiting_filing_selection"] = True
                    if session_memory:
                        session_memory.record_failure("run_embedding_pipeline", str(exc))
                    return {
                        "response": (
                            f"Downloaded filing #{entry.index} ({selected_filing.form} "
                            f"from {selected_filing.date}) to {download_path}, "
                            f"but embedding hit an unexpected error: {exc}"
                        )
                    }

            if entry.kind == "type" and entry.form:
                filtered = [
                    filing for filing in filings_cache if filing.form == entry.form
                ]
                if not filtered:
                    return {
                        "response": f"No {entry.form} filings found in the recent history."
                    }
                formatted = _format_and_store_filings(
                    filtered,
                    thread_state,
                    company_context,
                    thread_state["selected_cik"],
                    form_type=entry.form,
                    include_summary=False,
                )
                thread_state["awaiting_filing_selection"] = True
                instruction = _selection_instruction(thread_state)
                return {"response": f"{formatted}\n\n{instruction}"}

            if entry.kind == "year" and entry.year:
                if entry.year == "Unknown":
                    filtered = [
                        filing
                        for filing in filings_cache
                        if not (filing.date and len(filing.date) >= 4)
                    ]
                else:
                    filtered = [
                        filing
                        for filing in filings_cache
                        if filing.date and filing.date.startswith(entry.year)
                    ]
                if not filtered:
                    return {"response": f"No filings found for the year {entry.year}."}
                formatted = _format_and_store_filings(
                    filtered,
                    thread_state,
                    company_context,
                    thread_state["selected_cik"],
                    form_type=None,
                    include_summary=False,
                )
                thread_state["awaiting_filing_selection"] = True
                instruction = _selection_instruction(thread_state)
                return {"response": f"{formatted}\n\n{instruction}"}

        if selection_index is not None:
            return {
                "response": (
                    f"Choose a menu number between 1 and {max_menu_index}, or type 'cancel' to stop."
                )
            }

    if (
        thread_state.get("selected_cik")
        and _wants_filing_chat(prompt)
    ):
        filings_cache, retrieval_error = _ensure_recent_filings(settings, thread_state)
        if retrieval_error:
            return {"response": f"Unable to retrieve filings: {retrieval_error}"}
        if not filings_cache:
            return {
                "response": (
                    "No filings are available yet. Provide a valid ticker or CIK first."
                )
            }
        company_context = thread_state["company_search"] or thread_state["symbol"] or ""
        formatted = _format_and_store_filings(
            filings_cache,
            thread_state,
            company_context,
            thread_state["selected_cik"],
            form_type=None,
            include_summary=False,
        )
        thread_state["awaiting_filing_selection"] = True
        instruction = _selection_instruction(thread_state)
        return {
            "response": (
                f"{formatted}\n\n{instruction}"
            )
        }

    if (
        thread_state.get("selected_cik")
        and filings_cache
        and "filing" in prompt.lower()
    ):
        form_filter = _extract_form_filter(prompt, filings_cache)
        if form_filter:
            filtered = [filing for filing in filings_cache if filing.form == form_filter]
            if not filtered:
                return {
                    "response": f"No {form_filter} filings found in the recent history."
                }
            company_context = thread_state["company_search"] or thread_state["symbol"] or ""
            formatted = _format_and_store_filings(
                filtered,
                thread_state,
                company_context,
                thread_state["selected_cik"],
                form_type=form_filter,
                include_summary=False,
            )
            thread_state["awaiting_filing_selection"] = True
            instruction = _selection_instruction(thread_state)
            return {"response": f"{formatted}\n\n{instruction}"}

        year_filter = _extract_year_filter(prompt, filings_cache)
        if year_filter:
            filtered = [
                filing
                for filing in filings_cache
                if filing.date and filing.date.startswith(year_filter)
            ]
            if not filtered:
                return {
                    "response": f"No filings found for the year {year_filter}."
                }
            company_context = thread_state["company_search"] or thread_state["symbol"] or ""
            formatted = _format_and_store_filings(
                filtered,
                thread_state,
                company_context,
                thread_state["selected_cik"],
                form_type=None,
                include_summary=False,
            )
            thread_state["awaiting_filing_selection"] = True
            instruction = _selection_instruction(thread_state)
            return {"response": f"{formatted}\n\n{instruction}"}

    if not symbol and thread_state.get("selected_cik"):
        symbol = thread_state["symbol"]

    include_system_messages = thread_id not in _known_threads
    if include_system_messages:
        _known_threads.add(thread_id)
    messages = []
    if include_system_messages:
        messages.extend(_collect_system_messages(symbol))
    if session_memory:
        for memory_prompt in session_memory.render_system_messages():
            messages.append(SystemMessage(content=memory_prompt))
    if retrieved_long_term:
        formatted_snippets = "\n".join(f"- {snippet}" for snippet in retrieved_long_term)
        messages.append(
            SystemMessage(
                content=(
                    "AgentCore long-term memory highlights to factor into the response:\n"
                    f"{formatted_snippets}"
                )
            )
        )
    messages.append(HumanMessage(content=prompt))

    input_messages = {"messages": messages}

    if memory_store and memory_namespace:
        try:
            memory_store.put(
                memory_namespace,
                str(uuid.uuid4()),
                {"type": "human", "content": prompt},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("AgentCore Memory put (human) failed: %s", exc)

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
    if memory_store and memory_namespace and final_message:
        try:
            memory_store.put(
                memory_namespace,
                str(uuid.uuid4()),
                {"type": "assistant", "content": final_message},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("AgentCore Memory put (assistant) failed: %s", exc)
    return {"response": final_message}


def _parse_candidate_ciks(response: str) -> list[str]:
    candidates: list[str] = []
    for line in response.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            content = stripped[2:]
        else:
            match = re.match(r"\d+\.\s+(.*)", stripped)
            if not match:
                continue
            content = match.group(1)
        first_token = content.split(maxsplit=1)[0]
        digits = "".join(ch for ch in first_token if ch.isdigit())
        if len(digits) >= 7:
            candidates.append(digits.zfill(10))
    return candidates


def _find_menu_entry(
    menu_entries: Iterable[FilingMenuEntry],
    index: int,
) -> FilingMenuEntry | None:
    for entry in menu_entries:
        if entry.index == index:
            return entry
    return None


def _format_and_store_filings(
    filings: Iterable[SecFiling],
    thread_state: Dict[str, Any],
    company_search: str,
    cik: str,
    *,
    form_type: str | None,
    include_summary: bool,
) -> str:
    filings_list = list(filings)
    formatted_text, menu_entries = cast(
        tuple[str, List[FilingMenuEntry]],
        format_sec_filings(
            filings_list,
            form_type,
            company_search,
            cik,
            include_index=True,
            include_summary=include_summary,
            return_menu=True,
        ),
    )
    thread_state["filings_display"] = filings_list
    thread_state["filing_menu_entries"] = menu_entries
    return formatted_text


def _selection_instruction(thread_state: Dict[str, Any]) -> str:
    menu_entries: List[FilingMenuEntry] = thread_state.get("filing_menu_entries", [])
    if not menu_entries:
        return "Select a filing number or type 'cancel'."
    max_index = menu_entries[-1].index
    if all(entry.kind == "filing" for entry in menu_entries):
        return f"Select a filing number between 1 and {max_index}, or type 'cancel'."
    return (
        f"Select a menu number between 1 and {max_index} to continue "
        "(filing numbers open the document; type/year entries refine the list), or type 'cancel'."
    )


def _wants_filing_chat(prompt: str) -> bool:
    return bool(_CHAT_FILING_PATTERN.search(prompt))


def _extract_menu_index(text: str) -> Optional[int]:
    match = re.search(r"\b(\d{1,3})\b", text)
    if not match:
        return None
    return int(match.group(1))


def _extract_cik_index(
    prompt: str,
    candidates: Sequence[CikMatch],
) -> Optional[str]:
    index = _extract_menu_index(prompt)
    if index is None or not 1 <= index <= len(candidates):
        return None
    return candidates[index - 1].cik


def _extract_form_filter(prompt: str, filings: Iterable[SecFiling]) -> Optional[str]:
    lowered = prompt.lower()
    forms = {filing.form for filing in filings}
    for form in forms:
        if form.lower() in lowered:
            return form
    return None


def _extract_year_filter(prompt: str, filings: Iterable[SecFiling]) -> Optional[str]:
    years = {filing.date[:4] for filing in filings if filing.date and len(filing.date) >= 4}
    for year in years:
        if year and year in prompt:
            return year
    return None


def _ensure_recent_filings(
    settings,
    thread_state: Dict[str, Any],
) -> tuple[List[SecFiling], Optional[str]]:
    filings: List[SecFiling] = thread_state.get("filings", [])
    if filings:
        return filings, None
    selected_cik = thread_state.get("selected_cik")
    if not selected_cik:
        return [], "A CIK must be selected before fetching filings."
    try:
        filings = get_recent_filings(
            selected_cik,
            timeout=settings.http_timeout,
        )
    except SecLookupError as exc:
        return [], str(exc)
    thread_state["filings"] = filings
    return filings, None



def run_local_cli() -> None:
    logger.info("AgentCore POC CLI. Type 'exit' to quit.")
    settings = load_settings()
    vectorstore_summaries = _discover_vectorstore_summaries()
    menu_result = _vectorstore_menu(vectorstore_summaries, settings)
    if menu_result == "exit":
        return

    session_id = str(uuid.uuid4())
    context = SimpleNamespace(
        session_id=session_id,
        actor_id=settings.actor_id,
        identity={"actor_id": settings.actor_id},
    )
    intro = invoke({}, context=context)
    logger.info("Agent: %s", intro["response"])
    logger.info("Select enter to proceed with top result.")
    last_symbol: str | None = None
    awaiting_cik_selection = False
    awaiting_filing_selection = False
    candidate_ciks: list[str] = []
    require_symbol = True
    while True:
        prompt_label = "Enter response: "
        if awaiting_cik_selection and candidate_ciks:
            prompt_label = "Enter response (press Enter for top CIK): "
        elif awaiting_filing_selection:
            prompt_label = "Enter response (select filing index): "
        elif require_symbol:
            prompt_label = "Enter company or symbol: "

        raw_input = input(prompt_label)
        user_input = raw_input.strip()

        if (
            awaiting_cik_selection
            and not user_input
            and candidate_ciks
        ):
            user_input = candidate_ciks[0]

        if user_input.lower() in {"exit", "quit"}:
            break

        if require_symbol and not user_input:
            logger.info("Please provide a company name or ticker symbol to begin.")
            continue

        payload: Dict[str, Any] = {"prompt": user_input}
        detected_symbol = _extract_symbol_from_text(user_input)
        if detected_symbol:
            last_symbol = detected_symbol
            payload["symbol"] = detected_symbol
        elif last_symbol:
            payload["symbol"] = last_symbol
        result = invoke(payload, context=context)
        response_text = result["response"]
        logger.info("Agent: %s", response_text)
        candidate_ciks = _parse_candidate_ciks(response_text)
        awaiting_cik_selection = bool(
            candidate_ciks
            and (
                "Reply with the desired 10-digit CIK" in response_text
                or "Reply with the numbered option" in response_text
            )
        )
        awaiting_filing_selection = any(
            phrase in response_text
            for phrase in (
                "Select a filing index",
                "Select a filing number",
                "Select a menu number",
            )
        )
        if (
            "Provide a new ticker symbol" in response_text
            or "Cancelled CIK selection" in response_text
        ):
            last_symbol = None
        require_symbol = last_symbol is None and not awaiting_cik_selection


if __name__ == "__main__":
    run_local_cli()

