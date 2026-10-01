from __future__ import annotations

import json
import os

import fcntl

from .config import BASE_DIR, Settings
from .rag_pipeline import VECTORSTORE_DIR

COUNTER_PATH = BASE_DIR / "data" / "cost_guards.json"
_LOCK_PATH = COUNTER_PATH.with_suffix(".lock")

_DEFAULTS = {
    "filing_chat_prepares": 0,
    "bedrock_replies": 0,
    "embed_floor": 0,
    "pending_embeds": 0,
    "pending_filing_chats": 0,
    "pending_replies": 0,
}


class CostGuardBlocked(Exception):
    """Raised when a sandbagged AWS cap has been reached."""


def _count_vectorstores() -> int:
    if not VECTORSTORE_DIR.exists():
        return 0
    return sum(1 for path in VECTORSTORE_DIR.glob("*_vectorstore.jsonl") if path.is_file())


def _read_state() -> dict[str, int]:
    if not COUNTER_PATH.exists():
        return dict(_DEFAULTS)
    try:
        raw = json.loads(COUNTER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(_DEFAULTS)
    if not isinstance(raw, dict):
        return dict(_DEFAULTS)
    state = dict(_DEFAULTS)
    for key in _DEFAULTS:
        value = raw.get(key, 0)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            state[key] = value
    return state


def _write_state(state: dict[str, int]) -> None:
    COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = COUNTER_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, COUNTER_PATH)


class _CounterLock:
    def __enter__(self) -> "_CounterLock":
        _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._handle = _LOCK_PATH.open("a+")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        return False


def apply_startup_reset(settings: Settings) -> None:
    """Zero usage counters and allow the next embed cap on top of files already stored."""
    if not settings.cost_guards_reset:
        return
    with _CounterLock():
        state = dict(_DEFAULTS)
        state["embed_floor"] = _count_vectorstores()
        _write_state(state)


def embedded_filing_message(settings: Settings) -> str:
    return (
        "Sorry, no more documents can be chunked. "
        f"{settings.max_embedded_filings} filings are already embedded."
    )


def filing_chat_message(settings: Settings) -> str:
    return (
        "Sorry, no more filing chats can be prepared. "
        f"{settings.max_filing_chat_prepares} filing chats are already set up."
    )


def bedrock_reply_message(settings: Settings) -> str:
    return (
        "Sorry, no more assistant replies are available. "
        f"{settings.max_bedrock_replies} replies have already been sent."
    )


def reserve_embedded_filing(settings: Settings) -> bool:
    """Hold an embed slot until the pipeline finishes. False when guards are off."""
    if settings.cost_guards_disabled:
        return False
    with _CounterLock():
        state = _read_state()
        used = _count_vectorstores() + state["pending_embeds"]
        allowed = state["embed_floor"] + settings.max_embedded_filings
        if used >= allowed:
            raise CostGuardBlocked(embedded_filing_message(settings))
        state["pending_embeds"] += 1
        _write_state(state)
    return True


def release_embedded_filing(settings: Settings) -> None:
    if settings.cost_guards_disabled:
        return
    with _CounterLock():
        state = _read_state()
        state["pending_embeds"] = max(0, state["pending_embeds"] - 1)
        _write_state(state)


def reserve_filing_chat(settings: Settings) -> bool:
    if settings.cost_guards_disabled:
        return False
    with _CounterLock():
        state = _read_state()
        used = state["filing_chat_prepares"] + state["pending_filing_chats"]
        if used >= settings.max_filing_chat_prepares:
            raise CostGuardBlocked(filing_chat_message(settings))
        state["pending_filing_chats"] += 1
        _write_state(state)
    return True


def commit_filing_chat(settings: Settings) -> None:
    if settings.cost_guards_disabled:
        return
    with _CounterLock():
        state = _read_state()
        state["pending_filing_chats"] = max(0, state["pending_filing_chats"] - 1)
        state["filing_chat_prepares"] += 1
        _write_state(state)


def release_filing_chat(settings: Settings) -> None:
    if settings.cost_guards_disabled:
        return
    with _CounterLock():
        state = _read_state()
        state["pending_filing_chats"] = max(0, state["pending_filing_chats"] - 1)
        _write_state(state)


def reserve_bedrock_reply(settings: Settings) -> bool:
    if settings.cost_guards_disabled:
        return False
    with _CounterLock():
        state = _read_state()
        used = state["bedrock_replies"] + state["pending_replies"]
        if used >= settings.max_bedrock_replies:
            raise CostGuardBlocked(bedrock_reply_message(settings))
        state["pending_replies"] += 1
        _write_state(state)
    return True


def commit_bedrock_reply(settings: Settings) -> None:
    if settings.cost_guards_disabled:
        return
    with _CounterLock():
        state = _read_state()
        state["pending_replies"] = max(0, state["pending_replies"] - 1)
        state["bedrock_replies"] += 1
        _write_state(state)


def release_bedrock_reply(settings: Settings) -> None:
    if settings.cost_guards_disabled:
        return
    with _CounterLock():
        state = _read_state()
        state["pending_replies"] = max(0, state["pending_replies"] - 1)
        _write_state(state)
