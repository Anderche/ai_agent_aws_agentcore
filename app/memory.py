from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional


def _truncate(text: str, limit: int = 240) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


@dataclass(frozen=True)
class ToolEvent:
    name: str
    status: str
    detail: Optional[str] = None


@dataclass(frozen=True)
class FailureEvent:
    name: str
    detail: str


class SessionMemory:
    """Simple in-memory recorder for user intent, tool usage, and failures."""

    def __init__(
        self,
        *,
        goal_limit: int = 5,
        tool_history_limit: int = 10,
        failure_history_limit: int = 10,
    ) -> None:
        self._goals: Deque[str] = deque(maxlen=goal_limit)
        self._tool_events: Deque[ToolEvent] = deque(maxlen=tool_history_limit)
        self._failure_events: Deque[FailureEvent] = deque(maxlen=failure_history_limit)
        self._primary_symbol: Optional[str] = None

    def record_goal(self, goal_text: str) -> None:
        normalized = goal_text.strip()
        if not normalized:
            return
        lowered = normalized.lower()
        if lowered.startswith("/refresh"):
            return
        if normalized in self._goals:
            return

        # Heuristic: capture explicit goals or the very first prompt.
        if not self._goals or any(
            phrase in lowered
            for phrase in ("goal", "need to", "objective", "looking to", "task is")
        ):
            self._goals.append(_truncate(normalized))

    def record_tool_event(
        self,
        name: str,
        status: str,
        detail: Optional[str] = None,
    ) -> None:
        status = status.strip().lower()
        event = ToolEvent(name=name.strip(), status=status, detail=_truncate(detail) if detail else None)
        self._tool_events.append(event)

    def record_failure(self, name: str, detail: str) -> None:
        detail = _truncate(detail)
        self._failure_events.append(FailureEvent(name=name.strip(), detail=detail))

    def set_primary_symbol(self, symbol: Optional[str]) -> None:
        if not symbol:
            return
        upper = symbol.strip().upper()
        if not upper:
            return
        if self._primary_symbol is None:
            self._primary_symbol = upper

    def _format_tool_history(self) -> Optional[str]:
        if not self._tool_events:
            return None
        recent: Iterable[ToolEvent] = list(self._tool_events)[-3:]
        rendered: List[str] = []
        for event in recent:
            detail = f" ({event.detail})" if event.detail else ""
            rendered.append(f"{event.name} [{event.status}]{detail}")
        return "; ".join(rendered)

    def _format_failures(self) -> Optional[str]:
        if not self._failure_events:
            return None
        recent: Iterable[FailureEvent] = list(self._failure_events)[-3:]
        rendered = [f"{event.name}: {event.detail}" for event in recent]
        return "; ".join(rendered)

    def render_system_messages(self) -> List[str]:
        messages: List[str] = []
        if self._primary_symbol:
            messages.append(
                f"Persistent ticker context: {self._primary_symbol}. Keep recommendations focused on this company unless the user refreshes."
            )
        if self._goals:
            joined_goals = "; ".join(self._goals)
            messages.append(f"User stated goals to honor: {joined_goals}.")
        tool_history = self._format_tool_history()
        if tool_history:
            messages.append(
                f"Recent tool invocations to reference: {tool_history}."
            )
        failures = self._format_failures()
        if failures:
            messages.append(
                f"Known failure heuristics or blockers detected so far: {failures}. Avoid repeating the same failed route without new guidance."
            )
        return messages


