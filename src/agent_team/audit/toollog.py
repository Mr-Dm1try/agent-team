"""The tool-call log the auditor reads as evidence.

No contract in ``core`` reads events back, so the auditor keeps its own log, filled
from ``tool_call_after`` events. The event's ``data`` is read leniently: ``tool`` (or
``name``), ``args`` (or ``arguments``), ``result`` (or ``content``), ``is_error`` and
``blocked`` (True when the pre-check refused the call, so it never ran).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agent_team.core.interfaces import EventBus
from agent_team.core.models import Event, EventName, utcnow


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: str | None = None
    agent: str | None = None
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    is_error: bool = False
    blocked: bool = False  # refused by the pre-check: it never ran
    ts: datetime = Field(default_factory=utcnow)


class ToolCallLog(Protocol):
    async def for_task(self, task_id: str) -> list[ToolCall]: ...

    async def for_run(self, run_id: str) -> list[ToolCall]: ...


class InMemoryToolCallLog:
    """Implements ``ToolCallLog``. ``attach(bus)`` records every finished tool call."""

    def __init__(self) -> None:
        self._calls: list[ToolCall] = []

    def record(self, call: ToolCall) -> None:
        self._calls.append(call)

    def attach(self, bus: EventBus) -> None:
        bus.on_after(EventName.TOOL_CALL_AFTER, self._on_event)

    async def _on_event(self, event: Event) -> None:
        call = tool_call_from_event(event)
        if call is not None:
            self.record(call)

    async def for_task(self, task_id: str) -> list[ToolCall]:
        return [c for c in self._calls if c.task_id == task_id]

    async def for_run(self, run_id: str) -> list[ToolCall]:
        return [c for c in self._calls if c.run_id == run_id]


def tool_call_from_event(event: Event) -> ToolCall | None:
    data = event.data
    tool = data.get("tool", data.get("name"))
    if not isinstance(tool, str):
        return None
    args = data.get("args", data.get("arguments", {}))
    result = data.get("result", data.get("content", ""))
    return ToolCall(
        run_id=event.run_id,
        task_id=event.task_id,
        agent=event.agent.value if event.agent else None,
        tool=tool,
        args=args if isinstance(args, dict) else {"value": args},
        result=result if isinstance(result, str) else str(result),
        is_error=bool(data.get("is_error", False)),
        blocked=bool(data.get("blocked", False)),
        ts=event.ts,
    )
