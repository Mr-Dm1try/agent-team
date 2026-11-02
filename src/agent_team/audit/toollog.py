"""The tool-call log the auditor reads as evidence.

No contract in ``core`` reads events back, so the auditor keeps its own log, filled
from ``tool_call_after`` events. The event's ``data`` is read leniently: ``tool`` (or
``name``), ``args`` (or ``arguments``), ``result`` (or ``content``), ``is_error`` and
``blocked`` (True when the pre-check refused the call, so it never ran).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from agent_team.core.base import Contract
from agent_team.core.interfaces import EventBus
from agent_team.core.models import Event, EventName, utcnow
from agent_team.core.tracing import read_events


class ToolCall(Contract):
    """One finished (or refused) tool call, as the auditor reads it."""

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
    """Where the auditor reads tool calls from."""

    async def for_task(self, run_id: str, task_id: str) -> list[ToolCall]:
        """The task's calls, oldest first."""
        ...

    async def for_run(self, run_id: str) -> list[ToolCall]:
        """The run's calls, oldest first."""
        ...


class InMemoryToolCallLog:
    """Implements ``ToolCallLog`` in memory, for tests and short runs.
    ``attach(bus)`` records every finished tool call."""

    def __init__(self) -> None:
        self._calls: list[ToolCall] = []

    def record(self, call: ToolCall) -> None:
        """Add one call."""
        self._calls.append(call)

    def attach(self, bus: EventBus) -> None:
        """Record every ``tool_call_after`` event from the bus."""
        bus.on_after(EventName.TOOL_CALL_AFTER, self._on_event)

    async def _on_event(self, event: Event) -> None:
        call = tool_call_from_event(event)
        if call is not None:
            self.record(call)

    async def for_task(self, run_id: str, task_id: str) -> list[ToolCall]:
        """The task's calls, oldest first."""
        return [c for c in self._calls if c.run_id == run_id and c.task_id == task_id]

    async def for_run(self, run_id: str) -> list[ToolCall]:
        """The run's calls, oldest first."""
        return [c for c in self._calls if c.run_id == run_id]


class EventFileToolCallLog:
    """Implements ``ToolCallLog`` over the run's ``events.jsonl``, so the audit still
    sees every tool call after a crash and resume."""

    def __init__(self, runs_dir: str | Path) -> None:
        self.runs_dir = runs_dir

    async def for_run(self, run_id: str) -> list[ToolCall]:
        """The run's calls, oldest first."""
        events = await asyncio.to_thread(read_events, self.runs_dir, run_id)
        calls = (tool_call_from_event(e) for e in events if e.name is EventName.TOOL_CALL_AFTER)
        return [c for c in calls if c is not None]

    async def for_task(self, run_id: str, task_id: str) -> list[ToolCall]:
        """The task's calls, oldest first."""
        return [c for c in await self.for_run(run_id) if c.task_id == task_id]


def tool_call_from_event(event: Event) -> ToolCall | None:
    """Read a ``tool_call_after`` event into a ToolCall; None if it names no tool."""
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
