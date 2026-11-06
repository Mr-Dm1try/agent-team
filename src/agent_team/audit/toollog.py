"""The tool-call log the auditor reads as evidence.

No contract in ``core`` reads events back, so the auditor keeps its own log, filled
from ``tool_call_after`` events. The event's ``data`` is read leniently: ``tool`` (or
``name``), ``args`` (or ``arguments``), ``result`` (or ``content``), ``is_error``,
``blocked`` (True when the pre-check refused the call, so it never ran) and ``data``
(the tool's structured result, such as a command's ``exit_code``).
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
    attempt: int | None = None
    agent: str | None = None
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    is_error: bool = False
    blocked: bool = False  # refused by the pre-check: it never ran
    data: dict[str, Any] = Field(default_factory=dict)
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
        """Start with no calls."""
        self._calls: list[ToolCall] = []

    def record(self, call: ToolCall) -> None:
        """Add one call."""
        self._calls.append(call)

    def attach(self, bus: EventBus) -> None:
        """Record every ``tool_call_after`` event from the bus."""
        bus.on_after(EventName.TOOL_CALL_AFTER, self._on_event)

    async def _on_event(self, event: Event) -> None:
        """Record one tool_call_after event."""
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
        """Point at the runs folder that holds each run's events.jsonl."""
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
    extra = data.get("data")
    return ToolCall(
        run_id=event.run_id,
        task_id=event.task_id,
        attempt=event.attempt,
        agent=event.agent.value if event.agent else None,
        tool=tool,
        args=args if isinstance(args, dict) else {"value": args},
        result=result if isinstance(result, str) else str(result),
        is_error=bool(data.get("is_error", False)),
        blocked=bool(data.get("blocked", False)),
        data=extra if isinstance(extra, dict) else {},
        ts=event.ts,
    )


#: Commands listed in an attempt summary; earlier ones are counted, not listed.
SUMMARY_COMMANDS = 20
#: Longest command shown in an attempt summary.
SUMMARY_COMMAND_CHARS = 160
#: Tools whose calls an attempt summary lists one by one.
COMMAND_TOOLS = frozenset({"run_command", "install_deps"})


def _command_line(call: ToolCall) -> str:
    """One command and how it ended, such as ``run_command pytest -q: exit 1``."""
    if call.tool == "install_deps":
        packages = call.data.get("packages") or call.args.get("packages") or []
        shown = " ".join(str(p) for p in packages) or str(call.args.get("requirements", ""))
    else:
        argv = call.data.get("argv") or call.args.get("argv") or []
        shown = " ".join(str(a) for a in argv) if isinstance(argv, list) else str(argv)
    if len(shown) > SUMMARY_COMMAND_CHARS:
        shown = shown[: SUMMARY_COMMAND_CHARS - 3] + "..."
    if call.blocked:
        ended = "blocked by the pre-check, not run"
    elif isinstance(call.data.get("exit_code"), int):
        ended = f"exit {call.data['exit_code']}"
    else:
        first = call.result.strip().splitlines()[0] if call.result.strip() else "no result"
        ended = f"failed: {first[:SUMMARY_COMMAND_CHARS]}"
    return f"{call.tool} {shown}: {ended}"


def summarize_attempt(error: str, calls: list[ToolCall]) -> str:
    """What one failed attempt did, for the next attempt: the error it ended with, the
    commands it ran with their exit codes (repeats folded, the last
    ``SUMMARY_COMMANDS`` listed) and the files it wrote."""
    lines = [f"It ended with: {error}"]
    commands: list[tuple[str, int]] = []  # each line and how many times in a row
    for call in calls:
        if call.tool not in COMMAND_TOOLS:
            continue
        text = _command_line(call)
        if commands and commands[-1][0] == text:
            commands[-1] = (text, commands[-1][1] + 1)
        else:
            commands.append((text, 1))
    if commands:
        lines.append("Commands it ran:")
        left_out = len(commands) - SUMMARY_COMMANDS
        if left_out > 0:
            lines.append(f"- ({left_out} earlier commands left out)")
        for text, times in commands[-SUMMARY_COMMANDS:]:
            lines.append(f"- {text}" + (f" ({times} times)" if times > 1 else ""))
    else:
        lines.append("Commands it ran: none.")
    written = [
        str(c.args.get("path"))
        for c in calls
        if c.tool == "write_file" and not c.is_error and not c.blocked and c.args.get("path")
    ]
    written = list(dict.fromkeys(written))
    lines.append("Files it wrote: " + (", ".join(written) if written else "none") + ".")
    others: dict[str, int] = {}
    for call in calls:
        if call.tool not in COMMAND_TOOLS and call.tool != "write_file":
            others[call.tool] = others.get(call.tool, 0) + 1
    if others:
        lines.append(
            "Other calls: " + ", ".join(f"{n} {tool}" for tool, n in sorted(others.items())) + "."
        )
    return "\n".join(lines)
