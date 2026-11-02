"""One JSON Lines log per run, holding events and spans.

- ``JsonlTracer`` is an after hook on every event. It writes each event to one file,
  ``runs/<run>/logs/events.jsonl``. Every line carries its agent and task, so one
  agent's trail is a filter (``read_events``) rather than a separate file.
- ``Tracer.span`` opens a span (run, phase, task, attempt, llm, tool or audit). Spans
  nest through a context variable. Opening one emits ``span_started`` and closing it
  emits ``span_finished`` with status, duration, attributes and the traceback on error.
  Both go through the bus, so they land in ``events.jsonl`` like any other event.
- ``read_events`` reads the event log back, optionally filtered by agent or task.
- ``load_spans`` and ``span_tree`` rebuild the tree for ``agent-team trace``. A span
  that started but never finished (the process died) shows as ``open``.
"""

from __future__ import annotations

import json
import threading
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent_team.core.interfaces import EventBus
from agent_team.core.models import AgentName, Event, EventName, utcnow

SpanKind = Literal["run", "phase", "task", "attempt", "llm", "tool", "audit", "other"]
SpanStatus = Literal["ok", "error", "open"]


def new_trace_id() -> str:
    """A new 32-character trace id."""
    return uuid4().hex


def new_span_id() -> str:
    """A new 16-character span id."""
    return uuid4().hex[:16]


class _Appender:
    """Appends lines to files; one lock per process is plenty for our volume."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def append(self, path: Path, line: str) -> None:
        """Append one line to a file, creating folders as needed."""
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class JsonlTracer:
    """After hook that writes every event to the run's ``events.jsonl``."""

    def __init__(self, runs_dir: str | Path) -> None:
        self.runs_dir = Path(runs_dir)
        self._out = _Appender()

    def attach(self, bus: EventBus) -> None:
        """Subscribe to every event on the bus."""
        bus.on_after(None, self.on_event)

    async def on_event(self, event: Event) -> None:
        """After hook: write the event."""
        self.write(event)

    def write(self, event: Event) -> None:
        """Append one event to ``runs/<run>/logs/events.jsonl``."""
        path = events_path(self.runs_dir, event.run_id)
        self._out.append(path, event.model_dump_json(exclude_none=True))


def events_path(runs_dir: str | Path, run_id: str) -> Path:
    """``runs/<run>/logs/events.jsonl``."""
    return Path(runs_dir) / run_id / "logs" / "events.jsonl"


def read_events(
    runs_dir: str | Path,
    run_id: str,
    *,
    agent: AgentName | str | None = None,
    task_id: str | None = None,
) -> list[Event]:
    """A run's events in write order, optionally only one agent's or one task's.
    Unreadable lines (e.g. one cut off by a crash) are skipped."""
    path = events_path(runs_dir, run_id)
    if not path.exists():
        return []
    events: list[Event] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = Event.model_validate_json(line)
        except ValueError:
            continue
        if agent is not None and event.agent != agent:
            continue
        if task_id is not None and event.task_id != task_id:
            continue
        events.append(event)
    return events


# --- spans -------------------------------------------------------------------------


class SpanRecord(BaseModel):
    """One span rebuilt from its ``span_started`` and ``span_finished`` events. An
    ``open`` span has no end: the run stopped before it closed."""

    model_config = ConfigDict(extra="forbid")
    run_id: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str
    kind: SpanKind
    task_id: str | None = None
    agent: AgentName | None = None
    attempt: int | None = None
    start: datetime
    end: datetime | None = None
    duration_ms: float | None = None
    status: SpanStatus
    error_type: str | None = None
    error: str | None = None
    traceback: str | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)


@dataclass
class Span:
    """An open span. Attributes set on it are written when it closes."""

    run_id: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: SpanKind
    task_id: str | None = None
    agent: AgentName | None = None
    attempt: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    start: datetime = field(default_factory=utcnow)

    def set(self, **attrs: Any) -> None:
        """Set attributes on the span."""
        self.attrs.update(attrs)

    def add(self, key: str, amount: float) -> None:
        """Accumulate a number, e.g. tokens or cost across calls."""
        self.attrs[key] = self.attrs.get(key, 0) + amount

    def _event(self, name: EventName, ts: datetime, data: dict[str, Any]) -> Event:
        return Event(
            name=name,
            run_id=self.run_id,
            ts=ts,
            task_id=self.task_id,
            attempt=self.attempt,
            agent=self.agent,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            level="error" if data.get("status") == "error" else "debug",
            data={"name": self.name, "kind": self.kind, **data},
        )


_current: ContextVar[Span | None] = ContextVar("agent_team_span", default=None)


def current_span() -> Span | None:
    """The innermost open span in this task, or None."""
    return _current.get()


class Tracer:
    """Opens nested spans and emits them as ``span_started`` and ``span_finished``
    events. Attach a ``JsonlTracer`` to the same bus to write them."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus

    @asynccontextmanager
    async def span(
        self,
        run_id: str,
        name: str,
        kind: SpanKind = "other",
        *,
        task_id: str | None = None,
        agent: AgentName | str | None = None,
        attempt: int | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        **attrs: Any,
    ) -> AsyncIterator[Span]:
        """Open a span nested under the current one. It is emitted on open and again
        on close with its status, duration and any traceback."""
        parent = _current.get()
        if parent is not None and parent.run_id != run_id:
            parent = None
        span = Span(
            run_id=run_id,
            trace_id=trace_id or (parent.trace_id if parent else new_trace_id()),
            span_id=new_span_id(),
            parent_span_id=parent_span_id or (parent.span_id if parent else None),
            name=name,
            kind=kind,
            task_id=task_id or (parent.task_id if parent else None),
            agent=AgentName(agent) if agent is not None else (parent.agent if parent else None),
            attempt=attempt if attempt is not None else (parent.attempt if parent else None),
            attrs=dict(attrs),
        )
        await self.bus.emit(span._event(EventName.SPAN_STARTED, span.start, {}))
        token = _current.set(span)
        try:
            yield span
        except BaseException as exc:
            await self._close(span, exc)
            raise
        else:
            await self._close(span, None)
        finally:
            _current.reset(token)

    async def _close(self, span: Span, exc: BaseException | None) -> None:
        end = utcnow()
        data: dict[str, Any] = {
            "start": span.start.isoformat(),
            "end": end.isoformat(),
            "duration_ms": (end - span.start).total_seconds() * 1000,
            "status": "ok" if exc is None else "error",
            "attrs": _jsonable(span.attrs),
        }
        if exc is not None:
            data["error_type"] = type(exc).__name__
            data["error"] = str(exc)
            data["traceback"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        await self.bus.emit(span._event(EventName.SPAN_FINISHED, end, data))


def _jsonable(attrs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(attrs, default=str))
    return out


# --- reading traces ----------------------------------------------------------------


@dataclass
class SpanNode:
    """A span with its children, for printing the trace tree."""

    span: SpanRecord
    children: list[SpanNode] = field(default_factory=list)


def spans_from_events(events: list[Event]) -> list[SpanRecord]:
    """Spans rebuilt from ``span_started`` and ``span_finished`` events, in start
    order. A start with no finish is an ``open`` span; a finish whose start line was
    lost still counts. Events that are not spans, or that do not parse, are skipped."""
    spans: dict[str, SpanRecord] = {}
    for event in events:
        if event.name not in (EventName.SPAN_STARTED, EventName.SPAN_FINISHED):
            continue
        if event.span_id is None or event.trace_id is None:
            continue
        d = event.data
        fields: dict[str, Any] = {
            "run_id": event.run_id,
            "trace_id": event.trace_id,
            "span_id": event.span_id,
            "parent_span_id": event.parent_span_id,
            "name": d.get("name"),
            "kind": d.get("kind"),
            "task_id": event.task_id,
            "agent": event.agent,
            "attempt": event.attempt,
        }
        if event.name is EventName.SPAN_STARTED:
            fields |= {"start": event.ts, "status": "open"}
        else:
            fields |= {
                key: d.get(key)
                for key in (
                    "start",
                    "end",
                    "duration_ms",
                    "status",
                    "error_type",
                    "error",
                    "traceback",
                )
            }
            fields["attrs"] = d.get("attrs") or {}
            fields["start"] = fields["start"] or event.ts
        try:
            spans[event.span_id] = SpanRecord.model_validate(fields)
        except ValueError:
            continue
    return sorted(spans.values(), key=lambda s: s.start)


def load_spans(runs_dir: str | Path, run_id: str) -> list[SpanRecord]:
    """A run's spans, read from its ``events.jsonl``."""
    return spans_from_events(read_events(runs_dir, run_id))


def span_tree(spans: list[SpanRecord]) -> list[SpanNode]:
    """Roots in start order, each with children in start order. Spans whose parent
    is missing become roots."""
    nodes = {s.span_id: SpanNode(s) for s in sorted(spans, key=lambda s: s.start)}
    roots: list[SpanNode] = []
    for node in nodes.values():
        parent = nodes.get(node.span.parent_span_id or "")
        (parent.children if parent else roots).append(node)
    return roots


def render_tree(roots: list[SpanNode]) -> list[str]:
    """Plain-text tree: name, kind, status, duration, tokens and cost when present."""
    lines: list[str] = []

    def walk(node: SpanNode, depth: int) -> None:
        s = node.span
        extra = []
        for key in ("input_tokens", "output_tokens", "cost_usd"):
            if key in s.attrs:
                extra.append(f"{key}={s.attrs[key]}")
        mark = f"ERROR {s.error_type}: {s.error}" if s.status == "error" else s.status
        took = f" {s.duration_ms:.0f}ms" if s.duration_ms is not None else ""
        suffix = f" {' '.join(extra)}" if extra else ""
        lines.append(f"{'  ' * depth}{s.kind}:{s.name} [{mark}]{took}{suffix}")
        for child in node.children:
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return lines
