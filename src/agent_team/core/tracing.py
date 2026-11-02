"""JSON Lines logs and spans.

- ``JsonlTracer`` is an after hook on every event. It writes each event to
  ``runs/<run>/logs/<agent>.jsonl`` and to the combined ``runs/<run>/logs/events.jsonl``.
- ``Tracer.span`` opens a span (run, phase, task, attempt, llm or tool call). Spans nest
  through a context variable and are written to ``runs/<run>/logs/spans.jsonl`` when
  they close, with status, duration, attributes and the traceback on error.
- ``load_spans`` and ``span_tree`` rebuild the tree for ``agent-team trace``.
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
from agent_team.core.models import Event, utcnow

SpanKind = Literal["run", "phase", "task", "attempt", "llm", "tool", "other"]


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
    """After hook that writes every event to per-agent and combined JSON Lines logs."""

    def __init__(self, runs_dir: str | Path) -> None:
        self.runs_dir = Path(runs_dir)
        self._out = _Appender()

    def logs_dir(self, run_id: str) -> Path:
        """``runs/<run>/logs``."""
        return self.runs_dir / run_id / "logs"

    def attach(self, bus: EventBus) -> None:
        """Subscribe to every event on the bus."""
        bus.on_after(None, self.on_event)

    async def on_event(self, event: Event) -> None:
        """After hook: write the event."""
        self.write(event)

    def write(self, event: Event) -> None:
        """Write one event to its agent's log and to ``events.jsonl``."""
        line = event.model_dump_json(exclude_none=True)
        logs = self.logs_dir(event.run_id)
        agent = event.agent.value if event.agent else "system"
        self._out.append(logs / f"{agent}.jsonl", line)
        self._out.append(logs / "events.jsonl", line)


# --- spans -------------------------------------------------------------------------


class SpanRecord(BaseModel):
    """One closed span as written to ``spans.jsonl``."""

    model_config = ConfigDict(extra="forbid")
    run_id: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str
    kind: SpanKind
    task_id: str | None = None
    agent: str | None = None
    attempt: int | None = None
    start: datetime
    end: datetime
    duration_ms: float
    status: Literal["ok", "error"]
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
    agent: str | None = None
    attempt: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    start: datetime = field(default_factory=utcnow)

    def set(self, **attrs: Any) -> None:
        """Set attributes on the span."""
        self.attrs.update(attrs)

    def add(self, key: str, amount: float) -> None:
        """Accumulate a number, e.g. tokens or cost across calls."""
        self.attrs[key] = self.attrs.get(key, 0) + amount


_current: ContextVar[Span | None] = ContextVar("agent_team_span", default=None)


def current_span() -> Span | None:
    """The innermost open span in this task, or None."""
    return _current.get()


class Tracer:
    """Opens nested spans and writes them to ``spans.jsonl``."""

    def __init__(self, runs_dir: str | Path) -> None:
        self.runs_dir = Path(runs_dir)
        self._out = _Appender()

    def spans_path(self, run_id: str) -> Path:
        """``runs/<run>/logs/spans.jsonl``."""
        return self.runs_dir / run_id / "logs" / "spans.jsonl"

    @asynccontextmanager
    async def span(
        self,
        run_id: str,
        name: str,
        kind: SpanKind = "other",
        *,
        task_id: str | None = None,
        agent: str | None = None,
        attempt: int | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        **attrs: Any,
    ) -> AsyncIterator[Span]:
        """Open a span nested under the current one. It is written on close with its
        status, duration and any traceback."""
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
            agent=agent or (parent.agent if parent else None),
            attempt=attempt if attempt is not None else (parent.attempt if parent else None),
            attrs=dict(attrs),
        )
        token = _current.set(span)
        try:
            yield span
        except BaseException as exc:
            self._close(span, "error", exc)
            raise
        else:
            self._close(span, "ok", None)
        finally:
            _current.reset(token)

    def _close(self, span: Span, status: Literal["ok", "error"], exc: BaseException | None) -> None:
        end = utcnow()
        record = SpanRecord(
            run_id=span.run_id,
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            name=span.name,
            kind=span.kind,
            task_id=span.task_id,
            agent=span.agent,
            attempt=span.attempt,
            start=span.start,
            end=end,
            duration_ms=(end - span.start).total_seconds() * 1000,
            status=status,
            error=None if exc is None else f"{type(exc).__name__}: {exc}",
            traceback=None
            if exc is None
            else "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            attrs=_jsonable(span.attrs),
        )
        self._out.append(self.spans_path(span.run_id), record.model_dump_json(exclude_none=True))


def _jsonable(attrs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(attrs, default=str))
    return out


# --- reading traces ----------------------------------------------------------------


@dataclass
class SpanNode:
    """A span with its children, for printing the trace tree."""

    span: SpanRecord
    children: list[SpanNode] = field(default_factory=list)


def load_spans(path: str | Path) -> list[SpanRecord]:
    """Spans from a ``spans.jsonl``, skipping unreadable lines."""
    p = Path(path)
    if not p.exists():
        return []
    spans: list[SpanRecord] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            spans.append(SpanRecord.model_validate_json(line))
        except ValueError:
            continue  # a line cut off by a crash; keep the spans that survived
    return spans


def span_tree(spans: list[SpanRecord]) -> list[SpanNode]:
    """Roots in start order, each with children in start order. Spans whose parent
    is missing (e.g. a crash before the parent closed) become roots."""
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
        mark = "ok" if s.status == "ok" else f"ERROR {s.error}"
        suffix = f" {' '.join(extra)}" if extra else ""
        lines.append(f"{'  ' * depth}{s.kind}:{s.name} [{mark}] {s.duration_ms:.0f}ms{suffix}")
        for child in node.children:
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return lines
