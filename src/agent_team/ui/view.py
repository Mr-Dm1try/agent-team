"""The live view of a run, fed only by the event bus.

``RunView`` subscribes to every event and keeps what the user watches: each phase, each
task with its agent, status, attempt and current tool call, and a token and cost
ticker. It draws a Rich tree that redraws in place, or with ``plain`` it prints one
stable line per event that matters, for CI and logs.

The view must never break a run. Every event is handled inside a guard: if drawing
fails, the error is logged once and the view falls back to plain lines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text
from rich.tree import Tree

from agent_team.core.artifacts import plain_kind
from agent_team.core.interfaces import EventBus
from agent_team.core.models import PHASE_ORDER, Event, EventName, HookOutcome, Task

log = logging.getLogger(__name__)

STATUS_MARK = {
    "pending": "·",
    "ready": "·",
    "running": "▶",
    "auditing": "◆",
    "done": "✓",
    "retry": "↻",
    "escalated": "!",
    "waiting": "?",
    "superseded": "x",
}
STATUS_STYLE = {
    "running": "bold cyan",
    "auditing": "magenta",
    "done": "green",
    "retry": "yellow",
    "escalated": "bold red",
    "waiting": "bold yellow",
    "superseded": "dim",
}
#: Task changes worth a plain line: starting, finishing, retrying and waiting.
PLAIN_STATES = frozenset({"running", "done", "retry", "escalated", "waiting", "superseded"})


@dataclass
class TaskRow:
    """One task as the view draws it."""

    id: str
    phase: str
    agent: str
    kind: str
    objective: str
    status: str = "pending"
    attempt: int = 0
    tool: str | None = None
    origin: str = "plan"

    @property
    def label(self) -> str:
        """Who does what, in a few words."""
        if self.agent == "user":
            return f"you: {self.objective}"
        if self.agent == "auditor":
            return "auditor: check"
        return f"{self.agent}: {plain_kind(self.kind)}"


@dataclass
class Ticker:
    """Tokens and cost across the model calls seen so far."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def text(self) -> str:
        """The ticker as one line."""
        return (
            f"{self.calls} model calls · {self.input_tokens:,} tokens in · "
            f"{self.output_tokens:,} out · ${self.cost_usd:.4f}"
        )


@dataclass
class RunState:
    """What the events have said so far about one run."""

    run_id: str
    phases: dict[str, str] = field(default_factory=dict)  # phase -> running | done
    tasks: dict[str, TaskRow] = field(default_factory=dict)
    ticker: Ticker = field(default_factory=Ticker)
    finished: bool = False


def _short(value: Any, limit: int = 60) -> str:
    """A value cut to one short line."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tag(state: RunState, row: TaskRow) -> str:
    """A short objective after a task's plain lines when a sibling (another live task of
    the same phase, agent and output) would print the same label, such as two research
    questions running at once; empty otherwise."""
    siblings = [
        t
        for t in state.tasks.values()
        if t.id != row.id
        and (t.phase, t.agent, t.kind) == (row.phase, row.agent, row.kind)
        and t.status != "superseded"
    ]
    if not siblings or row.agent in ("user", "auditor") or not row.objective:
        return ""
    return f" [{_short(row.objective, 40)}]"


def tool_summary(tool: str, args: Any) -> str:
    """A tool call as a few words: the tool and its main argument."""
    if not isinstance(args, dict):
        return tool
    for key in ("argv", "command", "path", "url", "query", "packages", "pattern"):
        value = args.get(key)
        if value:
            shown = " ".join(map(str, value)) if isinstance(value, list) else value
            return f"{tool} {_short(shown, 50)}"
    return tool


class RunView:
    """Turns bus events into a live tree or plain lines. See the module docstring."""

    def __init__(self, console: Console, *, plain: bool = False) -> None:
        """Draw on ``console``; ``plain`` prints lines instead of a live tree."""
        self.console = console
        self.plain = plain
        self.runs: dict[str, RunState] = {}
        self._live: Live | None = None
        self._broken = False

    def attach(self, bus: EventBus) -> None:
        """Receive every event from the bus. Tool calls only pass the before hooks, so
        the view also watches those, after the pre-check, and always allows."""
        bus.on_after(None, self.on_event)
        bus.on_before(EventName.TOOL_CALL_BEFORE, self._watch_tool)

    async def _watch_tool(self, event: Event) -> HookOutcome:
        """Show a tool call the pre-check let through; never blocks it."""
        await self.on_event(event)
        return HookOutcome(allow=True)

    # --- the live display ------------------------------------------------------------

    def start(self) -> None:
        """Start redrawing in place, unless the view prints plain lines."""
        if self.plain or self._live is not None:
            return
        try:
            self._live = Live(
                self.render(), console=self.console, auto_refresh=False, transient=False
            )
            self._live.start()
        except Exception:
            self._fall_back("start the live view")

    def stop(self) -> None:
        """Stop redrawing and leave the last frame on screen."""
        live, self._live = self._live, None
        if live is not None:
            try:
                live.stop()
            except Exception:
                self._fall_back("stop the live view")

    def _fall_back(self, what: str) -> None:
        """Log the first failure and switch to plain lines for good."""
        if not self._broken:
            log.exception("the live view failed to %s; printing plain lines instead", what)
            self._broken = True
        self.plain = True
        live, self._live = self._live, None
        if live is not None:
            try:
                live.stop()
            except Exception:
                log.debug("the live view didn't stop cleanly", exc_info=True)

    def load(self, run_id: str, tasks: list[Task]) -> None:
        """Start from the tasks a run already has, so a resumed run shows its earlier
        work. A phase whose tasks are all finished shows as done, and one with work
        under way as running."""
        state = self._run(run_id)
        for task in tasks:
            state.tasks[task.id] = TaskRow(
                id=task.id,
                phase=task.phase.value,
                agent=task.agent.value,
                kind=task.output_kind,
                objective=task.objective,
                status=task.status.value,
                attempt=task.attempt,
                origin=task.origin.value,
            )
        for phase in {t.phase for t in state.tasks.values()}:
            rows = [t for t in state.tasks.values() if t.phase == phase]
            if all(t.status in ("done", "superseded") for t in rows):
                state.phases[phase] = "done"
            elif any(t.status not in ("pending", "ready") for t in rows):
                state.phases[phase] = "running"

    # --- events ----------------------------------------------------------------------

    async def on_event(self, event: Event) -> None:
        """Update the state from one event and redraw; never raises."""
        try:
            line = self.apply(event)
        except Exception:
            self._fall_back("read an event")
            return
        try:
            if self.plain:
                if line:
                    self.console.print(line, highlight=False, markup=False)
            elif self._live is not None:
                self._live.update(self.render(), refresh=True)
        except Exception:
            self._fall_back("draw")

    def _run(self, run_id: str) -> RunState:
        """The state of one run, created on its first event."""
        if run_id not in self.runs:
            self.runs[run_id] = RunState(run_id)
        return self.runs[run_id]

    def apply(self, event: Event) -> str | None:
        """Update the state from one event; return its plain line, if it gets one."""
        if event.run_id == "session":
            return None
        state = self._run(event.run_id)
        data = event.data
        name = event.name
        if name is EventName.RUN_STARTED:
            return f"run {event.run_id} started"
        if name is EventName.PHASE_STARTED:
            state.phases[str(data.get("phase"))] = "running"
            return f"phase {data.get('phase')} started"
        if name is EventName.PHASE_FINISHED:
            state.phases[str(data.get("phase"))] = "done"
            return f"phase {data.get('phase')} finished"
        if name is EventName.LLM_CALL_AFTER:
            ticker = state.ticker
            ticker.calls += 1
            ticker.input_tokens += int(data.get("input_tokens") or 0)
            ticker.output_tokens += int(data.get("output_tokens") or 0)
            ticker.cost_usd += float(data.get("cost_usd") or 0.0)
            return None
        if name is EventName.PROVIDER_DOWN:
            return (
                f"  {data.get('provider')} is unreachable; calls go to the fallback for "
                f"{data.get('retry_after_s')}s"
            )
        if name is EventName.PROVIDER_UP:
            return f"  {data.get('provider')} is reachable again"
        if name is EventName.RUN_FINISHED:
            state.finished = True
            return f"run {event.run_id} finished · {state.ticker.text()}"
        if event.task_id is None:
            return None
        return self._apply_task(state, event)

    def _apply_task(self, state: RunState, event: Event) -> str | None:
        """Update one task from an event about it; return its plain line, if any."""
        data = event.data
        name = event.name
        row = state.tasks.get(event.task_id or "")
        if name is EventName.TASK_STATE_CHANGED:
            if row is None:
                row = TaskRow(
                    id=event.task_id or "",
                    phase=str(data.get("phase", "")),
                    agent=event.agent.value if event.agent else "?",
                    kind=str(data.get("kind", "")),
                    objective=str(data.get("objective", "")),
                    origin=str(data.get("origin", "plan")),
                )
                state.tasks[row.id] = row
            row.status = str(data.get("to", row.status))
            row.attempt = event.attempt or row.attempt
            if row.status != "running":
                row.tool = None
            if row.status not in PLAIN_STATES:
                return None
            attempt = f" (attempt {row.attempt})" if row.attempt > 1 else ""
            return f"  {row.phase}: {row.label}{_tag(state, row)} {row.status}{attempt}"
        if row is None:
            return None
        if name is EventName.TASK_STARTED:
            row.attempt = event.attempt or row.attempt
            return None
        if name is EventName.TOOL_CALL_BEFORE:
            row.tool = tool_summary(str(data.get("tool", "")), data.get("args"))
            return f"    {row.agent}{_tag(state, row)} → {row.tool}"
        if name is EventName.TOOL_CALL_AFTER:
            row.tool = None
            if data.get("blocked"):
                return (
                    f"    {row.agent}{_tag(state, row)} ✗ {data.get('tool')} blocked by "
                    f"{data.get('rule_id')}"
                )
            return None
        if name is EventName.AUDIT_VERDICT:
            findings = data.get("findings") or []
            blockers = [f for f in findings if f.get("severity") == "blocker"]
            if data.get("passed"):
                return f"    audit passed: {row.label}{_tag(state, row)}"
            return f"    audit failed: {row.label}{_tag(state, row)} ({len(blockers)} blockers)"
        return None

    # --- drawing ---------------------------------------------------------------------

    def render(self) -> Group:
        """The tree of every run the view has seen, with its ticker."""
        trees = [self._tree(state) for state in self.runs.values()]
        return Group(*trees) if trees else Group(Text("starting…", style="dim"))

    def _tree(self, state: RunState) -> Tree:
        """One run as a tree: phases, then their tasks, then the ticker."""
        tree = Tree(Text(f"run {state.run_id}", style="bold"))
        phases = [p.value for p in PHASE_ORDER]
        extra = sorted({t.phase for t in state.tasks.values()} - set(phases))
        for phase in [*phases, *extra]:
            rows = [t for t in state.tasks.values() if t.phase == phase]
            status = state.phases.get(phase)
            if status is None and not rows:
                continue
            mark = "✓" if status == "done" else "▶" if status == "running" else "·"
            branch = tree.add(Text(f"{mark} {phase}", style="bold" if status else "dim"))
            for row in rows:
                branch.add(self._row(row))
        tree.add(Text(state.ticker.text(), style="dim"))
        return tree

    @staticmethod
    def _row(row: TaskRow) -> Text:
        """One task line: mark, who and what, attempt and current tool."""
        text = Text(f"{STATUS_MARK.get(row.status, '·')} ", style=STATUS_STYLE.get(row.status, ""))
        text.append(row.label, style="dim" if row.status == "superseded" else "")
        if row.attempt > 1:
            text.append(f" · attempt {row.attempt}", style="yellow")
        if row.agent not in ("user", "auditor") and row.objective:
            text.append(f" · {_short(row.objective, 50)}", style="dim")
        if row.tool:
            text.append(f" · {row.tool}", style="cyan")
        return text
