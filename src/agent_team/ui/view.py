"""The live view of a run, fed only by the event bus.

``RunView`` subscribes to every event and keeps what the user watches: each phase, each
task with its agent, status, attempt and current tool call, and a token and cost
ticker that keeps paid (Claude) and local tokens apart. It draws a Rich tree that
redraws in place while the team works and is cleared when it stops; the tree is then
printed once, only if it changed, so the scrollback holds one copy per change, not
one per reply. A reply printed while the team works pauses the tree instead, so the
screen keeps one tree, below the reply. With ``plain`` it prints one stable line per
event that matters, for CI and logs.

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
from agent_team.core.models import (
    PHASE_ORDER,
    STRONGER_MARK,
    DecisionKind,
    Event,
    EventName,
    HookOutcome,
    Task,
    attempt_text,
    checked_attempt,
)

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
#: How each provider's tokens are named in the ticker; Claude's are the paid ones.
PROVIDER_NAMES = {"anthropic": "Claude", "ollama": "local"}
#: A user task's short label, by the reason it was created for.
USER_LABELS = {
    "confirm_change": "confirm a change",
    "confirm_stop": "confirm a stop",
    "escalation": "decide how a stuck step goes on",
    "policy": "decide how a failed check goes on",
    "run_audit": "decide on the run's final check",
}
#: The fewest characters an objective tag is cut to, however narrow the screen.
MIN_TAG = 16
#: How far the tree indents a task row: two levels of guides.
TREE_INDENT = 8


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
    parent: str | None = None  # the task this one checks, asks about or revises
    reason: str | None = None  # why the task exists, such as "confirm_change"
    max_attempts: int = 0  # 0 when the events didn't say
    #: The first attempt on the stronger model, once the policy sent the task there.
    stronger_from: int | None = None

    def on_stronger(self, attempt: int | None = None) -> bool:
        """Whether ``attempt`` (the latest by default) ran on the stronger model."""
        attempt = self.attempt if attempt is None else attempt
        return self.stronger_from is not None and attempt >= self.stronger_from

    @property
    def label(self) -> str:
        """Who does what, in a few words. A check is labelled by what it checks in
        ``label``, which can see the other rows."""
        if self.agent == "user":
            if self.origin == "review":
                return f"you: review the {self.phase} phase"
            return f"you: {USER_LABELS.get(self.reason or '', 'answer a question')}"
        if self.agent == "auditor":
            return "check"
        return f"{self.agent}: {plain_kind(self.kind)}"


def compact(n: int) -> str:
    """A token count in a few characters: 950, 210k, 1.3M."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n // 1000}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


@dataclass
class Ticker:
    """Tokens and cost across the model calls seen so far, with the tokens kept per
    provider so paid and local ones read apart."""

    cost_usd: float = 0.0
    calls: int = 0
    #: Input and output tokens per provider, in the order providers were first seen.
    tokens: dict[str, list[int]] = field(default_factory=dict)

    def add(self, provider: str, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        """Count one model call."""
        self.calls += 1
        self.cost_usd += cost_usd
        counts = self.tokens.setdefault(provider, [0, 0])
        counts[0] += tokens_in
        counts[1] += tokens_out

    def text(self) -> str:
        """The ticker as one line, such as "4 model calls · Claude 210k in / 3.1k out ·
        local 1.3M in / 40k out · $0.3600"."""
        parts = [f"{self.calls} model calls"]
        for provider, (tokens_in, tokens_out) in self.tokens.items():
            name = PROVIDER_NAMES.get(provider, provider)
            parts.append(f"{name} {compact(tokens_in)} in / {compact(tokens_out)} out")
        parts.append(f"${self.cost_usd:.4f}")
        return " · ".join(parts)


@dataclass
class RunState:
    """What the events have said so far about one run."""

    run_id: str
    phases: dict[str, str] = field(default_factory=dict)  # phase -> running | done
    tasks: dict[str, TaskRow] = field(default_factory=dict)
    ticker: Ticker = field(default_factory=Ticker)
    finished: bool = False


def short(value: Any, limit: int = 60) -> str:
    """A value cut to one short line."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tag(state: RunState, row: TaskRow, room: int) -> str:
    """A short objective after a task's label when a sibling (another task of the same
    phase, agent and output that isn't superseded) shows the same label, such as two
    research questions running at once or work packages one after another; empty
    otherwise. The tag, brackets included, is cut to ``room`` characters, but never
    below ``MIN_TAG``."""
    siblings = [
        t
        for t in state.tasks.values()
        if t.id != row.id
        and (t.phase, t.agent, t.kind) == (row.phase, row.agent, row.kind)
        and t.status != "superseded"
    ]
    if not siblings or row.agent in ("user", "auditor") or not row.objective:
        return ""
    return f" [{short(row.objective, max(MIN_TAG, room - 3))}]"


def label(state: RunState, row: TaskRow, room: int) -> str:
    """A row's label. A check names the attempt and what it checks, cut to fit
    ``room`` characters, such as "check 2/3: the research [How do you run…]",
    "check 4 (stronger model): the research", or "check: the whole run"."""
    if row.agent != "auditor":
        return row.label
    subject = state.tasks.get(row.parent or "")
    if subject is not None:
        attempt = checked_attempt(row.reason)
        of = f"/{subject.max_attempts}" if attempt and subject.max_attempts else ""
        if attempt and subject.on_stronger(attempt):
            of = f" ({STRONGER_MARK})"
        head = f"check {attempt}{of}: " if attempt else "check: "
        head += plain_kind(subject.kind)
        return head + _tag(state, subject, room - len(head))
    if row.parent is None:
        return "check: the whole run"
    return f"check: {short(row.objective.removeprefix('Check '), max(MIN_TAG, room - 7))}"


def tagged(state: RunState, row: TaskRow, room: int) -> str:
    """A row's label and its sibling tag, together cut to fit ``room`` characters."""
    text = label(state, row, room)
    return text + _tag(state, row, room - len(text))


def waiting(state: RunState) -> list[TaskRow]:
    """The items of the run waiting on the user."""
    return [t for t in state.tasks.values() if t.agent == "user" and t.status == "waiting"]


def tool_summary(tool: str, args: Any) -> str:
    """A tool call as a few words: the tool and its main argument."""
    if not isinstance(args, dict):
        return tool
    for key in ("argv", "command", "path", "url", "query", "packages", "pattern"):
        value = args.get(key)
        if value:
            shown = " ".join(map(str, value)) if isinstance(value, list) else value
            return f"{tool} {short(shown, 50)}"
    return tool


class RunView:
    """Turns bus events into a live tree or plain lines. See the module docstring."""

    def __init__(self, console: Console, *, plain: bool = False) -> None:
        """Draw on ``console``; ``plain`` prints lines instead of a live tree."""
        self.console = console
        self.plain = plain
        self.runs: dict[str, RunState] = {}
        #: Model calls made outside any run, such as reading a message with no run open.
        self.session = Ticker()
        self._live: Live | None = None
        #: A dim line under the live tree, such as how to talk to the team meanwhile.
        self._hint: str | None = None
        self._broken = False
        #: What the last printed tree showed, so an unchanged tree isn't printed again.
        self._printed: tuple[object, ...] | None = None

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

    def start(self, *, hint: str | None = None) -> None:
        """Start redrawing in place, with ``hint`` under the tree, unless the view
        prints plain lines."""
        if self.plain or self._live is not None:
            return
        self._hint = hint
        try:
            self._live = Live(
                self._frame(), console=self.console, auto_refresh=False, transient=True
            )
            self._live.start()
        except Exception:
            self._fall_back("start the live view")

    def line_typed(self) -> None:
        """The user pressed Enter under the live tree, and the terminal moved the
        cursor down a line: move it back up, so the next redraw lands in place instead
        of leaving the tree's top line behind."""
        if self._live is None or not self.console.is_terminal:
            return
        try:
            self.console.file.write("\x1b[1A")
            self.console.file.flush()
        except Exception:
            self._fall_back("follow a typed line")

    def pause(self) -> None:
        """Stop redrawing, which clears the live frame, without printing the tree:
        ``start`` draws it again below whatever prints meanwhile, so a reply under the
        live view leaves no frozen copy of the tree above it."""
        live, self._live = self._live, None
        if live is None:
            return
        try:
            live.stop()
        except Exception:
            self._fall_back("pause the live view")

    def stop(self) -> None:
        """Stop redrawing, which clears the live frame, then print the tree once if it
        changed since it was last printed."""
        live, self._live = self._live, None
        if live is None:
            return
        try:
            live.stop()
            shown = self._shown()
            if self.runs and shown != self._printed:
                self.console.print(self.render())
                self._printed = shown
        except Exception:
            self._fall_back("stop the live view")

    def _shown(self) -> tuple[object, ...]:
        """What the tree shows, leaving out the ticker and tool calls, which change
        on every call."""
        return tuple(
            (
                state.run_id,
                tuple(sorted(state.phases.items())),
                tuple((t.id, t.status, t.attempt) for t in state.tasks.values()),
            )
            for state in self.runs.values()
        )

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
                parent=task.parent_task_id,
                reason=task.reason,
                max_attempts=task.max_attempts,
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
                self._live.update(self._frame(), refresh=True)
        except Exception:
            self._fall_back("draw")

    def _run(self, run_id: str) -> RunState:
        """The state of one run, created on its first event."""
        if run_id not in self.runs:
            self.runs[run_id] = RunState(run_id)
        return self.runs[run_id]

    def apply(self, event: Event) -> str | None:
        """Update the state from one event; return its plain line, if it gets one."""
        data = event.data
        name = event.name
        if event.run_id == "session":
            if name is EventName.LLM_CALL_AFTER:
                self.session.add(*_call_usage(data))
            return None
        state = self._run(event.run_id)
        if name is EventName.RUN_STARTED:
            return f"run {event.run_id} started"
        if name is EventName.PHASE_STARTED:
            state.phases[str(data.get("phase"))] = "running"
            return f"phase {data.get('phase')} started"
        if name is EventName.PHASE_FINISHED:
            state.phases[str(data.get("phase"))] = "done"
            return f"phase {data.get('phase')} finished"
        if name is EventName.LLM_CALL_AFTER:
            state.ticker.add(*_call_usage(data))
            return None
        if name is EventName.PROVIDER_DOWN:
            return (
                f"  {data.get('provider')} is unreachable; calls use the next provider and "
                f"try it again every {data.get('retry_after_s')}s"
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
                    parent=data.get("parent"),
                    reason=data.get("task_reason"),
                    max_attempts=int(data.get("max_attempts") or 0),
                )
                state.tasks[row.id] = row
            row.status = str(data.get("to", row.status))
            row.attempt = event.attempt or row.attempt
            # An answer or research can give a task another attempt.
            row.max_attempts = int(data.get("max_attempts") or row.max_attempts)
            if row.status != "running":
                row.tool = None
            if row.status not in PLAIN_STATES:
                return None
            attempt = f" ({self._attempt(row)})" if row.attempt > 1 else ""
            head, tail = f"  {row.phase}: ", f" {row.status}{attempt}"
            return head + tagged(state, row, self._room(head, tail)) + tail
        if row is None:
            return None
        if name is EventName.TASK_STARTED:
            row.attempt = event.attempt or row.attempt
            return None
        if name is EventName.DECISION and data.get("kind") == DecisionKind.STRONGER_MODEL:
            # Decided after an attempt failed: the next one runs on the stronger model.
            row.stronger_from = row.attempt + 1
            return None
        if name is EventName.TOOL_CALL_BEFORE:
            row.tool = tool_summary(str(data.get("tool", "")), data.get("args"))
            tail = f" → {row.tool}"
            room = self._room(f"    {row.agent}", tail)
            return f"    {row.agent}{_tag(state, row, room)}{tail}"
        if name is EventName.TOOL_CALL_AFTER:
            row.tool = None
            if data.get("blocked"):
                tail = f" ✗ {data.get('tool')} blocked by {data.get('rule_id')}"
                room = self._room(f"    {row.agent}", tail)
                return f"    {row.agent}{_tag(state, row, room)}{tail}"
            return None
        if name is EventName.AUDIT_VERDICT:
            findings = data.get("findings") or []
            blockers = [f for f in findings if f.get("severity") == "blocker"]
            if data.get("passed"):
                head, tail = "    audit passed: ", ""
            else:
                head, tail = "    audit failed: ", f" ({len(blockers)} blockers)"
            return head + tagged(state, row, self._room(head, tail)) + tail
        return None

    @staticmethod
    def _attempt(row: TaskRow) -> str:
        """A row's latest attempt, such as "attempt 2" or "attempt 4 (stronger model)"."""
        return attempt_text(row.attempt, stronger=row.on_stronger())

    def _room(self, head: str, tail: str) -> int:
        """The characters a line of the console's width leaves between ``head`` and
        ``tail``."""
        return self.console.width - len(head) - len(tail)

    # --- drawing ---------------------------------------------------------------------

    def _frame(self) -> Group:
        """What the live display draws: the tree, then the hint."""
        if self._hint is None:
            return self.render()
        return Group(self.render(), Text(self._hint, style="dim"))

    def render(self) -> Group:
        """The tree of every run the view has seen, with its ticker."""
        trees = [self._tree(state) for state in self.runs.values()]
        return Group(*trees) if trees else Group(Text("reading your message…", style="dim"))

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
                branch.add(self.row(state, row))
        asks = waiting(state)
        if asks:
            what = "1 question" if len(asks) == 1 else f"{len(asks)} questions"
            tree.add(Text(f"? waiting for you: {what}, shown below", style="bold yellow"))
        tree.add(Text(state.ticker.text(), style="dim"))
        return tree

    def row(self, state: RunState, row: TaskRow) -> Text:
        """One task line: mark, who and what, attempt and current tool. A check's label
        gets the room the line has left at the console's width."""
        attempt = f" · {self._attempt(row)}" if row.attempt > 1 else ""
        objective = ""
        if row.agent not in ("user", "auditor") and row.objective:
            objective = f" · {short(row.objective, 50)}"
        tool = f" · {row.tool}" if row.tool else ""
        mark = f"{STATUS_MARK.get(row.status, '·')} "
        room = self._room(" " * TREE_INDENT + mark, attempt + objective + tool)
        text = Text(mark, style=STATUS_STYLE.get(row.status, ""))
        text.append(label(state, row, room), style="dim" if row.status == "superseded" else "")
        if attempt:
            text.append(attempt, style="yellow")
        if objective:
            text.append(objective, style="dim")
        if tool:
            text.append(tool, style="cyan")
        return text


def _call_usage(data: dict[str, Any]) -> tuple[str, int, int, float]:
    """A model call's provider, tokens in and out, and cost, from llm_call_after."""
    return (
        str(data.get("provider") or "model"),
        int(data.get("input_tokens") or 0),
        int(data.get("output_tokens") or 0),
        float(data.get("cost_usd") or 0.0),
    )
