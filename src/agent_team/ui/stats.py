"""Run stats for ``agent-team stats``: tokens, cost, time, retries and the first-pass
audit rate, per run and per agent.

Tokens and cost come from each run's ``llm_call_after`` events, retries from
``task_retried`` events, audits from ``audit_verdict`` events, and time from the
attempt spans (``span_started``/``span_finished`` of kind ``attempt``); the agent
behind each task comes from the task store. An agent's time is the time inside its
attempts, checks included for the auditor. A run's time is the time any attempt was
running, so hours spent waiting on the user, or stopped, don't count. A task passes
first time when the first verdict on it passed and no attempt before it failed its
own output (``OWN_OUTPUT_ERRORS``, such as JSON that never validated), which counts
like a failed check. Every task counts toward that rate
except fixes, which exist because a check failed on that work: a retry is another
attempt of the same task, so only its first verdict counts, and a revision for a
change request is new work with a first verdict of its own.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rich.table import Column, Table
from rich.text import Text

from agent_team.core.errors import OWN_OUTPUT_ERRORS
from agent_team.core.interfaces import TaskStore
from agent_team.core.models import AgentName, Event, EventName, Run, Task, TaskOrigin
from agent_team.core.tracing import SpanRecord, read_events, spans_from_events
from agent_team.ui.view import compact, short

#: Agents whose tasks are audited work, not checks or questions.
WORK_AGENTS = frozenset(a for a in AgentName if a not in (AgentName.AUDITOR, AgentName.USER))


@dataclass
class Stats:
    """Counts for one run or one agent."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    tasks: int = 0
    retries: int = 0
    audited: int = 0  # planned tasks with a verdict or an output that failed validation
    first_pass: int = 0  # ...whose first verdict passed with no failed output before it

    @property
    def first_pass_rate(self) -> float | None:
        """Share of audited planned tasks that passed their first audit; None if none."""
        return self.first_pass / self.audited if self.audited else None

    def add(self, other: Stats) -> None:
        """Add another set of counts to this one."""
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd
        self.seconds += other.seconds
        self.tasks += other.tasks
        self.retries += other.retries
        self.audited += other.audited
        self.first_pass += other.first_pass


@dataclass
class RunStats:
    """One run's totals and its counts per agent."""

    run: Run
    total: Stats = field(default_factory=Stats)
    agents: dict[str, Stats] = field(default_factory=lambda: defaultdict(Stats))


def compute(run: Run, tasks: list[Task], events: list[Event]) -> RunStats:
    """The stats of one run from its tasks and its events."""
    out = RunStats(run)
    by_id = {t.id: t for t in tasks}
    for task in tasks:
        if task.agent in WORK_AGENTS:
            out.agents[task.agent.value].tasks += 1
    attempts = attempt_spans(events)
    for span in attempts:
        if span.agent is not None and span.duration_ms is not None:
            out.agents[span.agent.value].seconds += span.duration_ms / 1000
    first_verdict: dict[str, bool] = {}
    for event in events:
        who = event.agent.value if event.agent else "orchestrator"
        if event.name is EventName.LLM_CALL_AFTER:
            stats = out.agents[who]
            stats.calls += 1
            stats.input_tokens += int(event.data.get("input_tokens") or 0)
            stats.output_tokens += int(event.data.get("output_tokens") or 0)
            stats.cost_usd += float(event.data.get("cost_usd") or 0.0)
        elif event.name is EventName.TASK_RETRIED and event.task_id in by_id:
            out.agents[by_id[event.task_id].agent.value].retries += 1
        elif (
            event.name is EventName.AUDIT_VERDICT
            and event.data.get("scope") == "task"
            and event.task_id in by_id
            and event.task_id not in first_verdict
        ):
            first_verdict[event.task_id] = bool(event.data.get("passed"))
        elif (
            event.name is EventName.ERROR
            and event.data.get("error_type") in OWN_OUTPUT_ERRORS
            and event.task_id in by_id
            and event.task_id not in first_verdict
        ):
            first_verdict[event.task_id] = False  # its output failed before any check
    for task_id, passed in first_verdict.items():
        task = by_id[task_id]
        if task.origin is TaskOrigin.FIX or task.agent not in WORK_AGENTS:
            continue
        stats = out.agents[task.agent.value]
        stats.audited += 1
        stats.first_pass += int(passed)
    for stats in out.agents.values():
        out.total.add(stats)
    # Attempts run in parallel, so the run's time is not the sum of theirs.
    out.total.seconds = busy_seconds(attempts)
    return out


def attempt_spans(events: list[Event]) -> list[SpanRecord]:
    """The finished attempt spans in the events: one per attempt of a task or a check."""
    return [s for s in spans_from_events(events) if s.kind == "attempt" and s.end is not None]


def busy_seconds(attempts: list[SpanRecord]) -> float:
    """How long at least one attempt was running: their time spans merged."""
    spans = sorted((s.start, s.end) for s in attempts if s.end is not None)
    total = 0.0
    end = None
    for start, finish in spans:
        if end is None or start > end:
            total += (finish - start).total_seconds()
            end = finish
        elif finish > end:
            total += (finish - end).total_seconds()
            end = finish
    return total


async def run_stats(run_id: str, tasks: TaskStore, runs_dir: str | Path) -> RunStats:
    """The stats of one run, read from the store and its events.jsonl."""
    run = await tasks.get_run(run_id)
    events = await asyncio.to_thread(read_events, runs_dir, run_id)
    return compute(run, await tasks.list_tasks(run_id), events)


def _rate(stats: Stats) -> str:
    """The first-pass rate as a percentage, or a dash."""
    rate = stats.first_pass_rate
    return "-" if rate is None else f"{rate:.0%}"


def _time(seconds: float) -> str:
    """Seconds as minutes and seconds."""
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}m{secs:02d}s"


#: Two-line headers keep the number columns as narrow as their numbers.
HEADER = ("", "calls", "tokens\nin", "tokens\nout", "cost", "time", "tasks", "retries", "1st\npass")
#: Printed under the table, so the time column can't be misread.
FOOTNOTE = (
    "time: inside each agent's attempts (checks for the auditor); for a run, while any "
    "attempt ran. 1st pass: share of tasks, fixes left out, whose first check passed "
    "with no invalid output before it."
)
#: The least room a title column is worth; with less, titles go under the table.
MIN_TITLE = 12


def row(name: str, stats: Stats) -> tuple[str, ...]:
    """One table row."""
    return (
        name,
        str(stats.calls),
        compact(stats.input_tokens),
        compact(stats.output_tokens),
        f"${stats.cost_usd:.4f}",
        _time(stats.seconds),
        str(stats.tasks),
        str(stats.retries),
        _rate(stats),
    )


def layout(rows: list[tuple[str, ...]], width: int, *, titled: bool = False) -> list[Table | Text]:
    """The table for a terminal ``width`` wide. Names and numbers always show whole: each
    column is as wide as its longest cell. With ``titled``, each row ends in a run's
    title, which gets the room left, cut to fit; when too little is left, the titles go
    on their own lines under the table instead."""
    titles = [r[-1] for r in rows] if titled else []
    cells = [r[:-1] for r in rows] if titled else rows
    widths = [
        max(len(part) for text in (head, *(r[i] for r in cells)) for part in text.split("\n"))
        for i, head in enumerate(HEADER)
    ]
    # One space before every column but the first, and before the title.
    room = width - sum(widths) - len(widths)
    inline = titled and room >= MIN_TITLE
    name, *numbers = HEADER
    columns = [
        Column(name, width=widths[0], no_wrap=True),
        *(
            Column(h, justify="right", width=w, no_wrap=True)
            for h, w in zip(numbers, widths[1:], strict=True)
        ),
    ]
    if inline:
        longest = max(len("title"), *(len(t) for t in titles))
        columns.append(Column("title", width=min(room, longest), no_wrap=True))
    table = Table(*columns, box=None, padding=(0, 0, 0, 1), pad_edge=False)
    for cell, title in zip(cells, titles if inline else [None] * len(cells), strict=True):
        table.add_row(*cell, *(() if title is None else (title,)))
    out: list[Table | Text] = [table]
    if titled and not inline:
        for cell, title in zip(cells, titles, strict=True):
            if title:
                out.append(Text(f"{cell[0]}  {short(title, max(width - len(cell[0]) - 2, 1))}"))
    return out


def heading(stats: RunStats) -> str:
    """The line above one run's table: its id, status and title."""
    return f"run {stats.run.id} ({stats.run.status.value}): {stats.run.title}"


def run_table(stats: RunStats) -> list[tuple[str, ...]]:
    """Rows for one run: its total, then one per agent. The run's id goes in the
    ``heading``, so the name column stays narrow."""
    rows = [row("total", stats.total)]
    rows += [row(f"  {name}", s) for name, s in sorted(stats.agents.items())]
    return rows


def overview_table(all_stats: list[RunStats]) -> list[tuple[str, ...]]:
    """Rows for many runs: one per run (its id), then the totals per agent across them.
    Each row ends with the run's title (empty on the totals)."""
    rows = [(*row(s.run.id, s.total), s.run.title) for s in all_stats]
    agents: dict[str, Stats] = defaultdict(Stats)
    total = Stats()
    for s in all_stats:
        total.add(s.total)
        for name, a in s.agents.items():
            agents[name].add(a)
    rows.append((*row("all runs", total), ""))
    rows += [(*row(f"  {name}", a), "") for name, a in sorted(agents.items())]
    return rows
