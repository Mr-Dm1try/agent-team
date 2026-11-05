"""Run stats for ``agent-team stats``: tokens, cost, time, retries and the first-pass
audit rate, per run and per agent.

Tokens and cost come from each run's ``llm_call_after`` events, retries from
``task_retried`` events, and audits from ``audit_verdict`` events; time and the agent
behind each task come from the task store. A run's time is the time any task was
working, so hours spent waiting on the user, or stopped, don't count. A task passes
first time when the first verdict on it passed. Only planned work counts toward that
rate: retries, revisions and fixes are follow-ups of a task that already had its
first verdict.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from agent_team.core.interfaces import TaskStore
from agent_team.core.models import AgentName, Event, EventName, Run, Task, TaskOrigin
from agent_team.core.tracing import read_events

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
    audited: int = 0  # planned tasks with at least one verdict
    first_pass: int = 0  # ...whose first verdict passed

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
        if task.agent not in WORK_AGENTS:
            continue
        agent = out.agents[task.agent.value]
        agent.tasks += 1
        if task.started_at and task.finished_at:
            agent.seconds += (task.finished_at - task.started_at).total_seconds()
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
    for task_id, passed in first_verdict.items():
        task = by_id[task_id]
        if task.origin is not TaskOrigin.PLAN or task.agent not in WORK_AGENTS:
            continue
        stats = out.agents[task.agent.value]
        stats.audited += 1
        stats.first_pass += int(passed)
    for stats in out.agents.values():
        out.total.add(stats)
    # Tasks run in parallel, so the run's time is not the sum of theirs.
    out.total.seconds = busy_seconds([t for t in tasks if t.agent is not AgentName.USER])
    return out


def busy_seconds(tasks: list[Task]) -> float:
    """How long at least one of the tasks was working: their time spans merged."""
    spans = sorted((t.started_at, t.finished_at) for t in tasks if t.started_at and t.finished_at)
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


HEADER = ("", "calls", "tokens in", "tokens out", "cost", "time", "tasks", "retries", "first pass")


def row(name: str, stats: Stats) -> tuple[str, ...]:
    """One table row."""
    return (
        name,
        str(stats.calls),
        f"{stats.input_tokens:,}",
        f"{stats.output_tokens:,}",
        f"${stats.cost_usd:.4f}",
        _time(stats.seconds),
        str(stats.tasks),
        str(stats.retries),
        _rate(stats),
    )


def run_table(stats: RunStats) -> list[tuple[str, ...]]:
    """Rows for one run: its total, then one per agent."""
    rows = [row(f"run {stats.run.id} ({stats.run.status.value})", stats.total)]
    rows += [row(f"  {name}", s) for name, s in sorted(stats.agents.items())]
    return rows


def overview_table(all_stats: list[RunStats]) -> list[tuple[str, ...]]:
    """Rows for many runs: one per run, then the totals per agent across them."""
    rows = [row(f"{s.run.id} {s.run.title[:30]!r}", s.total) for s in all_stats]
    agents: dict[str, Stats] = defaultdict(Stats)
    total = Stats()
    for s in all_stats:
        total.add(s.total)
        for name, a in s.agents.items():
            agents[name].add(a)
    rows.append(row("all runs", total))
    rows += [row(f"  {name}", a) for name, a in sorted(agents.items())]
    return rows
