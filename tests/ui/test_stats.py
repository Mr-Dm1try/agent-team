"""Run stats from a fixture run: tokens and cost per agent, time inside attempt spans,
retries and the first-pass audit rate, which leaves fixes out."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from typing import Any

from rich.console import Console

from agent_team.core.models import (
    AgentName,
    Event,
    EventName,
    Phase,
    Run,
    Task,
    TaskOrigin,
)
from agent_team.ui.stats import Stats, compute, heading, layout, overview_table, row, run_table

T0 = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
RUN = Run(id="run_1", title="Books API", request="Build a books API", workspace="/w")


def _task(task_id: str, agent: AgentName, origin: TaskOrigin, start: int, end: int) -> Task:
    """A finished task that ran from ``start`` to ``end`` seconds after T0."""
    return Task(
        id=task_id,
        run_id=RUN.id,
        phase=Phase.IMPLEMENTATION,
        agent=agent,
        objective="o",
        output_kind="code_change",
        origin=origin,
        started_at=T0 + timedelta(seconds=start),
        finished_at=T0 + timedelta(seconds=end),
    )


def _event(name: EventName, at: int, **kw: Any) -> Event:
    """An event ``at`` seconds after T0."""
    return Event(name=name, run_id=RUN.id, ts=T0 + timedelta(seconds=at), **kw)


def _call(at: int, agent: AgentName | None, tokens: int, cost: float) -> Event:
    """One model call."""
    data = {"input_tokens": tokens, "output_tokens": tokens // 10, "cost_usd": cost}
    return _event(EventName.LLM_CALL_AFTER, at, agent=agent, data=data)


def _attempt(task_id: str, agent: AgentName, start: int, end: int) -> list[Event]:
    """The span_started and span_finished events of one attempt."""
    span = {"trace_id": "t", "span_id": f"{task_id}-{start}", "task_id": task_id, "agent": agent}
    data = {"name": "attempt", "kind": "attempt"}
    finish = {
        **data,
        "start": (T0 + timedelta(seconds=start)).isoformat(),
        "end": (T0 + timedelta(seconds=end)).isoformat(),
        "duration_ms": (end - start) * 1000.0,
        "status": "ok",
    }
    return [
        _event(EventName.SPAN_STARTED, start, data=data, **span),
        _event(EventName.SPAN_FINISHED, end, data=finish, **span),
    ]


def _verdict(at: int, task_id: str, passed: bool, scope: str = "task") -> Event:
    """An audit verdict on a task."""
    return _event(
        EventName.AUDIT_VERDICT, at, task_id=task_id, data={"scope": scope, "passed": passed}
    )


TASKS = [
    # The task records span far longer than the attempts: waits, audits and questions.
    _task("prd", AgentName.PRODUCT, TaskOrigin.PLAN, 0, 300),
    _task("code", AgentName.ENGINEER, TaskOrigin.PLAN, 40, 400),
    _task("fix", AgentName.ENGINEER, TaskOrigin.FIX, 110, 130),
    _task("change", AgentName.ENGINEER, TaskOrigin.REVISION, 200, 230),
    _task("check", AgentName.AUDITOR, TaskOrigin.CHECK, 100, 105),
    _task("ask", AgentName.USER, TaskOrigin.QUESTION, 130, 900),
]
EVENTS = [
    _event(EventName.RUN_STARTED, 0),
    *_attempt("prd", AgentName.PRODUCT, 0, 30),
    *_attempt("code", AgentName.ENGINEER, 40, 70),
    *_attempt("code", AgentName.ENGINEER, 75, 100),
    *_attempt("check", AgentName.AUDITOR, 100, 105),
    *_attempt("fix", AgentName.ENGINEER, 110, 130),
    *_attempt("change", AgentName.ENGINEER, 200, 210),
    _call(1, None, 500, 0.001),  # the orchestrator's planner
    _call(5, AgentName.PRODUCT, 1000, 0.01),
    _verdict(31, "prd", True),
    _call(45, AgentName.ENGINEER, 2000, 0.02),
    _event(EventName.TASK_RETRIED, 50, task_id="code"),
    _call(55, AgentName.ENGINEER, 2000, 0.02),
    _verdict(101, "code", False),
    _verdict(102, "code", True),  # a later verdict doesn't change the first pass
    _verdict(131, "fix", True),  # a fix is follow-up work, not counted
    _verdict(140, "fix", True, scope="run"),
    _verdict(211, "change", True),  # a change request's revision is new work
    _event(EventName.RUN_FINISHED, 150),
]


def test_stats_per_agent() -> None:
    stats = compute(RUN, TASKS, EVENTS)
    product, engineer = stats.agents["product"], stats.agents["engineer"]
    assert (product.calls, product.input_tokens, product.output_tokens) == (1, 1000, 100)
    assert (product.tasks, product.seconds, product.audited, product.first_pass) == (
        1,
        30.0,
        1,
        1,
    )
    # Time is inside attempts: 30 + 25 + 20 + 10 s, not the tasks' start to finish.
    assert (engineer.calls, engineer.tasks, engineer.retries, engineer.seconds) == (2, 3, 1, 85.0)
    # The planned task failed its first check, the change request's revision passed.
    assert (engineer.audited, engineer.first_pass) == (2, 1)
    assert engineer.first_pass_rate == 0.5
    assert stats.agents["orchestrator"].calls == 1
    auditor = stats.agents["auditor"]  # checks take time but aren't tasks of work
    assert (auditor.seconds, auditor.tasks, auditor.first_pass_rate) == (5.0, 0, None)


def test_run_time_counts_only_time_when_work_ran() -> None:
    total = compute(RUN, TASKS, EVENTS).total
    assert total.calls == 4 and total.input_tokens == 5500
    assert round(total.cost_usd, 4) == 0.051
    # 0-30, 40-70, 75-105 with the check right after the code, 110-130 and 200-210;
    # the user's wait and the gaps between attempts don't count.
    assert total.seconds == 120.0
    assert (total.audited, total.first_pass) == (3, 2)


def test_an_open_attempt_does_not_count() -> None:
    started = _attempt("prd", AgentName.PRODUCT, 0, 30)[0]
    assert compute(RUN, TASKS[:1], [started]).total.seconds == 0.0


def test_tables_show_each_run_and_agent() -> None:
    stats = compute(RUN, TASKS, EVENTS)
    assert heading(stats) == "run run_1 (active): Books API"
    rows = run_table(stats)
    assert rows[0] == ("total", "4", "5.5k", "550", "$0.0510", "2m00s", "4", "1", "67%")
    assert [r[0] for r in rows[1:]] == ["  auditor", "  engineer", "  orchestrator", "  product"]
    assert rows[3][-1] == "-"  # the orchestrator has no audited work
    overview = overview_table([stats, stats])
    assert overview[0][0] == "run_1" and overview[0][-1] == "Books API"
    assert overview[2][:3] == ("all runs", "8", "11k")


def test_a_run_with_no_events_has_zero_time() -> None:
    stats = compute(RUN, [], [])
    assert stats.total.seconds == 0.0 and stats.total.first_pass_rate is None


def test_an_output_that_failed_validation_is_not_a_first_pass() -> None:
    error = {"error_type": "OutputValidationError", "error": "files_written: none"}
    events = [
        _event(EventName.ERROR, 10, task_id="code", data=error),
        _event(EventName.ERROR, 20, task_id="prd", data={"error_type": "BudgetExceeded"}),
        _verdict(30, "code", True),
        _verdict(31, "prd", True),
    ]
    stats = compute(RUN, TASKS[:2], events)
    engineer, product = stats.agents["engineer"], stats.agents["product"]
    assert (engineer.audited, engineer.first_pass) == (1, 0)
    # A pause isn't the agent's output failing.
    assert (product.audited, product.first_pass) == (1, 1)


def test_token_counts_are_short() -> None:
    shown = [row("r", Stats(input_tokens=n)).__getitem__(2) for n in (950, 5_500, 812_400)]
    assert shown == ["950", "5.5k", "812k"]
    assert [row("r", Stats(input_tokens=n))[2] for n in (1_150_000, 20_396_212)] == ["1.1M", "20M"]


def test_the_all_runs_table_never_cuts_numbers_at_80_columns() -> None:
    big = Stats(
        calls=1234,
        input_tokens=260_396_212,
        output_tokens=1_512_334,
        cost_usd=12.6912,
        seconds=7385,
        tasks=40,
        retries=12,
        audited=3,
        first_pass=2,
    )
    title = "Build a library REST API with tests, then answer a long question about it"
    rows = [(*row(f"run_3f855756d562{i:04d}", big), title) for i in range(13)]
    rows += [(*row("all runs", big), ""), (*row("  orchestrator", big), "")]
    for width in (80, 120, 160):
        console = Console(width=width, record=True, file=io.StringIO(), color_system=None)
        for part in layout(rows, width, titled=True):
            console.print(part)
        lines = console.export_text().splitlines()
        assert all(len(line.rstrip()) <= width for line in lines)
        for r in rows:
            shown = next(line for line in lines if line.startswith(r[0] + " "))
            assert shown[len(r[0]) :].split()[: len(r) - 2] == list(r[1:-1]), shown
        if width == 80:  # no room for titles: each goes on its own line, cut to fit
            assert sum(title[:20] in line for line in lines) == 13
        else:
            assert all(title[:20] in line for line in lines if line.startswith("run_"))
