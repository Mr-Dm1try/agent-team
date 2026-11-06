"""The live view from a scripted event stream: the tree, the plain lines, and the
fallback to plain lines when drawing fails."""

from __future__ import annotations

import io
from typing import Any

import pytest
from rich.console import Console

from agent_team.core.models import AgentName, Event, EventName, Phase, Task, TaskStatus
from agent_team.ui.view import RunView, tool_summary

RUN = "run_1"


def _event(name: EventName, task: str | None = None, **data: Any) -> Event:
    """An event of the scripted run."""
    agent = data.pop("agent", None)
    attempt = data.pop("attempt", None)
    return Event(name=name, run_id=RUN, task_id=task, agent=agent, attempt=attempt, data=data)


def _state(task: str, agent: AgentName, to: str, **data: Any) -> Event:
    """A task moving to ``to``."""
    return _event(EventName.TASK_STATE_CHANGED, task, agent=agent, to=to, **data)


SPEC_TASK = {"phase": "spec", "kind": "spec", "objective": "Design the books API"}
CODE_TASK = {"phase": "implementation", "kind": "code_change", "objective": "Write app.py"}

STREAM = [
    _event(EventName.RUN_STARTED),
    _event(EventName.PHASE_STARTED, phase="spec"),
    _state("t1", AgentName.ARCHITECT, "running", attempt=1, **SPEC_TASK),
    _event(
        EventName.LLM_CALL_AFTER,
        provider="anthropic",
        input_tokens=1200,
        output_tokens=300,
        cost_usd=0.012,
    ),
    _state("t1", AgentName.ARCHITECT, "auditing", **SPEC_TASK),
    _event(EventName.AUDIT_VERDICT, "t1", scope="task", passed=True, findings=[]),
    _state("t1", AgentName.ARCHITECT, "done", **SPEC_TASK),
    _event(EventName.PHASE_FINISHED, phase="spec"),
    _event(EventName.PHASE_STARTED, phase="implementation"),
    _state("t2", AgentName.ENGINEER, "running", attempt=1, **CODE_TASK),
    _event(EventName.TOOL_CALL_BEFORE, "t2", tool="run_command", args={"argv": ["pytest", "-q"]}),
    _event(EventName.TOOL_CALL_AFTER, "t2", tool="run_command"),
    _event(
        EventName.AUDIT_VERDICT,
        "t2",
        scope="task",
        passed=False,
        findings=[{"severity": "blocker"}, {"severity": "minor"}],
    ),
    _state("t2", AgentName.ENGINEER, "retry", **CODE_TASK),
    _state("t2", AgentName.ENGINEER, "running", attempt=2, **CODE_TASK),
    _event(EventName.TOOL_CALL_BEFORE, "t2", tool="write_file", args={"path": "app.py"}),
]

PLAIN = [
    "run run_1 started",
    "phase spec started",
    "  spec: architect: the design running",
    "    audit passed: architect: the design",
    "  spec: architect: the design done",
    "phase spec finished",
    "phase implementation started",
    "  implementation: engineer: the code running",
    "    engineer → run_command pytest -q",
    "    audit failed: engineer: the code (1 blockers)",
    "  implementation: engineer: the code retry",
    "  implementation: engineer: the code running (attempt 2)",
    "    engineer → write_file app.py",
]


def _console() -> Console:
    """A wide console that records what it prints."""
    return Console(file=io.StringIO(), record=True, width=140, color_system=None)


async def _feed(view: RunView, events: list[Event]) -> None:
    """Pass each event to the view."""
    for event in events:
        await view.on_event(event)


async def test_the_tree_shows_phases_tasks_tools_and_the_ticker() -> None:
    console = _console()
    view = RunView(console)
    await _feed(view, STREAM)
    console.print(view.render())
    text = console.export_text()
    assert "run run_1" in text
    assert text.index("✓ spec") < text.index("▶ implementation")
    assert "✓ architect: the design · Design the books API" in text
    assert "▶ engineer: the code · attempt 2 · Write app.py · write_file app.py" in text
    assert "1 model calls · Claude 1.2k in / 300 out · $0.0120" in text


async def test_plain_lines_are_stable() -> None:
    console = _console()
    view = RunView(console, plain=True)
    await _feed(view, STREAM)
    assert console.export_text().splitlines() == PLAIN


async def test_a_blocked_tool_call_and_the_end_of_the_run_get_a_line() -> None:
    console = _console()
    view = RunView(console, plain=True)
    blocked = _event(EventName.TOOL_CALL_AFTER, "t2", tool="run_command", blocked=True)
    blocked.data["rule_id"] = "R6"
    await _feed(view, [*STREAM, blocked, _event(EventName.RUN_FINISHED)])
    lines = console.export_text().splitlines()
    assert lines[-2] == "    engineer ✗ run_command blocked by R6"
    assert lines[-1].startswith("run run_1 finished · 1 model calls")


async def test_a_view_that_fails_to_draw_falls_back_to_plain_lines(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    console = _console()
    view = RunView(console)
    view.start()

    def broken() -> None:
        """A renderer with a bug."""
        raise ValueError("bad frame")

    monkeypatch.setattr(view, "render", broken)
    await _feed(view, STREAM)
    view.stop()
    assert view.plain
    assert caplog.text.count("printing plain lines instead") == 1
    # The event that broke the drawing is lost; every later one gets its line.
    lines = console.export_text().splitlines()
    assert lines[-len(PLAIN) + 1].endswith(PLAIN[1])
    assert lines[-len(PLAIN) + 2 :] == PLAIN[2:]


async def test_an_event_the_view_cannot_read_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    console = _console()
    view = RunView(console)
    bad = _event(EventName.LLM_CALL_AFTER, input_tokens="lots")
    await view.on_event(bad)
    await view.on_event(_event(EventName.RUN_STARTED))
    assert view.plain and "run run_1 started" in console.export_text()
    assert "failed to read an event" in caplog.text


async def test_watching_a_tool_call_always_allows_it(monkeypatch: pytest.MonkeyPatch) -> None:
    view = RunView(_console())

    def broken(event: Event) -> str | None:
        """A state update with a bug."""
        raise KeyError("oops")

    monkeypatch.setattr(view, "apply", broken)
    call = _event(EventName.TOOL_CALL_BEFORE, "t2", tool="read_file", args={"path": "a.py"})
    outcome = await view._watch_tool(call)
    assert outcome.allow


def test_tool_summary_names_the_main_argument() -> None:
    assert tool_summary("run_command", {"argv": ["pytest", "-q"]}) == "run_command pytest -q"
    assert tool_summary("web_fetch", {"url": "https://x.org"}) == "web_fetch https://x.org"
    assert tool_summary("list_dir", {}) == "list_dir"
    assert tool_summary("odd", "not a dict") == "odd"


def _task(phase: Phase, agent: AgentName, kind: str, status: TaskStatus) -> Task:
    """A stored task of the run."""
    return Task(
        run_id=RUN, phase=phase, agent=agent, objective="o", output_kind=kind, status=status
    )


async def test_a_resumed_run_shows_the_work_done_before() -> None:
    console = _console()
    view = RunView(console)
    view.load(
        RUN,
        [
            _task(Phase.PRD, AgentName.PRODUCT, "prd", TaskStatus.DONE),
            _task(Phase.SPEC, AgentName.ARCHITECT, "spec", TaskStatus.SUPERSEDED),
            _task(Phase.SPEC, AgentName.ARCHITECT, "spec", TaskStatus.RUNNING),
            _task(Phase.IMPLEMENTATION, AgentName.ENGINEER, "code_change", TaskStatus.PENDING),
        ],
    )
    console.print(view.render())
    text = console.export_text()
    assert "✓ prd" in text and "✓ product: the requirements" in text
    assert "▶ spec" in text and "x architect: the design" in text
    assert "· implementation" in text and "· engineer: the code" in text


async def test_plain_lines_of_sibling_tasks_say_which_task_they_are() -> None:
    console = _console()
    view = RunView(console, plain=True)
    isbn = {"phase": "planning", "kind": "research_brief", "objective": "How are ISBNs checked?"}
    tests = {"phase": "planning", "kind": "research_brief", "objective": "How to test FastAPI?"}
    await _feed(
        view,
        [
            _state("r1", AgentName.RESEARCHER, "ready", **isbn),
            _state("r2", AgentName.RESEARCHER, "ready", **tests),
            _state("r1", AgentName.RESEARCHER, "running", attempt=1, **isbn),
            _event(EventName.TOOL_CALL_BEFORE, "r2", tool="web_search", args={"query": "q"}),
            _state("r1", AgentName.RESEARCHER, "done", **isbn),
        ],
    )
    lines = console.export_text().splitlines()
    assert lines == [
        "  planning: researcher: the research [How are ISBNs checked?] running",
        "    researcher [How to test FastAPI?] → web_search q",
        "  planning: researcher: the research [How are ISBNs checked?] done",
    ]


async def test_a_provider_going_down_and_back_gets_a_line() -> None:
    console = _console()
    view = RunView(console, plain=True)
    down = _event(EventName.PROVIDER_DOWN, "r1", provider="ollama", retry_after_s=60.0)
    await _feed(view, [down, _event(EventName.PROVIDER_UP, "r1", provider="ollama")])
    assert console.export_text().splitlines() == [
        "  ollama is unreachable; calls use the next provider and try it again every 60.0s",
        "  ollama is reachable again",
    ]


QUESTION = "How do you run a FastMCP server over stdio?"
BRIEF_TASK = {"phase": "planning", "kind": "research_brief", "objective": QUESTION}
OTHER_BRIEF = {"phase": "planning", "kind": "research_brief", "objective": "Which mcp version?"}
STUCK = (
    "The researcher is stuck on: How do you run a FastMCP server over stdio? (no valid "
    "output within 80 steps). Reply with guidance for another try, or 'stop'."
)


def _labelled_run() -> list[Event]:
    """Two briefs, a check of one of them, the run's check and a stuck step's question."""
    return [
        _event(EventName.RUN_STARTED),
        _event(EventName.PHASE_STARTED, phase="planning"),
        _state("b1", AgentName.RESEARCHER, "done", **BRIEF_TASK),
        _state("b2", AgentName.RESEARCHER, "running", **OTHER_BRIEF),
        _state(
            "c1",
            AgentName.AUDITOR,
            "done",
            phase="planning",
            kind="verdict",
            objective="Check research brief from the researcher",
            parent="b1",
        ),
        _state(
            "c2",
            AgentName.AUDITOR,
            "running",
            phase="planning",
            kind="verdict",
            objective="Check the whole run",
            parent=None,
        ),
        _state(
            "u1",
            AgentName.USER,
            "waiting",
            phase="planning",
            kind="user_answer",
            objective=STUCK,
            origin="question",
            parent="b2",
            task_reason="escalation",
        ),
    ]


async def test_checks_say_what_they_check_and_questions_are_short() -> None:
    console = _console()
    view = RunView(console)
    await _feed(view, _labelled_run())
    console.print(view.render())
    text = console.export_text()
    assert "✓ check: the research [How do you run a FastMCP server over st…]" in text
    assert "▶ check: the whole run" in text
    assert "auditor: check" not in text
    assert "? you: decide how a stuck step goes on" in text
    assert STUCK not in text  # the full question prints under the tree, with the reply
    assert "? waiting for you: 1 question, shown below" in text


async def test_plain_lines_label_checks_and_questions_too() -> None:
    console = _console()
    view = RunView(console, plain=True)
    await _feed(view, _labelled_run())
    lines = console.export_text().splitlines()
    assert (
        "  planning: check: the research [How do you run a FastMCP server over st…] done" in lines
    )
    assert "  planning: check: the whole run running" in lines
    assert "  planning: you: decide how a stuck step goes on waiting" in lines


async def test_the_ticker_keeps_paid_and_local_tokens_apart() -> None:
    console = _console()
    view = RunView(console)
    calls = [
        ("anthropic", 210_000, 2_000, 0.30),
        ("ollama", 1_300_000, 40_000, 0.0),
        ("anthropic", 400, 100, 0.06),
    ]
    await _feed(
        view,
        [_event(EventName.RUN_STARTED)]
        + [
            _event(
                EventName.LLM_CALL_AFTER,
                provider=p,
                input_tokens=i,
                output_tokens=o,
                cost_usd=c,
            )
            for p, i, o, c in calls
        ],
    )
    assert view.runs[RUN].ticker.text() == (
        "3 model calls · Claude 210k in / 2.1k out · local 1.3M in / 40k out · $0.3600"
    )


async def test_model_calls_outside_a_run_are_counted_apart() -> None:
    view = RunView(_console())
    outside = Event(
        name=EventName.LLM_CALL_AFTER,
        run_id="session",
        data={"provider": "anthropic", "input_tokens": 900, "output_tokens": 50, "cost_usd": 0.01},
    )
    await view.on_event(outside)
    assert view.runs == {}
    assert view.session.text() == "1 model calls · Claude 900 in / 50 out · $0.0100"


async def test_the_tree_prints_once_per_change_not_once_per_reply() -> None:
    console = _console()
    view = RunView(console)
    for _ in range(2):  # two replies that change nothing on the tree
        view.start()
        await _feed(view, STREAM)
        view.stop()
    view.start()
    await _feed(view, [_state("t2", AgentName.ENGINEER, "done", **CODE_TASK)])
    view.stop()
    text = console.export_text()
    assert text.count("run run_1\n") == 2
    assert "reading your message" not in text


def test_the_view_before_any_run_says_what_it_is_doing() -> None:
    console = _console()
    console.print(RunView(console).render())
    assert console.export_text().strip() == "reading your message…"
