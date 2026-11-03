from pathlib import Path

import pytest

from agent_team.core.events import InMemoryEventBus
from agent_team.core.models import AgentName, Event, EventName
from agent_team.core.tracing import (
    JsonlTracer,
    Tracer,
    current_span,
    events_path,
    load_spans,
    read_events,
    render_tree,
    span_tree,
)


async def test_jsonl_tracer_writes_one_file_filterable_by_agent_and_task(
    tmp_path: Path,
) -> None:
    bus = InMemoryEventBus()
    JsonlTracer(tmp_path).attach(bus)
    for agent, task in ((AgentName.PRODUCT, "t1"), (AgentName.ENGINEER, "t2"), (None, None)):
        await bus.emit(Event(name=EventName.TASK_STARTED, run_id="r1", agent=agent, task_id=task))
    await bus.emit(
        Event(name=EventName.TASK_FINISHED, run_id="r1", agent=AgentName.PRODUCT, task_id="t1")
    )
    logs = tmp_path / "r1" / "logs"
    assert [p.name for p in logs.iterdir()] == ["events.jsonl"]

    everything = read_events(tmp_path, "r1")
    assert [e.agent for e in everything] == [
        AgentName.PRODUCT,
        AgentName.ENGINEER,
        None,
        AgentName.PRODUCT,
    ]
    product = read_events(tmp_path, "r1", agent="product")
    assert [e.name for e in product] == [EventName.TASK_STARTED, EventName.TASK_FINISHED]
    assert [e.agent for e in read_events(tmp_path, "r1", task_id="t2")] == [AgentName.ENGINEER]
    assert read_events(tmp_path, "r1", agent=AgentName.QA) == []
    assert read_events(tmp_path, "missing") == []


async def test_read_events_skips_a_line_cut_off_by_a_crash(tmp_path: Path) -> None:
    bus = InMemoryEventBus()
    JsonlTracer(tmp_path).attach(bus)
    await bus.emit(Event(name=EventName.RUN_STARTED, run_id="r1"))
    with events_path(tmp_path, "r1").open("a") as fh:
        fh.write('{"name": "run_fin')
    assert [e.name for e in read_events(tmp_path, "r1")] == [EventName.RUN_STARTED]


def traced(tmp_path: Path) -> Tracer:
    """A tracer whose spans land in ``tmp_path``'s event logs."""
    bus = InMemoryEventBus()
    JsonlTracer(tmp_path).attach(bus)
    return Tracer(bus)


async def test_spans_nest_in_the_event_log_and_rebuild_as_a_tree(tmp_path: Path) -> None:
    tracer = traced(tmp_path)
    async with (
        tracer.span("r1", "run", "run") as run_span,
        tracer.span("r1", "prd", "task", task_id="t1", agent="product") as task,
    ):
        assert current_span() is task
        async with tracer.span("r1", "call", "llm") as call:
            call.add("input_tokens", 10)
            call.add("input_tokens", 5)
            call.set(cost_usd=0.01)
        async with tracer.span("r1", "check", "audit"):
            pass
    assert current_span() is None

    logs = tmp_path / "r1" / "logs"
    assert [p.name for p in logs.iterdir()] == ["events.jsonl"]
    events = read_events(tmp_path, "r1")
    assert [e.name for e in events].count(EventName.SPAN_STARTED) == 4
    assert [e.name for e in events].count(EventName.SPAN_FINISHED) == 4
    # Spans are tagged like any other line, so one agent's trail includes its spans.
    assert {e.data["name"] for e in read_events(tmp_path, "r1", agent="product")} == {
        "prd",
        "call",
        "check",
    }

    spans = load_spans(tmp_path, "r1")
    by_name = {s.name: s for s in spans}
    assert [s.name for s in spans] == ["run", "prd", "call", "check"]
    assert len({s.trace_id for s in spans}) == 1
    assert all(s.status == "ok" and s.end is not None for s in spans)
    assert by_name["prd"].parent_span_id == run_span.span_id
    assert by_name["call"].parent_span_id == task.span_id
    assert by_name["call"].task_id == "t1" and by_name["call"].agent is AgentName.PRODUCT
    assert by_name["call"].attrs == {"input_tokens": 15, "cost_usd": 0.01}
    assert by_name["check"].kind == "audit"

    roots = span_tree(spans)
    assert len(roots) == 1 and [c.span.name for c in roots[0].children] == ["prd"]
    assert [c.span.name for c in roots[0].children[0].children] == ["call", "check"]
    lines = render_tree(roots)
    assert lines[0].startswith("run:run [ok]")
    assert any("input_tokens=15" in line for line in lines)


async def test_error_span_keeps_type_message_and_traceback(tmp_path: Path) -> None:
    tracer = traced(tmp_path)
    with pytest.raises(RuntimeError):
        async with tracer.span("r1", "write_file", "tool"):
            raise RuntimeError("disk full")
    finished = [e for e in read_events(tmp_path, "r1") if e.name is EventName.SPAN_FINISHED]
    assert finished[0].level == "error"
    (tool,) = load_spans(tmp_path, "r1")
    assert tool.status == "error"
    assert tool.error_type == "RuntimeError" and tool.error == "disk full"
    assert "RuntimeError: disk full" in (tool.traceback or "")
    assert "test_tracing.py" in (tool.traceback or "")
    assert render_tree(span_tree([tool]))[0].startswith(
        "tool:write_file [ERROR RuntimeError: disk full]"
    )


async def test_span_left_open_by_a_crash_shows_as_open(tmp_path: Path) -> None:
    tracer = traced(tmp_path)
    async with tracer.span("r1", "run", "run"):
        pass
    # Simulate a process that died inside a span: the start is written, the finish never
    # is, and the last line is cut off mid-write.
    ctx = tracer.span("r1", "task", "task", task_id="t1")
    await ctx.__aenter__()
    try:
        with events_path(tmp_path, "r1").open("a") as fh:
            fh.write('{"name": "span_fin')

        spans = load_spans(tmp_path, "r1")
        assert [(s.name, s.status) for s in spans] == [("run", "ok"), ("task", "open")]
        task = spans[1]
        assert task.end is None and task.duration_ms is None and task.task_id == "t1"
        assert render_tree(span_tree(spans))[1] == "task:task [open]"
    finally:
        await ctx.__aexit__(None, None, None)


def test_load_spans_missing_run(tmp_path: Path) -> None:
    assert load_spans(tmp_path, "nope") == []
