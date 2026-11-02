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


async def test_spans_nest_and_record_errors(tmp_path: Path) -> None:
    tracer = Tracer(tmp_path)
    async with (
        tracer.span("r1", "run", "run") as run_span,
        tracer.span("r1", "prd", "task", task_id="t1", agent="product") as task,
    ):
        assert current_span() is task
        async with tracer.span("r1", "call", "llm") as call:
            call.add("input_tokens", 10)
            call.add("input_tokens", 5)
            call.set(cost_usd=0.01)
        with pytest.raises(RuntimeError):
            async with tracer.span("r1", "write_file", "tool"):
                raise RuntimeError("disk full")
    assert current_span() is None

    spans = load_spans(tracer.spans_path("r1"))
    by_name = {s.name: s for s in spans}
    assert len({s.trace_id for s in spans}) == 1
    assert by_name["prd"].parent_span_id == run_span.span_id
    assert by_name["call"].parent_span_id == task.span_id
    assert by_name["call"].task_id == "t1" and by_name["call"].agent == "product"
    assert by_name["call"].attrs == {"input_tokens": 15, "cost_usd": 0.01}
    tool = by_name["write_file"]
    assert tool.status == "error" and "disk full" in (tool.error or "")
    assert "RuntimeError" in (tool.traceback or "")

    roots = span_tree(spans)
    assert len(roots) == 1 and [c.span.name for c in roots[0].children] == ["prd"]
    lines = render_tree(roots)
    assert lines[0].startswith("run:run [ok]")
    assert any("tool:write_file [ERROR" in line for line in lines)
    assert any("input_tokens=15" in line for line in lines)


def test_load_spans_missing_file(tmp_path: Path) -> None:
    assert load_spans(tmp_path / "nope.jsonl") == []


async def test_load_spans_skips_a_line_cut_off_by_a_crash(tmp_path: Path) -> None:
    tracer = Tracer(tmp_path)
    async with tracer.span("r1", "run", "run"):
        pass
    path = tracer.spans_path("r1")
    with path.open("a") as fh:
        fh.write('{"run_id": "r1", "trace_id": "t", "span_')
    assert [s.name for s in load_spans(path)] == ["run"]
