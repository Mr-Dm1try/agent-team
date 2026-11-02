import json
from pathlib import Path

import pytest

from agent_team.core.events import InMemoryEventBus
from agent_team.core.models import AgentName, Event, EventName
from agent_team.core.tracing import (
    JsonlTracer,
    Tracer,
    current_span,
    load_spans,
    render_tree,
    span_tree,
)


async def test_jsonl_tracer_writes_agent_and_combined_logs(tmp_path: Path) -> None:
    bus = InMemoryEventBus()
    JsonlTracer(tmp_path).attach(bus)
    await bus.emit(
        Event(
            name=EventName.TASK_STARTED,
            run_id="r1",
            agent=AgentName.PRODUCT,
            task_id="t1",
            data={"tokens": 3},
        )
    )
    await bus.emit(Event(name=EventName.RUN_STARTED, run_id="r1"))
    logs = tmp_path / "r1" / "logs"
    product = [json.loads(x) for x in (logs / "product.jsonl").read_text().splitlines()]
    combined = (logs / "events.jsonl").read_text().splitlines()
    assert product[0]["task_id"] == "t1" and product[0]["data"] == {"tokens": 3}
    assert len(combined) == 2
    assert (logs / "system.jsonl").exists()


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
