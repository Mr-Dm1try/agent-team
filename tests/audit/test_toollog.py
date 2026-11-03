from __future__ import annotations

from pathlib import Path

from agent_team.audit.toollog import EventFileToolCallLog, InMemoryToolCallLog, tool_call_from_event
from agent_team.core.models import AgentName, Event, EventName
from agent_team.core.tracing import events_path


def _event(run_id: str, task_id: str, tool: str, **data: object) -> Event:
    return Event(name=EventName.TOOL_CALL_AFTER, run_id=run_id, task_id=task_id,
                 agent=AgentName.ENGINEER, data={"tool": tool, **data})  # fmt: skip


def test_event_parsing_is_lenient() -> None:
    call = tool_call_from_event(
        _event("r", "t", "run_command", arguments={"command": "ls"}, content="x", blocked=True)
    )
    assert call is not None and call.args == {"command": "ls"} and call.blocked
    nameless = Event(name=EventName.TOOL_CALL_AFTER, run_id="r", data={})
    assert tool_call_from_event(nameless) is None


async def test_file_log_reads_events_back_after_a_restart(tmp_path: Path) -> None:
    path = events_path(tmp_path, "run_1")
    path.parent.mkdir(parents=True)
    lines = [
        _event("run_1", "task_a", "read_file", args={"path": "a.py"}, result="..."),
        _event("run_1", "task_b", "run_command", args={"command": "pytest -q"}),
        Event(name=EventName.TASK_STARTED, run_id="run_1", task_id="task_a"),
    ]
    path.write_text("\n".join(e.model_dump_json() for e in lines) + "\n{cut off by a crash")
    log = EventFileToolCallLog(tmp_path)  # a fresh instance, as after a resume
    assert [c.tool for c in await log.for_run("run_1")] == ["read_file", "run_command"]
    assert [c.tool for c in await log.for_task("run_1", "task_b")] == ["run_command"]
    assert await log.for_run("run_missing") == []


async def test_memory_log_filters_by_run_and_task() -> None:
    log = InMemoryToolCallLog()
    for e in (_event("r1", "t", "a"), _event("r2", "t", "b")):
        call = tool_call_from_event(e)
        assert call is not None
        log.record(call)
    assert [c.tool for c in await log.for_task("r1", "t")] == ["a"]
