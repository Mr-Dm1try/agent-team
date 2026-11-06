from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_team.audit.toollog import (
    SUMMARY_COMMANDS,
    EventFileToolCallLog,
    InMemoryToolCallLog,
    ToolCall,
    summarize_attempt,
    tool_call_from_event,
)
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
    assert call.data == {}
    detailed = tool_call_from_event(
        _event("r", "t", "run_command", args={"argv": ["ls"]}, data={"exit_code": 2})
    )
    assert detailed is not None and detailed.data == {"exit_code": 2}
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


def test_attempt_summary_lists_commands_exit_codes_and_files() -> None:
    def call(tool: str, args: dict[str, Any], **kw: Any) -> ToolCall:
        """One logged call."""
        return ToolCall(run_id="r", task_id="t", attempt=1, tool=tool, args=args, **kw)

    pytest_argv = ["python", "-m", "pytest", "-q"]
    calls = [
        call(
            "install_deps",
            {"packages": ["mcp==2.3.0"]},
            data={"packages": ["mcp==2.3.0"], "exit_code": 0},
        ),
        call("run_command", {"argv": pytest_argv}, data={"argv": pytest_argv, "exit_code": 1}),
        call("run_command", {"argv": pytest_argv}, data={"argv": pytest_argv, "exit_code": 1}),
        call("run_command", {"argv": ["curl", "x"]}, blocked=True),
        call(
            "run_command",
            {"argv": "pytest"},
            is_error=True,
            result="run_command failed: argv must be a list",
        ),
        call("write_file", {"path": "app.py"}),
        call("write_file", {"path": "bad.py"}, is_error=True),
        call("read_file", {"path": "spec.md"}),
    ]
    text = summarize_attempt("OutputValidationError: no files", calls)
    assert text.splitlines() == [
        "It ended with: OutputValidationError: no files",
        "Commands it ran:",
        "- install_deps mcp==2.3.0: exit 0",
        "- run_command python -m pytest -q: exit 1 (2 times)",
        "- run_command curl x: blocked by the pre-check, not run",
        "- run_command pytest: failed: run_command failed: argv must be a list",
        "Files it wrote: app.py.",
        "Other calls: 1 read_file.",
    ]


def test_attempt_summary_keeps_the_last_commands() -> None:
    calls = [
        ToolCall(run_id="r", tool="run_command", data={"argv": [f"cmd{i}"], "exit_code": 0})
        for i in range(SUMMARY_COMMANDS + 5)
    ]
    text = summarize_attempt("E: x", calls)
    assert "(5 earlier commands left out)" in text
    assert "cmd4:" not in text and f"cmd{SUMMARY_COMMANDS + 4}: exit 0" in text
    assert "Files it wrote: none." in text
