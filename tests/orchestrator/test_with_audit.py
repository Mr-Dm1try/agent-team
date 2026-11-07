"""The orchestrator with the audit package's real verdict policy (the composition
root's default) and tool-call log, over the fake stores, to check the two agree on
decision and event shapes."""

from __future__ import annotations

import json
from pathlib import Path

from agents.fakes import tool_call
from agents.scenario import CODE, Responder, team_script

from agent_team.audit.toollog import ToolCall, tool_call_from_event
from agent_team.core.artifacts import ArtifactRef
from agent_team.core.models import (
    AgentName,
    DecisionKind,
    EventName,
    Finding,
    Phase,
    RuleScope,
    Severity,
    Task,
    TaskOrigin,
    TaskStatus,
    Verdict,
)
from agent_team.llm.types import ContentBlock, LLMRequest, ToolResultBlock
from orchestrator.harness import Team, make_team


def _blocker(rule: str, item: str) -> Verdict:
    """A failing task verdict with one blocker on ``item``."""
    return Verdict(
        run_id="x",
        scope=RuleScope.TASK,
        passed=False,
        findings=[Finding(rule_id=rule, severity=Severity.BLOCKER, item=item, message="bad")],
    )


async def test_failed_task_is_retried_by_the_real_policy(tmp_path: Path) -> None:
    team = await make_team(tmp_path, code_policy=True)
    team.auditor.fail_next["prd"].append(_blocker("R3", "REQ-1"))
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    assert reply.run_id
    prd = next(t for t in await team.tasks.list_tasks(reply.run_id) if t.output_kind == "prd")
    assert prd.status is TaskStatus.DONE and prd.attempt == 2
    assert any(d.kind is DecisionKind.RETRY for d in team.decisions.items)


async def test_qa_finding_on_code_is_fixed_by_the_real_policy(tmp_path: Path) -> None:
    team = await make_team(tmp_path, code_policy=True)
    team.auditor.fail_next["qa_report"].append(_blocker("S2", "app.py"))
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    assert reply.run_id
    kinds = [d.kind for d in team.decisions.items]
    assert DecisionKind.FIX in kinds and DecisionKind.REOPEN_PHASE in kinds
    tasks = await team.tasks.list_tasks(reply.run_id)
    for kind in ("code_change", "qa_report"):
        found = [t.status for t in tasks if t.output_kind == kind]
        assert found == [TaskStatus.SUPERSEDED, TaskStatus.DONE], kind
    fixed = next(t for t in tasks if t.output_kind == "code_change" and t.status is TaskStatus.DONE)
    assert fixed.origin is TaskOrigin.FIX
    # The PRD and spec were not touched.
    assert [t.status for t in tasks if t.output_kind in ("prd", "spec")] == [TaskStatus.DONE] * 2


async def test_qa_finding_on_the_spec_reopens_it(tmp_path: Path) -> None:
    team = await make_team(tmp_path, code_policy=True)
    gap = Finding(
        rule_id="R10",
        severity=Severity.BLOCKER,
        artifact=ArtifactRef(kind="spec", version=1),
        item="COMP-1",
        message="the spec contradicts the PRD",
    )
    team.auditor.fail_next["qa_report"].append(
        Verdict(run_id="x", scope=RuleScope.TASK, passed=False, findings=[gap])
    )
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    assert reply.run_id
    assert DecisionKind.UPSTREAM_RERUN in [d.kind for d in team.decisions.items]
    tasks = await team.tasks.list_tasks(reply.run_id)
    for kind in ("spec", "code_change", "qa_report"):
        found = [t.status for t in tasks if t.output_kind == kind]
        assert found == [TaskStatus.SUPERSEDED, TaskStatus.DONE], kind
    assert (await team.tasks.get_run(reply.run_id)).upstream_reruns == 1


async def test_tool_events_read_back_as_tool_calls(tmp_path: Path) -> None:
    team = await make_team(tmp_path)
    await team.orch.handle("Build a books API")
    after = [e for e in team.bus.events if e.name is EventName.TOOL_CALL_AFTER]
    calls = [tool_call_from_event(e) for e in after]
    assert calls and all(c is not None for c in calls)
    write = next(c for c in calls if c is not None and c.tool == "write_file")
    assert write.args["path"] and "created" in write.result and not write.blocked


async def test_a_dangerous_action_cannot_be_kept_as_it_is(tmp_path: Path) -> None:
    team = await make_team(tmp_path, code_policy=True)
    team.auditor.fail_next["prd"].append(_blocker("S5", "run_command"))
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "stopped" and reply.run_id, reply.text
    [asked] = [
        t
        for t in await team.tasks.list_tasks(reply.run_id, status=TaskStatus.WAITING)
        if t.agent is AgentName.USER
    ]
    assert "unsafe" in asked.objective and "keep the work" not in asked.objective
    reply = await team.orch.answer(reply.run_id, "ok")
    # Taken as guidance for one more attempt, never as accepting the flagged work.
    assert reply.kind == "done", reply.text
    prd = next(t for t in await team.tasks.list_tasks(reply.run_id) if t.output_kind == "prd")
    assert prd.attempt == 2
    assert all(v.model != "user" for v in team.verdicts.items)


class _BusToolLog:
    """The tool-call log read straight from the fake bus, which writes no events file."""

    def __init__(self, team: Team) -> None:
        """Read ``team``'s bus."""
        self.team = team

    async def for_run(self, run_id: str) -> list[ToolCall]:
        """The run's calls, oldest first."""
        calls = (
            tool_call_from_event(e)
            for e in self.team.bus.events
            if e.name is EventName.TOOL_CALL_AFTER and e.run_id == run_id
        )
        return [c for c in calls if c is not None]

    async def for_task(self, run_id: str, task_id: str) -> list[ToolCall]:
        """The task's calls, oldest first."""
        return [c for c in await self.for_run(run_id) if c.task_id == task_id]


def _engineer_without_files(team: Team, *, stronger_writes: bool) -> Responder:
    """An Engineer that runs one command and then replies with no files, so its output
    never validates; on the stronger model it writes app.py when ``stronger_writes``."""

    def respond(request: LLMRequest) -> list[ContentBlock] | str:
        """One scripted Engineer turn."""
        stronger = request.task_id in team.llm.overrides
        last = request.messages[-1]
        ran = any(isinstance(b, ToolResultBlock) for b in last.content)
        if stronger and stronger_writes:
            if not ran:
                return tool_call("write_file", {"path": "app.py", "content": "print('ok')\n"})
            return json.dumps(CODE)
        if len(request.messages) == 1:
            return tool_call("run_command", {"argv": ["python", "-c", "import mcp"]})
        return json.dumps({"summary": "Looked at the SDK.", "files_written": []})

    return respond


async def test_invalid_output_repeated_goes_to_the_stronger_model(tmp_path: Path) -> None:
    script = team_script()
    team = await make_team(tmp_path, code_policy=True, script=script)
    script[AgentName.ENGINEER] = _engineer_without_files(team, stronger_writes=True)
    team.dispatcher.tool_log = _BusToolLog(team)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    assert reply.run_id
    code = next(
        t for t in await team.tasks.list_tasks(reply.run_id) if t.output_kind == "code_change"
    )
    # Attempt 2 failed the same way as attempt 1, so attempt 3 ran on the stronger model.
    assert code.status is TaskStatus.DONE and code.attempt == 3
    kinds = [d.kind for d in team.decisions.items if d.task_id == code.id]
    assert DecisionKind.NO_PROGRESS in kinds and DecisionKind.STRONGER_MODEL in kinds
    assert kinds.index(DecisionKind.NO_PROGRESS) < kinds.index(DecisionKind.STRONGER_MODEL)
    # The second attempt was told what the first one did.
    prompts = [
        r.messages[0].content[0]
        for r in team.llm.requests
        if r.agent is AgentName.ENGINEER and len(r.messages) == 1
    ]
    second = getattr(prompts[1], "text", "")
    assert "Your previous attempt failed" in second
    assert "OutputValidationError" in second
    assert "run_command python -c import mcp: exit 0" in second
    assert "Files it wrote: none." in second
    assert "Your previous attempt failed" not in getattr(prompts[0], "text", "")


async def test_invalid_output_after_the_stronger_model_asks_the_user(tmp_path: Path) -> None:
    script = team_script()
    team = await make_team(tmp_path, code_policy=True, script=script)
    script[AgentName.ENGINEER] = _engineer_without_files(team, stronger_writes=False)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "waiting", reply.text
    assert reply.run_id
    tasks = await team.tasks.list_tasks(reply.run_id)
    code = next(t for t in tasks if t.output_kind == "code_change")
    assert code.status is TaskStatus.ESCALATED and code.attempt == 3
    kinds = [d.kind for d in team.decisions.items if d.task_id == code.id]
    assert kinds.count(DecisionKind.STRONGER_MODEL) == 1
    asks = [d for d in team.decisions.items if d.kind is DecisionKind.ASK_USER]
    assert asks[-1].rule == "policy.ask_after_stronger_model"
    question = next(t for t in tasks if t.agent is AgentName.USER and t.parent_task_id == code.id)
    assert question.id in code.depends_on


async def test_an_escalation_left_by_a_crash_still_gets_the_stronger_model(
    tmp_path: Path,
) -> None:
    # The process stopped right after the dispatcher escalated the task.
    team = await make_team(tmp_path, code_policy=True)
    run = await team.orch.start_run("Build a books API")
    error = "OutputValidationError: output was cut off: files_written: no files listed"
    task = await team.tasks.create_task(
        Task(
            run_id=run.id,
            phase=Phase.IMPLEMENTATION,
            agent=AgentName.ENGINEER,
            objective="Build it",
            output_kind="code_change",
            attempt=2,
            last_error=error,
            previous_error=error,
        )
    )
    for status in (TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.ESCALATED):
        await team.tasks.transition(task.id, status, reason="test")
    assert await team.orch._settle(run, await team.tasks.list_tasks(run.id))
    task = await team.tasks.get_task(task.id)
    assert task.status is TaskStatus.READY and task.max_attempts == 3
    kinds = [d.kind for d in team.decisions.items if d.task_id == task.id]
    assert kinds == [DecisionKind.NO_PROGRESS, DecisionKind.STRONGER_MODEL]
    assert task.id in team.llm.overrides
