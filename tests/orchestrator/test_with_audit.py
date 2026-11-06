"""The orchestrator with the audit package's real verdict policy (the composition
root's default) and tool-call log, over the fake stores, to check the two agree on
decision and event shapes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agents.fakes import tool_call
from agents.scenario import BRIEF, CODE, Responder, team_script

from agent_team.audit.toollog import ToolCall, tool_call_from_event
from agent_team.core.artifacts import ArtifactRef
from agent_team.core.models import (
    AgentName,
    Decision,
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
    assert "'stop'" not in asked.objective and "the run stays stopped" in asked.objective
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


async def test_research_requests_and_the_stronger_model_stay_within_the_limit(
    tmp_path: Path,
) -> None:
    # A planned research task fails its check, asks for research, keeps asking, and
    # keeps failing. Its attempts stop at the limit plus the one on the stronger model,
    # and the attempt that asked counts like any other.
    script = team_script()
    team = await make_team(tmp_path, code_policy=True, script=script)
    prompts: dict[int, str] = {}
    gap = {"question": "What does the WAL checkpoint cost?", "why": "trade-offs"}

    def researcher(request: LLMRequest) -> str:
        """A planned brief plain on attempt 1, then always asking for research."""
        text = request.messages[0].content[0].text  # type: ignore[union-attr]
        if "origin research_request" in text:
            return json.dumps(BRIEF)
        attempt = int(text.split("(attempt ", 1)[1].split(",", 1)[0])
        prompts[attempt] = text
        return json.dumps(BRIEF if attempt == 1 else dict(BRIEF, research_requests=[gap]))

    script[AgentName.RESEARCHER] = researcher
    passing = team.auditor.audit_task

    async def audit_task(task: Task, artifact: object, run: object) -> Verdict:
        """Fail every attempt of planned research on a new item, so it isn't no progress."""
        if task.origin is TaskOrigin.PLAN and task.output_kind == "research_brief":
            return _blocker("P2", f"item {task.attempt}").model_copy(
                update={"task_id": task.id, "run_id": task.run_id}
            )
        return await passing(task, artifact, run)  # type: ignore[arg-type]

    team.auditor.audit_task = audit_task  # type: ignore[method-assign]
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "waiting", reply.text
    assert reply.run_id
    tasks = await team.tasks.list_tasks(reply.run_id)
    planned = next(
        t for t in tasks if t.origin is TaskOrigin.PLAN and t.output_kind == "research_brief"
    )
    # 1 failed, 2 asked for research, 3 (the last) couldn't ask and failed, 4 ran on the
    # stronger model and failed: then the user is asked.
    assert planned.attempt == 4 and planned.max_attempts == 4
    assert planned.status is TaskStatus.ESCALATED
    children = [t for t in tasks if t.origin is TaskOrigin.RESEARCH_REQUEST]
    assert [c.parent_task_id for c in children] == [planned.id]
    assert "can't ask for more research" not in prompts[2]
    assert all("can't ask for more research" in prompts[n] for n in (3, 4))
    kinds = [d.kind for d in team.decisions.items if d.task_id == planned.id]
    assert kinds.count(DecisionKind.STRONGER_MODEL) == 1


async def test_the_stronger_model_decision_records_its_prompt_cap_and_cost(
    tmp_path: Path,
) -> None:
    team = await make_team(tmp_path, code_policy=True)
    for item in ("REQ-1", "REQ-2", "REQ-3"):
        team.auditor.fail_next["prd"].append(_blocker("R3", item))
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    [stronger] = [d for d in team.decisions.items if d.kind is DecisionKind.STRONGER_MODEL]
    cap = team.llm.context_budget(AgentName.PRODUCT, stronger.task_id)
    assert stronger.inputs["prompt_cap_tokens"] == cap
    # The default routing file prices the stronger model at $2 per million prompt tokens.
    assert stronger.inputs["max_prompt_usd"] == round(cap * 2.0 / 1_000_000, 4)
    prd = next(t for t in await team.tasks.list_tasks(stronger.run_id) if t.output_kind == "prd")
    assert prd.attempt == 4 and prd.max_attempts == 4 and prd.status is TaskStatus.DONE


async def test_routes_to_the_stronger_model_come_back_in_a_new_process(tmp_path: Path) -> None:
    team = await make_team(tmp_path, code_policy=True)
    run = await team.orch.start_run("Build a books API")
    open_task, done = [
        await team.tasks.create_task(
            Task(
                run_id=run.id,
                phase=Phase.PRD,
                agent=AgentName.PRODUCT,
                objective=objective,
                output_kind="prd",
            )
        )
        for objective in ("PRD", "old PRD")
    ]
    await team.tasks.transition(done.id, TaskStatus.READY, reason="test")
    await team.tasks.transition(done.id, TaskStatus.RUNNING, reason="test")
    await team.tasks.transition(done.id, TaskStatus.AUDITING, reason="test")
    await team.tasks.transition(done.id, TaskStatus.DONE, reason="test")
    for task in (open_task, done):
        await team.decisions.record(
            Decision(
                run_id=run.id,
                task_id=task.id,
                kind=DecisionKind.STRONGER_MODEL,
                inputs={"attempt": 4},
                choice="one last attempt",
                reason="no attempts left",
                made_by="code",
            )
        )
    team.llm.overrides.clear()  # routes live in memory: a new process has none
    await team.orch._restore_stronger_routes(run.id)
    assert set(team.llm.overrides) == {open_task.id}


async def test_a_stronger_model_decision_is_recorded_even_when_routing_fails(
    tmp_path: Path,
) -> None:
    team = await make_team(tmp_path, code_policy=True)
    run = await team.orch.start_run("Build a books API")
    task = await team.tasks.create_task(
        Task(
            run_id=run.id,
            phase=Phase.PRD,
            agent=AgentName.PRODUCT,
            objective="PRD",
            output_kind="prd",
        )
    )

    def no_stronger(task_id: str) -> object:
        """A routing file without a stronger model."""
        raise ValueError("config has no stronger_model")

    team.llm.use_stronger_model = no_stronger  # type: ignore[method-assign]
    sent = Decision(
        run_id=run.id,
        task_id=task.id,
        kind=DecisionKind.STRONGER_MODEL,
        inputs={"attempt": 4},
        choice="one last attempt",
        reason="no attempts left",
        made_by="code",
    )
    with pytest.raises(ValueError):
        await team.orch._apply(run, sent)
    assert [d.id for d in team.decisions.items] == [sent.id]  # so the policy asks next time
