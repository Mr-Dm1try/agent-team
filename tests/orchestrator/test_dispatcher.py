from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agents.scenario import team_script

from agent_team.agents import default_agents
from agent_team.core.artifacts import ArtifactRef
from agent_team.core.errors import BudgetExceeded
from agent_team.core.models import (
    AgentName,
    AgentResult,
    EventName,
    Phase,
    ResultStatus,
    RunStatus,
    Task,
    TaskEnvelope,
    TaskStatus,
)
from agent_team.orchestrator.dispatcher import INTERRUPTED
from orchestrator.harness import Team, make_team


async def _prd_task(team: Team, tmp_path: Path) -> Task:
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
    await team.tasks.transition(task.id, TaskStatus.READY, reason="test")
    return task


class Boom:
    name = AgentName.PRODUCT
    output_kind = "prd"

    def __init__(self, exc: BaseException | None = None, sleep: float = 0) -> None:
        self.exc, self.sleep = exc, sleep

    async def run(self, envelope: TaskEnvelope, ctx: Any) -> AgentResult:
        if self.sleep:
            await asyncio.sleep(self.sleep)
        if self.exc:
            raise self.exc
        raise AssertionError("unreachable")


async def test_ok_result_saves_artifact_and_moves_to_auditing(tmp_path: Path) -> None:
    team = await make_team(tmp_path)
    task = await _prd_task(team, tmp_path)
    result = await team.dispatcher.dispatch(task)
    assert result.status is ResultStatus.OK
    stored = await team.tasks.get_task(task.id)
    assert stored.status is TaskStatus.AUDITING
    assert stored.attempt == 1 and stored.output == ArtifactRef(kind="prd", version=1)
    assert stored.model == "fake-1"
    prd = await team.artifacts.load(task.run_id, stored.output)
    assert prd.task_id == task.id and prd.run_id == task.run_id
    names = team.bus.names()
    assert names.index(EventName.TASK_STARTED) < names.index(EventName.TASK_FINISHED)


async def test_exception_is_caught_with_traceback_and_retried(tmp_path: Path) -> None:
    team = await make_team(tmp_path)
    team.dispatcher.agents = {AgentName.PRODUCT: Boom(ValueError("bug in our code"))}  # type: ignore[dict-item]
    task = await _prd_task(team, tmp_path)
    result = await team.dispatcher.dispatch(task)
    assert result.status is ResultStatus.ERROR and result.error_type == "ValueError"
    stored = await team.tasks.get_task(task.id)
    assert stored.status is TaskStatus.RETRY
    assert stored.last_error == "ValueError: bug in our code"
    assert stored.last_traceback and "Traceback" in stored.last_traceback
    errors = [e for e in team.bus.events if e.name is EventName.ERROR]
    assert errors and "traceback" in errors[0].data


async def test_timeout_and_escalation_on_last_attempt(tmp_path: Path) -> None:
    team = await make_team(tmp_path)
    team.dispatcher.agents = {AgentName.PRODUCT: Boom(sleep=1)}  # type: ignore[dict-item]
    team.dispatcher.timeout_s = 0.01
    task = await _prd_task(team, tmp_path)
    stored = await team.tasks.get_task(task.id)
    stored.attempt = stored.max_attempts - 1
    await team.tasks.update_task(stored)
    result = await team.dispatcher.dispatch(task)
    assert result.error_type == "TimeoutError"
    assert (await team.tasks.get_task(task.id)).status is TaskStatus.ESCALATED


async def test_budget_pauses_without_using_an_attempt(tmp_path: Path) -> None:
    team = await make_team(tmp_path)
    team.dispatcher.agents = {AgentName.PRODUCT: Boom(BudgetExceeded("cap", limit="usd"))}  # type: ignore[dict-item]
    task = await _prd_task(team, tmp_path)
    await team.dispatcher.dispatch(task)
    stored = await team.tasks.get_task(task.id)
    assert stored.status is TaskStatus.READY and stored.last_error == INTERRUPTED
    # The next dispatch resumes the same attempt.
    team.dispatcher.agents = default_agents()
    await team.dispatcher.dispatch(stored)
    assert (await team.tasks.get_task(task.id)).attempt == 1


async def test_orchestrator_pauses_run_on_budget(tmp_path: Path) -> None:
    script = team_script()

    def broke(request: Any) -> str:
        raise BudgetExceeded("over budget", limit="max_usd_per_run")

    script[AgentName.PRODUCT] = broke
    team = await make_team(tmp_path, script=script)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "stopped"
    assert reply.run_id is not None
    assert (await team.tasks.get_run(reply.run_id)).status is RunStatus.PAUSED


async def test_envelope_errors_are_a_failed_attempt(tmp_path: Path) -> None:
    team = await make_team(tmp_path)
    task = await _prd_task(team, tmp_path)
    task.inputs = [ArtifactRef(kind="research_brief", version=7)]  # never saved
    await team.tasks.update_task(task)
    result = await team.dispatcher.dispatch(task)
    assert result.status is ResultStatus.ERROR and result.traceback
    assert (await team.tasks.get_task(task.id)).status is TaskStatus.RETRY


async def test_timeout_is_reported_not_raised(tmp_path: Path) -> None:
    team = await make_team(tmp_path)
    team.dispatcher.agents = {AgentName.PRODUCT: Boom(sleep=1)}  # type: ignore[dict-item]
    team.dispatcher.timeouts = {AgentName.PRODUCT: 0.01}
    task = await _prd_task(team, tmp_path)
    result = await team.dispatcher.dispatch(task)
    assert result.error_type == "TimeoutError" and "0s" in (result.error or "")
    assert (await team.tasks.get_task(task.id)).status is TaskStatus.RETRY


async def test_result_of_a_task_superseded_mid_attempt_is_dropped(tmp_path: Path) -> None:
    team = await make_team(tmp_path)
    task = await _prd_task(team, tmp_path)
    agent = default_agents()[AgentName.PRODUCT]

    class Reopened:
        name = AgentName.PRODUCT
        output_kind = "prd"

        async def run(self, envelope: TaskEnvelope, ctx: Any) -> AgentResult:
            await team.tasks.transition(task.id, TaskStatus.SUPERSEDED, reason="reopened")
            return await agent.run(envelope, ctx)

    team.dispatcher.agents = {AgentName.PRODUCT: Reopened()}  # type: ignore[dict-item]
    await team.dispatcher.dispatch(task)
    stored = await team.tasks.get_task(task.id)
    assert stored.status is TaskStatus.SUPERSEDED and stored.output is None
    assert team.artifacts.kinds(task.run_id) == []
    finished = [e for e in team.bus.events if e.name is EventName.TASK_FINISHED]
    assert finished[-1].data["status"] == "discarded"


async def test_feedback_reaches_only_its_own_line_of_work(tmp_path: Path) -> None:
    from agent_team.core.models import Feedback, FeedbackSource, TaskOrigin

    team = await make_team(tmp_path)
    run = await team.orch.start_run("Build a books API")

    def engineer(package: str, **kw: Any) -> Task:
        """An Engineer task for one work package."""
        return Task(
            run_id=run.id,
            phase=Phase.IMPLEMENTATION,
            agent=AgentName.ENGINEER,
            objective=f"Implement {package}",
            output_kind="code_change",
            reason=f"work package {package}",
            **kw,
        )

    from agent_team.core.artifacts import CodeChange

    async def change(path: str) -> ArtifactRef:
        """Save one package's code change."""
        return await team.artifacts.save(
            CodeChange(summary="s", run_id=run.id, files_written=[path])
        )

    a = await team.tasks.create_task(engineer("a", output=await change("a.py")))
    b = await team.tasks.create_task(engineer("b", output=await change("b.py")))
    a_fix = await team.tasks.create_task(engineer("a", origin=TaskOrigin.FIX, parent_task_id=a.id))

    def note(task: Task, text: str) -> Feedback:
        """Feedback on ``task``'s output, as the orchestrator files it."""
        return Feedback(
            artifact=task.output, task_id=task.id, source=FeedbackSource.AUDITOR, text=text
        )

    await team.feedback.add(run.id, note(a, "S2 on a.py"))
    await team.feedback.add(run.id, note(b, "R6 on b"))
    await team.feedback.add(run.id, note(a_fix, "keep the fix small"))

    fix_view = await team.dispatcher.build_envelope(a_fix, run)
    assert [f.text for f in fix_view.feedback] == ["S2 on a.py", "keep the fix small"]
    b_view = await team.dispatcher.build_envelope(b, run)
    assert [f.text for f in b_view.feedback] == ["R6 on b"]


async def test_the_stronger_model_gets_a_compact_envelope(tmp_path: Path) -> None:
    from agent_team.core.models import Decision, DecisionKind, Feedback, FeedbackSource

    team = await make_team(tmp_path)
    run = await team.orch.start_run("Build a books API")
    remembered: list[str] = []

    async def memory(task: Task, run: Any, inputs: Any) -> list[str]:
        """One memory for every task; records who asked."""
        remembered.append(task.id)
        return ["an old lesson"]

    team.dispatcher.memory = memory
    failed = "OutputValidationError: no files"
    task = await team.tasks.create_task(
        Task(
            run_id=run.id,
            phase=Phase.PRD,
            agent=AgentName.PRODUCT,
            objective="PRD",
            output_kind="prd",
            attempt=3,
            previous_error=failed,
            model="claude-sonnet-5-5",  # an agent may run on that model as its own
        )
    )
    for source, text in (
        (FeedbackSource.HARNESS, "No more research will be run for this task"),
        (FeedbackSource.USER_ANSWER, "Use SQLite"),
        (FeedbackSource.AUDITOR, "R3 on REQ-1"),
    ):
        await team.feedback.add(
            run.id, Feedback(artifact=None, task_id=task.id, source=source, text=text)
        )

    usual = await team.dispatcher.build_envelope(task, run)
    assert usual.memories == ["an old lesson"] and usual.previous_attempt is not None
    assert len(usual.feedback) == 3

    await team.decisions.record(
        Decision(
            run_id=run.id,
            task_id=task.id,
            kind=DecisionKind.STRONGER_MODEL,
            inputs={"attempt": 3, "model": "claude-sonnet-5-5"},
            choice="one last attempt on claude-sonnet-5-5",
            reason="no attempts left",
            made_by="code",
        )
    )
    compact = await team.dispatcher.build_envelope(task, run)
    assert compact.memories == [] and compact.previous_attempt is None
    assert compact.upstream_diff is None
    assert [f.text for f in compact.feedback] == ["Use SQLite", "R3 on REQ-1"]
    assert remembered == [task.id]  # memory isn't even searched for the stronger model
