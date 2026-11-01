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
    team = make_team(tmp_path)
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
    team = make_team(tmp_path)
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
    team = make_team(tmp_path)
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
    team = make_team(tmp_path)
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
    team = make_team(tmp_path, script=script)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "stopped"
    assert reply.run_id is not None
    assert (await team.tasks.get_run(reply.run_id)).status is RunStatus.PAUSED
