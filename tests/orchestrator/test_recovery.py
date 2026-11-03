"""Failure paths: restarts, failed checks the user keeps or comments on, budget stops
during audits, re-opened plans and questions left open by a re-open."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from agents.scenario import PRD, team_script

from agent_team.core.artifacts import Artifact
from agent_team.core.errors import BudgetExceeded
from agent_team.core.models import (
    AgentName,
    Phase,
    ReviewMode,
    Run,
    RunStatus,
    Task,
    TaskOrigin,
    TaskStatus,
    Verdict,
)
from agent_team.llm.types import LLMRequest
from agent_team.orchestrator import Orchestrator, OrchestratorConfig
from orchestrator.harness import Team, make_team


def _fresh_orchestrator(team: Team, tmp_path: Path) -> Orchestrator:
    """A new process over the same stores: nothing carried over in memory."""
    return Orchestrator(
        tasks=team.tasks,
        artifacts=team.artifacts,
        checkpoints=team.checkpoints,
        decisions=team.decisions,
        feedback=team.feedback,
        verdicts=team.verdicts,
        bus=team.bus,
        llm=team.llm,
        auditor=team.auditor,
        dispatcher=team.dispatcher,
        config=OrchestratorConfig(runs_dir=tmp_path / "runs"),
    )


def _live(tasks: list[Task]) -> Counter[str]:
    return Counter(t.output_kind for t in tasks if t.status is not TaskStatus.SUPERSEDED)


async def test_finished_phases_survive_a_restart(tmp_path: Path) -> None:
    team = make_team(tmp_path)
    from orchestrator.test_flows import failing

    team.auditor.fail_next["spec"] += [failing("R4", "REQ-1")] * 3
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "waiting" and reply.run_id
    tasks = await team.tasks.list_tasks(reply.run_id)
    early = {t.id for t in tasks if t.phase in (Phase.PLANNING, Phase.PRD)}
    assert all(t.status is TaskStatus.DONE for t in tasks if t.id in early)

    team.auditor.task_calls.clear()
    reply = await _fresh_orchestrator(team, tmp_path).answer(reply.run_id, "Use one module")
    assert reply.kind == "done", reply.text
    # The finished phases' tasks were neither run nor checked again.
    assert early.isdisjoint(team.auditor.task_calls)
    kinds = {t.output_kind for t in await team.tasks.list_tasks(reply.run_id)}
    assert {"code_change", "qa_report"} <= kinds


async def test_failed_check_kept_by_the_user(tmp_path: Path) -> None:
    team = make_team(tmp_path)
    from orchestrator.test_flows import failing

    team.auditor.fail_next["prd"] += [failing("R3", "REQ-1")] * 3
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "waiting" and reply.run_id
    asked = await team.tasks.get_task(reply.pending[0].task_id)
    assert asked.phase is Phase.PRD and "'approve' to keep the work" in asked.objective
    reply = await team.orch.answer(reply.run_id, "approve")
    assert reply.kind == "done", reply.text
    prds = [t for t in await team.tasks.list_tasks(reply.run_id) if t.output_kind == "prd"]
    assert len(prds) == 1 and prds[0].status is TaskStatus.DONE and prds[0].attempt == 3
    accepted = await team.verdicts.latest_for_task(prds[0].id)
    assert accepted is not None and accepted.passed and accepted.model == "user"
    assert accepted.summary and "R3" in accepted.summary
    assert any(d.choice == "accept as is" for d in team.decisions.items)


async def test_failed_check_comment_is_one_more_attempt(tmp_path: Path) -> None:
    team = make_team(tmp_path)
    from orchestrator.test_flows import failing

    team.auditor.fail_next["prd"] += [failing("R3", "REQ-1")] * 3
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    reply = await team.orch.answer(reply.run_id, "Add a latency target")
    assert reply.kind == "done", reply.text
    prds = [t for t in await team.tasks.list_tasks(reply.run_id) if t.output_kind == "prd"]
    assert len(prds) == 1 and prds[0].status is TaskStatus.DONE and prds[0].attempt == 4
    assert any(f.text == "Add a latency target" for _, f in team.feedback.items)


async def test_plan_that_fails_its_check_cannot_be_kept(tmp_path: Path) -> None:
    team = make_team(tmp_path)
    from orchestrator.test_flows import failing

    team.auditor.fail_next["plan"] += [failing("P1", "ASK-1")] * 3
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "waiting" and reply.run_id
    asked = await team.tasks.get_task(reply.pending[0].task_id)
    assert "keep the work" not in asked.objective
    reply = await team.orch.answer(reply.run_id, "approve")
    # Taken as guidance: the planner tries once more, then the pipeline is built.
    assert reply.kind == "done", reply.text
    plan = next(t for t in await team.tasks.list_tasks(reply.run_id) if t.output_kind == "plan")
    assert plan.attempt == 4


async def test_budget_during_an_audit_pauses_and_resumes(tmp_path: Path) -> None:
    team = make_team(tmp_path)
    passing = team.auditor.audit_task
    calls: list[str] = []

    async def audit(task: Task, artifact: Artifact | None, run: Run) -> Verdict:
        calls.append(task.output_kind)
        if task.output_kind == "prd" and calls.count("prd") == 1:
            raise BudgetExceeded("judge budget", limit="max_usd_per_run")
        return await passing(task, artifact, run)

    team.auditor.audit_task = audit  # type: ignore[method-assign]
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "stopped" and reply.run_id
    assert (await team.tasks.get_run(reply.run_id)).status is RunStatus.PAUSED
    prd = next(t for t in await team.tasks.list_tasks(reply.run_id) if t.output_kind == "prd")
    assert prd.status is TaskStatus.AUDITING  # nothing passed while the budget was out
    reply = await team.orch.resume(reply.run_id)
    assert reply.kind == "done", reply.text
    assert (await team.tasks.get_task(prd.id)).attempt == 1


async def test_reopening_the_plan_rebuilds_the_pipeline_once(tmp_path: Path) -> None:
    team = make_team(tmp_path)
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    before = _live(await team.tasks.list_tasks(reply.run_id))
    reply = await team.orch.request_change(reply.run_id, "Also list authors", target="plan")
    assert reply.kind == "done", reply.text
    tasks = await team.tasks.list_tasks(reply.run_id)
    assert _live(tasks) == before
    plans = [t for t in tasks if t.output_kind == "plan"]
    assert [t.status for t in plans] == [TaskStatus.SUPERSEDED, TaskStatus.DONE]


async def test_reopen_supersedes_open_questions(tmp_path: Path) -> None:
    script = team_script()
    asked = dict(PRD, questions=[{"id": "Q-1", "text": "Paginate?", "why": "list size"}])
    script[AgentName.PRODUCT] = lambda r: json.dumps(asked)
    team = make_team(tmp_path, script=script, review_mode=ReviewMode.QUESTIONS_ONLY)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "waiting" and reply.run_id
    run = await team.tasks.get_run(reply.run_id)
    waiting = await team.tasks.list_tasks(run.id, status=TaskStatus.WAITING)
    user = next(t for t in waiting if t.agent is AgentName.USER)
    prd = next(t for t in waiting if t.output_kind == "prd")
    revision = await team.orch.reopen(run, prd, origin=TaskOrigin.REVISION, reason="test")
    assert revision is not None and user.id not in revision.depends_on
    assert (await team.tasks.get_task(user.id)).status is TaskStatus.SUPERSEDED


async def test_answer_to_an_escalation_without_output_reaches_the_agent(tmp_path: Path) -> None:
    script = team_script()
    calls: list[LLMRequest] = []

    def flaky(request: LLMRequest) -> Any:
        calls.append(request)
        if len(calls) <= 3:
            raise RuntimeError("provider is down")
        return json.dumps(PRD)

    script[AgentName.PRODUCT] = flaky
    team = make_team(tmp_path, script=script)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "waiting" and reply.run_id
    reply = await team.orch.answer(reply.run_id, "Keep it to one endpoint")
    assert reply.kind == "done", reply.text
    [answer] = [f for _, f in team.feedback.items if f.text == "Keep it to one endpoint"]
    prd = next(t for t in await team.tasks.list_tasks(reply.run_id) if t.output_kind == "prd")
    assert answer.artifact is None and answer.task_id == prd.id  # no output yet: keyed by task
    assert "Keep it to one endpoint" in calls[-1].messages[0].content[0].text  # type: ignore[union-attr]
