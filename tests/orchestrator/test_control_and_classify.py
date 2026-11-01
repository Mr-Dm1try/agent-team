from __future__ import annotations

from pathlib import Path

import pytest
from agents.fakes import FakeBus, FakeLLM

from agent_team.core.errors import ToolError
from agent_team.core.models import AgentName, Phase, RequestKind, Run, TaskStatus
from agent_team.orchestrator.classify import classify, heuristic
from agent_team.orchestrator.control import control_tools
from agent_team.orchestrator.phases import PhaseMachine
from agent_team.tools import RunToolContext
from orchestrator.harness import make_team


async def _ctx(tmp_path: Path):  # type: ignore[no-untyped-def]
    team = make_team(tmp_path)
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    run = await team.tasks.get_run(reply.run_id)
    task = (await team.tasks.list_tasks(run.id))[0]
    tools = control_tools(team.tasks, team.artifacts, team.decisions, team.verdicts)
    return team, tools, RunToolContext(run=run, task=task, bus=FakeBus())


async def test_read_tools_show_run_state(tmp_path: Path) -> None:
    _team, tools, ctx = await _ctx(tmp_path)
    listing = await tools["list_tasks"]({"phase": "prd"}, ctx)
    assert "product done" in listing.content and listing.data == {"count": 1}
    art = await tools["get_artifact"]({"ref": "prd"}, ctx)
    assert '"REQ-1"' in art.content
    art = await tools["get_artifact"]({"ref": "spec@v1"}, ctx)
    assert '"CHK-1"' in art.content
    decisions = await tools["list_decisions"]({}, ctx)
    assert "classify_request" in decisions.content and "because" in decisions.content
    verdicts = await tools["get_verdicts"]({"task_id": ctx.task.id}, ctx)
    assert '"passed":true' in verdicts.content
    with pytest.raises(ToolError):
        await tools["get_artifact"]({"ref": "nothing"}, ctx)
    with pytest.raises(ToolError):
        await tools["list_tasks"]({"phase": "nope"}, ctx)


async def test_create_task_is_validated(tmp_path: Path) -> None:
    team, tools, ctx = await _ctx(tmp_path)
    ok = await tools["create_task"](
        {
            "agent": "engineer",
            "phase": "implementation",
            "objective": "rename",
            "output_kind": "code_change",
            "reason": "user asked",
        },
        ctx,
    )
    created = await team.tasks.get_task(ok.data["task_id"])  # type: ignore[index]
    assert created.agent is AgentName.ENGINEER and created.parent_task_id == ctx.task.id
    bad = [
        {
            "agent": "auditor",
            "phase": "spec",
            "objective": "x",
            "output_kind": "spec",
            "reason": "r",
        },
        {
            "agent": "engineer",
            "phase": "spec",
            "objective": "x",
            "output_kind": "code_change",
            "reason": "r",
        },
        {
            "agent": "engineer",
            "phase": "implementation",
            "objective": "x",
            "output_kind": "prd",
            "reason": "r",
        },
        {
            "agent": "engineer",
            "phase": "implementation",
            "objective": "x",
            "output_kind": "code_change",
            "reason": "r",
            "depends_on": ["ghost"],
        },
        {
            "agent": "engineer",
            "phase": "implementation",
            "objective": "x",
            "output_kind": "code_change",
            "reason": "r",
            "origin": "plan",
        },
    ]
    for args in bad:
        with pytest.raises(ToolError):
            await tools["create_task"](args, ctx)


async def test_ask_user_creates_a_waiting_user_task(tmp_path: Path) -> None:
    team, tools, ctx = await _ctx(tmp_path)
    out = await tools["ask_user"]({"question": "Which port?"}, ctx)
    task = await team.tasks.get_task(out.data["task_id"])  # type: ignore[index]
    assert task.agent is AgentName.USER and task.status is TaskStatus.WAITING


async def test_classifier_uses_the_model_and_sanitizes() -> None:
    llm = FakeLLM({AgentName.ORCHESTRATOR: ['{"kind": "change", "reason": "tweak"}']})
    result, made_by, _ = await classify(llm, "add auth", None)
    assert made_by == "model" and result.kind is RequestKind.BUILD  # no run to change


async def test_classifier_falls_back_to_keywords(tmp_path: Path) -> None:
    llm = FakeLLM({AgentName.ORCHESTRATOR: ["no idea", "still no"]})
    run = Run(title="t", request="r", workspace=str(tmp_path))
    result, made_by, _ = await classify(llm, "Why did the spec retry?", run)
    assert made_by == "code" and result.kind is RequestKind.RUN_QUESTION


def test_heuristic() -> None:
    assert heuristic("Build a todo API", False).kind is RequestKind.BUILD
    assert heuristic("What is MCP?", False).kind is RequestKind.RESEARCH_QUESTION
    assert heuristic("Add a delete endpoint", True).kind is RequestKind.CHANGE


def test_phase_machine() -> None:
    build = PhaseMachine(RequestKind.BUILD)
    assert (
        build.next(Phase.SPEC) is Phase.IMPLEMENTATION and build.next(Phase.IMPLEMENTATION) is None
    )
    gates = {Phase.PLANNING, Phase.PRD}
    assert build.current([], gates) is Phase.SPEC
    assert build.can_enter(Phase.SPEC, gates) and not build.can_enter(Phase.IMPLEMENTATION, gates)
    build.reopen(Phase.PRD, gates)
    assert gates == {Phase.PLANNING}
    assert PhaseMachine(RequestKind.RESEARCH_QUESTION).phases == (Phase.PLANNING,)
