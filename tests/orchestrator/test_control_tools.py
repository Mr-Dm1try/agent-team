from __future__ import annotations

from pathlib import Path

import pytest
from agents.fakes import FakeBus

from agent_team.core.errors import ToolError
from agent_team.core.interfaces import ToolContext
from agent_team.core.models import AgentName, Phase, RequestKind, Task, TaskStatus
from agent_team.orchestrator.control import control_tools
from agent_team.orchestrator.phases import PhaseMachine
from orchestrator.harness import make_team


async def _ctx(tmp_path: Path):  # type: ignore[no-untyped-def]
    team = make_team(tmp_path)
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    run = await team.tasks.get_run(reply.run_id)
    task = (await team.tasks.list_tasks(run.id))[0]
    tools = control_tools(team.tasks, team.artifacts, team.decisions, team.verdicts)
    return team, tools, ToolContext(run=run, task=task, bus=FakeBus())


async def test_read_tools_show_run_state(tmp_path: Path) -> None:
    _team, tools, ctx = await _ctx(tmp_path)
    listing = await tools["list_tasks"]({"phase": "prd"}, ctx)
    # The PRD task and the check of its output.
    assert "product done" in listing.content and "auditor done" in listing.content
    assert listing.data == {"count": 2}
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


def test_phase_machine() -> None:
    build = PhaseMachine(RequestKind.BUILD)
    assert (
        build.next(Phase.SPEC) is Phase.IMPLEMENTATION and build.next(Phase.IMPLEMENTATION) is None
    )

    def task(phase: Phase, agent: AgentName, status: TaskStatus) -> Task:
        """One task in ``phase`` with ``status``."""
        t = Task(run_id="r", phase=phase, agent=agent, objective="o", output_kind="x")
        return t.model_copy(update={"status": status})

    plan = task(Phase.PLANNING, AgentName.ORCHESTRATOR, TaskStatus.DONE)
    prd = task(Phase.PRD, AgentName.PRODUCT, TaskStatus.DONE)
    assert build.current([]) is Phase.PLANNING  # nothing created yet
    assert build.current([plan, prd]) is Phase.SPEC  # no spec task yet
    assert build.finished([plan, prd], Phase.PRD) and not build.finished([plan], Phase.PRD)
    done = [task(p, AgentName.ENGINEER, TaskStatus.DONE) for p in Phase]
    assert build.current(done) is None
    # Open work in an earlier phase wins, even when later phases are done.
    open_prd = task(Phase.PRD, AgentName.PRODUCT, TaskStatus.RUNNING)
    assert build.current([*done, open_prd]) is Phase.PRD
    # A superseded task doesn't count as work, and a waiting question keeps the phase open.
    gone = task(Phase.SPEC, AgentName.ARCHITECT, TaskStatus.SUPERSEDED)
    assert build.current([plan, prd, gone]) is Phase.SPEC
    question = task(Phase.PRD, AgentName.USER, TaskStatus.WAITING)
    assert build.current([plan, prd, question]) is Phase.PRD
    assert PhaseMachine(RequestKind.RESEARCH_QUESTION).phases == (Phase.PLANNING,)
