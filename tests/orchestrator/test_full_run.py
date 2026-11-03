"""With fake stores, a fake LLM and an always-pass auditor, a run
goes through all four phases and produces every artifact; a research question returns a
research_answer without PRD or code."""

from __future__ import annotations

from pathlib import Path

from agent_team.core.models import (
    PHASE_ORDER,
    AgentName,
    DecisionKind,
    EventName,
    Phase,
    RunStatus,
    TaskStatus,
)
from orchestrator.harness import make_team


async def test_build_runs_all_four_phases(tmp_path: Path) -> None:
    team = make_team(tmp_path)
    reply = await team.orch.handle("Build a REST API that lists books")
    assert reply.kind == "done", reply.text
    run_id = reply.run_id
    assert run_id is not None
    run = await team.tasks.get_run(run_id)
    assert run.status is RunStatus.DONE

    assert team.artifacts.kinds(run_id) == sorted(
        ["plan", "research_brief", "prd", "spec", "code_change", "qa_report"]
    )
    assert (Path(run.workspace) / "app.py").read_text() == "print('books')\n"

    tasks = await team.tasks.list_tasks(run_id)
    assert all(t.status is TaskStatus.DONE for t in tasks)
    assert {t.phase for t in tasks} == set(PHASE_ORDER)

    # Every phase finished, in order, and the run was audited once.
    finished = [e.data["phase"] for e in team.bus.events if e.name is EventName.PHASE_FINISHED]
    assert finished == [p.value for p in PHASE_ORDER]
    assert team.auditor.runs == 1
    started = [e.data["phase"] for e in team.bus.events if e.name is EventName.PHASE_STARTED]
    assert started == [p.value for p in PHASE_ORDER]
    assert team.bus.names()[-1] is EventName.RUN_FINISHED

    # Lineage: each artifact records what it was built from.
    spec = await team.artifacts.latest(run_id, "spec")
    assert spec is not None and {str(r) for r in spec.built_from} >= {"plan@v1", "prd@v1"}
    qa = await team.artifacts.latest(run_id, "qa_report")
    assert qa is not None and {r.kind for r in qa.built_from} >= {"prd", "spec", "code_change"}

    kinds = [d.kind for d in team.decisions.items]
    assert kinds[0] is DecisionKind.CLASSIFY_REQUEST
    plans = [d for d in team.decisions.items if d.kind is DecisionKind.PLAN]
    assert [d.choice.split(":")[0] for d in plans] == ["3 tasks", "2 tasks"]
    assert all(d.made_by == "code" and d.rule == "pipeline" for d in plans)
    assert kinds.count(DecisionKind.DELEGATE) == len(tasks)
    assert all(d.reason for d in team.decisions.items)
    assert "the QA check: All good" in reply.text and "@v" not in reply.text  # plain words


async def test_research_question_gets_an_answer_and_no_code(tmp_path: Path) -> None:
    team = make_team(tmp_path, kind="research_question")
    reply = await team.orch.handle("Which Python web framework is fastest?")
    assert reply.kind == "done", reply.text
    assert reply.run_id is not None
    assert team.artifacts.kinds(reply.run_id) == ["plan", "research_answer", "research_brief"]
    tasks = await team.tasks.list_tasks(reply.run_id)
    brief = next(t for t in tasks if t.output_kind == "research_brief")
    answer = next(t for t in tasks if t.output_kind == "research_answer")
    planner = next(t for t in tasks if t.output_kind == "plan")
    assert brief.objective == "Which benchmarks are current?"
    assert answer.depends_on == [planner.id, brief.id]
    assert reply.artifact is not None and reply.artifact.kind == "research_answer"
    assert "Starlette" in reply.text
    finished = [e.data["phase"] for e in team.bus.events if e.name is EventName.PHASE_FINISHED]
    assert finished == [Phase.PLANNING.value]
    agents = {t.agent for t in await team.tasks.list_tasks(reply.run_id)}
    assert agents == {AgentName.ORCHESTRATOR, AgentName.RESEARCHER}


async def test_work_packages_become_serial_engineer_tasks(tmp_path: Path) -> None:
    import json

    from agents.scenario import SPEC, team_script

    script = team_script()
    spec = dict(
        SPEC,
        components=[
            {
                "name": "db",
                "responsibility": "storage",
                "requirements": ["REQ-1"],
                "files": ["db.py"],
            },
            {
                "name": "api",
                "responsibility": "routes",
                "requirements": ["REQ-1"],
                "files": ["app.py"],
            },
        ],
        work_packages=[
            {"name": "storage", "components": ["db"]},
            {"name": "http", "components": ["api"]},
        ],
    )
    script[AgentName.ARCHITECT] = lambda r: json.dumps(spec)
    team = make_team(tmp_path, script=script)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    assert reply.run_id
    tasks = await team.tasks.list_tasks(reply.run_id, phase=Phase.IMPLEMENTATION)
    engineers = [t for t in tasks if t.agent is AgentName.ENGINEER]
    qa = next(t for t in tasks if t.agent is AgentName.QA)
    assert [t.objective for t in engineers] == [
        "Implement work package 'storage': components db",
        "Implement work package 'http': components api",
    ]
    assert engineers[1].depends_on == [engineers[0].id]
    assert qa.depends_on == [e.id for e in engineers]
    starts = [e.task_id for e in team.bus.events if e.name is EventName.TASK_STARTED]
    assert starts.index(engineers[0].id) < starts.index(engineers[1].id) < starts.index(qa.id)
