"""Done condition for 2b: with fake stores, a fake LLM and an always-pass auditor, a run
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

    # Every gate and the run were audited, in phase order.
    assert team.auditor.gates == list(PHASE_ORDER)
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
    assert DecisionKind.PLAN in kinds
    assert kinds.count(DecisionKind.DELEGATE) == len(tasks)
    assert all(d.reason for d in team.decisions.items)
    assert "qa_report@v1" in reply.text


async def test_research_question_gets_an_answer_and_no_code(tmp_path: Path) -> None:
    team = make_team(tmp_path, kind="research_question")
    reply = await team.orch.handle("Which Python web framework is fastest?")
    assert reply.kind == "done", reply.text
    assert reply.run_id is not None
    assert team.artifacts.kinds(reply.run_id) == ["plan", "research_answer"]
    assert reply.artifact is not None and reply.artifact.kind == "research_answer"
    assert "Starlette" in reply.text
    assert team.auditor.gates == [Phase.PLANNING]
    agents = {t.agent for t in await team.tasks.list_tasks(reply.run_id)}
    assert agents == {AgentName.ORCHESTRATOR, AgentName.RESEARCHER}
