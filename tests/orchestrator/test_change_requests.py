"""Change requests and stuck QA reviews: the change-request Engineer works from the
spec change, not from findings an earlier fix resolved; guidance that names the code
at a stuck QA review reruns the code, then QA; and local-model tasks queue before
they start."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from agents.scenario import PLAN_BUILD, PRD, QA, SPEC, orchestrator_responder, team_script

from agent_team.core.artifacts import ArtifactRef, CodeChange
from agent_team.core.models import (
    AgentName,
    DecisionKind,
    FeedbackSource,
    Finding,
    Phase,
    RuleScope,
    Severity,
    Task,
    TaskOrigin,
    TaskStatus,
    Verdict,
    new_id,
)
from agent_team.llm.types import LLMRequest
from agent_team.orchestrator.base import names_code, reruns_code
from orchestrator.harness import make_team

CHANGE = "Add a search route"
SEARCH_REQ = {
    "id": "REQ-2",
    "text": "Search books by title",
    "asks": ["ASK-1"],
    "acceptance_criteria": [{"id": "AC-2.1", "text": "GET /search?q= returns matches"}],
}
SEARCH = {
    "name": "search",
    "responsibility": "search route",
    "requirements": ["REQ-2"],
    "files": ["search.py"],
}


def prompt(request: LLMRequest) -> str:
    """The first message's text: the task envelope."""
    return getattr(request.messages[0].content[0], "text", "")


def changing_script() -> dict[AgentName, Any]:
    """The scripted team, where the PRD and spec gain the search requirement once the
    change request reaches them."""
    script = team_script()

    def product(request: LLMRequest) -> str:
        """The PRD, with REQ-2 once the user asked for search."""
        if CHANGE in prompt(request):
            return json.dumps(dict(PRD, requirements=[*PRD["requirements"], SEARCH_REQ]))
        return json.dumps(PRD)

    def architect(request: LLMRequest) -> str:
        """The spec, with the search component once the PRD has REQ-2."""
        if "REQ-2" in prompt(request):
            return json.dumps(dict(SPEC, components=[*SPEC["components"], SEARCH]))
        return json.dumps(SPEC)

    def qa(request: LLMRequest) -> str:
        """QA's report, with a result for the search criterion once the PRD has it."""
        if "AC-2.1" in prompt(request):
            search = {"criterion": "AC-2.1", "result": "pass", "evidence": "search.py:1"}
            return json.dumps(dict(QA, results=[*QA["results"], search]))
        return json.dumps(QA)

    script[AgentName.PRODUCT] = product
    script[AgentName.ARCHITECT] = architect
    script[AgentName.QA] = qa
    return script


def blocker(rule: str, message: str, *, item: str, artifact: ArtifactRef | None) -> Verdict:
    """A failed check with one blocker."""
    finding = Finding(
        rule_id=rule, severity=Severity.BLOCKER, item=item, message=message, artifact=artifact
    )
    return Verdict(run_id="x", scope=RuleScope.TASK, passed=False, findings=[finding])


def engineer_prompts(requests: list[LLMRequest]) -> list[str]:
    """The envelope of each Engineer attempt, in order."""
    return [prompt(r) for r in requests if r.agent is AgentName.ENGINEER and len(r.messages) == 1]


async def test_the_change_request_engineer_gets_the_spec_change_not_resolved_findings(
    tmp_path: Path,
) -> None:
    team = await make_team(tmp_path, script=changing_script(), code_policy=True)
    readme = blocker(
        "R5",
        "README.md missing",
        item="README.md",
        artifact=ArtifactRef(kind="code_change", version=1),
    )
    team.auditor.fail_next["qa_report"].append(readme)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    assert reply.run_id
    # QA's finding was fixed before the change: the fix saw it.
    assert "README.md missing" in engineer_prompts(team.llm.requests)[-1]
    fix = [t for t in await team.tasks.list_tasks(reply.run_id) if t.origin is TaskOrigin.FIX]
    assert [t.agent for t in fix] == [AgentName.ENGINEER]

    seen = len(team.llm.requests)
    reply = await team.orch.request_change(reply.run_id, CHANGE, target="prd")
    assert reply.kind == "done", reply.text
    [envelope] = engineer_prompts(team.llm.requests[seen:])
    assert f"Change request: {CHANGE}" in envelope
    # The requirements the change added are named before the full diff.
    assert "Requirements in spec@v2 since v1:\n- new: REQ-2 in search (search.py)" in envelope
    assert "Requirements in prd@v2 since v1:\n- new: REQ-2 Search books by title" in envelope
    # The README finding was resolved by the fix, so it doesn't follow the change.
    assert "README.md missing" not in envelope
    # The change revises the latest code in the line, the fix's.
    assert "Your previous output code_change@v2" in envelope
    tasks = await team.tasks.list_tasks(reply.run_id)
    change = [t for t in tasks if t.agent is AgentName.ENGINEER][-1]
    assert change.parent_task_id == fix[0].id
    qa = [t for t in tasks if t.agent is AgentName.QA][-1]
    assert qa.objective.endswith(f"Change request: {CHANGE}")


async def test_an_open_finding_still_follows_the_line(tmp_path: Path) -> None:
    team = await make_team(tmp_path, code_policy=True)
    readme = blocker(
        "R5",
        "README.md missing",
        item="README.md",
        artifact=ArtifactRef(kind="code_change", version=1),
    )
    # The fix's own check fails once, so the finding is still open on its retry.
    team.auditor.fail_next["qa_report"].append(readme)
    team.auditor.fail_next["code_change"] += [
        Verdict(run_id="x", scope=RuleScope.TASK, passed=True)
    ]
    team.auditor.fail_next["code_change"].append(
        blocker("R6", "no test run", item="tests", artifact=None)
    )
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    retry = engineer_prompts(team.llm.requests)[-1]
    assert "README.md missing" in retry


async def test_a_qa_blocker_on_a_planned_requirement_fixes_the_code_not_the_spec(
    tmp_path: Path,
) -> None:
    team = await make_team(tmp_path, script=changing_script(), code_policy=True)
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    missing = blocker(
        "R7",
        "the search route is not implemented; search.py is missing",
        item="REQ-2",
        artifact=ArtifactRef(kind="spec", version=2),
    )
    team.auditor.fail_next["qa_report"].append(missing)
    reply = await team.orch.request_change(reply.run_id, CHANGE, target="prd")
    assert reply.kind == "done", reply.text
    decisions = await team.decisions.list(reply.run_id)
    assert not [d for d in decisions if d.kind is DecisionKind.UPSTREAM_RERUN]
    [fix] = [d for d in decisions if d.kind is DecisionKind.FIX]
    assert fix.inputs["blamed_code"] == [missing.findings[0].id]
    tasks = await team.tasks.list_tasks(reply.run_id)
    assert len([t for t in tasks if t.output_kind == "spec"]) == 2  # no third spec
    assert next(t for t in tasks if t.origin is TaskOrigin.FIX).agent is AgentName.ENGINEER


def _stuck_qa(team: Any, verdict: Verdict, times: int) -> None:
    """Queue ``times`` QA failures like ``verdict``, each a verdict of its own."""
    team.auditor.fail_next["qa_report"] += [
        verdict.model_copy(update={"id": new_id("verdict")}) for _ in range(times)
    ]


async def test_guidance_that_names_the_code_reruns_the_code_then_qa(tmp_path: Path) -> None:
    team = await make_team(tmp_path, code_policy=True)
    # QA's own finding, not the code's: its attempts and the stronger model run out.
    own = blocker("R11", "a question is left open", item="Q-1", artifact=None)
    _stuck_qa(team, own, 3)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "waiting", reply.text
    assert reply.run_id
    [pending] = reply.pending
    asked = await team.tasks.get_task(pending.task_id)
    assert "another try of the QA check (guidance that names the code reruns" in asked.objective

    guidance = "Retry the code: implement the search route in search.py"
    reply = await team.orch.answer(reply.run_id, guidance)
    assert reply.kind == "done", reply.text
    tasks = await team.tasks.list_tasks(reply.run_id)
    engineers = [t for t in tasks if t.agent is AgentName.ENGINEER]
    assert engineers[-1].origin is TaskOrigin.FIX
    assert guidance in engineer_prompts(team.llm.requests)[-1]
    # QA reviews again after the code, as a revision of the stuck review.
    qa = [t for t in tasks if t.agent is AgentName.QA]
    assert qa[-1].status is TaskStatus.DONE
    assert qa[-1].created_at > engineers[-1].created_at
    assert engineers[-1].id in qa[-1].depends_on
    # The guidance went to the code and to QA's fresh review.
    given = [f.task_id for _, f in team.feedback.items if f.source is FeedbackSource.USER_ANSWER]
    assert given == [engineers[-2].id, qa[-2].id]
    assert (
        guidance
        in [
            prompt(r) for r in team.llm.requests if r.agent is AgentName.QA and len(r.messages) == 1
        ][-1]
    )


async def test_guidance_for_a_review_stuck_on_its_own_work_retries_qa(tmp_path: Path) -> None:
    team = await make_team(tmp_path, code_policy=True)
    own = blocker("R11", "a question is left open", item="Q-1", artifact=None)
    _stuck_qa(team, own, 3)
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    reply = await team.orch.answer(reply.run_id, "Record the open question as an assumption")
    assert reply.kind == "done", reply.text
    tasks = await team.tasks.list_tasks(reply.run_id)
    assert not [t for t in tasks if t.origin is TaskOrigin.FIX]
    assert len([t for t in tasks if t.agent is AgentName.ENGINEER]) == 1


async def test_a_review_stuck_on_code_says_a_retry_reruns_the_code(tmp_path: Path) -> None:
    team = await make_team(tmp_path, code_policy=True)
    issue = blocker(
        "R7",
        "the list route returns 500",
        item="app.py",
        artifact=ArtifactRef(kind="qa_report", version=1),
    )
    # The first failure fixes the code; the same failure again asks the user.
    _stuck_qa(team, issue, 2)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "waiting", reply.text
    assert reply.run_id
    asked = await team.tasks.get_task(reply.pending[0].task_id)
    assert "another try of the code, then the QA check" in asked.objective

    # Guidance that doesn't name the code still goes to the code: the check failed on it.
    reply = await team.orch.answer(reply.run_id, "Return an empty list when there are no books")
    assert reply.kind == "done", reply.text
    engineers = [
        t for t in await team.tasks.list_tasks(reply.run_id) if t.agent is AgentName.ENGINEER
    ]
    assert len(engineers) == 3 and engineers[-1].origin is TaskOrigin.FIX
    assert "Return an empty list" in engineer_prompts(team.llm.requests)[-1]


async def test_tasks_on_a_local_model_queue_before_their_timeout_starts(tmp_path: Path) -> None:
    plan = dict(
        PLAN_BUILD,
        research_questions=[
            {"question": q, "ask": "ASK-1"}
            for q in ("Which database?", "Which ORM?", "Which host?")
        ],
    )
    script = team_script()
    script[AgentName.ORCHESTRATOR] = orchestrator_responder(plan=plan)
    team = await make_team(tmp_path, script=script)
    assert team.orch.config.max_tasks_per_provider == {"ollama": 1}
    team.llm.providers_for = (  # type: ignore[method-assign]
        lambda agent, task_id: ["ollama"] if agent is AgentName.RESEARCHER else ["fake"]
    )
    running: list[str] = []
    most = 0
    run_task = team.orch._run_task

    async def counted(run: Any, task: Task) -> None:
        """Run one task, counting the Researcher tasks running at once."""
        nonlocal most
        if task.agent is AgentName.RESEARCHER:
            running.append(task.id)
            most = max(most, len(running))
        await asyncio.sleep(0)
        try:
            await run_task(run, task)
        finally:
            if task.id in running:
                running.remove(task.id)

    team.orch._run_task = counted  # type: ignore[method-assign]
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    researched = [
        t
        for t in await team.tasks.list_tasks(reply.run_id or "")
        if t.agent is AgentName.RESEARCHER
    ]
    assert len(researched) == 3
    assert most == 1

    # Without a cap the same three run together.
    team2 = await make_team(tmp_path / "uncapped", script=script)
    team2.orch.config.max_tasks_per_provider = {}
    team2.llm.providers_for = team.llm.providers_for  # type: ignore[method-assign]
    most = 0
    run_task = team2.orch._run_task
    team2.orch._run_task = counted  # type: ignore[method-assign]
    reply = await team2.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    assert most == 3


def test_guidance_names_the_code_unless_it_says_to_leave_it() -> None:
    assert names_code("Retry the code: implement list_recent_notes (recent.py)")
    assert names_code("ask the engineer to add recent.py")
    assert not names_code("don't touch the code, just rerun tests/test_api.py")
    assert not names_code("exit code 5 means no tests were collected")
    assert not names_code("Record the open question as an assumption")
    qa = Task(
        run_id="r",
        phase=Phase.IMPLEMENTATION,
        agent=AgentName.QA,
        objective="o",
        output_kind="qa_report",
    )
    on_code = blocker("R7", "500 on GET /books", item="app.py", artifact=None)
    assert reruns_code(qa, on_code)
    # Saying to leave the code alone wins over code blockers; other agents never rerun it.
    assert not reruns_code(qa, on_code, "leave the code as it is and check again")
    engineer = qa.model_copy(update={"agent": AgentName.ENGINEER})
    assert not reruns_code(engineer, on_code, "fix the code")


async def test_a_retry_of_the_code_reopens_the_engineer_that_wrote_the_file(
    tmp_path: Path,
) -> None:
    team = await make_team(tmp_path)
    run = await team.orch.start_run("Build a books API")
    engineers = []
    for path in ("a.py", "b.py"):
        ref = await team.artifacts.save(
            CodeChange(run_id=run.id, summary="s", files_written=[path])
        )
        engineers.append(
            Task(
                run_id=run.id,
                phase=Phase.IMPLEMENTATION,
                agent=AgentName.ENGINEER,
                objective="o",
                output_kind="code_change",
                output=ref,
            )
        )

    def at(item: str) -> list[Finding]:
        """One R7 blocker on ``item``."""
        return blocker("R7", "broken", item=item, artifact=None).findings

    assert await team.orch._code_at_fault(run, engineers, at("a.py:12")) is engineers[0]
    assert await team.orch._code_at_fault(run, engineers, at("b.py")) is engineers[1]
    assert await team.orch._code_at_fault(run, engineers, at("c.py")) is engineers[1]
