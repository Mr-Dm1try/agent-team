"""Retries, questions, reviews, research requests, re-runs, resume and run questions."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from agents.fakes import FakeTaskStore
from agents.scenario import PRD, SPEC, team_script

from agent_team.core.models import (
    AgentName,
    Decision,
    DecisionKind,
    Finding,
    Phase,
    ReviewMode,
    RuleScope,
    Run,
    RunStatus,
    Severity,
    Task,
    TaskOrigin,
    TaskStatus,
    Verdict,
)
from agent_team.llm.types import LLMRequest
from orchestrator.harness import make_team


def failing(rule: str = "R3", item: str = "REQ-1") -> Verdict:
    return Verdict(
        run_id="x",
        scope=RuleScope.TASK,
        passed=False,
        findings=[Finding(rule_id=rule, severity=Severity.BLOCKER, item=item, message="no AC")],
    )


async def _tasks(store: FakeTaskStore, run_id: str, kind: str) -> list[Task]:
    return [t for t in await store.list_tasks(run_id) if t.output_kind == kind]


async def test_audit_failure_retries_with_findings(tmp_path: Path) -> None:
    team = await make_team(tmp_path)
    team.auditor.fail_next["prd"].append(failing())
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done" and reply.run_id
    prd_task = (await _tasks(team.tasks, reply.run_id, "prd"))[0]
    assert prd_task.attempt == 2 and prd_task.output is not None and prd_task.output.version == 2
    retry = [d for d in team.decisions.items if d.kind is DecisionKind.RETRY]
    assert retry and retry[0].task_id == prd_task.id
    # The second attempt saw the findings and its previous output.
    product_calls = [r for r in team.llm.requests if r.agent is AgentName.PRODUCT]
    second = product_calls[1].messages[0].content[0].text  # type: ignore[union-attr]
    assert "Audit findings to fix" in second and "R3 blocker on REQ-1" in second
    assert "Your previous output prd@v1" in second


async def test_escalation_asks_the_user(tmp_path: Path) -> None:
    team = await make_team(tmp_path)
    team.auditor.fail_next["spec"] += [failing("R4", "REQ-1")] * 3
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "waiting" and reply.run_id
    assert "stuck" in reply.pending[0].question
    assert (await team.tasks.get_run(reply.run_id)).status is RunStatus.WAITING
    spec_task = (await _tasks(team.tasks, reply.run_id, "spec"))[0]
    assert spec_task.status is TaskStatus.ESCALATED
    reply = await team.orch.handle("Use one module and try again", run_id=reply.run_id)
    assert reply.kind == "done", reply.text
    spec_task = await team.tasks.get_task(spec_task.id)
    assert spec_task.attempt == 4 and spec_task.status is TaskStatus.DONE


async def test_questions_only_mode_waits_for_answers(tmp_path: Path) -> None:
    script = team_script()
    asked = dict(PRD, questions=[{"id": "Q-1", "text": "Paginate?", "why": "list size"}])
    answered = dict(
        PRD, questions=[{"id": "Q-1", "text": "Paginate?", "why": "list size", "answer": "yes"}]
    )
    calls: list[LLMRequest] = []

    def product(request: LLMRequest) -> str:
        calls.append(request)
        return json.dumps(asked if len(calls) == 1 else answered)

    script[AgentName.PRODUCT] = product
    team = await make_team(tmp_path, script=script, review_mode=ReviewMode.QUESTIONS_ONLY)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "waiting" and reply.run_id
    assert reply.pending[0].question.startswith("Paginate?")
    reply = await team.orch.answer(reply.run_id, "Yes, 20 per page")
    assert reply.kind == "done", reply.text
    second = calls[1].messages[0].content[0].text  # type: ignore[union-attr]
    assert "[user_answer Q-1] Yes, 20 per page" in second
    prd_task = (await _tasks(team.tasks, reply.run_id, "prd"))[0]
    assert prd_task.attempt == 2
    assert prd_task.max_attempts == 4  # asking didn't eat into the retry budget


async def test_every_phase_review_approve_and_comment(tmp_path: Path) -> None:
    team = await make_team(tmp_path, review_mode=ReviewMode.EVERY_PHASE)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "waiting" and reply.run_id
    run_id = reply.run_id
    assert reply.pending[0].origin is TaskOrigin.REVIEW and "planning" in reply.pending[0].question
    reply = await team.orch.answer(run_id, "approve")
    assert reply.kind == "waiting" and "prd" in reply.pending[0].question
    reply = await team.orch.answer(run_id, "Please add an author field")
    # The comment reopened the PRD: a new version, then the PRD review again.
    assert reply.kind == "waiting" and "prd" in reply.pending[0].question
    prds = await _tasks(team.tasks, run_id, "prd")
    assert [t.status for t in prds] == [TaskStatus.SUPERSEDED, TaskStatus.DONE]
    assert prds[1].output is not None and prds[1].output.version == 2
    assert any(f.text == "Please add an author field" for _, f in team.feedback.items)
    reply = await team.orch.answer(run_id, "lgtm")
    reply = await team.orch.answer(run_id, "ok")
    assert reply.kind == "done", reply.text


async def test_research_request_creates_a_researcher_task(tmp_path: Path) -> None:
    script = team_script()
    gap = dict(SPEC, research_requests=[{"question": "Which ORM is lighter?", "why": "deps"}])
    calls: list[LLMRequest] = []

    def architect(request: LLMRequest) -> str:
        calls.append(request)
        return json.dumps(gap if len(calls) == 1 else SPEC)

    script[AgentName.ARCHITECT] = architect
    team = await make_team(tmp_path, script=script)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    assert reply.run_id
    briefs = await _tasks(team.tasks, reply.run_id, "research_brief")
    extra = [t for t in briefs if t.origin is TaskOrigin.RESEARCH_REQUEST]
    assert len(extra) == 1 and extra[0].phase is Phase.SPEC
    assert extra[0].objective == "Which ORM is lighter?"
    second = calls[1].messages[0].content[0].text  # type: ignore[union-attr]
    assert "## Input research_brief@v2" in second


class UpstreamPolicy:
    """Fails QA once and blames the spec, to drive an upstream re-run."""

    def __init__(self) -> None:
        self.fired = False

    async def on_task_verdict(
        self, run: Run, task: Task, verdict: Verdict, history: list[Verdict]
    ) -> list[Decision]:
        if task.output_kind == "qa_report" and not self.fired:
            self.fired = True
            return [
                Decision(
                    run_id=run.id,
                    task_id=task.id,
                    kind=DecisionKind.UPSTREAM_RERUN,
                    inputs={"task_id": task.id, "artifact": "spec@v1"},
                    choice="reopen spec",
                    reason="spec gap found in QA",
                    made_by="code",
                    rule="test",
                )
            ]
        return []


async def test_upstream_rerun_supersedes_and_rebuilds_downstream(tmp_path: Path) -> None:
    team = await make_team(tmp_path, policy=UpstreamPolicy())
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    assert reply.run_id
    run = await team.tasks.get_run(reply.run_id)
    assert run.upstream_reruns == 1
    for kind in ("spec", "code_change", "qa_report"):
        tasks = await _tasks(team.tasks, run.id, kind)
        assert [t.status for t in tasks] == [TaskStatus.SUPERSEDED, TaskStatus.DONE], kind
        assert tasks[1].parent_task_id == tasks[0].id
    # The producer at fault re-runs as a fix; what was built on it re-runs as revisions.
    origins = {k: (await _tasks(team.tasks, run.id, k))[1].origin for k in ("spec", "qa_report")}
    assert origins == {"spec": TaskOrigin.FIX, "qa_report": TaskOrigin.REVISION}
    # PRD wasn't touched.
    assert len(await _tasks(team.tasks, run.id, "prd")) == 1
    kinds = [d.kind for d in team.decisions.items]
    assert DecisionKind.UPSTREAM_RERUN in kinds and DecisionKind.REOPEN_PHASE in kinds
    # The revised spec got a diff-aware envelope with its previous output.
    arch = [r for r in team.llm.requests if r.agent is AgentName.ARCHITECT]
    assert "Your previous output spec@v1" in arch[1].messages[0].content[0].text  # type: ignore[union-attr]


class FixPolicy:
    """Sends QA's first finding back to the Engineer, and re-runs QA after the fix."""

    def __init__(self) -> None:
        self.fired = False
        self.tasks = FakeTaskStore()  # the team's store, set once the team is built

    async def on_task_verdict(
        self, run: Run, task: Task, verdict: Verdict, history: list[Verdict]
    ) -> list[Decision]:
        if task.output_kind != "qa_report" or self.fired or verdict.passed:
            return []
        self.fired = True
        engineer = (await _tasks(self.tasks, run.id, "code_change"))[0]
        return [
            Decision(
                run_id=run.id,
                task_id=task.id,
                kind=DecisionKind.FIX,
                inputs={
                    "task_id": engineer.id,
                    "finding_ids": [f.id for f in verdict.findings],
                    "rerun_after": [task.id],
                },
                choice="send the finding back to the engineer",
                reason="QA found a fault in the code",
                made_by="code",
            )
        ]


async def test_fix_routes_findings_back_and_reruns_the_check(tmp_path: Path) -> None:
    policy = FixPolicy()
    team = await make_team(tmp_path, policy=policy)
    policy.tasks = team.tasks
    finding = Finding(
        id="f1", rule_id="S2", severity=Severity.BLOCKER, item="app.py", message="eval on input"
    )
    team.auditor.fail_next["qa_report"].append(
        Verdict(run_id="x", scope=RuleScope.TASK, passed=False, findings=[finding])
    )
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    assert reply.run_id
    code = await _tasks(team.tasks, reply.run_id, "code_change")
    assert [t.status for t in code] == [TaskStatus.SUPERSEDED, TaskStatus.DONE]
    assert code[1].origin is TaskOrigin.FIX and code[1].parent_task_id == code[0].id
    qa = await _tasks(team.tasks, reply.run_id, "qa_report")
    assert [t.status for t in qa] == [TaskStatus.SUPERSEDED, TaskStatus.DONE]
    assert code[1].id in qa[1].depends_on  # the check runs again after the fix
    fixes = [
        str(r.messages[0].content[0])
        for r in team.llm.requests
        if r.agent is AgentName.ENGINEER and "origin fix" in str(r.messages[0].content[0])
    ]
    assert fixes and "S2 on app.py: eval on input" in fixes[0]


async def test_resume_continues_an_interrupted_attempt(tmp_path: Path) -> None:
    team = await make_team(tmp_path)
    run = await team.orch.start_run("Build a books API")
    plan_task = (await team.tasks.list_tasks(run.id))[0]
    # Simulate a crash mid-attempt: task RUNNING with one checkpoint saved.
    for status in (TaskStatus.READY, TaskStatus.RUNNING):
        await team.tasks.transition(plan_task.id, status, reason="test")
    stored = await team.tasks.get_task(plan_task.id)
    stored.attempt = 1
    await team.tasks.update_task(stored)
    from agent_team.llm.types import Message

    await team.checkpoints.save_step(
        plan_task.id, 1, 1, [Message.user("ENVELOPE FROM BEFORE THE CRASH")]
    )
    reply = await team.orch.resume(run.id)
    assert reply.kind == "done", reply.text
    plan_task = await team.tasks.get_task(plan_task.id)
    assert plan_task.attempt == 1
    planner = [r for r in team.llm.requests if "planner" in r.system]
    assert planner[0].messages[0].content[0].text == "ENVELOPE FROM BEFORE THE CRASH"  # type: ignore[union-attr]


async def test_change_request_reopens_from_the_target(tmp_path: Path) -> None:
    team = await make_team(tmp_path)
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    reply = await team.orch.request_change(
        reply.run_id, "Rename the function to list_books", target="code_change"
    )
    assert reply.kind == "done", reply.text
    assert reply.run_id
    assert len(await _tasks(team.tasks, reply.run_id, "spec")) == 1
    codes = await _tasks(team.tasks, reply.run_id, "code_change")
    assert codes[1].origin is TaskOrigin.CHANGE_REQUEST
    assert len(await _tasks(team.tasks, reply.run_id, "qa_report")) == 2


async def test_auditor_crash_fails_closed(tmp_path: Path) -> None:
    team = await make_team(tmp_path)

    async def broken(task: Task, artifact: Any, run: Run) -> Verdict:
        raise RuntimeError("judge offline")

    team.auditor.audit_task = broken  # type: ignore[method-assign]
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "waiting"
    plan = (await team.tasks.list_tasks(reply.run_id or ""))[0]
    assert plan.status is TaskStatus.ESCALATED
    assert any(v.findings and v.findings[0].rule_id == "AUDIT" for v in team.verdicts.items)


def test_plan_mutation_helper_is_isolated() -> None:
    assert copy.deepcopy(PRD) == PRD


class EscalateThenStrongerPolicy(UpstreamPolicy):
    """Mimics the audit policy's shape: ESCALATE first, then one last try on the stronger model."""

    async def on_task_verdict(
        self, run: Run, task: Task, verdict: Verdict, history: list[Verdict]
    ) -> list[Decision]:
        if verdict.passed or task.output_kind != "prd":
            return []
        ids = [f.id for f in verdict.findings]
        base = {"task_id": task.id, "finding_ids": ids}
        return [
            Decision(
                run_id=run.id,
                task_id=task.id,
                kind=DecisionKind.ESCALATE,
                inputs=base,
                choice="escalate",
                reason="no attempts left",
                made_by="code",
            ),
            Decision(
                run_id=run.id,
                task_id=task.id,
                kind=DecisionKind.STRONGER_MODEL,
                inputs={**base, "model": "claude-sonnet-5-5"},
                choice="stronger",
                reason="own output at fault",
                made_by="code",
            ),
        ]


async def test_escalate_then_stronger_model(tmp_path: Path) -> None:
    team = await make_team(tmp_path, policy=EscalateThenStrongerPolicy())
    team.auditor.fail_next["prd"].append(failing())
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    assert reply.run_id
    prd = (await _tasks(team.tasks, reply.run_id, "prd"))[0]
    assert prd.attempt == 2 and prd.status is TaskStatus.DONE
    assert prd.id not in team.llm.overrides  # cleared once it passed
    moves = [(a, b) for tid, a, b, _ in team.tasks.transitions if tid == prd.id]
    assert (TaskStatus.AUDITING, TaskStatus.ESCALATED) in moves
    assert (TaskStatus.ESCALATED, TaskStatus.READY) in moves


async def test_tool_events_carry_the_auditor_keys(tmp_path: Path) -> None:
    team = await make_team(tmp_path)
    await team.orch.handle("Build a books API")
    after = [e for e in team.bus.events if e.name.value == "tool_call_after"]
    assert after and {"tool", "args", "result", "is_error", "blocked"} <= set(after[0].data)
