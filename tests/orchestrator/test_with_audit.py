"""The orchestrator with the audit package's real verdict policy (the composition
root's default) and tool-call log, over the fake stores, to check the two agree on
decision and event shapes."""

from __future__ import annotations

from pathlib import Path

from agent_team.audit.toollog import tool_call_from_event
from agent_team.core.artifacts import ArtifactRef
from agent_team.core.models import (
    AgentName,
    DecisionKind,
    EventName,
    Finding,
    RuleScope,
    Severity,
    TaskOrigin,
    TaskStatus,
    Verdict,
)
from orchestrator.harness import make_team


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
    reply = await team.orch.answer(reply.run_id, "ok")
    # Taken as guidance for one more attempt, never as accepting the flagged work.
    assert reply.kind == "done", reply.text
    prd = next(t for t in await team.tasks.list_tasks(reply.run_id) if t.output_kind == "prd")
    assert prd.attempt == 2
    assert all(v.model != "user" for v in team.verdicts.items)
