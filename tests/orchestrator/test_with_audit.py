"""The orchestrator with the audit package's real verdict policy and tool-call log,
over the fake stores, to check the two agree on decision and event shapes."""

from __future__ import annotations

from pathlib import Path

from agent_team.audit.policy import CodeVerdictPolicy
from agent_team.audit.toollog import tool_call_from_event
from agent_team.core.models import (
    DecisionKind,
    EventName,
    Finding,
    Phase,
    RuleScope,
    Run,
    Severity,
    TaskStatus,
    Verdict,
)
from orchestrator.harness import Team, make_team


def _with_policy(team: Team) -> Team:
    """Swap the fallback policy for the audit package's own."""
    team.orch.policy = CodeVerdictPolicy(
        tasks=team.tasks,
        artifacts=team.artifacts,
        decisions=team.decisions,
        verdicts=team.verdicts,
    )
    return team


def _blocker(rule: str, item: str) -> Verdict:
    """A failing task verdict with one blocker on ``item``."""
    return Verdict(
        run_id="x",
        scope=RuleScope.TASK,
        passed=False,
        findings=[Finding(rule_id=rule, severity=Severity.BLOCKER, item=item, message="bad")],
    )


async def test_failed_task_is_retried_by_the_real_policy(tmp_path: Path) -> None:
    team = _with_policy(make_team(tmp_path))
    team.auditor.fail_next["prd"].append(_blocker("R3", "REQ-1"))
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done", reply.text
    assert reply.run_id
    prd = next(t for t in await team.tasks.list_tasks(reply.run_id) if t.output_kind == "prd")
    assert prd.status is TaskStatus.DONE and prd.attempt == 2
    assert any(d.kind is DecisionKind.RETRY for d in team.decisions.items)


async def test_failed_gate_is_routed_by_the_real_policy(tmp_path: Path) -> None:
    team = _with_policy(make_team(tmp_path))
    passing = team.auditor.audit_gate
    failed: list[Phase] = []

    async def gate(run: Run, phase: Phase) -> Verdict:
        """Fail the PRD gate once, on a requirement the PRD defines."""
        if phase is Phase.PRD and not failed:
            failed.append(phase)
            return Verdict(
                run_id=run.id,
                scope=RuleScope.PHASE,
                phase=phase,
                passed=False,
                findings=[
                    Finding(rule_id="R3", severity=Severity.BLOCKER, item="REQ-1", message="vague")
                ],
            )
        return await passing(run, phase)

    team.auditor.audit_gate = gate  # type: ignore[method-assign]
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    kinds = [d.kind for d in team.decisions.items]
    assert reply.kind == "done", reply.text
    assert DecisionKind.GATE_FIX in kinds and DecisionKind.REOPEN_PHASE in kinds
    prds = [t for t in await team.tasks.list_tasks(reply.run_id) if t.output_kind == "prd"]
    assert [t.status for t in prds] == [TaskStatus.SUPERSEDED, TaskStatus.DONE]
    assert team.auditor.gates.count(Phase.PRD) == 1  # the fixed PRD passed its next gate


async def test_tool_events_read_back_as_tool_calls(tmp_path: Path) -> None:
    team = make_team(tmp_path)
    await team.orch.handle("Build a books API")
    after = [e for e in team.bus.events if e.name is EventName.TOOL_CALL_AFTER]
    calls = [tool_call_from_event(e) for e in after]
    assert calls and all(c is not None for c in calls)
    write = next(c for c in calls if c is not None and c.tool == "write_file")
    assert write.args["path"] and "created" in write.result and not write.blocked
