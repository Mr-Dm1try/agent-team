"""One test per branch of DESIGN.md's "What a verdict triggers" list, plus caps."""

from __future__ import annotations

from datetime import timedelta

from agent_team.audit.auditor import AUDIT_RULE
from agent_team.audit.policy import CodeVerdictPolicy, PolicyLimits, decide_gate, decide_task
from agent_team.core.artifacts import ArtifactRef, CodeChange, QAReport
from agent_team.core.models import (
    AgentName,
    Decision,
    DecisionKind,
    Finding,
    Phase,
    RuleScope,
    Run,
    Severity,
    Task,
    TaskStatus,
    Verdict,
    utcnow,
)

from .fakes import FakeArtifactStore, FakeDecisionLog, FakeTaskStore, make_task

LIMITS = PolicyLimits(max_upstream_reruns=2, stronger_model="strong-1")


def blocker(rule: str, item: str | None = None, artifact: str | None = None) -> Finding:
    ref = ArtifactRef.parse(artifact) if artifact else None
    return Finding(rule_id=rule, severity=Severity.BLOCKER, item=item, artifact=ref, message="bad")


def verdict(
    run: Run, *findings: Finding, task: Task | None = None, phase: Phase | None = None
) -> Verdict:
    return Verdict(
        run_id=run.id,
        scope=RuleScope.TASK if task else RuleScope.PHASE,
        task_id=task.id if task else None,
        phase=phase,
        passed=not any(f.severity is Severity.BLOCKER for f in findings),
        findings=list(findings),
    )


def kinds(decisions: list[Decision]) -> list[DecisionKind]:
    return [d.kind for d in decisions]


def spec_task(run: Run, **kw: object) -> Task:
    data: dict[str, object] = {
        "attempt": 1,
        "output": ArtifactRef.parse("spec@v1"),
        "model": "small",
    }
    data.update(kw)
    return make_task(run, AgentName.ARCHITECT, Phase.SPEC, "spec", **data)


# --- task verdicts -------------------------------------------------------------------


def test_pass_returns_no_decisions(run: Run) -> None:
    task = spec_task(run)
    assert decide_task(run, task, verdict(run, task=task), [], [], LIMITS) == []
    warning = Finding(rule_id="R12", severity=Severity.WARNING, message="w")
    assert decide_task(run, task, verdict(run, warning, task=task), [], [], LIMITS) == []


def test_fail_with_attempts_left_retries_as_a_revision(run: Run) -> None:
    task = spec_task(run)
    v = verdict(run, blocker("R4", "REQ-2", "spec@v1"), task=task)
    [d] = decide_task(run, task, v, [], [], LIMITS)
    assert d.kind is DecisionKind.RETRY and d.made_by == "code"
    assert d.task_id == d.inputs["task_id"] == task.id
    assert d.inputs["finding_ids"] == [f.id for f in v.findings]
    assert d.inputs["previous_output"] == "spec@v1"
    assert d.rule == "policy.retry_as_revision"


def test_same_failure_twice_is_no_progress_and_escalates(run: Run) -> None:
    task = spec_task(run, attempt=2)
    first = verdict(run, blocker("R4", "REQ-2", "spec@v1"), task=task)
    second = verdict(run, blocker("R4", "REQ-2", "spec@v2"), task=task)  # new version, same fault
    decisions = decide_task(run, task, second, [first], [], LIMITS)
    assert kinds(decisions) == [
        DecisionKind.NO_PROGRESS,
        DecisionKind.ESCALATE,
        DecisionKind.STRONGER_MODEL,
    ]
    # a different failure is progress, so it retries
    other = verdict(run, blocker("R4", "AC-1.1", "spec@v2"), task=task)
    assert kinds(decide_task(run, task, other, [first], [], LIMITS)) == [DecisionKind.RETRY]


def test_no_attempts_left_own_fault_gets_the_stronger_model(run: Run) -> None:
    task = spec_task(run, attempt=3)
    v = verdict(run, blocker("R4", "REQ-2", "spec@v3"), task=task)
    decisions = decide_task(run, task, v, [], [], LIMITS)
    assert kinds(decisions) == [DecisionKind.ESCALATE, DecisionKind.STRONGER_MODEL]
    assert decisions[1].inputs == {
        "task_id": task.id,
        "model": "strong-1",
        "finding_ids": [v.findings[0].id],
    }


def test_after_the_stronger_model_the_user_is_asked(run: Run) -> None:
    task = spec_task(run, attempt=4, model="strong-1")
    v = verdict(run, blocker("R4", "REQ-2", "spec@v4"), task=task)
    decisions = decide_task(run, task, v, [], [], LIMITS)
    assert kinds(decisions) == [DecisionKind.ESCALATE, DecisionKind.ASK_USER]
    assert decisions[1].inputs["task_id"] == task.id and "question" in decisions[1].inputs
    # also recognized from the decision log when the model name isn't on the task
    used = Decision(
        run_id=run.id,
        task_id=task.id,
        kind=DecisionKind.STRONGER_MODEL,
        choice="c",
        reason="r",
        made_by="code",
    )
    task2 = spec_task(run, attempt=4, id=task.id)
    assert kinds(decide_task(run, task2, v, [], [used], LIMITS))[-1] is DecisionKind.ASK_USER


def test_no_attempts_left_upstream_fault_reopens_the_earliest_artifact(run: Run) -> None:
    prd_task = make_task(
        run, AgentName.PRODUCT, Phase.PRD, "prd", output=ArtifactRef.parse("prd@v2")
    )
    eng = make_task(run, AgentName.ENGINEER, Phase.IMPLEMENTATION, "code_change", attempt=3)
    v = verdict(
        run,
        blocker("R1", "x", "spec@v1"),
        blocker("R1", "y", "prd@v2"),
        blocker("R1", "z", "code_change@v3"),
        task=eng,
    )
    decisions = decide_task(run, eng, v, [], [], LIMITS, tasks=[prd_task, eng])
    assert kinds(decisions) == [DecisionKind.ESCALATE, DecisionKind.UPSTREAM_RERUN]
    rerun = decisions[1].inputs
    assert rerun["task_id"] == eng.id
    assert rerun["artifact"] == "prd@v2"  # the earliest at fault
    assert rerun["phase"] == "prd" and rerun["producer_task_id"] == prd_task.id


def test_still_failing_after_an_upstream_rerun_asks_the_user(run: Run) -> None:
    eng = make_task(run, AgentName.ENGINEER, Phase.IMPLEMENTATION, "code_change", attempt=3)
    v = verdict(run, blocker("R1", "x", "spec@v2"), task=eng)
    earlier = Decision(
        run_id=run.id,
        task_id=eng.id,
        kind=DecisionKind.UPSTREAM_RERUN,
        inputs={"artifact": "spec@v1"},
        choice="c",
        reason="r",
        made_by="code",
    )
    decisions = decide_task(run, eng, v, [], [earlier], LIMITS)
    assert kinds(decisions) == [DecisionKind.ESCALATE, DecisionKind.ASK_USER]
    assert decisions[1].rule == "policy.ask_after_upstream_rerun"
    # a revision task created from the escalated one (parent link) counts too
    child = make_task(
        run,
        AgentName.ENGINEER,
        Phase.IMPLEMENTATION,
        "code_change",
        attempt=3,
        parent_task_id=eng.id,
    )
    assert kinds(decide_task(run, child, v, [], [earlier], LIMITS))[-1] is DecisionKind.ASK_USER


def test_upstream_rerun_cap_asks_the_user(run: Run) -> None:
    capped = run.model_copy(update={"upstream_reruns": 2})
    eng = make_task(capped, AgentName.ENGINEER, Phase.IMPLEMENTATION, "code_change", attempt=3)
    v = verdict(capped, blocker("R1", "x", "spec@v2"), task=eng)
    decisions = decide_task(capped, eng, v, [], [], LIMITS)
    assert (
        decisions[-1].kind is DecisionKind.ASK_USER and decisions[-1].rule == "policy.upstream_cap"
    )


def test_dangerous_action_stops_the_run(run: Run) -> None:
    task = make_task(run, AgentName.ENGINEER, Phase.IMPLEMENTATION, "code_change", attempt=1)
    v = verdict(run, blocker("S5", "run_command"), task=task)
    decisions = decide_task(run, task, v, [], [], LIMITS)
    assert kinds(decisions) == [DecisionKind.STOP, DecisionKind.ASK_USER]


def test_fail_closed_verdict_escalates_and_asks(run: Run) -> None:
    task = spec_task(run)
    v = verdict(run, blocker(AUDIT_RULE), task=task)
    assert kinds(decide_task(run, task, v, [], [], LIMITS)) == [
        DecisionKind.ESCALATE,
        DecisionKind.ASK_USER,
    ]


# --- gate verdicts -------------------------------------------------------------------


def impl_tasks(run: Run) -> tuple[list[Task], dict[str, CodeChange | QAReport]]:
    t0 = utcnow()
    e1 = make_task(
        run,
        AgentName.ENGINEER,
        Phase.IMPLEMENTATION,
        "code_change",
        attempt=1,
        status=TaskStatus.DONE,
        output=ArtifactRef.parse("code_change@v1"),
        created_at=t0,
    )
    e2 = make_task(
        run,
        AgentName.ENGINEER,
        Phase.IMPLEMENTATION,
        "code_change",
        attempt=1,
        status=TaskStatus.DONE,
        output=ArtifactRef.parse("code_change@v2"),
        depends_on=[e1.id],
        created_at=t0 + timedelta(seconds=1),
    )
    qa = make_task(
        run,
        AgentName.QA,
        Phase.IMPLEMENTATION,
        "qa_report",
        attempt=1,
        status=TaskStatus.DONE,
        output=ArtifactRef.parse("qa_report@v1"),
        depends_on=[e2.id],
        created_at=t0 + timedelta(seconds=2),
    )
    outputs: dict[str, CodeChange | QAReport] = {
        e1.id: CodeChange(summary="s", version=1, files_written=["app/db.py"]),
        e2.id: CodeChange(summary="s", version=2, files_written=["app/routes.py"]),
        qa.id: QAReport(summary="s", version=1, results=[], ran_end_to_end=True),
    }
    return [e1, e2, qa], outputs


def test_gate_pass_returns_no_decisions(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    v = verdict(run, phase=Phase.IMPLEMENTATION)
    assert decide_gate(run, Phase.IMPLEMENTATION, v, None, tasks, outputs, [], LIMITS) == []


def test_gate_failure_routes_each_finding_to_its_producer(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    e1, e2, qa = tasks
    v = verdict(run, blocker("S2", "app/db.py", "code_change@v1"), phase=Phase.IMPLEMENTATION)
    [fix] = decide_gate(run, Phase.IMPLEMENTATION, v, None, tasks, outputs, [], LIMITS)
    assert fix.kind is DecisionKind.GATE_FIX
    assert fix.inputs["task_id"] == e1.id
    assert fix.inputs["finding_ids"] == [v.findings[0].id]
    assert fix.inputs["rerun_after"] == [e2.id, qa.id]  # later tasks re-run too


def test_qa_blockers_become_engineer_fixes_then_a_fresh_qa(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    _, e2, qa = tasks
    v = verdict(
        run,
        blocker("R7", "app/routes.py", "qa_report@v1"),
        blocker("R7", "AC-1.1", "qa_report@v1"),
        phase=Phase.IMPLEMENTATION,
    )
    [fix] = decide_gate(run, Phase.IMPLEMENTATION, v, None, tasks, outputs, [], LIMITS)
    assert fix.inputs["task_id"] == e2.id  # wrote routes.py; the AC goes to the last engineer
    assert len(fix.inputs["finding_ids"]) == 2
    assert fix.inputs["rerun_after"] == [qa.id]


def test_gate_finding_by_file_without_artifact(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    v = verdict(run, blocker("R5", "app/db.py"), phase=Phase.IMPLEMENTATION)
    [fix] = decide_gate(run, Phase.IMPLEMENTATION, v, None, tasks, outputs, [], LIMITS)
    assert fix.inputs["task_id"] == tasks[0].id


def test_gate_upstream_fault_reopens_the_earlier_phase(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    arch = make_task(
        run, AgentName.ARCHITECT, Phase.SPEC, "spec", output=ArtifactRef.parse("spec@v1")
    )
    v = verdict(run, blocker("R10", "GET /books", "spec@v1"), phase=Phase.IMPLEMENTATION)
    [d] = decide_gate(run, Phase.IMPLEMENTATION, v, None, [*tasks, arch], outputs, [], LIMITS)
    assert d.kind is DecisionKind.UPSTREAM_RERUN
    assert d.inputs["artifact"] == "spec@v1" and d.inputs["producer_task_id"] == arch.id
    assert d.inputs["gate"] == "implementation"
    # and if the gate still fails after that re-run, ask the user
    again = verdict(run, blocker("R10", "GET /books", "spec@v2"), phase=Phase.IMPLEMENTATION)
    [ask] = decide_gate(
        run, Phase.IMPLEMENTATION, again, None, [*tasks, arch], outputs, [d], LIMITS
    )
    assert ask.kind is DecisionKind.ASK_USER


def test_gate_no_progress_asks_the_user(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    first = verdict(run, blocker("S2", "app/db.py", "code_change@v1"), phase=Phase.IMPLEMENTATION)
    second = verdict(run, blocker("S2", "app/db.py", "code_change@v3"), phase=Phase.IMPLEMENTATION)
    decisions = decide_gate(run, Phase.IMPLEMENTATION, second, first, tasks, outputs, [], LIMITS)
    assert kinds(decisions) == [DecisionKind.NO_PROGRESS, DecisionKind.ASK_USER]


def test_gate_fix_cap_asks_the_user(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    tasks[0] = tasks[0].model_copy(update={"attempt": 3})
    v = verdict(run, blocker("S2", "app/db.py", "code_change@v1"), phase=Phase.IMPLEMENTATION)
    [ask] = decide_gate(run, Phase.IMPLEMENTATION, v, None, tasks, outputs, [], LIMITS)
    assert ask.kind is DecisionKind.ASK_USER and ask.rule == "policy.gate_cap"


def test_gate_fail_closed_asks_the_user(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    v = verdict(run, blocker(AUDIT_RULE), phase=Phase.IMPLEMENTATION)
    assert kinds(decide_gate(run, Phase.IMPLEMENTATION, v, None, tasks, outputs, [], LIMITS)) == [
        DecisionKind.ASK_USER
    ]


# --- the VerdictPolicy implementation over the stores --------------------------------


async def test_policy_class_reads_the_stores(run: Run) -> None:
    store, artifacts, log = FakeTaskStore(), FakeArtifactStore(), FakeDecisionLog()
    tasks, outputs = impl_tasks(run)
    for t in tasks:
        await store.create_task(t)
        await artifacts.save(outputs[t.id].model_copy(update={"run_id": run.id}))
    policy = CodeVerdictPolicy(
        tasks=store,
        artifacts=artifacts,
        decisions=log,
        limits=PolicyLimits.from_config(
            {"limits": {"max_upstream_reruns": 1}, "stronger_model": {"model": "strong-2"}}
        ),
    )
    assert policy.limits == PolicyLimits(max_upstream_reruns=1, stronger_model="strong-2")

    gate = verdict(run, blocker("S2", "app/db.py", "code_change@v1"), phase=Phase.IMPLEMENTATION)
    assert kinds(await policy.on_gate_verdict(run, Phase.IMPLEMENTATION, gate)) == [
        DecisionKind.GATE_FIX
    ]
    # the policy remembers the last gate verdict, so the same failure again is no progress
    again = await policy.on_gate_verdict(run, Phase.IMPLEMENTATION, gate.model_copy())
    assert kinds(again) == [DecisionKind.NO_PROGRESS, DecisionKind.ASK_USER]

    task = tasks[1].model_copy(update={"attempt": 3})
    v = verdict(run, blocker("R1", "x", "code_change@v2"), task=task)
    out = await policy.on_task_verdict(run, task, v, [])
    assert kinds(out) == [DecisionKind.ESCALATE, DecisionKind.STRONGER_MODEL]
    assert out[1].inputs["model"] == "strong-2"
