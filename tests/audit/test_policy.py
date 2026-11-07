"""One test per branch of DESIGN.md's "What a verdict triggers" list, plus caps."""

from __future__ import annotations

from datetime import timedelta

from agent_team.audit.policy import (
    CodeVerdictPolicy,
    PolicyLimits,
    blame_code,
    decide_task,
    planned_in,
)
from agent_team.core.artifacts import (
    ArtifactRef,
    CodeChange,
    Component,
    QAReport,
    Spec,
    WorkPackage,
)
from agent_team.core.models import (
    AUDIT_RULE,
    AgentName,
    Decision,
    DecisionKind,
    Finding,
    Phase,
    RuleScope,
    Run,
    Severity,
    Task,
    TaskOrigin,
    TaskStatus,
    Verdict,
    utcnow,
)

from .fakes import FakeArtifactStore, FakeDecisionLog, FakeTaskStore, FakeVerdictStore, make_task

LIMITS = PolicyLimits(max_upstream_reruns=2, stronger_model="strong-1")


def blocker(rule: str, item: str | None = None, artifact: str | None = None) -> Finding:
    ref = ArtifactRef.parse(artifact) if artifact else None
    return Finding(rule_id=rule, severity=Severity.BLOCKER, item=item, artifact=ref, message="bad")


def verdict(run: Run, *findings: Finding, task: Task | None = None) -> Verdict:
    return Verdict(
        run_id=run.id,
        scope=RuleScope.TASK if task else RuleScope.RUN,
        task_id=task.id if task else None,
        phase=task.phase if task else None,
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
        "attempt": task.attempt + 1,  # the attempt that runs on the stronger model
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
        origin=TaskOrigin.REVISION,
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


def test_a_finding_on_a_call_that_already_ran_asks_instead_of_retrying(run: Run) -> None:
    task = make_task(run, AgentName.RESEARCHER, Phase.PLANNING, "research_brief", attempt=1)
    logged = blocker("S1", "tool call read_file:3").model_copy(update={"already_happened": True})
    decisions = decide_task(run, task, verdict(run, logged, task=task), [], [], LIMITS)
    assert kinds(decisions) == [DecisionKind.ESCALATE, DecisionKind.ASK_USER]
    assert decisions[1].rule == "policy.already_happened"
    assert "can't undo" in str(decisions[1].inputs["question"])


def test_fail_closed_verdict_escalates_and_asks(run: Run) -> None:
    task = spec_task(run)
    v = verdict(run, blocker(AUDIT_RULE), task=task)
    assert kinds(decide_task(run, task, v, [], [], LIMITS)) == [
        DecisionKind.ESCALATE,
        DecisionKind.ASK_USER,
    ]


# --- faults in other tasks' outputs (a QA task's end check) --------------------------


def impl_tasks(run: Run) -> tuple[list[Task], dict[str, CodeChange | QAReport]]:
    """Two Engineer tasks and the QA task whose end check is being decided."""
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
        status=TaskStatus.AUDITING,
        output=ArtifactRef.parse("qa_report@v1"),
        depends_on=[e1.id, e2.id],
        created_at=t0 + timedelta(seconds=2),
    )
    outputs: dict[str, CodeChange | QAReport] = {
        e1.id: CodeChange(summary="s", version=1, files_written=["app/db.py"]),
        e2.id: CodeChange(summary="s", version=2, files_written=["app/routes.py"]),
        qa.id: QAReport(summary="s", version=1, results=[], ran_end_to_end=True),
    }
    return [e1, e2, qa], outputs


def qa_check(
    run: Run,
    tasks: list[Task],
    outputs: dict[str, CodeChange | QAReport],
    *findings: Finding,
    history: list[Verdict] | None = None,
) -> list[Decision]:
    """Decide the end check of the QA task (the last of ``tasks``) with ``findings``."""
    qa = tasks[-1]
    v = verdict(run, *findings, task=qa)
    return decide_task(run, qa, v, history or [], [], LIMITS, tasks, outputs)


def test_a_fault_in_another_task_is_fixed_there_and_later_tasks_rerun(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    e1, e2, qa = tasks
    v = verdict(run, blocker("S2", "app/db.py", "code_change@v1"), task=qa)
    [fix] = decide_task(run, qa, v, [], [], LIMITS, tasks, outputs)
    assert fix.kind is DecisionKind.FIX and fix.rule == "policy.fix"
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
        task=qa,
    )
    [fix] = decide_task(run, qa, v, [], [], LIMITS, tasks, outputs)
    assert fix.inputs["task_id"] == e2.id  # wrote routes.py; the AC goes to the last engineer
    assert len(fix.inputs["finding_ids"]) == 2
    assert fix.inputs["rerun_after"] == [qa.id]


def test_qa_own_fault_is_retried_on_qa(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    check = qa_check(run, tasks, outputs, blocker("R1", "results", "qa_report@v1"))
    assert kinds(check) == [DecisionKind.RETRY]


def test_a_finding_by_file_without_artifact_goes_to_its_writer(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    [fix] = qa_check(run, tasks, outputs, blocker("R5", "app/db.py"))
    assert fix.inputs["task_id"] == tasks[0].id


def test_an_upstream_fault_reopens_the_earlier_phase(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    qa = tasks[-1]
    arch = make_task(
        run, AgentName.ARCHITECT, Phase.SPEC, "spec", output=ArtifactRef.parse("spec@v1")
    )
    v = verdict(run, blocker("R10", "GET /books", "spec@v1"), task=qa)
    decisions = decide_task(run, qa, v, [], [], LIMITS, [*tasks, arch], outputs)
    assert kinds(decisions) == [DecisionKind.ESCALATE, DecisionKind.UPSTREAM_RERUN]
    d = decisions[1]
    assert d.inputs["artifact"] == "spec@v1" and d.inputs["producer_task_id"] == arch.id
    assert d.inputs["phase"] == "spec" and "gate" not in d.inputs
    # the QA task re-run after that (its copy points back at it) still fails: ask
    copy = qa.model_copy(update={"id": "task_copy", "parent_task_id": qa.id})
    again = verdict(run, blocker("R10", "GET /books", "spec@v2"), task=copy)
    out = decide_task(run, copy, again, [], [d], LIMITS, [*tasks, arch], outputs)
    assert out[-1].kind is DecisionKind.ASK_USER


def test_the_same_failure_after_a_fix_is_no_progress_and_asks(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    first = verdict(run, blocker("S2", "app/db.py", "code_change@v1"), task=tasks[-1])
    again = blocker("S2", "app/db.py", "code_change@v3")
    decisions = qa_check(run, tasks, outputs, again, history=[first])
    assert kinds(decisions) == [DecisionKind.NO_PROGRESS, DecisionKind.ASK_USER]


def test_a_fix_for_a_task_without_attempts_left_asks_the_user(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    tasks[0] = tasks[0].model_copy(update={"attempt": 3})
    [ask] = qa_check(run, tasks, outputs, blocker("S2", "app/db.py", "code_change@v1"))
    assert ask.kind is DecisionKind.ASK_USER and ask.rule == "policy.fix_cap"


def test_qa_without_a_report_is_its_own_fault(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    for rule in ("R1", "R7"):  # no report, or one R7 can't read: nothing names an artifact
        check = qa_check(run, tasks, outputs, blocker(rule, "qa_report"))
        assert kinds(check) == [DecisionKind.RETRY], rule


def test_the_checkers_own_blockers_ride_on_the_fix(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    e1, _, qa = tasks
    v = verdict(
        run,
        blocker("S2", "app/db.py", "code_change@v1"),
        blocker("R5", "app/db.py"),
        blocker("R11", "Q-1", "qa_report@v1"),
        task=qa,
    )
    [fix] = decide_task(run, qa, v, [], [], LIMITS, tasks, outputs)
    assert fix.inputs["task_id"] == e1.id
    assert fix.inputs["finding_ids"] == [v.findings[0].id, v.findings[1].id]
    assert fix.inputs["checker"] == qa.id
    assert fix.inputs["checker_finding_ids"] == [v.findings[2].id]


def test_a_mixed_fix_with_one_task_out_of_attempts_only_asks(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    tasks[0] = tasks[0].model_copy(update={"attempt": 3})
    decisions = qa_check(
        run,
        tasks,
        outputs,
        blocker("S2", "app/db.py", "code_change@v1"),
        blocker("S2", "app/routes.py", "code_change@v2"),
    )
    # A fix for the other task would supersede the QA task the question hangs on.
    assert kinds(decisions) == [DecisionKind.ASK_USER]
    assert decisions[0].task_id == tasks[-1].id


def test_a_requested_brief_in_a_later_phase_is_not_upstream_of_itself(run: Run) -> None:
    brief = make_task(
        run,
        AgentName.RESEARCHER,
        Phase.PRD,
        "research_brief",
        attempt=1,
        output=ArtifactRef.parse("research_brief@v3"),
    )
    v = verdict(run, blocker("R2", "claim 1", "research_brief@v3"), task=brief)
    assert kinds(decide_task(run, brief, v, [], [], LIMITS, [brief], {})) == [DecisionKind.RETRY]


def test_a_brief_blaming_the_plan_asks_instead_of_rebuilding_it(run: Run) -> None:
    planner = make_task(
        run,
        AgentName.ORCHESTRATOR,
        Phase.PLANNING,
        "plan",
        status=TaskStatus.DONE,
        output=ArtifactRef.parse("plan@v1"),
    )
    brief = make_task(
        run,
        AgentName.RESEARCHER,
        Phase.PLANNING,
        "research_brief",
        attempt=3,
        output=ArtifactRef.parse("research_brief@v1"),
    )
    v = verdict(run, blocker("P2", "Which database?", "plan@v1"), task=brief)
    decisions = decide_task(run, brief, v, [], [], LIMITS, [planner, brief], {})
    assert DecisionKind.FIX not in kinds(decisions)
    assert kinds(decisions)[0] is DecisionKind.ESCALATE


def test_superseded_tasks_are_never_fix_targets(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    stale = tasks[0].model_copy(update={"status": TaskStatus.SUPERSEDED})
    finding = blocker("S2", "app/db.py", "code_change@v1")
    decisions = qa_check(run, [stale, *tasks[1:]], outputs, finding)
    assert all(d.task_id != stale.id for d in decisions)


# --- the VerdictPolicy implementation over the stores --------------------------------


async def test_policy_class_reads_the_stores(run: Run) -> None:
    store, artifacts, log = FakeTaskStore(), FakeArtifactStore(), FakeDecisionLog()
    verdicts = FakeVerdictStore()
    tasks, outputs = impl_tasks(run)
    for t in tasks:
        await store.create_task(t)
        await artifacts.save(outputs[t.id].model_copy(update={"run_id": run.id}))
    policy = CodeVerdictPolicy(
        tasks=store,
        artifacts=artifacts,
        decisions=log,
        verdicts=verdicts,
        limits=PolicyLimits.from_config(
            {"limits": {"max_upstream_reruns": 1}, "stronger_model": {"model": "strong-2"}}
        ),
    )
    assert policy.limits == PolicyLimits(max_upstream_reruns=1, stronger_model="strong-2")

    # The phase's outputs come from the store, so a finding by file finds its writer.
    qa = tasks[-1]
    check = verdict(run, blocker("R5", "app/routes.py"), task=qa)
    [fix] = await policy.on_task_verdict(run, qa, check, [])
    assert fix.kind is DecisionKind.FIX and fix.task_id == tasks[1].id

    task = tasks[1].model_copy(update={"attempt": 3})
    v = verdict(run, blocker("R1", "x", "code_change@v2"), task=task)
    out = await policy.on_task_verdict(run, task, v, [])
    assert kinds(out) == [DecisionKind.ESCALATE, DecisionKind.STRONGER_MODEL]
    assert out[1].inputs["model"] == "strong-2"


def test_dangerous_action_wins_over_a_fail_closed_verdict(run: Run) -> None:
    task = make_task(run, AgentName.ENGINEER, Phase.IMPLEMENTATION, "code_change", attempt=1)
    v = verdict(run, blocker(AUDIT_RULE), blocker("S5", "run_command"), task=task)
    assert kinds(decide_task(run, task, v, [], [], LIMITS)) == [
        DecisionKind.STOP,
        DecisionKind.ASK_USER,
    ]


def test_a_passed_retry_resets_no_progress(run: Run) -> None:
    task = spec_task(run, attempt=2)
    failed = verdict(run, blocker("R4", "REQ-2", "spec@v1"), task=task)
    passed = verdict(run, task=task)
    again = verdict(run, blocker("R4", "REQ-2", "spec@v3"), task=task)
    assert kinds(decide_task(run, task, again, [failed, passed], [], LIMITS)) == [
        DecisionKind.RETRY
    ]


def _planned_spec() -> Spec:
    """A spec of two work packages, where the api package plans REQ-10 in recent."""
    return Spec(
        summary="s",
        version=2,
        components=[
            Component(name="db", responsibility="r", requirements=["REQ-1"], files=["app/db.py"]),
            Component(
                name="recent",
                responsibility="r",
                requirements=["REQ-10"],
                files=["app/recent.py", "tests/test_recent.py"],
            ),
        ],
        checks=[],
        work_packages=[
            WorkPackage(name="db", components=["db"]),
            WorkPackage(name="api", components=["recent"]),
        ],
    )


def _packaged(tasks: list[Task]) -> list[Task]:
    """``impl_tasks`` with the Engineer tasks made for the db and api packages."""
    e1, e2, qa = tasks
    return [
        e1.model_copy(update={"reason": "work package db"}),
        e2.model_copy(update={"reason": "work package api"}),
        qa,
    ]


def missing(item: str, message: str, artifact: str = "spec@v2") -> Finding:
    """A QA blocker that something is missing, blamed on ``artifact``."""
    return Finding(
        rule_id="R7",
        severity=Severity.BLOCKER,
        item=item,
        artifact=ArtifactRef.parse(artifact),
        message=message,
    )


def test_a_missing_requirement_the_spec_plans_is_the_codes_fault(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    tasks = _packaged(tasks)
    qa = tasks[-1]
    f = missing("REQ-10", "list_recent_notes is not implemented; app/recent.py is missing")
    decisions = decide_task(
        run, qa, verdict(run, f, task=qa), [], [], LIMITS, tasks, outputs, spec=_planned_spec()
    )
    assert kinds(decisions) == [DecisionKind.FIX]
    # The Engineer of the package that builds the component, not the last one by chance.
    assert decisions[0].task_id == tasks[1].id
    assert decisions[0].inputs["blamed_code"] == [f.id]
    # Named by a planned file alone, or blamed on the PRD, it is the code's fault too.
    by_file = missing("server", "tests/test_recent.py does not exist", "prd@v2")
    out = decide_task(
        run, qa, verdict(run, by_file, task=qa), [], [], LIMITS, tasks, outputs, _planned_spec()
    )
    assert kinds(out) == [DecisionKind.FIX] and out[0].task_id == tasks[1].id


def test_a_requirement_the_spec_lacks_still_reopens_the_spec(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    qa = tasks[-1]
    arch = make_task(
        run, AgentName.ARCHITECT, Phase.SPEC, "spec", output=ArtifactRef.parse("spec@v2")
    )
    spec = _planned_spec()
    # REQ-11 isn't planned (and REQ-1 doesn't match it), and a wrong or incomplete
    # plan isn't missing code.
    for f in (
        missing("REQ-11", "REQ-11 is not implemented"),
        missing("REQ-10", "REQ-10 contradicts the PRD's REQ-1"),
        missing("REQ-10", "REQ-10 is missing an acceptance criterion", "prd@v2"),
    ):
        out = decide_task(
            run, qa, verdict(run, f, task=qa), [], [], LIMITS, [*tasks, arch], outputs, spec
        )
        assert kinds(out) == [DecisionKind.ESCALATE, DecisionKind.UPSTREAM_RERUN], f.message
        assert "blamed_code" not in out[0].inputs
    # Without the current spec, nothing is moved.
    f = missing("REQ-10", "REQ-10 is not implemented")
    out = decide_task(run, qa, verdict(run, f, task=qa), [], [], LIMITS, [*tasks, arch], outputs)
    assert kinds(out)[-1] is DecisionKind.UPSTREAM_RERUN


def test_an_engineers_own_check_moves_only_its_own_packages_misses(run: Run) -> None:
    tasks, outputs = impl_tasks(run)
    db, api, _ = _packaged(tasks)
    spec = _planned_spec()
    f = missing("REQ-10", "REQ-10 is not implemented; app/recent.py is missing")
    # The api Engineer builds recent, so its own check retries its own code.
    mine = decide_task(
        run, api, verdict(run, f, task=api), [], [], LIMITS, [db, api], outputs, spec
    )
    assert kinds(mine) == [DecisionKind.RETRY] and mine[0].inputs["blamed_code"] == [f.id]
    # The db Engineer doesn't: that stays a question for the spec, as before.
    arch = make_task(
        run, AgentName.ARCHITECT, Phase.SPEC, "spec", output=ArtifactRef.parse("spec@v2")
    )
    other = decide_task(
        run, db, verdict(run, f, task=db), [], [], LIMITS, [db, api, arch], outputs, spec
    )
    assert kinds(other) == [DecisionKind.ESCALATE, DecisionKind.UPSTREAM_RERUN]


def test_a_moved_finding_keeps_its_evidence(run: Run) -> None:
    tasks, _ = impl_tasks(run)
    qa = _packaged(tasks)[-1]
    f = missing("REQ-10", "REQ-10 is not implemented").model_copy(
        update={"evidence": "pytest: no tool list_recent_notes"}
    )
    blockers, moved = blame_code([f], qa, _planned_spec(), _packaged(tasks), _packaged(tasks))
    assert moved == [f.id]
    assert blockers[0].evidence is not None
    assert blockers[0].evidence.startswith("pytest: no tool list_recent_notes\nspec@v2 plans it")


def test_planned_in_matches_whole_ids_and_file_names() -> None:
    spec = _planned_spec()
    assert planned_in(spec, missing("REQ-10", "not implemented")) == ["recent"]
    # A requirement that lacks something in the PRD or spec isn't missing code.
    assert planned_in(spec, missing("REQ-10", "REQ-10 is missing an acceptance criterion")) == []
    assert planned_in(spec, missing("x", "recent.py is missing")) == ["recent"]
    assert planned_in(spec, missing("x", "REQ-100 is missing")) == []
    assert planned_in(spec, missing("x", "app/recent.pyc is missing")) == []
