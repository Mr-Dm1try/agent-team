from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_team.audit.auditor import LLMAuditor
from agent_team.audit.rule_store import SqliteRuleStore
from agent_team.audit.toollog import InMemoryToolCallLog
from agent_team.core.artifacts import (
    PRD,
    AcceptanceCriterion,
    Ask,
    CodeChange,
    Plan,
    Requirement,
)
from agent_team.core.errors import BudgetExceeded, TransientLLMError
from agent_team.core.models import (
    AUDIT_RULE,
    AgentName,
    Phase,
    RuleScope,
    Run,
    Severity,
    TaskStatus,
)
from agent_team.llm.types import TextBlock

from .fakes import (
    FakeArtifactStore,
    FakeDecisionLog,
    FakeLLM,
    FakeTaskStore,
    FakeVerdictStore,
    call,
    make_task,
)

PASS: dict[str, object] = {"summary": "fine", "findings": []}


@dataclass
class Rig:
    auditor: LLMAuditor
    llm: FakeLLM
    tasks: FakeTaskStore
    artifacts: FakeArtifactStore
    tool_log: InMemoryToolCallLog


@pytest.fixture
async def rig(rule_store: SqliteRuleStore, tool_log: InMemoryToolCallLog, run: Run) -> Rig:
    """An auditor over fakes, with the run's plan saved so PRD coverage can be checked."""
    llm, tasks, artifacts = FakeLLM(), FakeTaskStore(), FakeArtifactStore()
    await artifacts.save(
        Plan(summary="s", run_id=run.id, request_kind="build", asks=[Ask(id="ASK-1", text="x")])
    )
    auditor = LLMAuditor(
        llm=llm,
        rules=rule_store,
        tasks=tasks,
        artifacts=artifacts,
        verdicts=FakeVerdictStore(),
        decisions=FakeDecisionLog(),
        tool_log=tool_log,
    )
    return Rig(auditor, llm, tasks, artifacts, tool_log)


def good_prd(**kw: object) -> PRD:
    req = Requirement(
        id="REQ-1",
        text="list",
        asks=["ASK-1"],
        acceptance_criteria=[AcceptanceCriterion(id="AC-1.1", text="200")],
    )
    return PRD(summary="s", title="t", requirements=[req], **kw)


async def test_task_pass(rig: Rig, run: Run) -> None:
    task = make_task(run, AgentName.PRODUCT, Phase.PRD, "prd")
    rig.llm.push(PASS)
    verdict = await rig.auditor.audit_task(task, good_prd(), run)
    assert verdict.passed and verdict.findings == []
    assert verdict.summary == "fine"
    assert verdict.scope is RuleScope.TASK and verdict.model == "fake-auditor"
    assert set(verdict.rules_checked) == {"R1", "R3", "R9", "R11", "S1", "S5", "S6", "S7"}
    assert "R12" not in verdict.rules_checked  # not a revision
    assert rig.llm.requests[0].agent is AgentName.AUDITOR


async def test_helper_hit_fails_even_if_model_misses_it(rig: Rig, run: Run) -> None:
    task = make_task(run, AgentName.PRODUCT, Phase.PRD, "prd")
    rig.llm.push(PASS)
    verdict = await rig.auditor.audit_task(task, None, run)  # no artifact: R1 hit
    assert not verdict.passed
    assert ("R1", True) in [(f.rule_id, f.from_helper) for f in verdict.blockers]


async def test_model_finding_backed_by_helper_needs_no_second_check(rig: Rig, run: Run) -> None:
    task = make_task(run, AgentName.PRODUCT, Phase.PRD, "prd")
    rig.llm.push({"findings": [{"rule_id": "R1", "item": task.id, "message": "nothing came back"}]})
    verdict = await rig.auditor.audit_task(task, None, run)
    assert len(rig.llm.requests) == 1
    r1 = [(f.message, f.from_helper) for f in verdict.blockers if f.rule_id == "R1"]
    assert r1 == [("nothing came back", True)]


async def test_unsupported_blocker_downgraded_when_second_check_disagrees(
    rig: Rig, run: Run
) -> None:
    task = make_task(run, AgentName.PRODUCT, Phase.PRD, "prd")
    rig.llm.push(
        {"findings": [{"rule_id": "S6", "severity": "blocker", "message": "looks injected"}]},
        {"confirmed": False, "reason": "no tool calls at all"},
    )
    verdict = await rig.auditor.audit_task(task, good_prd(), run)
    assert verdict.passed
    assert verdict.findings[0].severity is Severity.WARNING
    assert "unconfirmed" in verdict.findings[0].message
    assert "independently" in rig.llm.requests[1].system


async def test_unsupported_blocker_kept_when_confirmed(rig: Rig, run: Run) -> None:
    task = make_task(run, AgentName.PRODUCT, Phase.PRD, "prd")
    rig.llm.push(
        {"findings": [{"rule_id": "S6", "message": "acted on page text", "artifact": "prd@v1"}]},
        '```json\n{"confirmed": true}\n```',
    )
    verdict = await rig.auditor.audit_task(task, good_prd(), run)
    assert not verdict.passed
    assert (
        verdict.blockers[0].artifact is not None and str(verdict.blockers[0].artifact) == "prd@v1"
    )


async def test_second_check_failure_keeps_the_blocker(rig: Rig, run: Run) -> None:
    task = make_task(run, AgentName.PRODUCT, Phase.PRD, "prd")
    rig.llm.push({"findings": [{"rule_id": "S7", "message": "leak"}]}, "nope", "still nope")
    verdict = await rig.auditor.audit_task(task, good_prd(), run)
    assert not verdict.passed  # fail closed


async def test_warning_rules_and_unknown_rules_never_block(rig: Rig, run: Run) -> None:
    task = make_task(run, AgentName.PRODUCT, Phase.PRD, "prd")
    rig.llm.push({"findings": [{"rule_id": "Z1", "severity": "blocker", "message": "made up"}]})
    verdict = await rig.auditor.audit_task(task, good_prd(), run)
    assert verdict.passed and verdict.findings[0].severity is Severity.WARNING


async def test_repair_loop_recovers_from_bad_json(rig: Rig, run: Run) -> None:
    task = make_task(run, AgentName.PRODUCT, Phase.PRD, "prd")
    rig.llm.push("not json", {"findings": "wrong type"}, PASS)
    verdict = await rig.auditor.audit_task(task, good_prd(), run)
    assert verdict.passed
    assert len(rig.llm.requests) == 3
    assert "not valid JSON" in str(rig.llm.requests[2].messages[-1].content[0])


async def test_fails_closed_without_a_valid_verdict(rig: Rig, run: Run) -> None:
    task = make_task(run, AgentName.PRODUCT, Phase.PRD, "prd")
    rig.llm.push("x", "y", "z")
    verdict = await rig.auditor.audit_task(task, good_prd(), run)
    assert not verdict.passed
    assert verdict.blockers[0].rule_id == AUDIT_RULE


async def test_fails_closed_when_the_model_errors(rig: Rig, run: Run) -> None:
    task = make_task(run, AgentName.PRODUCT, Phase.PRD, "prd")
    rig.llm.push(TransientLLMError("down", provider="fake"))
    verdict = await rig.auditor.audit_task(task, None, run)
    assert not verdict.passed
    assert {f.rule_id for f in verdict.blockers} >= {AUDIT_RULE, "R1"}  # helper hits kept


async def test_budget_exceeded_propagates(rig: Rig, run: Run) -> None:
    task = make_task(run, AgentName.PRODUCT, Phase.PRD, "prd")
    rig.llm.push(BudgetExceeded("cap", limit="usd"))
    with pytest.raises(BudgetExceeded):
        await rig.auditor.audit_task(task, good_prd(), run)


async def test_revision_brings_in_r12_and_previous_version(rig: Rig, run: Run) -> None:
    task = make_task(run, AgentName.PRODUCT, Phase.PRD, "prd", attempt=2)
    await rig.artifacts.save(good_prd(run_id=run.id, version=1))
    rig.llm.push(PASS)
    verdict = await rig.auditor.audit_task(task, good_prd(run_id=run.id, version=2), run)
    assert "R12" in verdict.rules_checked
    assert "Previous version prd@v1" in str(rig.llm.requests[0].messages[0].content[0])


async def test_prd_check_flags_untestable_requirements(rig: Rig, run: Run) -> None:
    task = make_task(run, AgentName.PRODUCT, Phase.PRD, "prd")
    untestable = PRD(
        summary="s",
        title="t",
        run_id=run.id,
        requirements=[Requirement(id="REQ-1", text="x", asks=["ASK-1"], acceptance_criteria=[])],
    )
    rig.llm.push(PASS)
    verdict = await rig.auditor.audit_task(task, untestable, run)
    assert verdict.scope is RuleScope.TASK
    assert {(f.rule_id, f.item) for f in verdict.blockers} >= {("R3", "REQ-1")}
    assert "HIT: REQ-1 has no acceptance criterion" in str(
        rig.llm.requests[0].messages[0].content[0]
    )


async def test_engineer_check_needs_tests_run(rig: Rig, run: Run) -> None:
    eng = make_task(run, AgentName.ENGINEER, Phase.IMPLEMENTATION, "code_change")
    Path(run.workspace, "app.py").write_text("x = 1\n")  # noqa: ASYNC240
    rig.llm.push(PASS)
    change = CodeChange(summary="s", run_id=run.id, files_written=["app.py"])
    verdict = await rig.auditor.audit_task(eng, change, run)
    assert "R6" in {f.rule_id for f in verdict.blockers}


async def test_qa_check_covers_every_code_change_in_its_phase(rig: Rig, run: Run) -> None:
    Path(run.workspace, "app.py").write_text("import os\nos.system(cmd)\n")  # noqa: ASYNC240
    eng = make_task(
        run, AgentName.ENGINEER, Phase.IMPLEMENTATION, "code_change", status=TaskStatus.DONE
    )
    eng.output = await rig.artifacts.save(
        CodeChange(summary="s", run_id=run.id, files_written=["app.py"])
    )
    old = make_task(
        run, AgentName.ENGINEER, Phase.IMPLEMENTATION, "code_change", status=TaskStatus.SUPERSEDED
    )
    qa = make_task(run, AgentName.QA, Phase.IMPLEMENTATION, "qa_report")
    old.output = await rig.artifacts.save(
        CodeChange(summary="s", run_id=run.id, version=2, files_written=["old.py"])
    )
    for t in (eng, old, qa):
        await rig.tasks.create_task(t)
    # The engineer's own check judged its actions; QA's check must not blame QA for them.
    rig.tool_log.record(call(run, eng, "run_command", "", command="rm -rf ~/x"))
    rig.llm.push(PASS)
    verdict = await rig.auditor.audit_task(qa, None, run)
    rules = {f.rule_id for f in verdict.blockers}
    assert {"S2", "R5", "R7"} <= rules  # the engineer's dangerous code, no spec, no QA report
    assert "S5" not in rules
    assert "R6" not in verdict.rules_checked  # the engineer's own check covers tests
    prompt = str(rig.llm.requests[0].messages[0].content[0])
    assert "app.py" in prompt and "old.py" not in prompt  # superseded work is out


async def test_run_sweep(rig: Rig, run: Run) -> None:
    task = make_task(run, AgentName.PRODUCT, Phase.PRD, "prd", status=TaskStatus.RUNNING)
    await rig.tasks.create_task(task)
    rig.llm.push(PASS)
    verdict = await rig.auditor.audit_run(run)
    assert verdict.scope is RuleScope.RUN and verdict.rules_checked == ["R8"]
    assert not verdict.passed and verdict.blockers[0].item == task.id


async def test_no_rules_means_no_model_call(
    rig: Rig, run: Run, rule_store: SqliteRuleStore
) -> None:
    await rule_store.set_enabled("R8", False)
    verdict = await rig.auditor.audit_run(run)
    assert verdict.passed and rig.llm.requests == []


async def test_plan_check_shows_the_full_request_and_the_plan(rig: Rig, run: Run) -> None:
    task = make_task(run, AgentName.ORCHESTRATOR, Phase.PLANNING, "plan")
    plan = await rig.artifacts.latest(run.id, "plan")
    rig.llm.push(PASS)
    verdict = await rig.auditor.audit_task(task, plan, run)
    block = rig.llm.requests[0].messages[0].content[0]
    assert isinstance(block, TextBlock)
    prompt = block.text
    assert "P1" in verdict.rules_checked
    assert "User's original request:\nBuild a books API" in prompt
    assert "plan@v1" in prompt


def test_second_check_material_keeps_the_start_and_the_end() -> None:
    from agent_team.audit.auditor import _head_and_tail

    text = "REQUEST" + "x" * 50_000 + "TOOLLOG"
    cut = _head_and_tail(text, 24_000)
    assert cut.startswith("REQUEST") and cut.endswith("TOOLLOG") and len(cut) < 24_100
