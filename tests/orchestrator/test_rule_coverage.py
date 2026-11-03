"""Every rule in the catalog runs somewhere: a task rule's phase and agent filters match
at least one task the pipeline really creates, so moving a rule can't silently switch
it off."""

from __future__ import annotations

from pathlib import Path

from agent_team.audit.catalog import load_catalog
from agent_team.audit.rule_store import matches
from agent_team.core.models import AgentName, Phase, RuleScope, TaskStatus
from orchestrator.harness import make_team


async def _pipeline_tasks(tmp_path: Path) -> set[tuple[Phase, AgentName]]:
    """The (phase, agent) of every checked task in a build and a research question run."""
    seen: set[tuple[Phase, AgentName]] = set()
    for kind, request in (
        ("build", "Build a books API"),
        ("research_question", "Which Python web framework is fastest?"),
    ):
        team = await make_team(tmp_path / kind, kind=kind)
        reply = await team.orch.handle(request)
        assert reply.kind == "done" and reply.run_id, reply.text
        for t in await team.tasks.list_tasks(reply.run_id):
            if t.agent not in (AgentName.USER, AgentName.AUDITOR):
                assert t.status is TaskStatus.DONE
                seen.add((t.phase, t.agent))
    return seen


async def test_every_task_rule_matches_a_pipeline_task(tmp_path: Path) -> None:
    tasks = await _pipeline_tasks(tmp_path)
    unmatched = [
        rule.id
        for rule in load_catalog()
        if rule.scope is RuleScope.TASK
        and not any(matches(rule, phase=p, agent=a) for p, a in tasks)
    ]
    assert unmatched == []


async def test_only_task_and_run_rules_exist() -> None:
    assert {rule.scope for rule in load_catalog()} == {RuleScope.TASK, RuleScope.RUN}


async def test_the_former_gate_rules_land_on_their_tasks(tmp_path: Path) -> None:
    tasks = await _pipeline_tasks(tmp_path)
    expected = {
        "P1": (Phase.PLANNING, AgentName.ORCHESTRATOR),
        "R3": (Phase.PRD, AgentName.PRODUCT),
        "R4": (Phase.SPEC, AgentName.ARCHITECT),
        "R6": (Phase.IMPLEMENTATION, AgentName.ENGINEER),
        "R7": (Phase.IMPLEMENTATION, AgentName.QA),
    }
    rules = {r.id: r for r in load_catalog()}
    for rule_id, (phase, agent) in expected.items():
        assert (phase, agent) in tasks
        hits = {pa for pa in tasks if matches(rules[rule_id], phase=pa[0], agent=pa[1])}
        assert hits == {(phase, agent)}, rule_id
    for rule_id in ("R5", "R10", "S2", "S3", "S4"):
        hits = {pa for pa in tasks if matches(rules[rule_id], phase=pa[0], agent=pa[1])}
        assert hits == {(Phase.IMPLEMENTATION, AgentName.QA)}, rule_id
    assert {pa for pa in tasks if matches(rules["R9"], phase=pa[0], agent=pa[1])} == {
        (Phase.PRD, AgentName.PRODUCT),
        (Phase.SPEC, AgentName.ARCHITECT),
    }
    assert all(matches(rules["R11"], phase=p, agent=a) for p, a in tasks)
