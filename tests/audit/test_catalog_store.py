from __future__ import annotations

from pathlib import Path

import pytest

from agent_team.audit.catalog import load_catalog
from agent_team.audit.evidence import HELPERS
from agent_team.audit.rule_store import SqliteRuleStore
from agent_team.core.models import AgentName, Phase, RuleScope, Severity

EXPECTED = {"P1", "P2", *(f"R{i}" for i in range(1, 13)), *(f"S{i}" for i in range(1, 8))}


def test_catalog_has_every_rule_with_a_helper() -> None:
    rules = load_catalog()
    assert {r.id for r in rules} == EXPECTED
    assert len(rules) == 21
    for rule in rules:
        assert rule.helper in HELPERS, rule.id
    warnings = {r.id for r in rules if r.severity is Severity.WARNING}
    assert warnings == {"R10", "R12", "S4"}
    assert {r.id for r in rules if r.precheck} == {"R5", "S1", "S4", "S5", "S7"}


def test_catalog_rejects_duplicates(tmp_path: Path) -> None:
    rule = "- {id: R1, title: t, scope: task, severity: blocker, statement: s}\n"
    path = tmp_path / "rules.yaml"
    path.write_text("rules:\n  " + rule + "  " + rule)
    with pytest.raises(ValueError, match="duplicate"):
        load_catalog(path)


async def test_store_filters_by_scope_phase_and_agent(rule_store: SqliteRuleStore) -> None:
    research = await rule_store.for_scope("task", phase=Phase.PLANNING, agent=AgentName.RESEARCHER)
    assert {r.id for r in research} == {"R1", "R2", "P2", "R12", "S1", "S5", "S6", "S7"}
    engineer = await rule_store.for_scope(
        "task", phase=Phase.IMPLEMENTATION, agent=AgentName.ENGINEER
    )
    assert {r.id for r in engineer} == {"R1", "R12", "S1", "S5", "S6", "S7"}
    prd_gate = await rule_store.for_scope("phase", phase=Phase.PRD, agent=None)
    assert {r.id for r in prd_gate} == {"R3", "R9", "R11"}
    impl_gate = await rule_store.for_scope("phase", phase=Phase.IMPLEMENTATION, agent=None)
    assert {r.id for r in impl_gate} == {"R5", "R6", "R7", "R10", "R11", "S2", "S3", "S4"}
    assert [r.id for r in await rule_store.for_scope("run", phase=None, agent=None)] == ["R8"]
    assert all(r.scope is RuleScope.PHASE for r in impl_gate)


async def test_reseed_keeps_disabled_rules_off(tmp_path: Path) -> None:
    store = SqliteRuleStore(tmp_path / "rules.db")
    await store.seed(load_catalog())
    await store.set_enabled("S6", False)
    await store.seed(load_catalog())
    ids = {r.id for r in await store.for_scope("task", phase=None, agent=None)}
    assert "S6" not in ids and "S5" in ids
    assert len(await store.all()) == 21
    with pytest.raises(KeyError):
        await store.set_enabled("Z9", True)
    store.close()


async def test_reseed_deletes_rules_the_catalog_dropped() -> None:
    store = SqliteRuleStore()
    await store.seed(load_catalog())
    await store.seed([r for r in load_catalog() if r.id != "R10"])
    assert "R10" not in {r.id for r in await store.all()}
    assert len(await store.all()) == 20
    store.close()
