from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from agent_team.agents import QA as QAAgent
from agent_team.agents import Architect, PlannerAgent, Product, default_agents, validate_plan
from agent_team.agents.planner import kind_from_objective, planning_objective
from agent_team.core.artifacts import PRD as PRDModel
from agent_team.core.artifacts import Plan
from agent_team.core.models import AgentName, RequestKind, ResultStatus, ReviewMode
from agents.fakes import FakeLLM
from agents.scenario import PLAN_BUILD, PLAN_RESEARCH, PRD, QA, SPEC
from agents.test_loop import ctx, envelope


def plan(data: dict[str, Any]) -> Plan:
    return Plan.model_validate(data)


def test_valid_plans_pass() -> None:
    assert validate_plan(plan(PLAN_BUILD), RequestKind.BUILD) == []
    assert validate_plan(plan(PLAN_RESEARCH), RequestKind.RESEARCH_QUESTION) == []


def _mutate(fn: Any) -> list[str]:
    data = copy.deepcopy(PLAN_BUILD)
    fn(data)
    return validate_plan(plan(data), RequestKind.BUILD)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda d: d.update(request_kind="change"), "either 'build' or 'research_question'"),
        (lambda d: d.update(request_kind="nonsense"), "must be 'build'"),
        (lambda d: d.update(request_kind="research_question"), "classified as 'build'"),
        (lambda d: d["asks"].append({"id": "two", "text": "x"}), "must look like ASK-1"),
        (lambda d: d["asks"].append({"id": "ASK-1", "text": "again"}), "unique"),
        (lambda d: d["asks"].append({"id": "ASK-2", "text": " "}), "say what the user asked"),
        (lambda d: d["research_questions"][0].update(ask="ASK-9"), "unknown ask 'ASK-9'"),
        (lambda d: d["research_questions"].append(dict(d["research_questions"][0])), "duplicate"),
    ],
)
def test_invalid_plans_are_explained(mutation: Any, expected: str) -> None:
    errors = _mutate(mutation)
    assert any(expected in e for e in errors), errors


def test_plan_needs_an_ask() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        plan(dict(PLAN_BUILD, asks=[]))


def test_kind_marker_round_trips() -> None:
    for kind in (RequestKind.BUILD, RequestKind.RESEARCH_QUESTION):
        assert kind_from_objective(planning_objective(kind)) is kind
    assert kind_from_objective("plan something") is None


async def test_planner_repairs_an_invalid_plan(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.ORCHESTRATOR, "plan")
    env.task.objective = planning_objective(RequestKind.BUILD)
    bad = copy.deepcopy(PLAN_BUILD)
    bad["research_questions"][0]["ask"] = "ASK-7"
    llm = FakeLLM({AgentName.ORCHESTRATOR: [json.dumps(bad), json.dumps(PLAN_BUILD)]})
    result = await PlannerAgent().run(env, ctx(llm))
    assert result.status is ResultStatus.OK
    assert "unknown ask 'ASK-7'" in llm.requests[1].messages[-1].content[0].text  # type: ignore[union-attr]


async def test_product_rejects_unknown_asks(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.PRODUCT, "prd")
    env.inputs.append(plan(PLAN_BUILD))
    bad = copy.deepcopy(PRD)
    bad["requirements"][0]["asks"] = ["ASK-9"]
    llm = FakeLLM({AgentName.PRODUCT: [json.dumps(bad), json.dumps(PRD)]})
    result = await Product().run(env, ctx(llm))
    assert result.status is ResultStatus.OK
    assert "unknown ask ASK-9" in llm.requests[1].messages[-1].content[0].text  # type: ignore[union-attr]


async def test_architect_and_qa_check_ids_against_the_prd(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.ARCHITECT, "spec")
    env.inputs.append(PRDModel.model_validate(PRD))
    bad = copy.deepcopy(SPEC)
    bad["checks"][0]["criteria"] = ["AC-7.1"]
    llm = FakeLLM({AgentName.ARCHITECT: [json.dumps(bad), json.dumps(SPEC)]})
    assert (await Architect().run(env, ctx(llm))).status is ResultStatus.OK

    env = envelope(tmp_path, AgentName.QA, "qa_report")
    env.inputs.append(PRDModel.model_validate(PRD))
    missing = dict(QA, results=[])
    llm = FakeLLM({AgentName.QA: [json.dumps(missing), json.dumps(QA)]})
    assert (await QAAgent().run(env, ctx(llm))).status is ResultStatus.OK
    assert "no result for AC-1.1" in llm.requests[1].messages[-1].content[0].text  # type: ignore[union-attr]


def test_default_agents_cover_every_planned_role() -> None:
    agents = default_agents()
    assert set(agents) == {
        AgentName.ORCHESTRATOR,
        AgentName.RESEARCHER,
        AgentName.PRODUCT,
        AgentName.ARCHITECT,
        AgentName.ENGINEER,
        AgentName.QA,
    }
    system = Product().system_prompt("prd", ReviewMode.AUTO_ASSUMPTIONS)
    assert '"requirements"' in system and '"run_id"' not in system
