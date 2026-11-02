"""The Orchestrator's planner: request -> request kind, numbered asks and research
questions. The tasks themselves come from a fixed pipeline in code
(agent_team.orchestrator.planning), so the model never designs the task graph.

``validate_plan`` is the code check behind "code validates every call": a bad plan goes
back to the model as a repair before the auditor's P1 sweep sees it.
"""

from __future__ import annotations

import re
from typing import ClassVar

from agent_team.agents import prompts
from agent_team.agents.base import BaseAgent
from agent_team.core.artifacts import ArtifactBase, Plan
from agent_team.core.models import AgentName, Phase, RequestKind, TaskEnvelope

#: Which phase and output kinds each expert works in (used to validate create_task).
AGENT_PLAN: dict[AgentName, tuple[Phase, tuple[str, ...]]] = {
    AgentName.RESEARCHER: (Phase.PLANNING, ("research_brief", "research_answer")),
    AgentName.PRODUCT: (Phase.PRD, ("prd",)),
    AgentName.ARCHITECT: (Phase.SPEC, ("spec",)),
    AgentName.ENGINEER: (Phase.IMPLEMENTATION, ("code_change",)),
    AgentName.QA: (Phase.IMPLEMENTATION, ("qa_report",)),
}
_ASK_ID = re.compile(r"^ASK-\d+$")
_KIND_MARKER = re.compile(r"\[request_kind=([a-z_]+)\]")


def planning_objective(kind: RequestKind) -> str:
    """The planning task's objective, with the request kind marker the code reads back."""
    what = "answer this research question" if kind is RequestKind.RESEARCH_QUESTION else "build"
    return f"Plan the asks and research questions to {what} [request_kind={kind.value}]"


def kind_from_objective(objective: str) -> RequestKind | None:
    """The request kind recorded in a planning objective, or None."""
    m = _KIND_MARKER.search(objective)
    if not m:
        return None
    try:
        return RequestKind(m.group(1))
    except ValueError:
        return None


def validate_plan(plan: Plan, expected: RequestKind | None = None) -> list[str]:
    """Code checks on a plan beyond its schema; each error goes back to the model."""
    errors: list[str] = []
    # request_kind's values are enforced by the schema; only the classification can clash.
    if expected is not None and plan.request_kind != expected.value:
        errors.append(f"request_kind: this request was classified as {expected.value!r}")

    ask_ids = [a.id for a in plan.asks]
    for ask in plan.asks:
        if not _ASK_ID.match(ask.id):
            errors.append(f"asks: {ask.id!r} must look like ASK-1")
        if not ask.text.strip():
            errors.append(f"asks.{ask.id}: say what the user asked for")
    if len(set(ask_ids)) != len(ask_ids):
        errors.append("asks: ids must be unique")

    seen: set[str] = set()
    for i, rq in enumerate(plan.research_questions, start=1):
        where = f"research_questions[{i}]"
        if not rq.question.strip():
            errors.append(f"{where}.question: write the question")
        if rq.ask not in ask_ids:
            errors.append(f"{where}.ask: unknown ask {rq.ask!r}")
        key = rq.question.strip().lower()
        if key in seen:
            errors.append(f"{where}: duplicate question")
        seen.add(key)
    return errors


class PlannerAgent(BaseAgent):
    """The orchestrator's planning call: turns the request into asks and research questions."""

    name: AgentName = AgentName.ORCHESTRATOR
    output_kind = "plan"
    role_prompt: ClassVar[str] = prompts.PLANNER
    tool_names: ClassVar[tuple[str, ...]] = ("list_dir", "read_file", "grep")

    async def check(self, artifact: ArtifactBase, envelope: TaskEnvelope) -> list[str]:
        """The plan matches the request kind and its ids are consistent."""
        errors = await super().check(artifact, envelope)
        if isinstance(artifact, Plan):
            errors += validate_plan(artifact, kind_from_objective(envelope.task.objective))
        return errors
