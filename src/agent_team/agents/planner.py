"""The Orchestrator's planner: request -> numbered asks -> a plan of tasks.

``validate_plan`` is the code check behind "code validates every call": a plan with an
unknown agent, a cycle or an uncovered ask goes back to the model as a repair, before
the auditor's P1 sweep ever sees it.
"""

from __future__ import annotations

import re
from typing import ClassVar

from agent_team.agents import prompts
from agent_team.agents.base import BaseAgent
from agent_team.core.artifacts import ArtifactBase, Plan
from agent_team.core.models import AgentName, Phase, RequestKind, TaskEnvelope

#: Which phase and output kinds each expert may be planned for.
AGENT_PLAN: dict[AgentName, tuple[Phase, tuple[str, ...]]] = {
    AgentName.RESEARCHER: (Phase.PLANNING, ("research_brief", "research_answer")),
    AgentName.PRODUCT: (Phase.PRD, ("prd",)),
    AgentName.ARCHITECT: (Phase.SPEC, ("spec",)),
    AgentName.ENGINEER: (Phase.IMPLEMENTATION, ("code_change",)),
    AgentName.QA: (Phase.IMPLEMENTATION, ("qa_report",)),
}
_PHASE_INDEX = {p: i for i, p in enumerate(Phase)}
_PHASE_VALUES = frozenset(p.value for p in Phase)
_ASK_ID = re.compile(r"^ASK-\d+$")
_BUILD_KINDS = ("prd", "spec", "code_change", "qa_report")
_KIND_MARKER = re.compile(r"\[request_kind=([a-z_]+)\]")


def planning_objective(kind: RequestKind) -> str:
    what = "answer this research question" if kind is RequestKind.RESEARCH_QUESTION else "build"
    return f"Plan the tasks to {what} [request_kind={kind.value}]"


def kind_from_objective(objective: str) -> RequestKind | None:
    m = _KIND_MARKER.search(objective)
    if not m:
        return None
    try:
        return RequestKind(m.group(1))
    except ValueError:
        return None


def _has_cycle(graph: dict[str, list[str]]) -> list[str] | None:
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for dep in graph.get(node, []):
            if state.get(dep) == 1:
                return [*stack[stack.index(dep) :], dep]
            if dep not in state:
                found = visit(dep)
                if found:
                    return found
        stack.pop()
        state[node] = 2
        return None

    for node in graph:
        if node not in state:
            found = visit(node)
            if found:
                return found
    return None


def validate_plan(plan: Plan, expected: RequestKind | None = None) -> list[str]:
    errors: list[str] = []
    try:
        kind = RequestKind(plan.request_kind)
    except ValueError:
        return [
            f"request_kind: {plan.request_kind!r} is not one of {[k.value for k in RequestKind]}"
        ]
    if kind not in (RequestKind.BUILD, RequestKind.RESEARCH_QUESTION):
        errors.append("request_kind: a plan is either 'build' or 'research_question'")
    if expected is not None and kind is not expected:
        errors.append(f"request_kind: this request was classified as {expected.value!r}")

    ask_ids = [a.id for a in plan.asks]
    if not ask_ids:
        errors.append("asks: list at least one ask (ASK-1)")
    for aid in ask_ids:
        if not _ASK_ID.match(aid):
            errors.append(f"asks: {aid!r} must look like ASK-1")
    if len(set(ask_ids)) != len(ask_ids):
        errors.append("asks: ids must be unique")

    keys = [t.key for t in plan.tasks]
    if not keys:
        errors.append("tasks: plan at least one task")
    if len(set(keys)) != len(keys):
        errors.append("tasks: keys must be unique")
    by_key = {t.key: t for t in plan.tasks}
    covered: set[str] = set()
    for t in plan.tasks:
        where = f"tasks.{t.key}"
        try:
            agent = AgentName(t.agent)
        except ValueError:
            errors.append(f"{where}.agent: unknown agent {t.agent!r}")
            continue
        if agent not in AGENT_PLAN:
            errors.append(f"{where}.agent: {agent.value} can't be planned; use an expert")
            continue
        phase, kinds = AGENT_PLAN[agent]
        if t.phase != phase.value:
            errors.append(f"{where}.phase: {agent.value} tasks belong to phase {phase.value!r}")
        if t.output_kind not in kinds:
            errors.append(f"{where}.output_kind: {agent.value} produces one of {list(kinds)}")
        if not t.objective.strip():
            errors.append(f"{where}.objective: say what to produce")
        for aid in t.covers_asks:
            if aid not in ask_ids:
                errors.append(f"{where}.covers_asks: unknown ask {aid!r}")
        covered.update(t.covers_asks)
        for dep in t.depends_on:
            if dep == t.key:
                errors.append(f"{where}.depends_on: a task can't depend on itself")
            elif dep not in by_key:
                errors.append(f"{where}.depends_on: unknown task key {dep!r}")
            elif (
                t.phase in _PHASE_VALUES
                and by_key[dep].phase in _PHASE_VALUES
                and _PHASE_INDEX[Phase(by_key[dep].phase)] > _PHASE_INDEX[Phase(t.phase)]
            ):
                errors.append(f"{where}.depends_on: {dep!r} is in a later phase")
    cycle = _has_cycle({t.key: [d for d in t.depends_on if d in by_key] for t in plan.tasks})
    if cycle:
        errors.append("tasks: dependency cycle " + " -> ".join(cycle))
    for aid in ask_ids:
        if aid not in covered:
            errors.append(f"asks: {aid} is not covered by any task")

    kinds_planned = {t.output_kind for t in plan.tasks}
    if kind is RequestKind.BUILD:
        for needed in _BUILD_KINDS:
            if needed not in kinds_planned:
                errors.append(f"tasks: a build plan needs a {needed} task")
        if "research_answer" in kinds_planned:
            errors.append("tasks: research_answer is only for research questions")
    elif kind is RequestKind.RESEARCH_QUESTION:
        if "research_answer" not in kinds_planned:
            errors.append("tasks: a research question plan needs a research_answer task")
        extra = kinds_planned & set(_BUILD_KINDS)
        if extra:
            errors.append(f"tasks: a research question gets no {sorted(extra)}")
    if "research_brief" in kinds_planned and not plan.research_questions:
        errors.append("research_questions: list the questions the research tasks answer")
    return errors


class PlannerAgent(BaseAgent):
    name: AgentName = AgentName.ORCHESTRATOR
    output_kind = "plan"
    role_prompt: ClassVar[str] = prompts.PLANNER
    tool_names: ClassVar[tuple[str, ...]] = ("list_dir", "read_file", "grep")

    async def check(self, artifact: ArtifactBase, envelope: TaskEnvelope) -> list[str]:
        errors = await super().check(artifact, envelope)
        if isinstance(artifact, Plan):
            errors += validate_plan(artifact, kind_from_objective(envelope.task.objective))
        return errors
