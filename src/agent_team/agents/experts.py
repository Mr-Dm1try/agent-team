"""The five experts. Each is a BaseAgent with its prompt, tools and code checks."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from agent_team.agents import prompts
from agent_team.agents.base import BaseAgent
from agent_team.core.artifacts import (
    PRD,
    ArtifactBase,
    CodeChange,
    Plan,
    QAReport,
    ResearchAnswer,
    ResearchBrief,
    Spec,
)
from agent_team.core.interfaces import Agent
from agent_team.core.models import AgentName, TaskEnvelope
from agent_team.tools.base import resolve_path


def _input[T](envelope: TaskEnvelope, kind: type[T]) -> list[T]:
    """The envelope's input artifacts of one type, such as every PRD it was given."""
    return [a for a in envelope.inputs if isinstance(a, kind)]


class Researcher(BaseAgent):
    """Writes research briefs for the pipeline and the final research answer for a question."""

    name: AgentName = AgentName.RESEARCHER
    output_kind = "research_brief"
    output_kinds: ClassVar[tuple[str, ...]] = ("research_brief", "research_answer")
    role_prompt: ClassVar[str] = prompts.RESEARCHER
    tool_names: ClassVar[tuple[str, ...]] = (
        "read_file",
        "list_dir",
        "grep",
        "web_search",
        "web_fetch",
    )

    async def check(self, artifact: ArtifactBase, envelope: TaskEnvelope) -> list[str]:
        """Every brief answers its question or names the gap."""
        errors = await super().check(artifact, envelope)
        if isinstance(artifact, ResearchBrief):
            # The question is the task's objective, so there is always one to answer.
            if not artifact.answers:
                errors.append(f"answers: answer the research question: {envelope.task.objective}")
            for a in artifact.answers:
                if a.answer is None and not a.gap:
                    errors.append(f"answers: {a.question!r} has no answer; say why in `gap`")
        if isinstance(artifact, ResearchAnswer) and not artifact.answer.strip():
            errors.append("answer: write the answer")
        return errors


class Product(BaseAgent):
    """Writes the PRD from the plan's asks and the research briefs."""

    name: AgentName = AgentName.PRODUCT
    output_kind = "prd"
    role_prompt: ClassVar[str] = prompts.PRODUCT
    tool_names: ClassVar[tuple[str, ...]] = ("memory_search",)

    async def check(self, artifact: ArtifactBase, envelope: TaskEnvelope) -> list[str]:
        """Requirement ids are unique, each has acceptance criteria and known asks."""
        errors = await super().check(artifact, envelope)
        if not isinstance(artifact, PRD):
            return errors
        asks = {a.id for plan in _input(envelope, Plan) for a in plan.asks}
        seen: set[str] = set()
        for req in artifact.requirements:
            if req.id in seen:
                errors.append(f"requirements: duplicate id {req.id}")
            seen.add(req.id)
            if not req.acceptance_criteria:
                errors.append(f"requirements.{req.id}: add at least one acceptance criterion")
            for aid in req.asks:
                if asks and aid not in asks:
                    errors.append(f"requirements.{req.id}.asks: unknown ask {aid}")
        return errors


class Architect(BaseAgent):
    """Writes the Spec, including the work packages the Engineer tasks follow."""

    name: AgentName = AgentName.ARCHITECT
    output_kind = "spec"
    role_prompt: ClassVar[str] = prompts.ARCHITECT
    tool_names: ClassVar[tuple[str, ...]] = ("memory_search", "read_file", "list_dir")

    async def check(self, artifact: ArtifactBase, envelope: TaskEnvelope) -> list[str]:
        """Components cite known requirements and relative paths; checks cite known criteria."""
        errors = await super().check(artifact, envelope)
        if not isinstance(artifact, Spec):
            return errors
        prds = _input(envelope, PRD)
        if not prds:
            return errors
        prd = prds[-1]
        reqs = {r.id for r in prd.requirements}
        acs = {c.id for r in prd.requirements for c in r.acceptance_criteria}
        for comp in artifact.components:
            for rid in comp.requirements:
                if rid not in reqs:
                    errors.append(f"components.{comp.name}: unknown requirement {rid}")
            for f in comp.files:
                if Path(f).is_absolute() or ".." in Path(f).parts:
                    errors.append(f"components.{comp.name}: {f!r} must be workspace-relative")
        for chk in artifact.checks:
            for cid in chk.criteria:
                if cid not in acs:
                    errors.append(f"checks.{chk.id}: unknown acceptance criterion {cid}")
        return errors


class Engineer(BaseAgent):
    """One Engineer task per spec work package (or one for the whole spec)."""

    name: AgentName = AgentName.ENGINEER
    output_kind = "code_change"
    role_prompt: ClassVar[str] = prompts.ENGINEER
    tool_names: ClassVar[tuple[str, ...]] = (
        "write_file",
        "read_file",
        "list_dir",
        "grep",
        "run_command",
        "install_deps",
    )

    async def check(self, artifact: ArtifactBase, envelope: TaskEnvelope) -> list[str]:
        """The change lists the files it wrote, and each exists in the workspace."""
        errors = await super().check(artifact, envelope)
        if not isinstance(artifact, CodeChange):
            return errors
        if not artifact.files_written:
            errors.append(
                "files_written: no files listed; write the spec's files with write_file "
                "first, then list their workspace-relative paths here"
            )
        for f in artifact.files_written:
            try:
                path = resolve_path(f, envelope.run, write=True, tool="check")
            except Exception:
                errors.append(f"files_written: {f!r} is outside the workspace")
                continue
            if not path.is_file():
                errors.append(f"files_written: {f!r} doesn't exist; write it with write_file")
        return errors


class QA(BaseAgent):
    """Runs the checks and reports a result for every acceptance criterion."""

    name: AgentName = AgentName.QA
    output_kind = "qa_report"
    role_prompt: ClassVar[str] = prompts.QA
    tool_names: ClassVar[tuple[str, ...]] = (
        "read_file",
        "list_dir",
        "grep",
        "run_command",
        "install_deps",
    )

    async def check(self, artifact: ArtifactBase, envelope: TaskEnvelope) -> list[str]:
        """Every PRD acceptance criterion has a result, and no result names an unknown one."""
        errors = await super().check(artifact, envelope)
        if not isinstance(artifact, QAReport):
            return errors
        acs = {
            c.id
            for prd in _input(envelope, PRD)
            for r in prd.requirements
            for c in r.acceptance_criteria
        }
        reported = {r.criterion for r in artifact.results}
        for cid in sorted(acs - reported):
            errors.append(f"results: no result for {cid}; use not_checked if you couldn't")
        for cid in sorted(reported - acs) if acs else []:
            errors.append(f"results: {cid} is not an acceptance criterion in the PRD")
        return errors


def default_agents() -> dict[AgentName, Agent]:
    """One instance of each agent, keyed by name."""
    from agent_team.agents.planner import PlannerAgent

    agents: list[Agent] = [PlannerAgent(), Researcher(), Product(), Architect(), Engineer(), QA()]
    return {a.name: a for a in agents}
