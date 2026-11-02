"""Typed artifacts the experts produce, and the id chain that links them.

The chain: Plan.asks (ASK-1) -> PRD.requirements (REQ-1, cite asks) and
acceptance criteria (AC-1.1) -> Spec.components (cover REQ ids) and planned checks
(cover AC ids) -> CodeChange files -> QAReport results per AC id. Rules P1, R3, R4
and R9 check these links with code, so field names here are part of the contract.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, get_args

from pydantic import Field

from agent_team.core.base import Contract

_REF = re.compile(r"^(?P<kind>[a-z_]+)@v(?P<version>\d+)$")


class ArtifactRef(Contract):
    """Points at one version of one artifact kind within a run, e.g. ``prd@v2``."""

    kind: str = Field(pattern=r"^[a-z_]+$")
    version: int = Field(ge=1)

    def __str__(self) -> str:
        return f"{self.kind}@v{self.version}"

    @classmethod
    def parse(cls, text: str) -> ArtifactRef:
        """Parse ``prd@v2``. Raises ValueError on anything else."""
        m = _REF.fullmatch(text)
        if not m:
            raise ValueError(f"not an artifact ref: {text!r}")
        return cls(kind=m["kind"], version=int(m["version"]))


class Question(Contract):
    """Something an agent can't decide without the user. In auto_assumptions mode
    the agent records an assumption instead."""

    id: str  # e.g. "Q-1"
    text: str
    why: str  # what the agent can't decide without it
    answer: str | None = None
    assumption: str | None = None  # recorded instead of asking in auto_assumptions mode


class MemoryProposal(Contract):
    """A fact an agent suggests keeping for future runs. The auditor decides
    whether it is saved."""

    kind: Literal["decision", "lesson", "preference", "research_fact"]
    text: str
    source: str | None = None  # URL or file for research facts


class ResearchRequest(Contract):
    """An agent's request for facts it lacks. Becomes a Researcher task, and the
    asking task waits for it."""

    question: str
    why: str


class Source(Contract):
    """A file or URL the Researcher actually opened in this attempt."""

    ref: str  # URL or path the Researcher actually opened
    title: str | None = None


class ArtifactBase(Contract):
    """Fields every artifact carries. The dispatcher fills run_id/task_id/version."""

    run_id: str = ""
    task_id: str = ""
    version: int = 1
    built_from: list[ArtifactRef] = Field(default_factory=list)
    summary: str  # two or three sentences shown in reviews and the trace
    questions: list[Question] = Field(default_factory=list)
    research_requests: list[ResearchRequest] = Field(default_factory=list)
    memory_proposals: list[MemoryProposal] = Field(default_factory=list)


# --- planning ----------------------------------------------------------------------


class Ask(Contract):
    """One numbered thing the user asked for. The root of the id chain."""

    id: str  # "ASK-1"
    text: str


class ResearchQuestion(Contract):
    """A fact needed before the PRD, tied to the ask it serves. Becomes one
    Researcher task."""

    question: str
    ask: str  # the ASK id this question serves


class Plan(ArtifactBase):
    """The Orchestrator's reading of the request. The phase order is fixed in code, so
    the plan holds no tasks: each research question becomes one Researcher task."""

    kind: Literal["plan"] = "plan"
    # Only these two RequestKind values start a plan; changes and run questions don't.
    request_kind: Literal["build", "research_question"]
    asks: list[Ask] = Field(min_length=1)
    research_questions: list[ResearchQuestion] = Field(default_factory=list)


# --- research ----------------------------------------------------------------------


class ResearchFinding(Contract):
    """A claim the Researcher found, with at least one source."""

    claim: str
    sources: list[Source] = Field(min_length=1)


class AnsweredQuestion(Contract):
    """The Researcher's answer to one question, or the gap that kept it from
    answering."""

    question: str
    answer: str | None  # None means it couldn't be answered; say why in `gap`
    gap: str | None = None
    sources: list[Source] = Field(default_factory=list)


class ResearchBrief(ArtifactBase):
    """The Researcher's notes for later agents: answers, findings, file excerpts
    and constraints. A run can have several."""

    kind: Literal["research_brief"] = "research_brief"
    answers: list[AnsweredQuestion]
    findings: list[ResearchFinding] = Field(default_factory=list)
    file_excerpts: list[Source] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


# --- PRD and spec ------------------------------------------------------------------


class AcceptanceCriterion(Contract):
    """One testable condition a requirement must meet."""

    id: str  # "AC-1.1"
    text: str


class Requirement(Contract):
    """One numbered requirement, the asks it covers, and its acceptance criteria."""

    id: str  # "REQ-1"
    text: str
    asks: list[str] = Field(min_length=1)  # ASK ids this requirement covers
    acceptance_criteria: list[AcceptanceCriterion]


class PRD(ArtifactBase):
    """Product's requirements document: what to build, and what is out of scope."""

    kind: Literal["prd"] = "prd"
    title: str
    requirements: list[Requirement]
    out_of_scope: list[str] = Field(default_factory=list)


class Component(Contract):
    """One part of the design, the requirements it covers and the files it owns."""

    name: str
    responsibility: str
    requirements: list[str]  # REQ ids
    files: list[str]  # workspace-relative paths


class Interface(Contract):
    """A function, route or tool a component exposes, with its signature."""

    name: str  # function, route or tool name
    signature: str
    component: str


class PlannedCheck(Contract):
    """A test the Architect plans, and the acceptance criteria it proves."""

    id: str  # "CHK-1"
    criteria: list[str]  # AC ids
    kind: Literal["unit_test", "e2e"]
    description: str


class WorkPackage(Contract):
    """A group of components one Engineer task builds."""

    name: str
    components: list[str]  # Component names


class Spec(ArtifactBase):
    """The Architect's design: components, interfaces, planned checks, pinned
    dependencies, allowed hosts and decisions."""

    kind: Literal["spec"] = "spec"
    components: list[Component]
    interfaces: list[Interface] = Field(default_factory=list)
    checks: list[PlannedCheck]
    dependencies: list[str] = Field(default_factory=list)  # pinned, e.g. "fastapi==0.115.0"
    allowed_hosts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)  # set on revisions
    # Empty means one Engineer task for the whole spec. Set it only when the parts are
    # independent; each package becomes its own Engineer task, run one at a time.
    work_packages: list[WorkPackage] = Field(default_factory=list)


# --- implementation ----------------------------------------------------------------


class TestRun(Contract):
    """The result of one test command in the sandbox."""

    command: str
    passed: int
    failed: int
    exit_code: int
    output_tail: str


class CodeChange(ArtifactBase):
    """What one Engineer attempt wrote: files, the commit, and its test run."""

    kind: Literal["code_change"] = "code_change"
    files_written: list[str]
    commit: str | None = None  # git sha in the run workspace
    tests: TestRun | None = None


class CriterionResult(Contract):
    """QA's result for one acceptance criterion, with evidence."""

    criterion: str  # AC id
    result: Literal["pass", "fail", "not_checked"]
    evidence: str


class QAIssue(Contract):
    """A problem QA found that isn't tied to a single criterion."""

    severity: Literal["blocker", "warning"]
    text: str
    files: list[str] = Field(default_factory=list)


class QAReport(ArtifactBase):
    """QA's review of the whole solution: a result per criterion, issues, and
    whether it ran end to end."""

    kind: Literal["qa_report"] = "qa_report"
    results: list[CriterionResult]
    issues: list[QAIssue] = Field(default_factory=list)
    ran_end_to_end: bool


# --- answers -----------------------------------------------------------------------


class ResearchAnswer(ArtifactBase):
    """Final artifact of a research-question run: an answer with sources, no code."""

    kind: Literal["research_answer"] = "research_answer"
    question: str
    answer: str
    sources: list[Source] = Field(min_length=1)


Artifact = Annotated[
    Plan | ResearchBrief | PRD | Spec | CodeChange | QAReport | ResearchAnswer,
    Field(discriminator="kind"),
]

#: Artifact class by kind, derived from the union so the two can't drift apart.
ARTIFACT_TYPES: dict[str, type[ArtifactBase]] = {
    cls.model_fields["kind"].default: cls for cls in get_args(get_args(Artifact)[0])
}
