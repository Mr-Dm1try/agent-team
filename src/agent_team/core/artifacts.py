"""Typed artifacts the experts produce, and the id chain that links them.

The chain: Plan.asks (ASK-1) -> PRD.requirements (REQ-1, cite asks) and
acceptance criteria (AC-1.1) -> Spec.components (cover REQ ids) and planned checks
(cover AC ids) -> CodeChange files -> QAReport results per AC id. Rules P1, R3, R4
and R9 check these links with code, so field names here are part of the contract.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

_REF = re.compile(r"^(?P<kind>[a-z_]+)@v(?P<version>\d+)$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactRef(_Strict):
    """Points at one version of one artifact kind within a run, e.g. ``prd@v2``."""

    kind: str
    version: int = Field(ge=1)

    def __str__(self) -> str:
        return f"{self.kind}@v{self.version}"

    @classmethod
    def parse(cls, text: str) -> ArtifactRef:
        m = _REF.match(text)
        if not m:
            raise ValueError(f"not an artifact ref: {text!r}")
        return cls(kind=m["kind"], version=int(m["version"]))


class Question(_Strict):
    id: str  # e.g. "Q-1"
    text: str
    why: str  # what the agent can't decide without it
    answer: str | None = None
    assumption: str | None = None  # recorded instead of asking in auto_assumptions mode


class MemoryProposal(_Strict):
    kind: Literal["decision", "lesson", "preference", "research_fact"]
    text: str
    source: str | None = None  # URL or file for research facts


class ResearchRequest(_Strict):
    question: str
    why: str


class Source(_Strict):
    ref: str  # URL or path the Researcher actually opened
    title: str | None = None


class ArtifactBase(_Strict):
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


class Ask(_Strict):
    id: str  # "ASK-1"
    text: str


class PlannedTask(_Strict):
    key: str  # local key used by depends_on inside the plan, e.g. "research-1"
    agent: str  # an AgentName value
    phase: str  # a Phase value
    objective: str
    output_kind: str
    covers_asks: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class Plan(ArtifactBase):
    kind: Literal["plan"] = "plan"
    request_kind: str  # a RequestKind value
    asks: list[Ask]
    research_questions: list[str] = Field(default_factory=list)
    tasks: list[PlannedTask]


# --- research ----------------------------------------------------------------------


class ResearchFinding(_Strict):
    claim: str
    sources: list[Source] = Field(min_length=1)


class AnsweredQuestion(_Strict):
    question: str
    answer: str | None  # None means it couldn't be answered; say why in `gap`
    gap: str | None = None
    sources: list[Source] = Field(default_factory=list)


class ResearchBrief(ArtifactBase):
    kind: Literal["research_brief"] = "research_brief"
    answers: list[AnsweredQuestion]
    findings: list[ResearchFinding] = Field(default_factory=list)
    file_excerpts: list[Source] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


# --- PRD and spec ------------------------------------------------------------------


class AcceptanceCriterion(_Strict):
    id: str  # "AC-1.1"
    text: str


class Requirement(_Strict):
    id: str  # "REQ-1"
    text: str
    asks: list[str] = Field(min_length=1)  # ASK ids this requirement covers
    acceptance_criteria: list[AcceptanceCriterion]


class PRD(ArtifactBase):
    kind: Literal["prd"] = "prd"
    title: str
    requirements: list[Requirement]
    out_of_scope: list[str] = Field(default_factory=list)


class Component(_Strict):
    name: str
    responsibility: str
    requirements: list[str]  # REQ ids
    files: list[str]  # workspace-relative paths


class Interface(_Strict):
    name: str  # function, route or tool name
    signature: str
    component: str


class PlannedCheck(_Strict):
    id: str  # "CHK-1"
    criteria: list[str]  # AC ids
    kind: Literal["unit_test", "e2e"]
    description: str


class Spec(ArtifactBase):
    kind: Literal["spec"] = "spec"
    components: list[Component]
    interfaces: list[Interface] = Field(default_factory=list)
    checks: list[PlannedCheck]
    dependencies: list[str] = Field(default_factory=list)  # pinned, e.g. "fastapi==0.115.0"
    allowed_hosts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)  # set on revisions


# --- implementation ----------------------------------------------------------------


class TestRun(_Strict):
    command: str
    passed: int
    failed: int
    exit_code: int
    output_tail: str


class CodeChange(ArtifactBase):
    kind: Literal["code_change"] = "code_change"
    files_written: list[str]
    commit: str | None = None  # git sha in the run workspace
    tests: TestRun | None = None


class CriterionResult(_Strict):
    criterion: str  # AC id
    result: Literal["pass", "fail", "not_checked"]
    evidence: str


class QAIssue(_Strict):
    severity: Literal["blocker", "warning"]
    text: str
    files: list[str] = Field(default_factory=list)


class QAReport(ArtifactBase):
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

ARTIFACT_TYPES: dict[str, type[ArtifactBase]] = {
    "plan": Plan,
    "research_brief": ResearchBrief,
    "prd": PRD,
    "spec": Spec,
    "code_change": CodeChange,
    "qa_report": QAReport,
    "research_answer": ResearchAnswer,
}
