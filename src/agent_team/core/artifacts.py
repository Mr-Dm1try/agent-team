"""Typed artifacts the experts produce, and the id chain that links them.

The chain: Plan.asks (ASK-1) -> PRD.requirements (REQ-1, cite asks) and
acceptance criteria (AC-1.1) -> Spec.components (cover REQ ids) and planned checks
(cover AC ids) -> CodeChange files -> QAReport results per AC id. Rules P1, R3, R4
and R9 check these links with code, so field names here are part of the contract.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, get_args

from pydantic import Field, model_validator

from agent_team.core.base import Contract

_REF = re.compile(r"^(?P<kind>[a-z_]+)@v(?P<version>\d+)$")


class ArtifactRef(Contract):
    """Points at one version of one artifact kind within a run, e.g. ``prd@v2``."""

    kind: str = Field(pattern=r"^[a-z_]+$")
    version: int = Field(ge=1)

    def __str__(self) -> str:
        """Short form for logs, e.g. ``prd@v2``."""
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
    """Something an agent suggests keeping for later runs. The memory writer saves a
    decision, or a research fact with a source the output cites, once the output passes
    its check; lessons and preferences come from the auditor and the user instead, so
    those proposals are left out."""

    kind: Literal["decision", "lesson", "preference", "research_fact"]
    text: str
    source: str | None = None  # the URL or file a research fact cites


class ResearchRequest(Contract):
    """An agent's request for facts it lacks. Becomes a Researcher task, and the
    asking task waits for it."""

    question: str
    why: str


#: Words that mark a note about how a source was read, not its title: "opened in this
#: attempt at offset 361000", "first 10,000 of 27,246 characters", "header only".
_READ_NOTE = re.compile(
    r"\boffset\s*=?\s*\d|\bcharacters\s+\d[\d,]*\s*(?:-|to)\s*\d"
    r"|\b\d[\d,]*\s+of\s+\d[\d,]*\s+characters\b|\bthis attempt\b|\bheader only\b"
    r"|\bnot (?:reached|read|re-?verified)\b",
    re.IGNORECASE,
)
#: A bracketed aside in a title after a space, such as "(opened for its header only)";
#: "seek(offset)" is part of the name.
_ASIDE = re.compile(r"\s+[(\[]([^()\[\]]*)[)\]]")
#: Where a title splits into parts: a semicolon, a comma, or a spaced dash.
_TITLE_PARTS = re.compile(r"(\s*;\s*|,\s+|\s+[-\u2013\u2014]\s+)")


def split_read_notes(title: str) -> tuple[str, list[str]]:
    """``title`` without its notes on how the source was read, and those notes. Only
    asides and trailing parts are taken, so the title's first part always stays."""
    notes: list[str] = []

    def aside(m: re.Match[str]) -> str:
        """Drop a bracketed aside that is a read note, keeping it in ``notes``."""
        if _READ_NOTE.search(m[1]):
            notes.append(m[1].strip())
            return ""
        return m[0]

    parts = _TITLE_PARTS.split(_ASIDE.sub(aside, title).strip())
    tail: list[str] = []
    while len(parts) > 1 and _READ_NOTE.search(parts[-1]):
        tail.insert(0, parts[-1].strip())
        del parts[-2:]
    return "".join(parts).strip(), notes + tail


class Source(Contract):
    """A file or URL the Researcher actually opened in this attempt."""

    ref: str  # URL or path the Researcher actually opened
    title: str | None = Field(
        default=None,
        description="The page or file title only, as the user would read it. How you read "
        "it goes in `read`, not here.",
    )
    read: str | None = Field(
        default=None,
        description="Which part you read, such as 'characters 0-10000 of 27246'. Kept "
        "for the audit trail; the user isn't shown it.",
    )

    @model_validator(mode="before")
    @classmethod
    def _read_notes_out_of_the_title(cls, data: Any) -> Any:
        """Move notes on how the source was read from ``title`` into ``read``, so the
        title the user sees is the page's own."""
        if not isinstance(data, dict) or not isinstance(data.get("title"), str):
            return data
        read = data.get("read")
        if read is not None and not isinstance(read, str):
            return data  # left for the field's own check to reject
        title, notes = split_read_notes(data["title"])
        if not notes:
            return data
        return {**data, "title": title or None, "read": "; ".join([r for r in [read] if r] + notes)}


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

    command: str = Field(
        description="The exact argv of the run_command call you report, joined by spaces, "
        "such as 'python -m pytest -q'. Only the command: no notes."
    )
    passed: int
    failed: int
    exit_code: int
    output_tail: str = Field(description="The last few lines of the test output, not all of it.")


class CodeChange(ArtifactBase):
    """What one Engineer attempt wrote: files, the commit, and its test run."""

    kind: Literal["code_change"] = "code_change"
    files_written: list[str] = Field(
        description="Workspace-relative paths of the files you wrote with write_file in this "
        "attempt. Paths only, never file contents."
    )
    commit: str | None = None  # git sha in the run workspace
    tests: TestRun | None = None
    spec_deviations: list[str] = Field(
        default_factory=list,
        description="Where the code departs from the spec and why, one line each, such as "
        "'mcp==1.30.0, not the spec's mcp==2.3.0: 2.3.0 has no mcp.server.fastmcp'. The "
        "next spec revision takes these in.",
    )


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
    answer: str = Field(
        description="The answer as the user reads it. Say plainly if part of the question "
        "stays unanswered, but leave out how you researched it, such as which parts of a "
        "page you read or what you didn't re-check; that goes in `gaps`.",
    )
    sources: list[Source] = Field(min_length=1)
    gaps: list[str] = Field(
        default_factory=list,
        description="Notes on how the answer was researched, one line each, such as a part "
        "of a source you didn't read or a claim you didn't re-check. Kept for the audit "
        "trail; the user isn't shown it.",
    )


Artifact = Annotated[
    Plan | ResearchBrief | PRD | Spec | CodeChange | QAReport | ResearchAnswer,
    Field(discriminator="kind"),
]

#: Artifact class by kind, derived from the union so the two can't drift apart.
ARTIFACT_TYPES: dict[str, type[ArtifactBase]] = {
    cls.model_fields["kind"].default: cls for cls in get_args(get_args(Artifact)[0])
}

#: Each artifact kind as the user would say it.
PLAIN_KINDS = {
    "plan": "the plan",
    "research_brief": "the research",
    "research_answer": "the answer",
    "prd": "the requirements",
    "spec": "the design",
    "code_change": "the code",
    "qa_report": "the QA check",
}


def plain_kind(kind: str) -> str:
    """An artifact kind as the user would say it."""
    return PLAIN_KINDS.get(kind, kind.replace("_", " "))
