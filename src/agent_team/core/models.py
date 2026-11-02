"""Shared data contracts: tasks, runs, envelopes, results, findings, decisions, events.

These types are the seams between the packages. Additive changes (new optional
fields) are safe; renames and removals touch every package that uses them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent_team.core.artifacts import Artifact, ArtifactRef


def new_id(prefix: str) -> str:
    """Short, sortable-enough ids such as ``task_3f9a1c2e``."""
    return f"{prefix}_{uuid4().hex[:8]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Contract(BaseModel):
    """Base for every contract model: strict about unknown fields."""

    model_config = ConfigDict(extra="forbid")


# --- enums -------------------------------------------------------------------------


class Phase(StrEnum):
    PLANNING = "planning"
    PRD = "prd"
    SPEC = "spec"
    IMPLEMENTATION = "implementation"


PHASE_ORDER: tuple[Phase, ...] = (Phase.PLANNING, Phase.PRD, Phase.SPEC, Phase.IMPLEMENTATION)


class AgentName(StrEnum):
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    PRODUCT = "product"
    ARCHITECT = "architect"
    ENGINEER = "engineer"
    QA = "qa"
    AUDITOR = "auditor"
    USER = "user"  # reviews and questions are tasks assigned to the user


class TaskStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    AUDITING = "auditing"
    DONE = "done"
    RETRY = "retry"
    ESCALATED = "escalated"
    WAITING = "waiting"  # waiting on the user's answer or review
    SUPERSEDED = "superseded"  # inputs went stale; kept for history


ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.READY, TaskStatus.SUPERSEDED}),
    TaskStatus.READY: frozenset({TaskStatus.RUNNING, TaskStatus.SUPERSEDED}),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.AUDITING,
            TaskStatus.RETRY,
            TaskStatus.ESCALATED,
            TaskStatus.WAITING,
            TaskStatus.READY,  # resumed after a crash or a pause
        }
    ),
    TaskStatus.AUDITING: frozenset(
        {TaskStatus.DONE, TaskStatus.RETRY, TaskStatus.ESCALATED, TaskStatus.WAITING}
    ),
    TaskStatus.RETRY: frozenset({TaskStatus.READY, TaskStatus.SUPERSEDED}),
    TaskStatus.WAITING: frozenset({TaskStatus.READY, TaskStatus.DONE, TaskStatus.SUPERSEDED}),
    TaskStatus.ESCALATED: frozenset({TaskStatus.READY, TaskStatus.SUPERSEDED}),
    TaskStatus.DONE: frozenset({TaskStatus.SUPERSEDED}),
    TaskStatus.SUPERSEDED: frozenset(),
}


def can_transition(src: TaskStatus, dst: TaskStatus) -> bool:
    return dst in ALLOWED_TRANSITIONS[src]


class TaskOrigin(StrEnum):
    PLAN = "plan"
    RETRY = "retry"
    REVISION = "revision"  # upstream changed; update, don't rebuild
    RESEARCH_REQUEST = "research_request"
    REVIEW = "review"
    QUESTION = "question"
    CHANGE_REQUEST = "change_request"
    GATE_FIX = "gate_fix"  # a failed phase gate routed a finding back to this task
    USER_RERUN = "user_rerun"


class RunStatus(StrEnum):
    ACTIVE = "active"
    WAITING = "waiting"
    PAUSED = "paused"
    STOPPED = "stopped"
    DONE = "done"


class RequestKind(StrEnum):
    """How the orchestrator classifies each user message."""

    BUILD = "build"
    CHANGE = "change"
    RUN_QUESTION = "run_question"
    RESEARCH_QUESTION = "research_question"


class ReviewMode(StrEnum):
    EVERY_PHASE = "every_phase"
    QUESTIONS_ONLY = "questions_only"
    AUTO_ASSUMPTIONS = "auto_assumptions"


class Severity(StrEnum):
    BLOCKER = "blocker"
    WARNING = "warning"


class RuleScope(StrEnum):
    TASK = "task"
    PHASE = "phase"
    RUN = "run"


# --- runs and tasks ----------------------------------------------------------------


class Usage(Contract):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class Run(Contract):
    id: str = Field(default_factory=lambda: new_id("run"))
    title: str
    request: str
    status: RunStatus = RunStatus.ACTIVE
    review_mode: ReviewMode = ReviewMode.AUTO_ASSUMPTIONS
    workspace: str  # absolute path to runs/<run_id>/workspace
    context_dirs: list[str] = Field(default_factory=list)  # read-only folders from --context
    upstream_reruns: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Task(Contract):
    id: str = Field(default_factory=lambda: new_id("task"))
    run_id: str
    phase: Phase
    agent: AgentName
    objective: str
    inputs: list[ArtifactRef] = Field(default_factory=list)
    output_kind: str  # artifact kind this task must produce, e.g. "prd"
    depends_on: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    attempt: int = 0
    max_attempts: int = 3
    origin: TaskOrigin = TaskOrigin.PLAN
    parent_task_id: str | None = None  # the task this one came from (retry, revision...)
    reason: str | None = None  # why it exists, in words, for the decision log
    model: str | None = None  # the model that ran the latest attempt
    output: ArtifactRef | None = None
    finding_ids: list[str] = Field(default_factory=list)  # findings from the last verdict
    last_error: str | None = None
    last_traceback: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None


# --- delegation --------------------------------------------------------------------


class FeedbackSource(StrEnum):
    USER_ANSWER = "user_answer"
    USER_COMMENT = "user_comment"
    AUDITOR = "auditor"
    QA = "qa"


class Feedback(Contract):
    id: str = Field(default_factory=lambda: new_id("fb"))
    artifact: ArtifactRef | None
    source: FeedbackSource
    text: str
    question_id: str | None = None  # set when this answers an agent question
    created_at: datetime = Field(default_factory=utcnow)


class TaskEnvelope(Contract):
    """Everything an expert gets for one attempt. Built by the dispatcher, never by agents."""

    task: Task
    run: Run
    inputs: list[Artifact] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    memories: list[str] = Field(default_factory=list)  # pre-rendered "Relevant memory" lines
    feedback: list[Feedback] = Field(default_factory=list)  # answers, comments, findings
    findings: list[Finding] = Field(default_factory=list)  # from the last failed verdict
    previous_output: Artifact | None = None  # set on retries and revisions
    upstream_diff: str | None = None  # set on revisions
    trace_id: str
    span_id: str


class ResultStatus(StrEnum):
    OK = "ok"  # artifact produced
    QUESTIONS = "questions"  # agent can't proceed safely without the user's answers
    RESEARCH_REQUEST = "research_request"  # agent hit a knowledge gap
    ERROR = "error"


class AgentResult(Contract):
    task_id: str
    status: ResultStatus
    artifact: Artifact | None = None
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
    error_type: str | None = None
    error: str | None = None
    traceback: str | None = None


# --- auditing ----------------------------------------------------------------------


class Rule(Contract):
    id: str  # e.g. "R3", "S2", "P1"
    title: str
    scope: RuleScope
    phases: list[Phase] = Field(default_factory=list)  # empty = every phase
    agents: list[AgentName] = Field(default_factory=list)  # empty = every agent
    severity: Severity
    statement: str  # plain-language rule the auditor model applies
    helper: str | None = None  # name of the evidence helper, if any
    precheck: bool = False  # also enforced before tool calls
    enabled: bool = True


class Finding(Contract):
    id: str = Field(default_factory=lambda: new_id("finding"))
    rule_id: str
    severity: Severity
    artifact: ArtifactRef | None = None  # the artifact at fault (may be upstream)
    item: str | None = None  # e.g. "REQ-3", "src/app/routes.py"
    message: str
    evidence: str | None = None
    suggested_fix: str | None = None
    from_helper: bool = False  # True when an evidence helper produced or confirmed it


class Verdict(Contract):
    id: str = Field(default_factory=lambda: new_id("verdict"))
    run_id: str
    scope: RuleScope
    task_id: str | None = None
    phase: Phase | None = None
    passed: bool
    summary: str | None = None  # the auditor model's short explanation of the verdict
    findings: list[Finding] = Field(default_factory=list)
    rules_checked: list[str] = Field(default_factory=list)
    model: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.BLOCKER]


class PrecheckDecision(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    ASK_USER = "ask_user"


class PrecheckResult(Contract):
    decision: PrecheckDecision
    rule_id: str | None = None
    reason: str | None = None


# --- decisions and events ------------------------------------------------------------


class DecisionKind(StrEnum):
    CLASSIFY_REQUEST = "classify_request"
    PLAN = "plan"
    DELEGATE = "delegate"
    RETRY = "retry"
    ESCALATE = "escalate"
    STRONGER_MODEL = "stronger_model"
    UPSTREAM_RERUN = "upstream_rerun"
    GATE_FIX = "gate_fix"
    REOPEN_PHASE = "reopen_phase"
    ASK_USER = "ask_user"
    NO_PROGRESS = "no_progress"
    STOP = "stop"


class Decision(Contract):
    id: str = Field(default_factory=lambda: new_id("decision"))
    run_id: str
    task_id: str | None = None
    kind: DecisionKind
    inputs: dict[str, Any] = Field(default_factory=dict)
    choice: str
    reason: str
    made_by: Literal["code", "model"]
    rule: str | None = None  # the policy or rule that applied, for code-made choices
    created_at: datetime = Field(default_factory=utcnow)


class EventName(StrEnum):
    RUN_STARTED = "run_started"
    PHASE_STARTED = "phase_started"
    PHASE_FINISHED = "phase_finished"
    TASK_STARTED = "task_started"
    TASK_FINISHED = "task_finished"
    TASK_STATE_CHANGED = "task_state_changed"
    LLM_CALL_BEFORE = "llm_call_before"
    LLM_CALL_AFTER = "llm_call_after"
    TOOL_CALL_BEFORE = "tool_call_before"
    TOOL_CALL_AFTER = "tool_call_after"
    AUDIT_VERDICT = "audit_verdict"
    DECISION = "decision"
    TASK_RETRIED = "task_retried"
    TASK_ESCALATED = "task_escalated"
    QUESTION_ASKED = "question_asked"
    REVIEW_REQUESTED = "review_requested"
    USER_MESSAGE = "user_message"
    MEMORY_FETCHED = "memory_fetched"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CONTEXT_COMPACTED = "context_compacted"
    ERROR = "error"
    RUN_FINISHED = "run_finished"


class Event(Contract):
    name: EventName
    run_id: str
    ts: datetime = Field(default_factory=utcnow)
    task_id: str | None = None
    attempt: int | None = None
    agent: AgentName | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    level: Literal["debug", "info", "warning", "error"] = "info"
    data: dict[str, Any] = Field(default_factory=dict)


TaskEnvelope.model_rebuild()
