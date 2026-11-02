"""Shared data contracts: tasks, runs, envelopes, results, findings, decisions, events.

These types are the seams between the packages. Additive changes (new optional
fields) are safe; renames and removals touch every package that uses them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from agent_team.core.artifacts import Artifact, ArtifactRef
from agent_team.core.base import Contract


def new_id(prefix: str) -> str:
    """Random ids such as ``task_3f9a1c2e4b5d6a7f``: 64 random bits, so collisions are
    negligible across every run in one database. Not sortable; order by ``created_at``."""
    return f"{prefix}_{uuid4().hex[:16]}"


def utcnow() -> datetime:
    """The current time in UTC, timezone-aware. Every timestamp in the store uses it."""
    return datetime.now(UTC)


# --- enums -------------------------------------------------------------------------


class Phase(StrEnum):
    """The four phases a build goes through, in order (see PHASE_ORDER)."""

    PLANNING = "planning"
    PRD = "prd"
    SPEC = "spec"
    IMPLEMENTATION = "implementation"


PHASE_ORDER: tuple[Phase, ...] = (Phase.PLANNING, Phase.PRD, Phase.SPEC, Phase.IMPLEMENTATION)


class AgentName(StrEnum):
    """Who a task is assigned to: an expert, the orchestrator, the auditor or the user."""

    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    PRODUCT = "product"
    ARCHITECT = "architect"
    ENGINEER = "engineer"
    QA = "qa"
    AUDITOR = "auditor"
    USER = "user"  # reviews and questions are tasks assigned to the user


class TaskStatus(StrEnum):
    """Where a task is in its life. Changes only through TaskStore.transition,
    which checks ALLOWED_TRANSITIONS."""

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
            TaskStatus.SUPERSEDED,  # a change made its inputs stale mid-attempt
        }
    ),
    TaskStatus.AUDITING: frozenset(
        {
            TaskStatus.DONE,
            TaskStatus.RETRY,
            TaskStatus.ESCALATED,
            TaskStatus.WAITING,
            TaskStatus.SUPERSEDED,
        }
    ),
    TaskStatus.RETRY: frozenset({TaskStatus.READY, TaskStatus.SUPERSEDED}),
    TaskStatus.WAITING: frozenset({TaskStatus.READY, TaskStatus.DONE, TaskStatus.SUPERSEDED}),
    TaskStatus.ESCALATED: frozenset({TaskStatus.READY, TaskStatus.SUPERSEDED}),
    TaskStatus.DONE: frozenset({TaskStatus.SUPERSEDED}),
    TaskStatus.SUPERSEDED: frozenset(),
}


def can_transition(src: TaskStatus, dst: TaskStatus) -> bool:
    """Whether a task may move from ``src`` to ``dst``."""
    return dst in ALLOWED_TRANSITIONS[src]


class TaskOrigin(StrEnum):
    """Why a task exists: planned by the pipeline, or created later by a retry,
    revision, research request, review, question, change request or gate fix."""

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
    """Where a run is: working, waiting on the user, paused, stopped or done."""

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
    """How much the run involves the user: review after every phase, only answer
    agents' questions, or record assumptions and never stop."""

    EVERY_PHASE = "every_phase"
    QUESTIONS_ONLY = "questions_only"
    AUTO_ASSUMPTIONS = "auto_assumptions"


class Severity(StrEnum):
    """How much a finding matters: a blocker fails the step, a warning doesn't."""

    BLOCKER = "blocker"
    WARNING = "warning"


class RuleScope(StrEnum):
    """What an audit covers: one task's output, a whole phase at its gate, or the
    whole run at the end."""

    TASK = "task"
    PHASE = "phase"
    RUN = "run"


# --- runs and tasks ----------------------------------------------------------------


class Usage(Contract):
    """Tokens and cost for one model call, or summed over many with ``+``."""

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
    """One user request and everything done for it, with its own workspace,
    artifacts, tasks and budget."""

    id: str = Field(default_factory=lambda: new_id("run"))
    title: str
    request: str
    status: RunStatus = RunStatus.ACTIVE
    review_mode: ReviewMode = ReviewMode.AUTO_ASSUMPTIONS
    workspace: str  # absolute path to runs/<run_id>/workspace
    context_dirs: list[str] = Field(default_factory=list)  # read-only folders from --context
    upstream_reruns: int = 0
    # Tokens and cost spent so far, saved with the run so a resumed run keeps counting
    # against its budget instead of starting from zero.
    usage: Usage = Field(default_factory=Usage)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Task(Contract):
    """One unit of delegated work: an agent, an objective, inputs and the artifact
    kind it must produce. Tracks its attempts, output, last error and lineage."""

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
    """Who gave a piece of feedback."""

    USER_ANSWER = "user_answer"
    USER_COMMENT = "user_comment"
    AUDITOR = "auditor"
    QA = "qa"


class Feedback(Contract):
    """A comment on an artifact, handed to the next attempt that revises it: a user
    answer, review comment, or a note from the auditor or QA."""

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
    """How an agent's attempt ended."""

    OK = "ok"  # artifact produced
    QUESTIONS = "questions"  # agent can't proceed safely without the user's answers
    RESEARCH_REQUEST = "research_request"  # agent hit a knowledge gap
    ERROR = "error"


class AgentResult(Contract):
    """What an agent returns for one attempt: the artifact, or why there is none,
    plus usage and, on error, the error type, message and traceback."""

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
    """One auditor rule from config/rules.yaml: what it checks, where it applies,
    how severe a miss is, and which code helper gathers its evidence."""

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
    """One rule the auditor found broken, with the artifact and item at fault, the
    evidence and a suggested fix."""

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
    """The outcome of one audit: pass or fail, the findings, and which rules were
    checked. Can't pass while it holds a blocker."""

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
        """The findings with blocker severity."""
        return [f for f in self.findings if f.severity is Severity.BLOCKER]

    @model_validator(mode="after")
    def _no_pass_with_blockers(self) -> Verdict:
        """Fail closed: a verdict that holds a blocker can't be a pass, whoever built it."""
        if self.passed and self.blockers:
            raise ValueError("a verdict with blocker findings cannot pass")
        return self


class PrecheckDecision(StrEnum):
    """The pre-check's answer for a tool call."""

    ALLOW = "allow"
    BLOCK = "block"
    ASK_USER = "ask_user"


class PrecheckResult(Contract):
    """The pre-check's answer for a tool call, with the rule and reason when it
    blocks or asks the user."""

    decision: PrecheckDecision
    rule_id: str | None = None
    reason: str | None = None


# --- hooks and tools -----------------------------------------------------------------


class HookOutcome(Contract):
    """What a before hook returns. ``allow=False`` blocks the action."""

    allow: bool = True
    reason: str | None = None
    rule_id: str | None = None
    replacement: dict[str, Any] | None = None  # modified payload, if the hook rewrote it


class ToolResult(Contract):
    """What a tool returns. ``content`` goes back to the model; ``data`` is structured
    detail for the trace and the auditor's evidence helpers."""

    content: str
    is_error: bool = False
    data: dict[str, Any] | None = None


# --- decisions and events ------------------------------------------------------------


class DecisionKind(StrEnum):
    """The kinds of choices the orchestrator records in the decision log."""

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
    """One recorded choice: what was decided, from which inputs, why, and whether
    code (with the rule) or the model made it."""

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
    """Every event the bus carries. The tracer writes them all to the run's JSONL
    trace."""

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
    """One thing that happened in a run, with trace ids for the span tree and
    free-form ``data``."""

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
