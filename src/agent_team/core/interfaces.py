"""Interfaces between the packages. Implementations live in:

- agent_team.core: EventBus, TaskStore, ArtifactStore, CheckpointStore, DecisionLog,
  FeedbackStore, VerdictStore; agent_team.llm: the LLMClient gateway.
- agent_team.agents: Agent implementations; agent_team.orchestrator: Dispatcher and
  Orchestrator; agent_team.tools: tools.
- agent_team.audit: Auditor, VerdictPolicy, rule store and evidence helpers.

Each package codes against these Protocols and tests with its own fakes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from agent_team.core.artifacts import Artifact, ArtifactRef
from agent_team.core.models import (
    AgentName,
    AgentResult,
    Decision,
    Event,
    EventName,
    Feedback,
    HookOutcome,
    Phase,
    PrecheckResult,
    Rule,
    Run,
    Task,
    TaskEnvelope,
    TaskStatus,
    ToolResult,
    Verdict,
)
from agent_team.llm.types import LLMClient, Message, ToolSpec

# --- events and hooks --------------------------------------------------------------


BeforeHook = Callable[[Event], Awaitable[HookOutcome]]
AfterHook = Callable[[Event], Awaitable[None]]


class EventBus(Protocol):
    """In-process pub/sub for everything that happens in a run. Before hooks can veto an
    action (the auditor's tool pre-check); after hooks observe it (tracer, live view,
    tool-call log)."""

    async def emit(self, event: Event) -> None:
        """Deliver to every after hook. Hook errors are logged, never raised."""
        ...

    async def check(self, event: Event) -> HookOutcome:
        """Run before hooks in order; the first block wins."""
        ...

    def on_before(self, name: EventName, hook: BeforeHook) -> None:
        """Register a hook that ``check`` runs for events named ``name``, in
        registration order."""
        ...

    def on_after(self, name: EventName | None, hook: AfterHook) -> None:
        """``name=None`` subscribes to every event (tracer, live view)."""
        ...


# --- stores ------------------------------------------------------------------------


class TaskStore(Protocol):
    """Runs and their tasks: the task tracker. Status lives here and changes only
    through ``transition``, so the tracker and the event log always agree."""

    async def create_run(self, run: Run) -> Run:
        """Insert a new run and return it as stored."""
        ...

    async def get_run(self, run_id: str) -> Run:
        """The run with this id. Raises KeyError if there is none."""
        ...

    async def update_run(self, run: Run) -> Run:
        """Overwrite the stored run (status, budget spent, review mode and so on)."""
        ...

    async def list_runs(self) -> list[Run]:
        """Every run, oldest first. Used to list and switch between runs."""
        ...

    async def create_task(self, task: Task) -> Task:
        """Insert a new task, normally in ``pending``, and return it as stored."""
        ...

    async def get_task(self, task_id: str) -> Task:
        """The task with this id. Raises KeyError if there is none."""
        ...

    async def update_task(self, task: Task) -> Task:
        """Persist non-status fields. Status changes must go through transition()."""
        ...

    async def transition(self, task_id: str, to: TaskStatus, *, reason: str) -> Task:
        """The only way to change status. Checks ALLOWED_TRANSITIONS (raises
        InvalidTransition), stamps timestamps and emits task_state_changed."""
        ...

    async def list_tasks(
        self,
        run_id: str,
        *,
        status: TaskStatus | None = None,
        phase: Phase | None = None,
    ) -> list[Task]:
        """A run's tasks in creation order, optionally filtered by status and phase."""
        ...

    async def ready_tasks(self, run_id: str) -> list[Task]:
        """Pending tasks whose depends_on are all done get promoted and returned."""
        ...


class ArtifactStore(Protocol):
    """Versioned artifacts (plan, PRD, spec, ...) and the lineage between them, which
    is how a change marks downstream work stale."""

    async def save(self, artifact: Artifact) -> ArtifactRef:
        """Assigns the next version for (run_id, kind), writes runs/<run>/artifacts/
        <kind>.v<N>.json and records lineage from artifact.built_from."""
        ...

    async def load(self, run_id: str, ref: ArtifactRef) -> Artifact:
        """One exact version, e.g. ``prd@v2``. Raises KeyError if it doesn't exist."""
        ...

    async def latest(self, run_id: str, kind: str) -> Artifact | None:
        """The highest version of ``kind`` in the run, or None if there is none yet."""
        ...

    async def dependents(self, run_id: str, ref: ArtifactRef) -> list[ArtifactRef]:
        """Every artifact version built, directly or not, from ``ref``."""
        ...


class CheckpointStore(Protocol):
    """An agent's conversation saved after every model and tool step, so a crashed or
    stopped attempt resumes where it was instead of starting over."""

    async def save_step(
        self, task_id: str, attempt: int, step: int, messages: list[Message]
    ) -> None:
        """Save the full message list after ``step`` of this attempt, replacing any
        earlier save of the same step, and emit ``checkpoint_saved``."""
        ...

    async def load_latest(self, task_id: str, attempt: int) -> tuple[int, list[Message]] | None:
        """(step, messages) of the last saved step, or None."""
        ...


class DecisionLog(Protocol):
    """Why the run went the way it did: every delegation, retry, escalation, re-run and
    question to the user, with its inputs and reason."""

    async def record(self, decision: Decision) -> None:
        """Stores the decision and emits a ``decision`` event."""
        ...

    async def list(self, run_id: str) -> list[Decision]:
        """A run's decisions in the order they were recorded."""
        ...


class FeedbackStore(Protocol):
    """Comments on artifacts from the user (reviews, answers, change requests) and from
    agents, handed to the next attempt that revises the artifact."""

    async def add(self, run_id: str, feedback: Feedback) -> None:
        """Store one piece of feedback, keyed by the kind of artifact it is about."""
        ...

    async def for_artifact(self, run_id: str, kind: str) -> list[Feedback]:
        """All feedback on any version of ``kind`` in the run, oldest first."""
        ...


class VerdictStore(Protocol):
    """Audit verdicts, kept so the policy can see whether a retry made progress."""

    async def save(self, verdict: Verdict) -> None:
        """Append a verdict. Verdicts are never overwritten."""
        ...

    async def latest_for_task(self, task_id: str) -> Verdict | None:
        """The most recent verdict on this task, or None if it was never audited."""
        ...

    async def for_task(self, task_id: str) -> list[Verdict]:
        """Every verdict on this task, oldest first: one per audited attempt."""
        ...


# --- tools and agents --------------------------------------------------------------


class ToolContext(Protocol):
    """The run and task a tool call belongs to, and the bus to emit events on."""

    run: Run
    task: Task
    bus: EventBus


class Tool(Protocol):
    """A function the model can call. ``spec`` is what the model sees."""

    spec: ToolSpec
    #: Safe to run again after a crash mid-call (reads are, writes and commands aren't).
    safe_to_repeat: bool

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Run the tool. Expected failures (bad path, command error) come back as
        ``is_error=True`` so the model can react, rather than as exceptions."""
        ...


class AgentContext(Protocol):
    """Services an agent may use during one attempt. Built by the dispatcher."""

    llm: LLMClient
    bus: EventBus
    checkpoints: CheckpointStore
    tools: dict[str, Tool]


class Agent(Protocol):
    """An expert (Researcher, Product, Architect, Engineer, QA) that turns one task
    envelope into one artifact of ``output_kind``."""

    name: AgentName
    output_kind: str

    async def run(self, envelope: TaskEnvelope, ctx: AgentContext) -> AgentResult:
        """One attempt. Must not raise for expected failures: return an AgentResult
        with status=error. The dispatcher still catches anything that escapes."""
        ...


# --- auditing ----------------------------------------------------------------------


class RuleStore(Protocol):
    """The auditor's rule catalog, loaded from config/rules.yaml into the database so
    rules can be switched on and off without editing the file."""

    async def seed(self, rules: list[Rule]) -> None:
        """Insert or update the catalog on start. A rule that was switched off stays
        off; everything else is taken from the file."""
        ...

    async def for_scope(
        self, scope: str, *, phase: Phase | None, agent: AgentName | None
    ) -> list[Rule]:
        """The enabled rules that apply to one audit: ``scope`` is task, phase or run,
        and a rule with no phases or agents listed applies to all of them. ``None``
        skips that filter."""
        ...


class Auditor(Protocol):
    """Checks work against the rule catalog. Code evidence helpers gather facts and
    the model judges each rule. It never runs code and never changes artifacts."""

    async def audit_task(self, task: Task, artifact: Artifact | None, run: Run) -> Verdict:
        """Task sweep. Fails closed: if no valid verdict can be produced, return a
        failing verdict with an ``AUDIT`` finding rather than raising."""
        ...

    async def audit_gate(self, run: Run, phase: Phase) -> Verdict:
        """Phase-gate sweep once every task in ``phase`` is done: checks the phase's
        artifacts together, e.g. that every requirement has a planned check. Fails
        closed like ``audit_task``."""
        ...

    async def audit_run(self, run: Run) -> Verdict:
        """Final sweep over the whole run before it is reported done: the process
        trail, i.e. every task has timestamps, a verdict and attempts within the limit,
        and every decision has a reason. Fails closed like ``audit_task``."""
        ...

    async def precheck(
        self, tool: str, args: dict[str, Any], task: Task, run: Run
    ) -> PrecheckResult:
        """Decide whether a tool call may run, before it runs. Clear violations
        are blocked by code; unclear ones go to the model. Called from the
        tool_call_before hook."""
        ...


class VerdictPolicy(Protocol):
    """Turns verdicts into Decisions (agent_team.audit); the orchestrator applies them.

    The orchestrator, not the auditor, saves verdicts to VerdictStore and emits
    ``audit_verdict``. Each Decision's ``inputs`` carries ``task_id`` (None on gate-level
    decisions) and usually ``finding_ids``, plus, by kind:

    - RETRY: "previous_output"
    - STRONGER_MODEL: "model"
    - UPSTREAM_RERUN: "artifact" (e.g. "spec@v1"), "phase", "gate", "producer_task_id"
    - GATE_FIX: "rerun_after" (ids of later tasks to re-run after the fix)
    - ASK_USER: "question"
    - ESCALATE, NO_PROGRESS, STOP: nothing extra

    A passing verdict returns an empty list; the orchestrator marks the task done.
    """

    async def on_task_verdict(
        self, run: Run, task: Task, verdict: Verdict, history: list[Verdict]
    ) -> list[Decision]:
        """What to do after a task audit. ``history`` is the task's earlier verdicts,
        used to spot a retry that failed the same rules on the same items."""
        ...

    async def on_gate_verdict(self, run: Run, phase: Phase, verdict: Verdict) -> list[Decision]:
        """What to do after a phase-gate audit: usually a GATE_FIX for each task whose
        artifact is at fault, or ASK_USER once fixes stop making progress."""
        ...


class Dispatcher(Protocol):
    """Runs one task on its agent and records the outcome."""

    async def dispatch(self, task: Task) -> AgentResult:
        """Build the envelope, run the agent with a timeout, catch and classify errors,
        save the artifact, move the task to auditing (or retry/escalated)."""
        ...
