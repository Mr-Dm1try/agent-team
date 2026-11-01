"""Interfaces between the build steps. Implementations live in their own packages:

- 2a: EventBus, TaskStore, ArtifactStore, CheckpointStore, DecisionLog, FeedbackStore
  (agent_team.core) and the LLMClient gateway (agent_team.llm).
- 2b: Agent implementations (agent_team.agents), Dispatcher and Orchestrator
  (agent_team.orchestrator), file tools (agent_team.tools).
- 2c: Auditor, rule store and evidence helpers (agent_team.audit).

Each step codes against these Protocols and tests with its own fakes, so the three
can be built in parallel. Change a Protocol only through the orchestrating thread.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from agent_team.core.artifacts import Artifact, ArtifactRef
from agent_team.core.models import (
    AgentName,
    AgentResult,
    Decision,
    Event,
    EventName,
    Feedback,
    Phase,
    PrecheckResult,
    Rule,
    Run,
    Task,
    TaskEnvelope,
    TaskStatus,
    Verdict,
)
from agent_team.llm.types import LLMClient, Message, ToolSpec

# --- events and hooks --------------------------------------------------------------


class HookOutcome(BaseModel):
    """What a before hook returns. ``allow=False`` blocks the action."""

    model_config = ConfigDict(extra="forbid")
    allow: bool = True
    reason: str | None = None
    rule_id: str | None = None
    replacement: dict[str, Any] | None = None  # modified payload, if the hook rewrote it


BeforeHook = Callable[[Event], Awaitable[HookOutcome]]
AfterHook = Callable[[Event], Awaitable[None]]


class EventBus(Protocol):
    async def emit(self, event: Event) -> None:
        """Deliver to every after hook. Hook errors are logged, never raised."""
        ...

    async def check(self, event: Event) -> HookOutcome:
        """Run before hooks in order; the first block wins."""
        ...

    def on_before(self, name: EventName, hook: BeforeHook) -> None: ...

    def on_after(self, name: EventName | None, hook: AfterHook) -> None:
        """``name=None`` subscribes to every event (tracer, live view)."""
        ...


# --- stores ------------------------------------------------------------------------


class TaskStore(Protocol):
    async def create_run(self, run: Run) -> Run: ...

    async def get_run(self, run_id: str) -> Run: ...

    async def update_run(self, run: Run) -> Run: ...

    async def list_runs(self) -> list[Run]: ...

    async def create_task(self, task: Task) -> Task: ...

    async def get_task(self, task_id: str) -> Task: ...

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
    ) -> list[Task]: ...

    async def ready_tasks(self, run_id: str) -> list[Task]:
        """Pending tasks whose depends_on are all done get promoted and returned."""
        ...


class ArtifactStore(Protocol):
    async def save(self, artifact: Artifact) -> ArtifactRef:
        """Assigns the next version for (run_id, kind), writes runs/<run>/artifacts/
        <kind>.v<N>.json and records lineage from artifact.built_from."""
        ...

    async def load(self, run_id: str, ref: ArtifactRef) -> Artifact: ...

    async def latest(self, run_id: str, kind: str) -> Artifact | None: ...

    async def dependents(self, run_id: str, ref: ArtifactRef) -> list[ArtifactRef]:
        """Every artifact version built, directly or not, from ``ref``."""
        ...


class CheckpointStore(Protocol):
    async def save_step(
        self, task_id: str, attempt: int, step: int, messages: list[Message]
    ) -> None: ...

    async def load_latest(self, task_id: str, attempt: int) -> tuple[int, list[Message]] | None:
        """(step, messages) of the last saved step, or None."""
        ...


class DecisionLog(Protocol):
    async def record(self, decision: Decision) -> None:
        """Stores the decision and emits a ``decision`` event."""
        ...

    async def list(self, run_id: str) -> list[Decision]: ...


class FeedbackStore(Protocol):
    async def add(self, run_id: str, feedback: Feedback) -> None: ...

    async def for_artifact(self, run_id: str, kind: str) -> list[Feedback]: ...


class VerdictStore(Protocol):
    async def save(self, verdict: Verdict) -> None: ...

    async def latest_for_task(self, task_id: str) -> Verdict | None: ...

    async def for_task(self, task_id: str) -> list[Verdict]: ...


# --- tools and agents --------------------------------------------------------------


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str
    is_error: bool = False
    data: dict[str, Any] | None = None


class ToolContext(Protocol):
    run: Run
    task: Task
    bus: EventBus


class Tool(Protocol):
    spec: ToolSpec
    #: Safe to run again after a crash mid-call (reads are, writes and commands aren't).
    safe_to_repeat: bool

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...


class AgentContext(Protocol):
    """Services an agent may use during one attempt. Built by the dispatcher."""

    llm: LLMClient
    bus: EventBus
    checkpoints: CheckpointStore
    tools: dict[str, Tool]


class Agent(Protocol):
    name: AgentName
    output_kind: str

    async def run(self, envelope: TaskEnvelope, ctx: AgentContext) -> AgentResult:
        """One attempt. Must not raise for expected failures: return an AgentResult
        with status=error. The dispatcher still catches anything that escapes."""
        ...


# --- auditing ----------------------------------------------------------------------


class RuleStore(Protocol):
    async def seed(self, rules: list[Rule]) -> None: ...

    async def for_scope(
        self, scope: str, *, phase: Phase | None, agent: AgentName | None
    ) -> list[Rule]: ...


class Auditor(Protocol):
    async def audit_task(self, task: Task, artifact: Artifact | None, run: Run) -> Verdict:
        """Task sweep. Fails closed: if no valid verdict can be produced, return a
        failing verdict with an ``AUDIT`` finding rather than raising."""
        ...

    async def audit_gate(self, run: Run, phase: Phase) -> Verdict: ...

    async def audit_run(self, run: Run) -> Verdict: ...

    async def precheck(
        self, tool: str, args: dict[str, Any], task: Task, run: Run
    ) -> PrecheckResult:
        """Called from the tool_call_before hook."""
        ...


class Dispatcher(Protocol):
    async def dispatch(self, task: Task) -> AgentResult:
        """Build the envelope, run the agent with a timeout, catch and classify errors,
        save the artifact, move the task to auditing (or retry/escalated)."""
        ...
