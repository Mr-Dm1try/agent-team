"""Shared state and helpers for the orchestrator's parts: the services it holds, the
decision log, run status, feedback routing and re-opening work."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from agent_team.core.artifacts import Artifact
from agent_team.core.errors import BudgetExceeded, RunStopped
from agent_team.core.interfaces import (
    ArtifactStore,
    Auditor,
    CheckpointStore,
    DecisionLog,
    Dispatcher,
    EventBus,
    FeedbackStore,
    TaskStore,
    VerdictPolicy,
    VerdictStore,
)
from agent_team.core.models import (
    AUDIT_RULE,
    DANGEROUS_ACTION_RULES,
    AgentName,
    Decision,
    Event,
    EventName,
    Feedback,
    FeedbackSource,
    Finding,
    Phase,
    RequestKind,
    ReviewMode,
    RuleScope,
    Run,
    RunStatus,
    Severity,
    Task,
    TaskOrigin,
    TaskStatus,
    Verdict,
)
from agent_team.llm.types import LLMClient
from agent_team.orchestrator.lineage import reopen
from agent_team.orchestrator.policy import SimplePolicy
from agent_team.orchestrator.replies import Reply

APPROVE = re.compile(
    r"^\s*(approve[ds]?|ok(ay)?|yes|lgtm|looks good|accept(ed)?|ship it)[\s.!]*$", re.I
)
CHILD_ORIGINS = frozenset({TaskOrigin.QUESTION, TaskOrigin.RESEARCH_REQUEST})
#: The user-task reason for a question about the whole-run audit, which isn't tied to
#: one task.
RUN_QUESTION = "run_audit"
#: The user-task reason for a question the verdict policy asks about one task's check.
TASK_QUESTION = "policy"
#: The user-task reason for a question about a task that is stuck.
ESCALATION = "escalation"
#: Added to questions about work the user may keep as it is.
KEEP_HINT = " Reply 'approve' to keep the work as it is."
PAUSING = (BudgetExceeded, RunStopped)


@dataclass
class OrchestratorConfig:
    """Where runs live, the review mode and the caps that make every run end."""

    runs_dir: Path
    review_mode: ReviewMode = ReviewMode.AUTO_ASSUMPTIONS
    max_attempts: int = 3
    max_upstream_reruns: int = 2
    max_loops: int = 500  # safety valve for drive()


class OrchestratorBase:
    """The services and helpers that the orchestrator's mixins share."""

    def __init__(
        self,
        *,
        tasks: TaskStore,
        artifacts: ArtifactStore,
        checkpoints: CheckpointStore,
        decisions: DecisionLog,
        feedback: FeedbackStore,
        verdicts: VerdictStore,
        bus: EventBus,
        llm: LLMClient,
        auditor: Auditor,
        dispatcher: Dispatcher,
        config: OrchestratorConfig,
        policy: VerdictPolicy | None = None,
    ) -> None:
        """Keep the stores and collaborators the mixins share."""
        self.tasks = tasks
        self.artifacts = artifacts
        self.checkpoints = checkpoints
        self.decisions = decisions
        self.feedback = feedback
        self.verdicts = verdicts
        self.bus = bus
        self.llm = llm
        self.auditor = auditor
        self.dispatcher = dispatcher
        self.config = config
        self.policy: VerdictPolicy = policy or SimplePolicy()
        self._started: dict[str, set[Phase]] = {}  # only to emit phase_started once
        #: (phase, newest task id) pairs already reported finished, per run.
        self._finished: dict[str, set[tuple[Phase, str]]] = {}

    async def _audit(self, run: Run, task: Task, artifact: Artifact | None) -> None:
        """The task sweep; the mixins call it and Orchestrator implements it."""
        raise NotImplementedError  # the task sweep lives in Orchestrator

    async def drive(self, run_id: str) -> Reply:
        """Run until the run is done, waits on the user, or stops; see Orchestrator."""
        raise NotImplementedError  # the drive loop lives in Orchestrator

    async def start_run(
        self,
        request: str,
        *,
        kind: RequestKind = RequestKind.BUILD,
        title: str | None = None,
        context_dirs: list[str] | None = None,
        review_mode: ReviewMode | None = None,
    ) -> Run:
        """Create a run and its planning task; see Orchestrator."""
        raise NotImplementedError  # run setup lives in Orchestrator

    async def stop(self, run_id: str, reason: str = "stopped by the user") -> None:
        """The kill switch; see Orchestrator."""
        raise NotImplementedError  # the kill switch lives in Orchestrator

    async def _pending(self, run_id: str) -> list[Task]:
        """User tasks waiting on an answer or review."""
        return [
            t
            for t in await self.tasks.list_tasks(run_id, status=TaskStatus.WAITING)
            if t.agent is AgentName.USER
        ]

    async def _save_verdict(self, verdict: Verdict) -> None:
        """Store a verdict and emit audit_verdict for it."""
        await self.verdicts.save(verdict)
        await self.bus.emit(
            Event(
                name=EventName.AUDIT_VERDICT,
                run_id=verdict.run_id,
                task_id=verdict.task_id,
                agent=AgentName.AUDITOR,
                level="info" if verdict.passed else "warning",
                data={
                    "verdict_id": verdict.id,
                    "scope": verdict.scope.value,
                    "phase": verdict.phase.value if verdict.phase else None,
                    "passed": verdict.passed,
                    "findings": [f.model_dump(mode="json") for f in verdict.findings],
                },
            )
        )

    async def reopen(
        self,
        run: Run,
        task: Task,
        *,
        origin: TaskOrigin,
        reason: str | None = None,
        finding_ids: list[str] | None = None,
    ) -> Task | None:
        """Re-run ``task`` and everything built on it as revisions (see lineage.py)."""
        return await reopen(
            self.tasks,
            self.decisions,
            run,
            task,
            origin=origin,
            reason=reason,
            finding_ids=finding_ids,
        )

    async def _record(self, decision: Decision) -> None:
        """Add a decision to the run's decision log."""
        await self.decisions.record(decision)

    async def _set_status(self, run_id: str, status: RunStatus) -> Run:
        # Re-read first, so fields others keep up to date (usage) aren't overwritten.
        """Change the run's status, re-reading it first so fields others keep up to date (usage)
        aren't overwritten."""
        run = await self.tasks.get_run(run_id)
        if run.status is status:
            return run
        return await self.tasks.update_run(run.model_copy(update={"status": status}))

    async def _pause(self, run_id: str, exc: Exception) -> None:
        """Stop on RunStopped, pause on BudgetExceeded; either way the run can resume."""
        status = RunStatus.STOPPED if isinstance(exc, RunStopped) else RunStatus.PAUSED
        await self._set_status(run_id, status)

    @staticmethod
    def _feedback(
        task: Task, source: FeedbackSource, text: str, *, question_id: str | None = None
    ) -> Feedback:
        """Feedback for ``task``'s next attempt: on its output when it has one, and for
        the task itself either way, so a task with no output yet still gets it."""
        return Feedback(
            artifact=task.output,
            task_id=task.id,
            source=source,
            text=text,
            question_id=question_id,
        )

    async def _emit_task(self, name: EventName, task: Task, data: dict[str, object]) -> None:
        """Emit an event about one task."""
        await self.bus.emit(
            Event(
                name=name,
                run_id=task.run_id,
                task_id=task.id,
                attempt=task.attempt,
                agent=task.agent,
                data=dict(data),
            )
        )


def can_keep(task: Task, verdict: Verdict | None) -> bool:
    """Whether the user may keep this task's output although ``verdict`` failed it: the
    output exists and isn't the plan (the pipeline is built from it), and the check
    didn't fail closed or find a dangerous action, which are never waved through."""
    if task.output is None or task.output_kind == "plan":
        return False
    if verdict is None or verdict.passed:
        return False
    return not any(
        f.rule_id == AUDIT_RULE or f.rule_id in DANGEROUS_ACTION_RULES for f in verdict.blockers
    )


def failed_verdict(run: Run, message: str, *, task_id: str | None = None) -> Verdict:
    """A failing verdict with one AUDIT blocker, used when no real verdict can be had."""
    return Verdict(
        run_id=run.id,
        scope=RuleScope.TASK if task_id else RuleScope.RUN,
        task_id=task_id,
        passed=False,
        findings=[Finding(rule_id=AUDIT_RULE, severity=Severity.BLOCKER, message=message)],
    )
