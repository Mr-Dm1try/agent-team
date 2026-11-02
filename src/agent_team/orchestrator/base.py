"""Shared state and helpers for the orchestrator's parts: the services it holds, the
decision log, run status, feedback routing and re-opening work."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from agent_team.core.artifacts import Artifact, ArtifactRef
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
    Decision,
    Event,
    EventName,
    Finding,
    Phase,
    ReviewMode,
    RuleScope,
    Run,
    RunStatus,
    Severity,
    Task,
    TaskOrigin,
    Verdict,
)
from agent_team.llm.types import LLMClient
from agent_team.orchestrator.lineage import reopen
from agent_team.orchestrator.policy import SimplePolicy

APPROVE = re.compile(
    r"^\s*(approve[ds]?|ok(ay)?|yes|lgtm|looks good|accept(ed)?|ship it)[\s.!]*$", re.I
)
CHILD_ORIGINS = frozenset({TaskOrigin.QUESTION, TaskOrigin.RESEARCH_REQUEST})
#: User-task reasons for questions that aren't tied to one task.
GATE_QUESTION = "gate"
RUN_QUESTION = "run_audit"
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

    async def _audit(self, run: Run, task: Task, artifact: Artifact | None) -> None:
        raise NotImplementedError  # the task sweep lives in Orchestrator

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
        await self.decisions.record(decision)

    async def _set_status(self, run_id: str, status: RunStatus) -> Run:
        # Re-read first, so fields others keep up to date (usage) aren't overwritten.
        run = await self.tasks.get_run(run_id)
        if run.status is status:
            return run
        return await self.tasks.update_run(run.model_copy(update={"status": status}))

    async def _pause(self, run_id: str, exc: Exception) -> None:
        status = RunStatus.STOPPED if isinstance(exc, RunStopped) else RunStatus.PAUSED
        await self._set_status(run_id, status)

    async def _feedback_ref(self, run: Run, task: Task) -> ArtifactRef:
        """Feedback is looked up by artifact kind, so feedback on a task that has no
        output yet points at the latest version of its kind (or v1 before any)."""
        if task.output is not None:
            return task.output
        latest = await self.artifacts.latest(run.id, task.output_kind)
        return ArtifactRef(kind=task.output_kind, version=latest.version if latest else 1)

    async def _emit_task(self, name: EventName, task: Task, data: dict[str, object]) -> None:
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


def failed_verdict(
    run: Run, message: str, *, task_id: str | None = None, phase: Phase | None = None
) -> Verdict:
    """A failing verdict with one AUDIT blocker, used when no real verdict can be had."""
    scope = RuleScope.TASK if task_id else RuleScope.PHASE if phase else RuleScope.RUN
    return Verdict(
        run_id=run.id,
        scope=scope,
        task_id=task_id,
        phase=phase,
        passed=False,
        findings=[Finding(rule_id="AUDIT", severity=Severity.BLOCKER, message=message)],
    )
