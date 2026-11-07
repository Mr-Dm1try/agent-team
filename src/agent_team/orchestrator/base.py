"""Shared state and helpers for the orchestrator's parts: the services it holds, the
decision log, run status, feedback routing and re-opening work."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from agent_team.core.artifacts import Artifact, plain_kind
from agent_team.core.errors import BudgetExceeded, RunStopped
from agent_team.core.interfaces import (
    ArtifactStore,
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
    SEEN_CALLS,
    AgentName,
    Decision,
    DecisionKind,
    Event,
    EventName,
    Feedback,
    FeedbackSource,
    Finding,
    Phase,
    RequestKind,
    ReviewMode,
    Run,
    RunStatus,
    Task,
    TaskOrigin,
    TaskStatus,
    Verdict,
    is_safety_stop,
)
from agent_team.core.tracing import Tracer
from agent_team.llm.types import LLMClient
from agent_team.orchestrator.lineage import reopen
from agent_team.orchestrator.policy import SimplePolicy
from agent_team.orchestrator.replies import Reply
from agent_team.orchestrator.research import ResearchLimits

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


#: QA rules whose blockers the code must fix: planned files that don't exist (R5) and
#: the blocking issues in QA's review (R7).
CODE_RULES = frozenset({"R5", "R7"})
#: Guidance that names the code, so another try of QA's review reruns the code first.
NAMES_CODE = re.compile(r"\b(code|engineer|implement\w*)\b|\.py\b", re.I)
#: Words that name a code without meaning the code, such as "exit code 5".
OTHER_CODES = re.compile(r"\b(exit|status|error|return|http|response)\s+codes?\b", re.I)
#: Guidance that says to leave the code alone, such as "don't touch the code".
LEAVE_CODE = re.compile(
    r"\b(don't|do not|not|never|without|leave)\b(\W+\w+){0,3}?\W+(code|engineer)\b", re.I
)


def code_blockers(verdict: Verdict | None) -> list[Finding]:
    """The blockers of a failed check that the code must fix: findings on a code
    change, and QA's own code rules."""
    if verdict is None or verdict.passed:
        return []
    return [
        f
        for f in verdict.blockers
        if (f.artifact is not None and f.artifact.kind == "code_change") or f.rule_id in CODE_RULES
    ]


def names_code(guidance: str) -> bool:
    """Whether guidance asks for work on the code: it names the code, an Engineer or a
    Python file, and doesn't say to leave the code alone."""
    text = OTHER_CODES.sub("", guidance)
    return NAMES_CODE.search(text) is not None and LEAVE_CODE.search(text) is None


def reruns_code(task: Task, verdict: Verdict | None, guidance: str = "") -> bool:
    """Whether guidance for another try of a stuck QA review reruns the code first:
    its check failed on the code, or the guidance names the code. Guidance that says
    to leave the code alone never does."""
    if task.agent is not AgentName.QA or LEAVE_CODE.search(guidance):
        return False
    return bool(code_blockers(verdict)) or names_code(guidance)


def rerun_of(task: Task, verdict: Verdict | None) -> str:
    """The work that guidance for another try of ``task`` reruns, as the user says it."""
    if task.agent is not AgentName.QA:
        return plain_kind(task.output_kind)
    if reruns_code(task, verdict):
        return "the code, then the QA check"
    return "the QA check (guidance that names the code reruns the code first)"


def reply_options(*, keep: bool, stopped: bool = False, rerun: str | None = None) -> str:
    """The replies a question about one stuck or failing task takes, in one sentence:
    guidance for another try (of ``rerun``, when given), 'approve' when the work may
    be kept as it is, and 'stop' unless the run is already stopped."""
    retry = (
        f"Reply with guidance for another try of {rerun}"
        if rerun
        else ("Reply with guidance for another try")
    )
    approve = "'approve' to keep the work as it is" if keep else ""
    if stopped:
        tail = f", or {approve}" if approve else ""
        return f"{retry}{tail}; until then the run stays stopped."
    middle = f" {approve}," if approve else ""
    return f"{retry},{middle} or 'stop' to stop the run."


PAUSING = (BudgetExceeded, RunStopped)


@dataclass
class OrchestratorConfig:
    """Where runs live, the review mode and the caps that make every run end."""

    runs_dir: Path
    review_mode: ReviewMode = ReviewMode.AUTO_ASSUMPTIONS
    max_attempts: int = 3
    #: Attempts of each check (auditor task) before it fails closed.
    max_check_attempts: int = 3
    max_upstream_reruns: int = 2
    research: ResearchLimits = field(default_factory=ResearchLimits)
    #: The stronger model's price per million prompt tokens, for its decision's cost.
    stronger_input_usd_per_mtok: float | None = None
    #: Agent tasks that may run at once on each provider, by name; a provider not
    #: named has no cap of its own.
    max_tasks_per_provider: dict[str, int] = field(default_factory=dict)
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
        self.dispatcher = dispatcher
        self.config = config
        self.policy: VerdictPolicy = policy or SimplePolicy()
        self._started: dict[str, set[Phase]] = {}  # only to emit phase_started once
        #: (phase, newest task id) pairs already reported finished, per run.
        self._finished: dict[str, set[tuple[Phase, str]]] = {}

    async def _audit(
        self, run: Run, task: Task, artifact: Artifact | None, *, recheck: bool = False
    ) -> None:
        """The task sweep; the mixins call it and Orchestrator implements it."""
        raise NotImplementedError  # the task sweep lives in Orchestrator

    async def _note_seen_calls(self, run: Run, task: Task, question: Task) -> bool:
        """The user answered ``question`` about ``task``'s check. If that check stopped
        on a safety finding about a tool call that already ran, record that the user
        has seen the calls it judged, so later checks of the task don't stop on them
        again. Returns whether it recorded that; once per question."""
        judged = [
            v for v in await self.verdicts.for_task(task.id) if v.created_at <= question.created_at
        ]
        if not judged or judged[-1].passed:
            return False
        verdict = judged[-1]
        flagged = [f for f in verdict.blockers if is_safety_stop(f)]
        if not flagged:
            return False
        if any(
            d.rule == SEEN_CALLS and d.inputs.get("user_task") == question.id
            for d in await self.decisions.list(run.id)
        ):
            return False
        await self._record(
            Decision(
                run_id=run.id,
                task_id=task.id,
                kind=DecisionKind.ASK_USER,
                inputs={
                    "user_task": question.id,
                    "verdict_id": verdict.id,
                    "finding_ids": [f.id for f in flagged],
                    "seen_before": verdict.created_at.isoformat(),
                },
                choice="count the tool calls that check judged as seen",
                reason="the user answered the check's safety finding: "
                + "; ".join(f"{f.rule_id} {f.message}" for f in flagged)[:500],
                made_by="code",
                rule=SEEN_CALLS,
            )
        )
        return True

    async def _apply(self, run: Run, d: Decision, verdict: Verdict | None = None) -> None:
        """Carry out one decision; the mixins call it and DecisionsMixin implements it."""
        raise NotImplementedError  # decisions are applied in DecisionsMixin

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

    async def _save_verdict(self, verdict: Verdict, *, check: Task | None = None) -> None:
        """Store a verdict and emit audit_verdict for it, naming the check that made it."""
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
                    "check_task_id": check.id if check else None,
                    "scope": verdict.scope.value,
                    "phase": verdict.phase.value if verdict.phase else None,
                    "passed": verdict.passed,
                    "model": verdict.model,
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
        """Change the run's status, re-reading it first so fields others keep up to date (usage)
        aren't overwritten."""
        run = await self.tasks.get_run(run_id)
        if run.status is status:
            return run
        return await self.tasks.update_run(run.model_copy(update={"status": status}))

    async def _run_done(self, run: Run, data: dict[str, object]) -> Run:
        """Mark the run done, close its root span and emit run_finished."""
        run = await self._set_status(run.id, RunStatus.DONE)
        await Tracer(self.bus).finish_root(run)
        await self.bus.emit(Event(name=EventName.RUN_FINISHED, run_id=run.id, data=dict(data)))
        return run

    async def _pause(self, run_id: str, exc: Exception) -> None:
        """Stop on RunStopped, pause on BudgetExceeded; either way the run can resume."""
        await self._halt(run_id, stopped=isinstance(exc, RunStopped), reason=str(exc))

    async def _halt(self, run_id: str, *, stopped: bool, reason: str) -> None:
        """Mark the run stopped, with the reason in the decision log, or paused."""
        if stopped:
            await mark_stopped(self.tasks, self.decisions, run_id, reason)
        else:
            await self._set_status(run_id, RunStatus.PAUSED)

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
    didn't fail closed or find a dangerous action, which are never waved through. The
    user's guidance on a dangerous action retries the work instead, and the calls that
    check judged count as seen (``_note_seen_calls``)."""
    if task.output is None or task.output_kind == "plan":
        return False
    if verdict is None or verdict.passed:
        return False
    return not any(
        f.rule_id == AUDIT_RULE or f.rule_id in DANGEROUS_ACTION_RULES for f in verdict.blockers
    )


async def mark_stopped(tasks: TaskStore, decisions: DecisionLog, run_id: str, reason: str) -> Run:
    """Mark a run stopped and log why, unless it is already done or stopped. Used by
    the orchestrator and by ``agent-team stop`` for a run no session is driving."""
    run = await tasks.get_run(run_id)
    if run.status in (RunStatus.DONE, RunStatus.STOPPED):
        return run
    run = await tasks.update_run(run.model_copy(update={"status": RunStatus.STOPPED}))
    await decisions.record(
        Decision(
            run_id=run_id,
            kind=DecisionKind.STOP,
            choice="stop",
            reason=reason,
            made_by="code",
            rule="kill_switch",
        )
    )
    return run
