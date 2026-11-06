"""The dispatcher: the only way an agent runs (docs/DESIGN.md, "Delegation").

It builds the task envelope, runs the agent with a timeout, catches and classifies
every error, saves the artifact and moves the task on: to auditing when an artifact is
ready, to waiting when the agent asked questions or for research, and to retry or
escalated when the attempt failed. One agent's error never escapes into the run.

Checks run here too, as auditor tasks with their own attempts, timeout and events: an
auditor that crashes, times out or returns no valid verdict retries on the check's
attempts, never the checked task's, and fails closed once they run out. Each attempt
is an ``attempt`` span in the run's trace.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import traceback
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from agent_team.core.artifacts import Artifact, ArtifactRef
from agent_team.core.errors import BudgetExceeded, RunStopped
from agent_team.core.interfaces import (
    Agent,
    AgentContext,
    ArtifactStore,
    Auditor,
    CheckpointStore,
    EventBus,
    FeedbackStore,
    TaskStore,
    Tool,
    VerdictStore,
)
from agent_team.core.models import (
    AUDIT_RULE,
    AgentName,
    AgentResult,
    Event,
    EventName,
    Feedback,
    Finding,
    ResultStatus,
    RuleScope,
    Run,
    Severity,
    Task,
    TaskEnvelope,
    TaskStatus,
    Verdict,
)
from agent_team.core.tracing import Tracer
from agent_team.llm.types import LLMClient
from agent_team.orchestrator.lineage import ancestry, reopens_parent
from agent_team.orchestrator.research import ResearchLimits

#: Marks a task whose attempt was cut short (crash, budget, kill switch). Its next
#: dispatch resumes the same attempt from the checkpoint instead of starting a new one.
INTERRUPTED = "interrupted"

#: Artifact kinds each agent reads, besides its explicit inputs and dependencies.
INPUT_KINDS: dict[AgentName, tuple[str, ...]] = {
    AgentName.RESEARCHER: ("plan",),
    AgentName.PRODUCT: ("plan", "research_brief"),
    AgentName.ARCHITECT: ("plan", "prd", "research_brief"),
    AgentName.ENGINEER: ("plan", "spec", "prd", "research_brief"),
    AgentName.QA: ("plan", "prd", "spec", "code_change"),
}

#: Errors that pause the run rather than fail the attempt.
PAUSING_ERRORS = frozenset({"BudgetExceeded", "RunStopped"})

#: The "Relevant memory" lines for a task, given its run and its input artifacts.
MemoryFetcher = Callable[[Task, Run, list[Artifact]], Awaitable[list[str]]]
ConstraintSource = Callable[[Task, Run], list[str]]


@dataclass
class Dispatcher:
    """Runs tasks on their agents and records each outcome in the stores."""

    tasks: TaskStore
    artifacts: ArtifactStore
    checkpoints: CheckpointStore
    bus: EventBus
    llm: LLMClient
    agents: Mapping[AgentName, Agent]
    tools: dict[str, Tool] = field(default_factory=dict)
    feedback: FeedbackStore | None = None
    verdicts: VerdictStore | None = None
    memory: MemoryFetcher | None = None
    constraints: ConstraintSource | None = None
    timeout_s: float = 600.0
    timeouts: dict[AgentName, float] = field(default_factory=dict)
    auditor: Auditor | None = None
    #: Limits on research agents ask for; None leaves it unlimited.
    research: ResearchLimits | None = None

    async def dispatch(self, task: Task) -> AgentResult:
        """Run one attempt of ``task`` and move it on; never raises for agent errors."""
        task = await self.tasks.get_task(task.id)
        run = await self.tasks.get_run(task.run_id)
        resume = task.last_error == INTERRUPTED and task.attempt > 0
        attempt = task.attempt if resume else task.attempt + 1
        async with Tracer(self.bus).in_run(
            run, f"attempt {attempt}", "attempt", task_id=task.id, agent=task.agent, attempt=attempt
        ) as span:
            return await self._attempt(task, run, resume=resume, span_id=span.span_id)

    async def _attempt(self, task: Task, run: Run, *, resume: bool, span_id: str) -> AgentResult:
        """One attempt inside its span: start it, run the agent, record the outcome."""
        agent = self.agents.get(task.agent)
        if not resume:
            task.attempt += 1
        task.last_error = None
        task.last_traceback = None
        task.trace_id = run.trace_id
        task.span_id = span_id
        task = await self.tasks.update_task(task)
        task = await self.tasks.transition(
            task.id,
            TaskStatus.RUNNING,
            reason=f"{'resumed' if resume else 'started'} attempt {task.attempt}",
        )
        await self._emit(EventName.TASK_STARTED, task, {"resume": resume})

        if agent is None:
            result = AgentResult(
                task_id=task.id,
                status=ResultStatus.ERROR,
                error_type="UnknownAgent",
                error=f"no agent registered for {task.agent.value}",
            )
            return await self._finish(task, result, None)

        timeout = self.timeouts.get(task.agent, self.timeout_s)
        envelope: TaskEnvelope | None = None
        try:
            envelope = await self.build_envelope(task, run)
            ctx = AgentContext(
                llm=self.llm,
                bus=self.bus,
                checkpoints=self.checkpoints,
                tools=self.tools,
                resume=resume,
            )
            try:
                result = await asyncio.wait_for(agent.run(envelope, ctx), timeout=timeout)
            except TimeoutError:
                result = AgentResult(
                    task_id=task.id,
                    status=ResultStatus.ERROR,
                    error_type="TimeoutError",
                    error=f"{task.agent.value} took longer than {timeout:.0f}s",
                )
        except Exception as exc:
            # Anything else, from building the envelope or from the agent, is a failed
            # attempt with its traceback; it never escapes into the run.
            result = AgentResult(
                task_id=task.id,
                status=ResultStatus.ERROR,
                error_type=type(exc).__name__,
                error=str(exc) or type(exc).__name__,
                traceback=traceback.format_exc(),
            )
        return await self._finish(task, result, envelope)

    # --- checks ------------------------------------------------------------------

    async def check(self, check: Task, subject: Task | None, artifact: Artifact | None) -> Verdict:
        """Run a check (an auditor task) to a verdict on ``subject``'s output, or on the
        whole run when ``subject`` is None. An auditor that raises, times out or
        returns no valid verdict uses the check's attempts; once they run out the last
        failed-closed verdict is returned and the check is escalated. BudgetExceeded
        and RunStopped propagate and leave the check to resume on the same attempt."""
        if self.auditor is None:
            raise RuntimeError("the dispatcher has no auditor to run checks")
        run = await self.tasks.get_run(check.run_id)
        while True:
            check = await self.tasks.get_task(check.id)
            # Still running means the process that ran it died: continue that attempt.
            stopped = check.last_error == INTERRUPTED or check.status is TaskStatus.RUNNING
            resume = stopped and check.attempt > 0
            attempt = check.attempt if resume else check.attempt + 1
            async with Tracer(self.bus).in_run(
                run,
                f"check attempt {attempt}",
                "attempt",
                task_id=check.id,
                agent=AgentName.AUDITOR,
                attempt=attempt,
            ) as span:
                verdict, error = await self._check_once(
                    check, run, subject, artifact, resume=resume, span_id=span.span_id
                )
            check = await self.tasks.get_task(check.id)
            if error is None:
                await self.tasks.transition(check.id, TaskStatus.DONE, reason="verdict ready")
                await self._emit(
                    EventName.TASK_FINISHED, check, {"status": "ok", "passed": verdict.passed}
                )
                return verdict
            check.last_error = error
            await self.tasks.update_task(check)
            await self._emit(EventName.ERROR, check, {"error": error}, level="error")
            if check.attempt < check.max_attempts:
                await self.tasks.transition(check.id, TaskStatus.RETRY, reason="check failed")
                await self._emit(EventName.TASK_RETRIED, check, {"error": error})
                await self.tasks.transition(check.id, TaskStatus.READY, reason="retry the check")
                continue
            await self.tasks.transition(
                check.id, TaskStatus.ESCALATED, reason="no check attempts left"
            )
            await self._emit(EventName.TASK_ESCALATED, check, {"error": error})
            return verdict

    async def _check_once(
        self,
        check: Task,
        run: Run,
        subject: Task | None,
        artifact: Artifact | None,
        *,
        resume: bool,
        span_id: str,
    ) -> tuple[Verdict, str | None]:
        """One attempt of a check: the verdict, and why the attempt failed (None when
        it produced a real verdict)."""
        assert self.auditor is not None
        if not resume:
            check.attempt += 1
        check.last_error = None
        check.trace_id = run.trace_id
        check.span_id = span_id
        check = await self.tasks.update_task(check)
        if check.status in (TaskStatus.PENDING, TaskStatus.RETRY):
            check = await self.tasks.transition(check.id, TaskStatus.READY, reason="check due")
        if check.status is not TaskStatus.RUNNING:
            check = await self.tasks.transition(
                check.id, TaskStatus.RUNNING, reason=f"started check attempt {check.attempt}"
            )
        await self._emit(EventName.TASK_STARTED, check, {"resume": resume})
        timeout = self.timeouts.get(AgentName.AUDITOR, self.timeout_s)
        task_id = subject.id if subject else None
        try:
            name = f"check of {subject.output_kind}" if subject else "check of the run"
            async with Tracer(self.bus).span(run.id, name, "audit", task_id=check.id):
                pending = (
                    self.auditor.audit_task(subject, artifact, run)
                    if subject is not None
                    else self.auditor.audit_run(run)
                )
                verdict = await asyncio.wait_for(pending, timeout=timeout)
        except (BudgetExceeded, RunStopped):
            check.last_error = INTERRUPTED
            await self.tasks.update_task(check)
            await self.tasks.transition(check.id, TaskStatus.READY, reason="paused mid-check")
            raise
        except TimeoutError:
            message = f"the check took longer than {timeout:.0f}s"
            return failed_verdict(run, message, task_id=task_id), message
        except Exception as exc:
            message = f"auditor error: {type(exc).__name__}: {exc}"
            return failed_verdict(run, message, task_id=task_id), message
        unfinished = [f.message for f in verdict.blockers if f.rule_id == AUDIT_RULE]
        if unfinished:
            return verdict, f"no valid verdict: {unfinished[0]}"
        return verdict, None

    # --- envelope ----------------------------------------------------------------

    async def _resolve_inputs(self, task: Task) -> list[ArtifactRef]:
        """The task's inputs plus its dependencies' outputs and the kinds its agent reads, newest
        version per kind."""
        refs: list[ArtifactRef] = list(task.inputs)
        all_tasks = await self.tasks.list_tasks(task.run_id)
        by_id = {t.id: t for t in all_tasks}
        for dep_id in task.depends_on:
            dep = by_id.get(dep_id)
            if dep is not None and dep.output is not None:
                refs.append(dep.output)
        wanted = INPUT_KINDS.get(task.agent, ())
        for other in all_tasks:
            if (
                other.id != task.id
                and other.status is TaskStatus.DONE
                and other.output is not None
                and other.output.kind in wanted
            ):
                refs.append(other.output)
        # Newest version per kind wins when two versions of one kind are present,
        # except kinds that legitimately come in several (briefs, code changes).
        unique: dict[str, ArtifactRef] = {}
        for ref in refs:
            unique.setdefault(str(ref), ref)
        latest: dict[str, int] = {}
        for ref in unique.values():
            latest[ref.kind] = max(latest.get(ref.kind, 0), ref.version)
        multi = {"research_brief", "code_change"}
        return [r for r in unique.values() if r.kind in multi or r.version == latest[r.kind]]

    async def build_envelope(self, task: Task, run: Run) -> TaskEnvelope:
        """Everything the agent sees: inputs, previous output, feedback, findings and diffs."""
        refs = await self._resolve_inputs(task)
        if refs != task.inputs:
            task.inputs = refs
            task = await self.tasks.update_task(task)
        inputs: list[Artifact] = [await self.artifacts.load(run.id, r) for r in refs]

        previous: Artifact | None = None
        parent: Task | None = None
        if task.parent_task_id and reopens_parent(task):
            parent = await self.tasks.get_task(task.parent_task_id)
        prev_ref = task.output or (parent.output if parent else None)
        if prev_ref is not None and prev_ref.kind == task.output_kind:
            previous = await self.artifacts.load(run.id, prev_ref)

        all_tasks = await self.tasks.list_tasks(run.id)
        feedback: list[Feedback] = []
        if self.feedback is not None:
            # This line of work only: the task, the tasks it was reopened from, and the
            # versions they wrote. Sibling work of the same kind keeps its own.
            chain = ancestry(task, all_tasks)
            feedback = await self.feedback.for_work(
                run.id,
                refs=[t.output for t in chain if t.output is not None],
                task_ids=[t.id for t in chain],
            )

        findings: list[Finding] = []
        if self.verdicts is not None:
            verdict = await self.verdicts.latest_for_task(task.id)
            if verdict is None and parent is not None:
                verdict = await self.verdicts.latest_for_task(parent.id)
            if verdict is not None and not verdict.passed:
                wanted = set(task.finding_ids)
                findings = [f for f in verdict.findings if not wanted or f.id in wanted]

        memories: list[str] = []
        if self.memory is not None:
            memories = await self.memory(task, run, inputs)
        constraints = self.constraints(task, run) if self.constraints else []

        return TaskEnvelope(
            task=task,
            run=run,
            inputs=inputs,
            constraints=constraints,
            memories=memories,
            feedback=feedback,
            findings=findings,
            previous_output=previous,
            upstream_diff=await self._upstream_diff(run.id, previous, inputs) if previous else None,
            research_allowed=self.research is None
            or self.research.refusal(task, all_tasks) is None,
            trace_id=task.trace_id or run.trace_id,
            span_id=task.span_id or run.span_id,
        )

    async def _upstream_diff(
        self, run_id: str, previous: Artifact, inputs: list[Artifact]
    ) -> str | None:
        """For a revision: a diff of every input that moved since ``previous`` was built."""
        old = {r.kind: r for r in previous.built_from}
        parts: list[str] = []
        for art in inputs:
            before = old.get(art.kind)
            if before is None or before.version == art.version:
                continue
            try:
                old_art = await self.artifacts.load(run_id, before)
            except Exception:
                parts.append(f"{art.kind}: v{before.version} -> v{art.version}")
                continue
            parts.append(json_diff(old_art, art))
        return "\n\n".join(parts) or None

    # --- outcome -----------------------------------------------------------------

    async def _finish(
        self, task: Task, result: AgentResult, envelope: TaskEnvelope | None
    ) -> AgentResult:
        """Save the artifact and move the task to auditing, waiting, retry, escalated or ready
        (paused)."""
        task = await self.tasks.get_task(task.id)
        if task.status is TaskStatus.SUPERSEDED:
            # Reopened while it ran: its inputs are stale, so the result is dropped.
            await self._emit(
                EventName.TASK_FINISHED,
                task,
                {"status": "discarded", "reason": "superseded during the attempt"},
            )
            return result
        task.model = result.model or task.model
        if result.artifact is not None and result.status is not ResultStatus.ERROR:
            built_from = (
                [ArtifactRef(kind=a.kind, version=a.version) for a in envelope.inputs]
                if envelope
                else []
            )
            artifact = result.artifact.model_copy(
                update={"run_id": task.run_id, "task_id": task.id, "built_from": built_from}
            )
            ref = await self.artifacts.save(artifact)
            artifact = artifact.model_copy(update={"version": ref.version})
            result = result.model_copy(update={"artifact": artifact})
            task.output = ref

        if result.status is ResultStatus.ERROR:
            paused = result.error_type in PAUSING_ERRORS
            task.last_error = INTERRUPTED if paused else f"{result.error_type}: {result.error}"
            task.last_traceback = result.traceback
            await self.tasks.update_task(task)
            await self._emit(
                EventName.ERROR,
                task,
                {
                    "error_type": result.error_type,
                    "error": result.error,
                    "traceback": result.traceback,
                },
                level="error",
            )
            if paused:
                await self.tasks.transition(
                    task.id, TaskStatus.READY, reason=f"paused: {result.error_type}"
                )
            elif task.attempt < task.max_attempts:
                await self.tasks.transition(
                    task.id, TaskStatus.RETRY, reason=f"attempt failed: {result.error_type}"
                )
                await self._emit(EventName.TASK_RETRIED, task, {"error_type": result.error_type})
            else:
                await self.tasks.transition(
                    task.id,
                    TaskStatus.ESCALATED,
                    reason=f"no attempts left after {result.error_type}",
                )
                await self._emit(EventName.TASK_ESCALATED, task, {"error_type": result.error_type})
        else:
            await self.tasks.update_task(task)
            if result.status is ResultStatus.OK:
                await self.tasks.transition(task.id, TaskStatus.AUDITING, reason="output ready")
            else:
                if result.status is ResultStatus.QUESTIONS and result.artifact is not None:
                    for q in result.artifact.questions:
                        if q.answer is None and q.assumption is None:
                            await self._emit(
                                EventName.QUESTION_ASKED,
                                task,
                                {"question_id": q.id, "text": q.text, "why": q.why},
                            )
                await self.tasks.transition(
                    task.id, TaskStatus.WAITING, reason=f"agent returned {result.status.value}"
                )

        await self._emit(
            EventName.TASK_FINISHED,
            task,
            {
                "status": result.status.value,
                "output": str(task.output) if task.output else None,
                "usage": result.usage.model_dump(),
                "model": result.model,
            },
        )
        return result

    async def _emit(
        self,
        name: EventName,
        task: Task,
        data: dict[str, object],
        *,
        level: str = "info",
    ) -> None:
        """Emit an event about one task attempt."""
        await self.bus.emit(
            Event(
                name=name,
                run_id=task.run_id,
                task_id=task.id,
                attempt=task.attempt,
                agent=task.agent,
                trace_id=task.trace_id,
                span_id=task.span_id,
                level="error" if level == "error" else "info",
                data=dict(data),
            )
        )


def json_diff(old: Artifact, new: Artifact) -> str:
    """Unified diff of two artifact versions, for revisions and reviews."""
    a = json.dumps(old.model_dump(exclude={"version", "task_id"}), indent=1, sort_keys=True)
    b = json.dumps(new.model_dump(exclude={"version", "task_id"}), indent=1, sort_keys=True)
    return "\n".join(
        difflib.unified_diff(
            a.splitlines(),
            b.splitlines(),
            fromfile=f"{old.kind}@v{old.version}",
            tofile=f"{new.kind}@v{new.version}",
            lineterm="",
        )
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
