"""The dispatcher: the only way an agent runs (docs/DESIGN.md, "Delegation").

It builds the task envelope, runs the agent with a timeout, catches and classifies
every error, saves the artifact and moves the task on: to auditing when an artifact is
ready, to waiting when the agent asked questions or for research, and to retry or
escalated when the attempt failed. One agent's error never escapes into the run.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import traceback
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from agent_team.core.artifacts import Artifact, ArtifactRef
from agent_team.core.interfaces import (
    Agent,
    AgentContext,
    ArtifactStore,
    CheckpointStore,
    EventBus,
    FeedbackStore,
    TaskStore,
    Tool,
    VerdictStore,
)
from agent_team.core.models import (
    AgentName,
    AgentResult,
    Event,
    EventName,
    Feedback,
    Finding,
    ResultStatus,
    Run,
    Task,
    TaskEnvelope,
    TaskStatus,
    new_id,
)
from agent_team.llm.types import LLMClient

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
_PAUSING = frozenset({"BudgetExceeded", "RunStopped"})

MemoryFetcher = Callable[[Task, Run], Awaitable[list[str]]]
ConstraintSource = Callable[[Task, Run], list[str]]


@dataclass
class Dispatcher:
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

    async def dispatch(self, task: Task) -> AgentResult:
        task = await self.tasks.get_task(task.id)
        run = await self.tasks.get_run(task.run_id)
        agent = self.agents.get(task.agent)
        resume = task.last_error == INTERRUPTED and task.attempt > 0
        if not resume:
            task.attempt += 1
        task.last_error = None
        task.last_traceback = None
        task.trace_id = task.trace_id or new_id("trace")
        task.span_id = new_id("span")
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

        try:
            envelope = await self.build_envelope(task, run)
            ctx = AgentContext(
                llm=self.llm,
                bus=self.bus,
                checkpoints=self.checkpoints,
                tools=self.tools,
                resume=resume,
            )
            timeout = self.timeouts.get(task.agent, self.timeout_s)
            result = await asyncio.wait_for(agent.run(envelope, ctx), timeout=timeout)
        except TimeoutError:
            result = AgentResult(
                task_id=task.id,
                status=ResultStatus.ERROR,
                error_type="TimeoutError",
                error=f"{task.agent.value} took longer than {timeout:.0f}s",
            )
            envelope = None
        except Exception as exc:
            result = AgentResult(
                task_id=task.id,
                status=ResultStatus.ERROR,
                error_type=type(exc).__name__,
                error=str(exc) or type(exc).__name__,
                traceback=traceback.format_exc(),
            )
            envelope = None
        return await self._finish(task, result, envelope)

    # --- envelope ----------------------------------------------------------------

    async def _resolve_inputs(self, task: Task) -> list[ArtifactRef]:
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
        refs = await self._resolve_inputs(task)
        if refs != task.inputs:
            task.inputs = refs
            task = await self.tasks.update_task(task)
        inputs: list[Artifact] = [await self.artifacts.load(run.id, r) for r in refs]

        previous: Artifact | None = None
        parent: Task | None = None
        if task.parent_task_id:
            parent = await self.tasks.get_task(task.parent_task_id)
        prev_ref = task.output or (parent.output if parent else None)
        if prev_ref is not None and prev_ref.kind == task.output_kind:
            previous = await self.artifacts.load(run.id, prev_ref)

        feedback: list[Feedback] = []
        if self.feedback is not None:
            feedback = await self.feedback.for_artifact(run.id, task.output_kind)

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
            memories = await self.memory(task, run)
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
            trace_id=task.trace_id or new_id("trace"),
            span_id=task.span_id or new_id("span"),
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
        task = await self.tasks.get_task(task.id)
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
            paused = result.error_type in _PAUSING
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
