"""The orchestrator: the only agent you talk to.

It classifies each message, starts runs, and drives a run through the fixed phases:
plan, dispatch ready tasks (in parallel, Engineer tasks one at a time), audit each
output, apply the verdict policy's decisions, audit each phase gate, and finish with a
whole-run audit. Every choice goes to the decision log. Run state lives in the task
store, so ``drive`` can be called again after a crash and picks up where it left off.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_team.agents.loop import ToolLoop
from agent_team.agents.planner import kind_from_objective, planning_objective
from agent_team.agents.prompts import RUN_QA
from agent_team.core.artifacts import Artifact, ArtifactRef, Plan, Spec
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
    ResultStatus,
    ReviewMode,
    RuleScope,
    Run,
    RunStatus,
    Severity,
    Task,
    TaskOrigin,
    TaskStatus,
    Verdict,
    new_id,
)
from agent_team.llm.types import LLMClient, Message
from agent_team.orchestrator.classify import classify
from agent_team.orchestrator.control import create_user_task, read_tools
from agent_team.orchestrator.dispatcher import INTERRUPTED
from agent_team.orchestrator.phases import PhaseMachine, live
from agent_team.orchestrator.planning import RULE as PIPELINE
from agent_team.orchestrator.planning import expand_plan, implementation_tasks
from agent_team.orchestrator.policy import SimplePolicy
from agent_team.tools.base import RunToolContext

_APPROVE = re.compile(
    r"^\s*(approve[ds]?|ok(ay)?|yes|lgtm|looks good|accept(ed)?|ship it)[\s.!]*$", re.I
)
_RUN_QA_FORMAT = '\nWhen done, reply with only JSON: {"answer": "...", "cited": ["task_..."]}'
_CHILD_ORIGINS = frozenset({TaskOrigin.QUESTION, TaskOrigin.RESEARCH_REQUEST})


@dataclass
class OrchestratorConfig:
    runs_dir: Path
    review_mode: ReviewMode = ReviewMode.AUTO_ASSUMPTIONS
    max_attempts: int = 3
    max_upstream_reruns: int = 2
    max_loops: int = 500  # safety valve for drive()


@dataclass
class PendingQuestion:
    task_id: str
    question: str
    origin: TaskOrigin


@dataclass
class Reply:
    kind: Literal["done", "waiting", "answer", "stopped", "error"]
    text: str
    run_id: str | None = None
    pending: list[PendingQuestion] = field(default_factory=list)
    artifact: Artifact | None = None


class RunAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    cited: list[str] = Field(default_factory=list)


class Orchestrator:
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
        self._gates: dict[str, set[Phase]] = {}
        self._started: dict[str, set[Phase]] = {}
        self._gate_log: dict[str, list[Verdict]] = {}

    # --- entry points ----------------------------------------------------------------

    async def handle(self, message: str, *, run_id: str | None = None) -> Reply:
        """One user message: answer a pending question, or classify and act."""
        run = await self.tasks.get_run(run_id) if run_id else None
        await self.bus.emit(
            Event(
                name=EventName.USER_MESSAGE,
                run_id=run.id if run else "session",
                agent=AgentName.USER,
                data={"text": message},
            )
        )
        if run is not None and await self._pending(run.id):
            return await self.answer(run.id, message)

        result, made_by, _ = await classify(self.llm, message, run)
        decision = Decision(
            run_id=run.id if run else "session",
            kind=DecisionKind.CLASSIFY_REQUEST,
            inputs={"message": message[:500]},
            choice=result.kind.value,
            reason=result.reason,
            made_by="model" if made_by == "model" else "code",
            rule=None if made_by == "model" else "classify.heuristic",
        )
        if result.kind in (RequestKind.BUILD, RequestKind.RESEARCH_QUESTION):
            new = await self.start_run(message, kind=result.kind, title=result.title)
            # The classification belongs to the run it started.
            await self._record(decision.model_copy(update={"run_id": new.id}))
            return await self.drive(new.id)
        await self._record(decision)
        assert run is not None  # _sanitize guarantees a run for these kinds
        if result.kind is RequestKind.RUN_QUESTION:
            return await self.answer_run_question(run.id, message)
        return await self.request_change(run.id, message, target=result.target or "prd")

    async def start_run(
        self,
        request: str,
        *,
        kind: RequestKind = RequestKind.BUILD,
        title: str | None = None,
        context_dirs: list[str] | None = None,
        review_mode: ReviewMode | None = None,
    ) -> Run:
        run_id = new_id("run")
        workspace, dirs = _prepare_workspace(self.config.runs_dir, run_id, context_dirs or [])
        run = await self.tasks.create_run(
            Run(
                id=run_id,
                title=(title or request.strip().splitlines()[0])[:80],
                request=request,
                review_mode=review_mode or self.config.review_mode,
                workspace=str(workspace),
                context_dirs=dirs,
            )
        )
        await self.bus.emit(
            Event(name=EventName.RUN_STARTED, run_id=run.id, data={"kind": kind.value})
        )
        await self.tasks.create_task(
            Task(
                run_id=run.id,
                phase=Phase.PLANNING,
                agent=AgentName.ORCHESTRATOR,
                objective=planning_objective(kind),
                output_kind="plan",
                max_attempts=self.config.max_attempts,
                reason="every run starts with a plan",
            )
        )
        return run

    async def resume(self, run_id: str) -> Reply:
        """After a crash: interrupted attempts continue from their checkpoints."""
        for task in await self.tasks.list_tasks(run_id, status=TaskStatus.RUNNING):
            task.last_error = INTERRUPTED
            await self.tasks.update_task(task)
            await self.tasks.transition(task.id, TaskStatus.READY, reason="resumed after a stop")
        await self._set_status(run_id, RunStatus.ACTIVE)
        return await self.drive(run_id)

    async def stop(self, run_id: str, reason: str = "stopped by the user") -> None:
        await self._set_status(run_id, RunStatus.STOPPED)
        await self._record(
            Decision(
                run_id=run_id,
                kind=DecisionKind.STOP,
                choice="stop",
                reason=reason,
                made_by="code",
                rule="kill_switch",
            )
        )

    # --- the drive loop -------------------------------------------------------------

    async def drive(self, run_id: str) -> Reply:
        for _ in range(self.config.max_loops):
            run = await self.tasks.get_run(run_id)
            if run.status in (RunStatus.STOPPED, RunStatus.PAUSED):
                return Reply(
                    kind="stopped", text=f"Run {run.id} is {run.status.value}.", run_id=run.id
                )
            if run.status is RunStatus.DONE:
                return await self._done_reply(run)
            if run.status is RunStatus.WAITING:
                run = await self._set_status(run.id, RunStatus.ACTIVE)

            tasks = await self.tasks.list_tasks(run.id)
            if await self._settle(run, tasks):
                continue
            machine = PhaseMachine(self._kind(tasks))
            gates = self._gates.setdefault(run.id, set())
            phase = machine.current(tasks, gates)
            if phase is None:
                return await self._finish_run(run)
            await self._phase_started(run, phase)

            open_tasks = machine.open_tasks(tasks, phase)
            if not open_tasks:
                await self._gate(run, phase, machine, tasks)
                continue

            ready = await self._ready(run.id, phase)
            if not ready:
                pending = await self._pending(run.id)
                if pending:
                    await self._set_status(run.id, RunStatus.WAITING)
                    return self._waiting_reply(run, pending)
                stuck = ", ".join(f"{t.id} ({t.status.value})" for t in open_tasks)
                return Reply(
                    kind="error", text=f"Run is stuck in {phase.value}: {stuck}", run_id=run.id
                )

            others = [t for t in ready if t.agent is not AgentName.ENGINEER]
            engineers = [t for t in ready if t.agent is AgentName.ENGINEER]
            batch = others + engineers[:1]  # Engineer tasks share the workspace
            await asyncio.gather(*(self._run_task(run, t) for t in batch))
        return Reply(kind="error", text="Stopped: too many orchestration steps.", run_id=run_id)

    async def _ready(self, run_id: str, phase: Phase) -> list[Task]:
        found = {t.id: t for t in await self.tasks.ready_tasks(run_id)}
        for t in await self.tasks.list_tasks(run_id, status=TaskStatus.READY):
            found.setdefault(t.id, t)
        return [t for t in found.values() if t.phase is phase and t.agent is not AgentName.USER]

    async def _pending(self, run_id: str) -> list[Task]:
        return [
            t
            for t in await self.tasks.list_tasks(run_id, status=TaskStatus.WAITING)
            if t.agent is AgentName.USER
        ]

    def _kind(self, tasks: list[Task]) -> RequestKind:
        for t in tasks:
            if t.agent is AgentName.ORCHESTRATOR and t.output_kind == "plan":
                return kind_from_objective(t.objective) or RequestKind.BUILD
        return RequestKind.BUILD

    async def _phase_started(self, run: Run, phase: Phase) -> None:
        started = self._started.setdefault(run.id, set())
        if phase not in started:
            started.add(phase)
            await self.bus.emit(
                Event(name=EventName.PHASE_STARTED, run_id=run.id, data={"phase": phase.value})
            )

    # --- settling: retries, questions, research requests, escalations ----------------

    async def _settle(self, run: Run, tasks: list[Task]) -> bool:
        """Moves tasks that are blocked on something that has since happened. Returns
        True if anything changed, so the loop re-reads the store."""
        changed = False
        by_id = {t.id: t for t in tasks}
        for task in tasks:
            if task.agent is AgentName.USER:
                continue
            if task.status is TaskStatus.RETRY:
                await self.tasks.transition(task.id, TaskStatus.READY, reason="retry with feedback")
                changed = True
            elif task.status is TaskStatus.AUDITING:
                # The process stopped between output and verdict: audit it now.
                artifact = await self.artifacts.load(run.id, task.output) if task.output else None
                await self._audit(run, task, artifact)
                changed = True
            elif task.status in (TaskStatus.WAITING, TaskStatus.ESCALATED):
                children = [
                    by_id[d]
                    for d in task.depends_on
                    if d in by_id
                    and by_id[d].origin in _CHILD_ORIGINS
                    and by_id[d].parent_task_id == task.id
                ]
                live_children = [c for c in children if c.status is not TaskStatus.SUPERSEDED]
                if live_children:
                    if all(c.status is TaskStatus.DONE for c in live_children):
                        await self._unblock(task, children)
                        changed = True
                elif task.status is TaskStatus.WAITING:
                    changed = await self._open_children(run, task) or changed
                else:
                    await self._ask_about_escalation(run, task)
                    changed = True
        return changed

    async def _unblock(self, task: Task, children: list[Task]) -> None:
        ids = {c.id for c in children}
        task.depends_on = [d for d in task.depends_on if d not in ids]
        for c in children:
            if c.output is not None and c.output not in task.inputs:
                task.inputs.append(c.output)
        if task.status is TaskStatus.ESCALATED:
            # The user's guidance buys exactly one more attempt.
            task.max_attempts = max(task.max_attempts, task.attempt + 1)
        else:
            # Asking a question or for research doesn't use up a retry.
            task.max_attempts += 1
        await self.tasks.update_task(task)
        await self.tasks.transition(task.id, TaskStatus.READY, reason="answers and research are in")

    async def _open_children(self, run: Run, task: Task) -> bool:
        """The agent asked questions or for research: ask the user, or plan research."""
        artifact = await self.artifacts.load(run.id, task.output) if task.output else None
        if artifact is None:
            await self._ask_about_escalation(run, task)
            return True
        child_ids: list[str] = []
        open_qs = [q for q in artifact.questions if q.answer is None and q.assumption is None]
        if open_qs and run.review_mode is ReviewMode.AUTO_ASSUMPTIONS:
            for q in open_qs:
                await self.feedback.add(
                    run.id,
                    Feedback(
                        artifact=task.output,
                        source=FeedbackSource.USER_ANSWER,
                        question_id=q.id,
                        text="Review mode is auto_assumptions: no answer will come. Record "
                        "your best assumption in this question's `assumption` and continue.",
                    ),
                )
            await self._record(
                Decision(
                    run_id=run.id,
                    task_id=task.id,
                    kind=DecisionKind.ASK_USER,
                    inputs={"questions": [q.id for q in open_qs]},
                    choice="record assumptions instead of asking",
                    reason="review mode is auto_assumptions",
                    made_by="code",
                    rule="review_mode",
                )
            )
        else:
            for q in open_qs:
                child = await create_user_task(
                    self.tasks,
                    run,
                    phase=task.phase,
                    question=f"{q.text} (asked because: {q.why})",
                    origin=TaskOrigin.QUESTION,
                    parent=task,
                    reason=q.id,
                )
                child_ids.append(child.id)
                await self._record(
                    Decision(
                        run_id=run.id,
                        task_id=task.id,
                        kind=DecisionKind.ASK_USER,
                        inputs={"question_id": q.id, "user_task": child.id},
                        choice="ask the user",
                        reason=q.why,
                        made_by="model",
                    )
                )
        for i, req in enumerate(artifact.research_requests, start=1):
            child = await self.tasks.create_task(
                Task(
                    run_id=run.id,
                    phase=task.phase,
                    agent=AgentName.RESEARCHER,
                    objective=req.question,
                    inputs=[task.output] if task.output else [],
                    output_kind="research_brief",
                    origin=TaskOrigin.RESEARCH_REQUEST,
                    parent_task_id=task.id,
                    reason=f"research request {i} from {task.agent.value}: {req.why}",
                    max_attempts=self.config.max_attempts,
                )
            )
            child_ids.append(child.id)
            await self._record(
                Decision(
                    run_id=run.id,
                    task_id=task.id,
                    kind=DecisionKind.DELEGATE,
                    inputs={"question": req.question, "research_task": child.id},
                    choice="researcher",
                    reason=req.why,
                    made_by="code",
                    rule="research_request",
                )
            )
        task = await self.tasks.get_task(task.id)
        if child_ids:
            task.depends_on = [*task.depends_on, *child_ids]
            await self.tasks.update_task(task)
        else:
            task.max_attempts = max(task.max_attempts, task.attempt + 1)
            await self.tasks.update_task(task)
            await self.tasks.transition(task.id, TaskStatus.READY, reason="assumptions recorded")
        return True

    async def _ask_about_escalation(self, run: Run, task: Task) -> None:
        why = task.last_error or "it failed its audit with no attempts left"
        child = await create_user_task(
            self.tasks,
            run,
            phase=task.phase,
            question=(
                f"Task {task.id} ({task.agent.value}: {task.objective}) is stuck: {why}. "
                "Reply with guidance to retry it, or 'stop'."
            ),
            origin=TaskOrigin.QUESTION,
            parent=task,
            reason="escalation",
        )
        task = await self.tasks.get_task(task.id)
        task.depends_on = [*task.depends_on, child.id]
        await self.tasks.update_task(task)
        await self._record(
            Decision(
                run_id=run.id,
                task_id=task.id,
                kind=DecisionKind.ASK_USER,
                inputs={"user_task": child.id, "last_error": task.last_error},
                choice="ask the user",
                reason="task escalated with nothing left to try automatically",
                made_by="code",
                rule="escalation",
            )
        )

    # --- one task: dispatch, audit, apply --------------------------------------------

    async def _run_task(self, run: Run, task: Task) -> None:
        await self._record(
            Decision(
                run_id=run.id,
                task_id=task.id,
                kind=DecisionKind.DELEGATE,
                inputs={"objective": task.objective, "attempt": task.attempt + 1},
                choice=task.agent.value,
                reason=task.reason or f"{task.output_kind} is due in {task.phase.value}",
                made_by="code",
                rule="phase_order",
            )
        )
        result = await self.dispatcher.dispatch(task)
        if result.status is ResultStatus.ERROR and result.error_type in (
            "BudgetExceeded",
            "RunStopped",
        ):
            status = RunStatus.STOPPED if result.error_type == "RunStopped" else RunStatus.PAUSED
            await self._set_status(run.id, status)
            return
        task = await self.tasks.get_task(task.id)
        if task.status is TaskStatus.AUDITING:
            await self._audit(run, task, result.artifact)

    async def _audit(self, run: Run, task: Task, artifact: Artifact | None) -> None:
        try:
            verdict = await self.auditor.audit_task(task, artifact, run)
        except Exception as exc:  # fail closed
            verdict = _failed_verdict(run, f"auditor error: {exc}", task_id=task.id)
        await self._save_verdict(verdict)
        task = await self.tasks.get_task(task.id)
        task.finding_ids = [f.id for f in verdict.findings if f.severity is Severity.BLOCKER]
        task = await self.tasks.update_task(task)
        # Earlier verdicts only; the policy compares them with this one (no-progress).
        history = [v for v in await self.verdicts.for_task(task.id) if v.id != verdict.id]
        decisions = await self.policy.on_task_verdict(run, task, verdict, history)
        if not decisions and not verdict.passed:
            decisions = await SimplePolicy().on_task_verdict(run, task, verdict, history)
        if not decisions:
            await self.tasks.transition(task.id, TaskStatus.DONE, reason="audit passed")
            self.llm.set_override(task.id, None)
            if isinstance(artifact, Plan):
                await self._after_plan(run, task, artifact)
            return
        for decision in decisions:
            await self._apply(run, decision, verdict)

    async def _after_plan(self, run: Run, task: Task, plan: Plan) -> None:
        assert task.output is not None
        new_tasks = expand_plan(plan, task.output, run, task, max_attempts=self.config.max_attempts)
        for t in new_tasks:
            await self.tasks.create_task(t)
        await self._record_pipeline(
            run,
            task,
            new_tasks,
            inputs={
                "plan": str(task.output),
                "asks": [a.id for a in plan.asks],
                "research_questions": [rq.question for rq in plan.research_questions],
            },
            reason=f"{plan.request_kind} pipeline with one Researcher task per research question",
        )

    async def _start_implementation(self, run: Run, tasks: list[Task]) -> None:
        """When the Spec gate passes: one Engineer task per work package, then QA."""
        live_tasks = live(tasks)
        if any(
            t.phase is Phase.IMPLEMENTATION and t.agent is not AgentName.USER for t in live_tasks
        ):
            return
        spec_tasks = [
            t
            for t in live_tasks
            if t.agent is AgentName.ARCHITECT and t.status is TaskStatus.DONE and t.output
        ]
        if not spec_tasks:
            return
        spec_task = spec_tasks[-1]
        assert spec_task.output is not None
        spec = await self.artifacts.load(run.id, spec_task.output)
        assert isinstance(spec, Spec)
        previous = [
            t
            for t in tasks
            if t.phase is Phase.IMPLEMENTATION
            and t.status is TaskStatus.SUPERSEDED
            and t.agent is not AgentName.USER
        ]
        new_tasks = implementation_tasks(
            spec, spec_task, run, previous=previous, max_attempts=self.config.max_attempts
        )
        for t in new_tasks:
            await self.tasks.create_task(t)
        packages = [p.name for p in spec.work_packages] or ["all"]
        await self._record_pipeline(
            run,
            spec_task,
            new_tasks,
            inputs={"spec": str(spec_task.output), "work_packages": packages},
            reason=f"spec gate passed; {len(packages)} work package(s), run one at a time",
        )

    async def _record_pipeline(
        self,
        run: Run,
        task: Task,
        new_tasks: list[Task],
        *,
        inputs: dict[str, object],
        reason: str,
    ) -> None:
        await self._record(
            Decision(
                run_id=run.id,
                task_id=task.id,
                kind=DecisionKind.PLAN,
                inputs=dict(inputs),
                choice=f"{len(new_tasks)} tasks: "
                + ", ".join(f"{t.agent.value}->{t.output_kind}" for t in new_tasks),
                reason=reason,
                made_by="code",
                rule=PIPELINE,
            )
        )

    async def _gate_verdicts(self, run_id: str) -> list[Verdict]:
        return self._gate_log.get(run_id, [])

    async def _save_verdict(self, verdict: Verdict) -> None:
        await self.verdicts.save(verdict)
        if verdict.task_id is None:
            self._gate_log.setdefault(verdict.run_id, []).append(verdict)
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

    # --- applying decisions -----------------------------------------------------------

    async def _apply(self, run: Run, d: Decision, verdict: Verdict | None = None) -> None:
        await self._record(d)
        task_id = d.inputs.get("task_id") or d.task_id
        task = await self.tasks.get_task(task_id) if isinstance(task_id, str) else None

        if d.kind is DecisionKind.RETRY and task is not None:
            if task.status is TaskStatus.AUDITING:
                await self.tasks.transition(task.id, TaskStatus.RETRY, reason=d.reason)
                await self._emit_task(EventName.TASK_RETRIED, task, {"reason": d.reason})
        elif d.kind in (DecisionKind.ESCALATE, DecisionKind.NO_PROGRESS) and task is not None:
            if task.status is TaskStatus.AUDITING:
                await self.tasks.transition(task.id, TaskStatus.ESCALATED, reason=d.reason)
                await self._emit_task(EventName.TASK_ESCALATED, task, {"reason": d.reason})
        elif d.kind is DecisionKind.STRONGER_MODEL and task is not None:
            route = self.llm.use_stronger_model(task.id)
            model = d.inputs.get("model")
            task.model = model if isinstance(model, str) else str(route)
            task.max_attempts = max(task.max_attempts, task.attempt + 1)
            task = await self.tasks.update_task(task)
            if task.status is TaskStatus.AUDITING:
                await self.tasks.transition(task.id, TaskStatus.RETRY, reason=d.reason)
            elif task.status is TaskStatus.ESCALATED:  # the policy escalated first
                await self.tasks.transition(task.id, TaskStatus.READY, reason=d.reason)
        elif d.kind is DecisionKind.UPSTREAM_RERUN:
            await self._upstream_rerun(run, d, task)
        elif d.kind is DecisionKind.GATE_FIX and task is not None:
            ids = d.inputs.get("finding_ids") or []
            findings = [f for f in (verdict.findings if verdict else []) if f.id in ids]
            await self._findings_to_feedback(run, task, findings)
            root = await self.reopen(run, task, origin=TaskOrigin.GATE_FIX, finding_ids=list(ids))
            await self._rerun_after(run, root, d.inputs.get("rerun_after") or [])
        elif d.kind is DecisionKind.ASK_USER:
            question = str(d.inputs.get("question") or d.reason)
            if task is not None and task.status is TaskStatus.AUDITING:
                await self.tasks.transition(task.id, TaskStatus.ESCALATED, reason=d.reason)
            phase = (
                task.phase
                if task
                else (verdict.phase if verdict and verdict.phase else Phase.PLANNING)
            )
            child = await create_user_task(
                self.tasks,
                run,
                phase=phase,
                question=question,
                origin=TaskOrigin.QUESTION,
                parent=task,
                reason="policy",
            )
            if task is not None:
                task = await self.tasks.get_task(task.id)
                task.depends_on = [*task.depends_on, child.id]
                await self.tasks.update_task(task)
        elif d.kind is DecisionKind.STOP:
            await self._set_status(run.id, RunStatus.STOPPED)

    async def _upstream_rerun(self, run: Run, d: Decision, task: Task | None) -> None:
        if task is not None and task.status is TaskStatus.AUDITING:
            await self.tasks.transition(task.id, TaskStatus.ESCALATED, reason=d.reason)
        raw = d.inputs.get("artifact")
        producer: Task | None = None
        producer_id = d.inputs.get("producer_task_id")
        if isinstance(producer_id, str):
            producer = await self.tasks.get_task(producer_id)
        elif isinstance(raw, str):
            ref = ArtifactRef.parse(raw)
            for t in await self.tasks.list_tasks(run.id):
                if t.output == ref:
                    producer = t
        run = await self.tasks.get_run(run.id)
        if (
            producer is None
            or producer.output_kind == "plan"
            or run.upstream_reruns >= self.config.max_upstream_reruns
        ):
            # Nothing safe to re-run automatically: the escalation asks the user.
            return
        run.upstream_reruns += 1
        await self.tasks.update_run(run)
        await self.reopen(run, producer, origin=TaskOrigin.REVISION, reason=d.reason)

    async def _rerun_after(self, run: Run, root: Task | None, ids: list[str]) -> None:
        """Tasks the policy wants re-run once a gate fix passes, such as a fresh QA
        review after Engineer fixes. Those not already reopened as downstream of the fix
        are reopened here and wait for the fix."""
        if root is None:
            return
        for tid in ids:
            other = await self.tasks.get_task(tid)
            if other.status is TaskStatus.SUPERSEDED:
                continue
            copy = await self.reopen(
                run, other, origin=TaskOrigin.REVISION, reason=f"re-run after gate fix {root.id}"
            )
            if copy is not None and root.id not in copy.depends_on:
                copy.depends_on = [*copy.depends_on, root.id]
                await self.tasks.update_task(copy)

    async def _findings_to_feedback(self, run: Run, task: Task, findings: list[Finding]) -> None:
        for f in findings:
            fix = f" Suggested fix: {f.suggested_fix}" if f.suggested_fix else ""
            await self.feedback.add(
                run.id,
                Feedback(
                    artifact=task.output,
                    source=FeedbackSource.AUDITOR,
                    text=f"{f.rule_id} on {f.item or 'the artifact'}: {f.message}.{fix}",
                ),
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
        """Re-run ``task`` as a revision, and everything built on it after it.

        Affected tasks are superseded, never deleted, and replaced by copies that point
        back at them, so the history of every attempt stays in the trace."""
        tasks = await self.tasks.list_tasks(run.id)
        current = next((t for t in tasks if t.id == task.id), task)
        if current.status is TaskStatus.SUPERSEDED:
            return None
        downstream: set[str] = set()
        frontier = {current.id}
        while frontier:
            nxt = {
                t.id
                for t in tasks
                if t.id not in downstream
                and t.id != current.id
                and any(d in frontier for d in t.depends_on)
            }
            downstream |= nxt
            frontier = nxt
        affected = [current] + [
            t for t in tasks if t.id in downstream and t.status is not TaskStatus.SUPERSEDED
        ]
        affected_ids = {t.id for t in affected}
        # Pending questions on affected tasks go too.
        affected += [
            t
            for t in tasks
            if t.agent is AgentName.USER
            and t.parent_task_id in affected_ids
            and t.status is TaskStatus.WAITING
        ]
        # Implementation tasks downstream of an earlier phase aren't copied: the Spec
        # gate rebuilds them from the new spec's work packages.
        rebuilt_later = current.phase is not Phase.IMPLEMENTATION
        new_ids = {
            t.id: new_id("task")
            for t in affected
            if t.agent is not AgentName.USER
            and not (rebuilt_later and t.phase is Phase.IMPLEMENTATION)
        }
        for t in affected:
            if t.status in (TaskStatus.AUDITING, TaskStatus.RUNNING):
                await self.tasks.transition(t.id, TaskStatus.ESCALATED, reason="reopened")
            await self.tasks.transition(
                t.id, TaskStatus.SUPERSEDED, reason=f"reopened from {current.id}"
            )
        root: Task | None = None
        for t in affected:
            if t.id not in new_ids:
                continue
            copy = await self.tasks.create_task(
                Task(
                    id=new_ids[t.id],
                    run_id=run.id,
                    phase=t.phase,
                    agent=t.agent,
                    objective=t.objective,
                    inputs=[r for r in t.inputs if r.kind == "plan"],
                    output_kind=t.output_kind,
                    depends_on=[
                        new_ids.get(d, d)
                        for d in t.depends_on
                        if d not in affected_ids or d in new_ids
                    ],
                    max_attempts=t.max_attempts,
                    origin=origin if t.id == current.id else TaskOrigin.REVISION,
                    parent_task_id=t.id,
                    reason=reason if t.id == current.id else f"upstream {current.id} reopened",
                    finding_ids=(finding_ids or []) if t.id == current.id else [],
                )
            )
            if t.id == current.id:
                root = copy
        machine = PhaseMachine(self._kind(tasks))
        if current.phase in machine.phases:
            machine.reopen(current.phase, self._gates.setdefault(run.id, set()))
        await self._record(
            Decision(
                run_id=run.id,
                task_id=current.id,
                kind=DecisionKind.REOPEN_PHASE,
                inputs={"superseded": sorted(affected_ids), "new": sorted(new_ids.values())},
                choice=f"reopen {current.phase.value} from {current.output or current.id}",
                reason=reason or origin.value,
                made_by="code",
                rule="lineage",
            )
        )
        return root

    # --- gates and the end of a run -------------------------------------------------

    async def _gate(self, run: Run, phase: Phase, machine: PhaseMachine, tasks: list[Task]) -> None:
        try:
            verdict = await self.auditor.audit_gate(run, phase)
        except Exception as exc:
            verdict = _failed_verdict(run, f"auditor error: {exc}", phase=phase)
        await self._save_verdict(verdict)
        decisions = await self.policy.on_gate_verdict(run, phase, verdict)
        if not decisions and not verdict.passed:
            decisions = await SimplePolicy().on_gate_verdict(run, phase, verdict)
        if decisions:
            for d in decisions:
                await self._apply(run, d, verdict)
            return
        gates = self._gates.setdefault(run.id, set())
        gates.add(phase)
        await self.bus.emit(
            Event(name=EventName.PHASE_FINISHED, run_id=run.id, data={"phase": phase.value})
        )
        if phase is Phase.SPEC:
            await self._start_implementation(run, tasks)
        if (
            run.review_mode is ReviewMode.EVERY_PHASE
            and machine.next(phase) is not None
            and not _reviewed(tasks, phase)
        ):
            await self._request_review(run, phase, tasks)

    async def _request_review(self, run: Run, phase: Phase, tasks: list[Task]) -> None:
        outputs = [str(t.output) for t in PhaseMachine.expert_tasks(tasks, phase) if t.output]
        summaries = []
        gate = next(
            (v for v in reversed(await self._gate_verdicts(run.id)) if v.phase is phase), None
        )
        if gate is not None and gate.summary:
            summaries.append(f"Audit: {gate.summary}")
        warnings = [f for f in (gate.findings if gate else []) if f.severity is Severity.WARNING]
        summaries += [f"Warning {f.rule_id}: {f.message}" for f in warnings]
        for t in PhaseMachine.expert_tasks(tasks, phase):
            if t.output:
                art = await self.artifacts.load(run.id, t.output)
                summaries.append(f"{t.output}: {art.summary}")
        child = await create_user_task(
            self.tasks,
            run,
            phase=phase,
            question=f"Review the {phase.value} phase. "
            + " ".join(summaries)
            + " Reply 'approve' or comment to request changes.",
            origin=TaskOrigin.REVIEW,
            reason="phase review",
        )
        await self.bus.emit(
            Event(
                name=EventName.REVIEW_REQUESTED,
                run_id=run.id,
                task_id=child.id,
                data={"phase": phase.value, "artifacts": outputs},
            )
        )

    async def _finish_run(self, run: Run) -> Reply:
        try:
            verdict = await self.auditor.audit_run(run)
        except Exception as exc:
            verdict = _failed_verdict(run, f"auditor error: {exc}")
        await self._save_verdict(verdict)
        if not verdict.passed:
            lines = "; ".join(f"{f.rule_id}: {f.message}" for f in verdict.blockers)
            last_phase = PhaseMachine(self._kind(await self.tasks.list_tasks(run.id))).phases[-1]
            await self._apply(
                run,
                Decision(
                    run_id=run.id,
                    kind=DecisionKind.ASK_USER,
                    inputs={"question": f"The whole-run audit failed: {lines}. How to proceed?"},
                    choice="ask the user",
                    reason="run audit failed",
                    made_by="code",
                    rule="run_audit",
                ),
                verdict.model_copy(update={"phase": last_phase}),
            )
            return await self.drive(run.id)
        run = await self._set_status(run.id, RunStatus.DONE)
        await self.bus.emit(
            Event(name=EventName.RUN_FINISHED, run_id=run.id, data={"passed": True})
        )
        return await self._done_reply(run)

    async def _done_reply(self, run: Run) -> Reply:
        tasks = live(await self.tasks.list_tasks(run.id))
        outputs = [t.output for t in tasks if t.output is not None]
        lines = [f"Run {run.id} ({run.title}) is done."]
        final: Artifact | None = None
        assumptions: list[str] = []
        for ref in outputs:
            art = await self.artifacts.load(run.id, ref)
            lines.append(f"- {ref}: {art.summary}")
            assumptions += [f"{ref} {q.id}: {q.assumption}" for q in art.questions if q.assumption]
            if art.kind in ("research_answer", "qa_report"):
                final = art
        if final is not None and final.kind == "research_answer":
            lines.insert(1, final.answer)
        if assumptions:
            lines.append("Assumptions to check:")
            lines += [f"- {a}" for a in assumptions]
        return Reply(kind="done", text="\n".join(lines), run_id=run.id, artifact=final)

    def _waiting_reply(self, run: Run, pending: list[Task]) -> Reply:
        items = [
            PendingQuestion(task_id=t.id, question=t.objective, origin=t.origin) for t in pending
        ]
        text = "I need your input:\n" + "\n".join(f"- {p.question}" for p in items)
        return Reply(kind="waiting", text=text, run_id=run.id, pending=items)

    # --- the user's answers, comments and questions ----------------------------------

    async def answer(self, run_id: str, text: str, *, task_id: str | None = None) -> Reply:
        """Answers the given pending question or review (the oldest by default)."""
        run = await self.tasks.get_run(run_id)
        pending = await self._pending(run_id)
        target = next((t for t in pending if t.id == task_id), None) if task_id else None
        target = target or (pending[0] if pending else None)
        if target is None:
            return Reply(kind="error", text="Nothing is waiting on you.", run_id=run_id)
        if target.reason == "escalation" and text.strip().lower() == "stop":
            await self.tasks.transition(target.id, TaskStatus.DONE, reason="user said stop")
            await self.stop(run_id, "the user stopped the run at an escalation")
            return Reply(kind="stopped", text="Stopped the run.", run_id=run_id)

        if target.origin is TaskOrigin.REVIEW:
            await self.tasks.transition(target.id, TaskStatus.DONE, reason="reviewed")
            if not _APPROVE.match(text):
                await self._comment(run, target.phase, text)
        else:
            parent = (
                await self.tasks.get_task(target.parent_task_id) if target.parent_task_id else None
            )
            await self.feedback.add(
                run_id,
                Feedback(
                    artifact=parent.output if parent else None,
                    source=FeedbackSource.USER_ANSWER,
                    question_id=target.reason
                    if target.reason and target.reason.startswith("Q-")
                    else None,
                    text=text,
                ),
            )
            await self.tasks.transition(target.id, TaskStatus.DONE, reason="answered")
        await self._set_status(run_id, RunStatus.ACTIVE)
        return await self.drive(run_id)

    async def _comment(self, run: Run, phase: Phase, text: str) -> None:
        tasks = await self.tasks.list_tasks(run.id)
        experts = PhaseMachine.expert_tasks(tasks, phase)
        in_phase = {t.id for t in experts}
        roots = [t for t in experts if not any(d in in_phase for d in t.depends_on)]
        for t in roots:
            await self.feedback.add(
                run.id,
                Feedback(artifact=t.output, source=FeedbackSource.USER_COMMENT, text=text),
            )
            await self.reopen(run, t, origin=TaskOrigin.REVISION, reason=f"user comment: {text}")

    async def request_change(self, run_id: str, text: str, *, target: str = "prd") -> Reply:
        run = await self.tasks.get_run(run_id)
        tasks = live(await self.tasks.list_tasks(run_id))
        producers = [t for t in tasks if t.output is not None and t.output.kind == target]
        if not producers:
            return Reply(
                kind="error", text=f"This run has no {target} to change yet.", run_id=run_id
            )
        producer = producers[-1]
        await self.feedback.add(
            run_id,
            Feedback(artifact=producer.output, source=FeedbackSource.USER_COMMENT, text=text),
        )
        await self.reopen(
            run, producer, origin=TaskOrigin.CHANGE_REQUEST, reason=f"change request: {text}"
        )
        if run.status is RunStatus.DONE:
            await self._set_status(run_id, RunStatus.ACTIVE)
        return await self.drive(run_id)

    async def answer_run_question(self, run_id: str, question: str) -> Reply:
        """Answers from the task store, verdicts, decisions and the trace."""
        run = await self.tasks.get_run(run_id)
        pseudo = Task(
            id=new_id("ask"),
            run_id=run.id,
            phase=Phase.PLANNING,
            agent=AgentName.ORCHESTRATOR,
            objective=question,
            output_kind="run_answer",
        )
        tools = read_tools(self.tasks, self.artifacts, self.decisions, self.verdicts)
        loop: ToolLoop[RunAnswer] = ToolLoop(
            agent=AgentName.ORCHESTRATOR,
            system=RUN_QA + _RUN_QA_FORMAT,
            output_model=RunAnswer,
            llm=self.llm,
            bus=self.bus,
            checkpoints=self.checkpoints,
            tools=tools,
            tool_ctx=RunToolContext(run=run, task=pseudo, bus=self.bus),
            run=run,
            task_id=pseudo.id,
            attempt=1,
        )
        outcome = await loop.run_loop(
            Message.user(f"Run {run.id}: {run.title} ({run.status.value}).\nQuestion: {question}")
        )
        if outcome.output is None:
            return Reply(
                kind="error", text=f"I couldn't answer that: {outcome.error}", run_id=run_id
            )
        cited = f"\n(from {', '.join(outcome.output.cited)})" if outcome.output.cited else ""
        return Reply(kind="answer", text=outcome.output.answer + cited, run_id=run_id)

    # --- helpers ----------------------------------------------------------------------

    async def _record(self, decision: Decision) -> None:
        await self.decisions.record(decision)

    async def _set_status(self, run_id: str, status: RunStatus) -> Run:
        run = await self.tasks.get_run(run_id)
        if run.status is status:
            return run
        return await self.tasks.update_run(run.model_copy(update={"status": status}))

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


def _prepare_workspace(
    runs_dir: Path, run_id: str, context_dirs: list[str]
) -> tuple[Path, list[str]]:
    workspace = (runs_dir / run_id / "workspace").resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace, [str(Path(d).resolve()) for d in context_dirs]


def _reviewed(tasks: list[Task], phase: Phase) -> bool:
    experts = PhaseMachine.expert_tasks(tasks, phase)
    latest = max((t.created_at for t in experts), default=None)
    return any(
        t.agent is AgentName.USER
        and t.origin is TaskOrigin.REVIEW
        and t.phase is phase
        and t.status is TaskStatus.DONE
        and (latest is None or t.created_at > latest)
        for t in tasks
    )


def _failed_verdict(
    run: Run, message: str, *, task_id: str | None = None, phase: Phase | None = None
) -> Verdict:
    scope = RuleScope.TASK if task_id else RuleScope.PHASE if phase else RuleScope.RUN
    return Verdict(
        run_id=run.id,
        scope=scope,
        task_id=task_id,
        phase=phase,
        passed=False,
        findings=[Finding(rule_id="AUDIT", severity=Severity.BLOCKER, message=message)],
    )
