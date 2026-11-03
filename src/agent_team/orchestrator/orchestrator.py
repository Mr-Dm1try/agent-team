"""The orchestrator: the only agent you talk to.

It works out what each message means (orchestrator/messages.py), starts runs, and
drives a run through the fixed phases: plan, dispatch ready tasks (in parallel,
Engineer tasks one at a time), check each task's output at its end, apply the verdict
policy's decisions, move to the next phase once all of a phase's tasks are done, and
finish with a whole-run audit. Every choice goes to the decision log. Run state lives
in the task store, so ``drive`` can be called again after a crash and picks up where
it left off.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_team.agents.planner import kind_from_objective, planning_objective
from agent_team.core.artifacts import Artifact, Plan, Spec, plain_kind
from agent_team.core.models import (
    AgentName,
    Decision,
    DecisionKind,
    Event,
    EventName,
    Phase,
    RequestKind,
    ResultStatus,
    ReviewMode,
    Run,
    RunStatus,
    Severity,
    Task,
    TaskOrigin,
    TaskStatus,
    new_id,
)
from agent_team.orchestrator.base import (
    PAUSING,
    OrchestratorConfig,
    failed_verdict,
)
from agent_team.orchestrator.control import create_user_task
from agent_team.orchestrator.conversation import ConversationMixin
from agent_team.orchestrator.decisions import DecisionsMixin
from agent_team.orchestrator.dispatcher import INTERRUPTED, PAUSING_ERRORS
from agent_team.orchestrator.lineage import ancestry
from agent_team.orchestrator.phases import PhaseMachine, expert_tasks, reviewed
from agent_team.orchestrator.planning import RULE as PIPELINE
from agent_team.orchestrator.planning import expand_plan, implementation_tasks
from agent_team.orchestrator.policy import SimplePolicy
from agent_team.orchestrator.replies import (
    Reply,
    done_reply,
    upper_first,
    waiting_reply,
)
from agent_team.orchestrator.settle import SettleMixin

__all__ = ["Orchestrator", "OrchestratorConfig"]


class Orchestrator(ConversationMixin, SettleMixin, DecisionsMixin):
    """Drives runs through the phases and answers the user. See the module docstring."""

    # --- entry points ----------------------------------------------------------------

    async def start_run(
        self,
        request: str,
        *,
        kind: RequestKind = RequestKind.BUILD,
        title: str | None = None,
        context_dirs: list[str] | None = None,
        review_mode: ReviewMode | None = None,
    ) -> Run:
        """Create the run, its workspace and the planning task."""
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
        """The kill switch: the run halts at the next safe point and can be resumed."""
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
        """Run until the run is done, waits on the user, or stops."""
        for _ in range(self.config.max_loops):
            run = await self.tasks.get_run(run_id)
            if run.status in (RunStatus.STOPPED, RunStatus.PAUSED):
                return Reply(
                    kind="stopped", text=f"Run {run.id} is {run.status.value}.", run_id=run.id
                )
            if run.status is RunStatus.DONE:
                return await done_reply(run, await self.tasks.list_tasks(run.id), self.artifacts)
            if run.status is RunStatus.WAITING:
                run = await self._set_status(run.id, RunStatus.ACTIVE)

            tasks = await self.tasks.list_tasks(run.id)
            if await self._settle(run, tasks):
                continue
            machine = PhaseMachine(self._kind(tasks))
            phase = machine.current(tasks)
            if await self._phases_finished(run, machine, tasks, before=phase):
                continue  # a review was requested
            if phase is None:
                return await self._finish_run(run)
            await self._phase_started(run, phase)

            open_tasks = machine.open_tasks(tasks, phase)
            if not open_tasks:
                # The phase has no work yet: Implementation is built from the spec.
                if phase is Phase.IMPLEMENTATION and await self._start_implementation(run, tasks):
                    continue
                return Reply(
                    kind="error", text=f"Run has no work for {phase.value}.", run_id=run.id
                )

            ready = await self._ready(run.id, phase)
            if not ready:
                pending = await self._pending(run.id)
                if pending:
                    await self._set_status(run.id, RunStatus.WAITING)
                    return waiting_reply(run, pending)
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
        """Agent tasks in ``phase`` that can run now."""
        found = {t.id: t for t in await self.tasks.ready_tasks(run_id)}
        for t in await self.tasks.list_tasks(run_id, status=TaskStatus.READY):
            found.setdefault(t.id, t)
        return [t for t in found.values() if t.phase is phase and t.agent is not AgentName.USER]

    def _kind(self, tasks: list[Task]) -> RequestKind:
        """The run's request kind, read back from the planning task's objective."""
        for t in tasks:
            if t.agent is AgentName.ORCHESTRATOR and t.output_kind == "plan":
                return kind_from_objective(t.objective) or RequestKind.BUILD
        return RequestKind.BUILD

    async def _phase_started(self, run: Run, phase: Phase) -> None:
        """Emit phase_started once per phase in this process."""
        started = self._started.setdefault(run.id, set())
        if phase not in started:
            started.add(phase)
            await self.bus.emit(
                Event(name=EventName.PHASE_STARTED, run_id=run.id, data={"phase": phase.value})
            )

    # --- one task: dispatch, audit, apply --------------------------------------------

    async def _run_task(self, run: Run, task: Task) -> None:
        """Log the delegation, dispatch the task, and audit the output unless the run paused or the
        task was reopened."""
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
        if result.status is ResultStatus.ERROR and result.error_type in PAUSING_ERRORS:
            status = RunStatus.STOPPED if result.error_type == "RunStopped" else RunStatus.PAUSED
            await self._set_status(run.id, status)
            return
        task = await self.tasks.get_task(task.id)
        if task.status is TaskStatus.AUDITING:  # not superseded while it ran
            await self._audit(run, task, result.artifact)

    async def _audit(self, run: Run, task: Task, artifact: Artifact | None) -> None:
        """Audit one output and apply the policy's decisions; fails closed and pauses on budget or
        stop."""
        try:
            verdict = await self.auditor.audit_task(task, artifact, run)
        except PAUSING as exc:
            await self._pause(run.id, exc)  # the task stays auditing and is audited on resume
            return
        except Exception as exc:  # fail closed
            verdict = failed_verdict(run, f"auditor error: {exc}", task_id=task.id)
        await self._save_verdict(verdict)
        task = await self.tasks.get_task(task.id)
        if task.status is not TaskStatus.AUDITING:
            return  # superseded by a reopen while the auditor ran
        task.finding_ids = [f.id for f in verdict.findings if f.severity is Severity.BLOCKER]
        task = await self.tasks.update_task(task)
        # Earlier verdicts of the task and the tasks it was reopened from; the policy
        # compares the last one with this one (no progress).
        history = [
            v
            for t in ancestry(task, await self.tasks.list_tasks(run.id))
            for v in await self.verdicts.for_task(t.id)
            if v.id != verdict.id
        ]
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
        """Expand a passed plan into the fixed pipeline of tasks."""
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

    async def _start_implementation(self, run: Run, tasks: list[Task]) -> bool:
        """Once the spec is done: one Engineer task per work package, then QA. Returns
        whether tasks were created."""
        live_tasks = [t for t in tasks if t.status is not TaskStatus.SUPERSEDED]
        if expert_tasks(live_tasks, Phase.IMPLEMENTATION):
            return False
        spec_tasks = [
            t
            for t in live_tasks
            if t.agent is AgentName.ARCHITECT and t.status is TaskStatus.DONE and t.output
        ]
        if not spec_tasks:
            return False
        spec_task = spec_tasks[-1]
        assert spec_task.output is not None
        spec = await self.artifacts.load(run.id, spec_task.output)
        assert isinstance(spec, Spec)
        previous = [
            t
            for t in tasks
            if t.phase is Phase.IMPLEMENTATION
            and t.status is TaskStatus.SUPERSEDED
            and t.agent in (AgentName.ENGINEER, AgentName.QA)
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
            reason=f"spec done; {len(packages)} work package(s), run one at a time",
        )
        return True

    async def _record_pipeline(
        self,
        run: Run,
        task: Task,
        new_tasks: list[Task],
        *,
        inputs: dict[str, object],
        reason: str,
    ) -> None:
        """Log a code-made pipeline expansion as a PLAN decision."""
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

    # --- the end of a phase and of a run --------------------------------------------

    async def _phases_finished(
        self, run: Run, machine: PhaseMachine, tasks: list[Task], *, before: Phase | None
    ) -> bool:
        """For each finished phase before ``before`` (all of them when None): emit
        phase_finished once per process, like phase_started, and in ``every_phase``
        mode ask for a review of it if its current output wasn't reviewed. Returns
        whether a review was requested."""
        for phase in machine.phases:
            if phase is before:
                break
            newest = max(expert_tasks(tasks, phase), key=lambda t: t.created_at)
            key = (phase, newest.id)
            if key not in self._finished.setdefault(run.id, set()):
                self._finished[run.id].add(key)
                await self.bus.emit(
                    Event(name=EventName.PHASE_FINISHED, run_id=run.id, data={"phase": phase.value})
                )
            if (
                run.review_mode is ReviewMode.EVERY_PHASE
                and machine.next(phase) is not None
                and not reviewed(tasks, phase)
            ):
                await self._request_review(run, phase, tasks)
                return True
        return False

    async def _request_review(self, run: Run, phase: Phase, tasks: list[Task]) -> None:
        """Ask the user to review a phase, with each output's summary and the warnings
        from the checks at the end of its tasks."""
        experts = expert_tasks(tasks, phase)
        outputs = [str(t.output) for t in experts if t.output]
        summaries: list[str] = []
        for t in experts:
            if t.output:
                art = await self.artifacts.load(run.id, t.output)
                summaries.append(f"{upper_first(plain_kind(t.output.kind))}: {art.summary}")
            latest = await self.verdicts.latest_for_task(t.id)
            summaries += [
                f"Warning: {f.message}"
                for f in (latest.findings if latest else [])
                if f.severity is Severity.WARNING
            ]
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
        """Run the whole-run audit: finish the run on a pass, or ask the user how to proceed."""
        try:
            verdict = await self.auditor.audit_run(run)
        except PAUSING as exc:
            await self._pause(run.id, exc)
            return await self.drive(run.id)
        except Exception as exc:
            verdict = failed_verdict(run, f"auditor error: {exc}")
        await self._save_verdict(verdict)
        if not verdict.passed:
            lines = "; ".join(f.message for f in verdict.blockers)
            last_phase = PhaseMachine(self._kind(await self.tasks.list_tasks(run.id))).phases[-1]
            await self._apply(
                run,
                Decision(
                    run_id=run.id,
                    kind=DecisionKind.ASK_USER,
                    inputs={
                        "question": f"The final check of the whole run found problems: {lines}. "
                        "Reply 'approve' to accept it as it is, tell me what to fix, or 'stop'."
                    },
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
        return await done_reply(run, await self.tasks.list_tasks(run.id), self.artifacts)


def _prepare_workspace(
    runs_dir: Path, run_id: str, context_dirs: list[str]
) -> tuple[Path, list[str]]:
    """Create the run's workspace and resolve the context folders to absolute paths."""
    workspace = (runs_dir / run_id / "workspace").resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace, [str(Path(d).resolve()) for d in context_dirs]
