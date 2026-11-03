"""Applying the verdict policy's decisions: retry, escalate, the stronger model,
upstream re-runs, fixes, questions for the user and stop."""

from __future__ import annotations

from agent_team.core.artifacts import ArtifactRef
from agent_team.core.models import (
    Decision,
    DecisionKind,
    EventName,
    FeedbackSource,
    Finding,
    Phase,
    Run,
    RunStatus,
    Task,
    TaskOrigin,
    TaskStatus,
    Verdict,
)
from agent_team.orchestrator.base import (
    KEEP_HINT,
    RUN_QUESTION,
    TASK_QUESTION,
    OrchestratorBase,
    can_keep,
)
from agent_team.orchestrator.control import create_user_task


class DecisionsMixin(OrchestratorBase):
    """Turns each Decision into task moves, feedback and re-opened work."""

    async def _apply(self, run: Run, d: Decision, verdict: Verdict | None = None) -> None:
        """Record the decision, then carry it out on the task store."""
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
        elif d.kind is DecisionKind.FIX and task is not None:
            ids = d.inputs.get("finding_ids") or []
            findings = [f for f in (verdict.findings if verdict else []) if f.id in ids]
            await self._findings_to_feedback(run, task, findings)
            root = await self.reopen(run, task, origin=TaskOrigin.FIX, finding_ids=list(ids))
            await self._rerun_after(run, root, d.inputs.get("rerun_after") or [])
            checker_id = d.inputs.get("checker")
            if isinstance(checker_id, str):
                # The checking task's own blockers, for its fresh run after the fix.
                own = d.inputs.get("checker_finding_ids") or []
                checker = await self.tasks.get_task(checker_id)
                found = [f for f in (verdict.findings if verdict else []) if f.id in own]
                await self._findings_to_feedback(run, checker, found)
        elif d.kind is DecisionKind.ASK_USER:
            question = str(d.inputs.get("question") or d.reason)
            if task is not None and can_keep(task, verdict):
                question += KEEP_HINT
            if task is not None and task.status is TaskStatus.AUDITING:
                await self.tasks.transition(task.id, TaskStatus.ESCALATED, reason=d.reason)
            phase = (
                task.phase
                if task
                else (verdict.phase if verdict and verdict.phase else Phase.PLANNING)
            )
            why = TASK_QUESTION if task is not None else RUN_QUESTION
            child = await create_user_task(
                self.tasks,
                run,
                phase=phase,
                question=question,
                origin=TaskOrigin.QUESTION,
                parent=task,
                reason=why,
            )
            if task is not None:
                task = await self.tasks.get_task(task.id)
                task.depends_on = [*task.depends_on, child.id]
                await self.tasks.update_task(task)
        elif d.kind is DecisionKind.STOP:
            await self._set_status(run.id, RunStatus.STOPPED)

    async def _upstream_rerun(self, run: Run, d: Decision, task: Task | None) -> None:
        """Reopen the producer of a faulty upstream artifact, with the finding as feedback, and
        count the re-run."""
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
        run = await self.tasks.update_run(
            run.model_copy(update={"upstream_reruns": run.upstream_reruns + 1})
        )
        await self.reopen(run, producer, origin=TaskOrigin.REVISION, reason=d.reason)

    async def _rerun_after(self, run: Run, root: Task | None, ids: list[str]) -> None:
        """Tasks the policy wants re-run once a fix passes, such as a fresh QA review
        after Engineer fixes. Those not already reopened as downstream of the fix
        are reopened here and wait for the fix."""
        if root is None:
            return
        for tid in ids:
            other = await self.tasks.get_task(tid)
            if other.status is TaskStatus.SUPERSEDED:
                continue
            copy = await self.reopen(
                run, other, origin=TaskOrigin.REVISION, reason=f"re-run after fix {root.id}"
            )
            if copy is not None and root.id not in copy.depends_on:
                copy.depends_on = [*copy.depends_on, root.id]
                await self.tasks.update_task(copy)

    async def _findings_to_feedback(self, run: Run, task: Task, findings: list[Finding]) -> None:
        """Turn audit findings into feedback the task's next attempt will read."""
        for f in findings:
            fix = f" Suggested fix: {f.suggested_fix}" if f.suggested_fix else ""
            text = f"{f.rule_id} on {f.item or 'the artifact'}: {f.message}.{fix}"
            await self.feedback.add(run.id, self._feedback(task, FeedbackSource.AUDITOR, text))
