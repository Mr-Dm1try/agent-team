"""Settling the run between dispatch rounds: retries go back to ready, agents'
questions go to the user, research requests become Researcher tasks, and escalated
tasks ask the user for guidance."""

from __future__ import annotations

from agent_team.core.models import (
    AgentName,
    Decision,
    DecisionKind,
    Feedback,
    FeedbackSource,
    ReviewMode,
    Run,
    Task,
    TaskOrigin,
    TaskStatus,
)
from agent_team.orchestrator.base import (
    CHILD_ORIGINS,
    ESCALATION,
    KEEP_HINT,
    OrchestratorBase,
    can_keep,
)
from agent_team.orchestrator.control import create_user_task
from agent_team.orchestrator.replies import plain_agent, upper_first


class SettleMixin(OrchestratorBase):
    """Moves blocked tasks on once what they wait for has happened."""

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
                    and by_id[d].origin in CHILD_ORIGINS
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
        """Feed the children's answers back as inputs and make the task ready again."""
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
        """Ask the user how to get an escalated task unstuck."""
        why = task.last_error or "it failed its check with no attempts left"
        failed = await self.verdicts.latest_for_task(task.id)
        keep = KEEP_HINT if can_keep(task, failed) else ""
        child = await create_user_task(
            self.tasks,
            run,
            phase=task.phase,
            question=(
                f"{upper_first(plain_agent(task.agent))} is stuck on: {task.objective} "
                f"({why}). Reply with guidance to retry it, or 'stop'.{keep}"
            ),
            origin=TaskOrigin.QUESTION,
            parent=task,
            reason=ESCALATION,
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
