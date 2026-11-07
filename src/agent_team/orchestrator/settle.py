"""Settling the run between dispatch rounds: retries go back to ready, agents'
questions go to the user, research requests become Researcher tasks, and escalated
tasks get the policy's next step after a failed attempt, or ask the user for
guidance."""

from __future__ import annotations

from agent_team.core.artifacts import ResearchRequest
from agent_team.core.models import (
    AgentName,
    Decision,
    DecisionKind,
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
    OrchestratorBase,
    can_keep,
    reply_options,
    rerun_of,
)
from agent_team.orchestrator.control import create_user_task
from agent_team.orchestrator.lineage import ancestry
from agent_team.orchestrator.phases import is_work
from agent_team.orchestrator.replies import plain_agent, upper_first
from agent_team.orchestrator.research import find_same, is_requested, waits_on


class SettleMixin(OrchestratorBase):
    """Moves blocked tasks on once what they wait for has happened."""

    async def _settle(self, run: Run, tasks: list[Task]) -> bool:
        """Moves tasks that are blocked on something that has since happened. Returns
        True if anything changed, so the loop re-reads the store."""
        changed = False
        by_id = {t.id: t for t in tasks}
        for task in tasks:
            if not is_work(task):
                continue  # user tasks are answered; checks run inside their task's audit
            if task.status is TaskStatus.RETRY:
                await self.tasks.transition(task.id, TaskStatus.READY, reason="retry with feedback")
                changed = True
            elif task.status is TaskStatus.AUDITING:
                # The process stopped between output and verdict: audit it now.
                artifact = await self.artifacts.load(run.id, task.output) if task.output else None
                await self._audit(run, task, artifact)
                changed = True
            elif task.status in (TaskStatus.WAITING, TaskStatus.ESCALATED):
                # Its own questions and research, and research another task asked
                # for first that this one waits on instead of asking again.
                children = [
                    by_id[d]
                    for d in task.depends_on
                    if d in by_id
                    and (
                        (by_id[d].origin in CHILD_ORIGINS and by_id[d].parent_task_id == task.id)
                        or is_requested(by_id[d], by_id)
                    )
                ]
                live_children = [c for c in children if c.status is not TaskStatus.SUPERSEDED]
                if live_children:
                    if all(c.status is TaskStatus.DONE for c in live_children):
                        await self._unblock(task, children)
                        changed = True
                elif task.status is TaskStatus.WAITING:
                    changed = await self._open_children(run, task) or changed
                else:
                    await self._escalated(run, task)
                    changed = True
        return changed

    async def _unblock(self, task: Task, children: list[Task]) -> None:
        """Feed the children's answers back as inputs and make the task ready again."""
        ids = {c.id for c in children}
        task.depends_on = [d for d in task.depends_on if d not in ids]
        for c in children:
            if c.output is not None and c.output not in task.inputs:
                task.inputs.append(c.output)
        # The attempt that asked counts like any other. The answers buy one more only
        # when none is left: the user's guidance on an escalation, or answers to
        # questions asked on the last attempt (research can't be asked for then).
        task.max_attempts = max(task.max_attempts, task.attempt + 1)
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
                text = (
                    "Review mode is auto_assumptions: no answer will come. Record your "
                    "best assumption in this question's `assumption` and continue."
                )
                feedback = self._feedback(task, FeedbackSource.USER_ANSWER, text, question_id=q.id)
                await self.feedback.add(run.id, feedback)
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
        if artifact.research_requests:
            child_ids += await self._plan_research(run, task, artifact.research_requests)
        task = await self.tasks.get_task(task.id)
        if child_ids:
            task.depends_on = [*task.depends_on, *child_ids]
            await self.tasks.update_task(task)
        else:
            task.max_attempts = max(task.max_attempts, task.attempt + 1)
            await self.tasks.update_task(task)
            await self.tasks.transition(
                task.id, TaskStatus.READY, reason="assumptions recorded or research reused"
            )
        return True

    async def _plan_research(
        self, run: Run, task: Task, requests: list[ResearchRequest]
    ) -> list[str]:
        """Turn ``task``'s research requests into the tasks it waits for, and return
        their ids. A request the run already asked is answered by that task: a finished
        brief joins the inputs, and open requested research is waited for unless it
        waits on this task. Past a limit, for a question the task already has the answer
        to, or one planned research is still on, the request is refused and the task
        told to answer with what it has. Either way the task may not ask again."""
        limits = self.config.research
        tasks = await self.tasks.list_tasks(run.id)
        by_id = {t.id: t for t in tasks}
        own = {t.id for t in ancestry(task, tasks)}
        waits: list[str] = []
        refused: list[str] = []
        task = await self.tasks.get_task(task.id)
        for i, req in enumerate(requests, start=1):
            match = find_same(req.question, [t for t in tasks if t.id not in own])
            choice, why, target = "", req.why, None
            if match is not None and match.answered:
                assert match.task.output is not None
                if match.task.output in task.inputs:
                    refused.append(f"{req.question} (already answered in your inputs)")
                    choice, target = "refuse: already answered", match.task.id
                else:
                    task.inputs.append(match.task.output)
                    choice, target = "reuse the finished research", match.task.id
            elif (
                match is not None
                and is_requested(match.task, by_id)
                and not waits_on(task, match.task, by_id)
            ):
                waits.append(match.task.id)
                choice, target = "wait for the same open research", match.task.id
            elif match is not None:
                refused.append(f"{req.question} (another task is researching it now)")
                choice, target = "refuse: being researched", match.task.id
            elif (limit := limits.refusal(task, tasks)) is not None:
                refused.append(f"{req.question} ({limit})")
                choice, why = "refuse: research limit", limit
            else:
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
                tasks.append(child)
                by_id[child.id] = child
                waits.append(child.id)
                choice, target = "researcher", child.id
            await self._record(
                Decision(
                    run_id=run.id,
                    task_id=task.id,
                    kind=DecisionKind.DELEGATE,
                    inputs={"question": req.question, "research_task": target},
                    choice=choice,
                    reason=why,
                    made_by="code",
                    rule="research_request",
                )
            )
        # One round of research per task: the next attempt answers with what it has.
        task.may_request_research = False
        if refused:
            text = (
                "No more research will be run for this task: "
                + "; ".join(refused)
                + ". Answer with what you have, and list what is still unknown as gaps "
                "or as questions with an assumption, not as research_requests."
            )
            await self.feedback.add(run.id, self._feedback(task, FeedbackSource.HARNESS, text))
        await self.tasks.update_task(task)
        return list(dict.fromkeys(waits))

    async def _escalated(self, run: Run, task: Task) -> None:
        """An escalated task with nothing open: the policy's next step after a failed
        attempt (the stronger model, then the user), else ask the user. Done here, not
        right after the attempt, so a crash in between still gets the same step."""
        decisions = await self.policy.on_task_error(run, task)
        for decision in decisions:
            await self._apply(run, decision)
        if not decisions:
            await self._ask_about_escalation(run, task)

    async def _ask_about_escalation(self, run: Run, task: Task) -> None:
        """Ask the user how to get an escalated task unstuck."""
        why = task.last_error or "it failed its check with no attempts left"
        failed = await self.verdicts.latest_for_task(task.id)
        options = reply_options(keep=can_keep(task, failed), rerun=rerun_of(task, failed))
        child = await create_user_task(
            self.tasks,
            run,
            phase=task.phase,
            question=(
                f"{upper_first(plain_agent(task.agent))} is stuck on: {task.objective} "
                f"({why}). {options}"
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
