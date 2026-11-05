"""The conversation with the user: reading each message, answering what waits on
them, confirming and applying changes, and answering questions about a run.

``handle`` runs the message loop (orchestrator/messages.py), which works out what a
message means from the run's state, then carries out the actions it validated. A change
is confirmed before anything reopens. Replies are in plain words, with internal ids
and artifact names removed.
"""

from __future__ import annotations

import traceback

from agent_team.core.artifacts import plain_kind
from agent_team.core.errors import BudgetExceeded
from agent_team.core.models import (
    SESSION,
    AgentName,
    Decision,
    DecisionKind,
    Event,
    EventName,
    Feedback,
    FeedbackSource,
    Phase,
    RequestKind,
    RuleScope,
    Run,
    RunStatus,
    Task,
    TaskOrigin,
    TaskStatus,
    Verdict,
)
from agent_team.core.tracing import Tracer
from agent_team.orchestrator.base import (
    APPROVE,
    ESCALATION,
    PAUSING,
    RUN_QUESTION,
    TASK_QUESTION,
    OrchestratorBase,
    can_keep,
)
from agent_team.orchestrator.control import create_user_task
from agent_team.orchestrator.lineage import downstream
from agent_team.orchestrator.messages import (
    CHANGE_TARGETS,
    CONFIRM_CHANGE,
    NO,
    YES,
    Action,
    Interpretation,
    describe_pending,
    interpret,
    scrub,
)
from agent_team.orchestrator.phases import expert_tasks
from agent_team.orchestrator.replies import Reply, pending_items

#: The order actions are carried out in: answers and a new run, then preferences (kept
#: under the new run when there is one), changes, and a stop last.
_ORDER = {
    "answer_pending": 0,
    "start_run": 0,
    "remember_preference": 1,
    "request_change": 2,
    "stop_run": 3,
}


class ConversationMixin(OrchestratorBase):
    """Everything the orchestrator does with a user message."""

    async def handle(
        self, message: str, *, run_id: str | None = None, context_dirs: list[str] | None = None
    ) -> Reply:
        """One user message. An orchestrator tool loop works out what it means from the
        run's state and picks actions; code then carries them out. If the loop fails,
        nothing is acted on and the user is asked again. A message about a run is a
        span in that run's trace. ``context_dirs`` are the read-only folders a run the
        message starts gets."""
        run = await self.tasks.get_run(run_id) if run_id else None
        if run is None:
            return await self._handle(message, None, context_dirs or [])
        async with Tracer(self.bus).in_run(run, "user message", "other", agent=AgentName.USER):
            return await self._handle(message, run)

    async def _handle(
        self, message: str, run: Run | None, context_dirs: list[str] | None = None
    ) -> Reply:
        """Read one message and carry out what it means; see ``handle``."""
        await self.bus.emit(
            Event(
                name=EventName.USER_MESSAGE,
                run_id=run.id if run else SESSION,
                agent=AgentName.USER,
                data={"text": message},
            )
        )
        waiting = await self._pending(run.id) if run else []
        pending = [(t, await self._asker(t)) for t in waiting]
        try:
            meant = await interpret(
                message,
                run,
                pending,
                await self._changeable(run) if run else set(),
                tasks=self.tasks,
                artifacts=self.artifacts,
                decisions=self.decisions,
                verdicts=self.verdicts,
                checkpoints=self.checkpoints,
                bus=self.bus,
                llm=self.llm,
                workspace=str(self.config.runs_dir),
            )
        except PAUSING as exc:
            return await self._paused_while_reading(run, exc)
        except Exception as exc:  # a failed read is never an action
            await self.bus.emit(
                Event(
                    name=EventName.ERROR,
                    run_id=run.id if run else SESSION,
                    agent=AgentName.ORCHESTRATOR,
                    data={
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
            meant = None
        if meant is None:
            return self._didnt_catch(run, waiting)
        return await self._carry_out(run, message, meant, context_dirs or [])

    async def _paused_while_reading(self, run: Run | None, exc: Exception) -> Reply:
        """The budget ran out or the run was stopped while reading the message: pause the
        run (it can resume; a finished run stays finished) and act on nothing."""
        paused = run is not None and run.status is not RunStatus.DONE
        if paused and run is not None:
            await self._pause(run.id, exc)
        if isinstance(exc, BudgetExceeded):
            text = "The budget ran out before I could read that"
            text += ", so I've paused. Raise the budget and resume" if paused else ". Raise it"
            text += ", then send it again."
        else:
            text = "The run was stopped before I could read that. Resume it and send it again."
        return Reply(kind="stopped", text=text, run_id=run.id if run else None)

    async def _carry_out(
        self,
        run: Run | None,
        message: str,
        meant: Interpretation,
        context_dirs: list[str] | None = None,
    ) -> Reply:
        """Carry out the loop's actions (answers first, then changes, a stop last, so a
        change sees the run as the answers left it), log each as a decision, then drive
        what they touched and reply in plain words. Once the run is stopped, the rest of
        the actions on it are dropped (and logged as dropped); a new run still starts."""
        if not meant.actions:
            await self._log_action(run.id if run else SESSION, message, meant, None)
        touched, started, stopped = False, None, None
        notes: list[str] = []
        for action in sorted(meant.actions, key=lambda a: _ORDER[a.tool]):
            args = action.args
            if action.tool == "start_run":
                started = await self.start_run(
                    args["request"],
                    kind=RequestKind(args["kind"]),
                    context_dirs=context_dirs,
                )
                await self._log_action(started.id, message, meant, action)
                continue
            if action.tool == "remember_preference":
                await self._remember(started or run, message, args["text"])
                continue
            assert run is not None  # the tools refuse these without a run
            await self._log_action(run.id, message, meant, action, dropped=stopped is not None)
            if stopped is not None:
                continue
            if action.tool == "answer_pending":
                live = {t.id: t for t in await self._pending(run.id)}
                target = live.get(args["task_id"])
                if target is None:
                    continue  # an earlier answer in this message closed it
                stopped = await self._record_answer(run, target, args["text"], notes)
                touched = touched or stopped is None
            elif action.tool == "request_change":
                note = await self._propose_change(run, args["text"], args["target"])
                notes += [note] if note else []
                touched = True
            else:
                await self.stop(run.id, args["reason"])
                stopped = Reply(kind="stopped", text="", run_id=run.id)
        states: list[Reply] = []
        if stopped is not None:
            states.append(stopped)
        elif run is not None and touched:
            states.append(await self._continue(run.id))
        if started is not None:
            states.append(await self.drive(started.id))
        if not states:
            items = pending_items(await self._pending(run.id)) if run else []
            states.append(
                Reply(kind="answer", text="", run_id=run.id if run else None, pending=items)
            )
        return self._say(message, [meant.reply, *notes], states)

    async def _remember(self, run: Run | None, message: str, text: str) -> None:
        """Record a preference the user stated as a ``remember`` decision; the memory
        writer saves it from there. It needs no run, so it stands even after a stop."""
        await self._record(
            Decision(
                run_id=run.id if run else SESSION,
                kind=DecisionKind.REMEMBER,
                inputs={"message": message[:500], "preference": text},
                choice="remember preference",
                reason="the user stated a preference",
                made_by="model",
            )
        )

    async def _log_action(
        self,
        run_id: str,
        message: str,
        meant: Interpretation,
        action: Action | None,
        *,
        dropped: bool = False,
    ) -> None:
        """Record one action the loop chose (or that it chose none) as a decision; an
        action dropped because the run was stopped first is logged as dropped."""
        choice: str = "reply only" if action is None else action.tool
        if action is not None and "task_id" in action.args:
            choice += f" {action.args['task_id']}"
        elif action is not None and "target" in action.args:
            choice += f" {action.args['target']}"
        if dropped:
            choice = f"dropped {choice}: the run was stopped first"
        await self._record(
            Decision(
                run_id=run_id,
                kind=DecisionKind.CLASSIFY_REQUEST,
                inputs={"message": message[:500], **(action.args if action else {})},
                choice=choice,
                reason=meant.reply[:500],
                made_by="model",
            )
        )

    def _say(self, message: str, said: list[str], states: list[Reply]) -> Reply:
        """The loop's reply and notes, then each run's state, with internal names removed.
        With two runs (one stopped or touched, one just started) the new run's reply is
        returned, after the old run's text, and only its items are listed as pending."""
        state = states[-1]
        parts = [scrub(t, message) for t in [*said, *(r.text for r in states)]]
        state.text = "\n\n".join(p for p in parts if p)
        return state

    def _didnt_catch(self, run: Run | None, waiting: list[Task]) -> Reply:
        """The fallback when the loop fails: act on nothing and list what's waiting."""
        text = "I didn't catch that. Could you say it another way?"
        if waiting:
            text += " Here's what's waiting on you:\n" + "\n".join(
                f"- {t.objective}" for t in waiting
            )
        return Reply(
            kind="waiting" if waiting else "answer",
            text=text,
            run_id=run.id if run else None,
            pending=pending_items(waiting),
        )

    async def _asker(self, task: Task) -> str:
        """Who is asking for one waiting item, in plain words."""
        parent = await self.tasks.get_task(task.parent_task_id) if task.parent_task_id else None
        return describe_pending(task, parent)

    async def answer(self, run_id: str, text: str, *, task_id: str | None = None) -> Reply:
        """Answers the given pending question or review (the oldest by default)."""
        run = await self.tasks.get_run(run_id)
        pending = await self._pending(run_id)
        target = next((t for t in pending if t.id == task_id), None) if task_id else None
        target = target or (pending[0] if pending else None)
        if target is None:
            return Reply(kind="error", text="Nothing is waiting on you.", run_id=run_id)
        notes: list[str] = []
        stopped = await self._record_answer(run, target, text, notes)
        if stopped is not None:
            return stopped
        reply = await self._continue(run_id)
        reply.text = "\n\n".join([*notes, reply.text]) if notes else reply.text
        return reply

    async def _continue(self, run_id: str) -> Reply:
        """Drive the run on after answers. A run the user just accepted as done stays
        done, so its final audit doesn't run again."""
        run = await self.tasks.get_run(run_id)
        if run.status is not RunStatus.DONE:
            await self._set_status(run_id, RunStatus.ACTIVE)
        return await self.drive(run_id)

    async def _record_answer(
        self, run: Run, target: Task, text: str, notes: list[str]
    ) -> Reply | None:
        """Close one pending item with the user's words, without driving the run. Returns
        a reply only when the answer stopped the run; anything else the user should hear
        is added to ``notes``."""
        if target.reason == ESCALATION and text.strip().lower() == "stop":
            await self.tasks.transition(target.id, TaskStatus.DONE, reason="user said stop")
            await self.stop(run.id, "the user stopped the run at an escalation")
            return Reply(kind="stopped", text="Stopped the run.", run_id=run.id)
        if target.origin is TaskOrigin.REVIEW:
            await self.tasks.transition(target.id, TaskStatus.DONE, reason="reviewed")
            if not APPROVE.match(text):
                await self._comment(run, target.phase, text)
            return None
        if target.reason == RUN_QUESTION:
            return await self._answer_audit_question(run, target, text)
        if target.reason == CONFIRM_CHANGE:
            await self._confirm_change(run, target, text, notes)
            return None
        parent = await self.tasks.get_task(target.parent_task_id) if target.parent_task_id else None
        if (
            parent is not None
            and target.reason in (TASK_QUESTION, ESCALATION)
            and APPROVE.match(text)
            and await self._keep_as_is(run, target, parent)
        ):
            return None
        question_id = target.reason if target.reason and target.reason.startswith("Q-") else None
        await self.feedback.add(
            run.id,
            self._feedback(parent, FeedbackSource.USER_ANSWER, text, question_id=question_id)
            if parent
            else Feedback(
                artifact=None,
                task_id=target.id,
                source=FeedbackSource.USER_ANSWER,
                question_id=question_id,
                text=text,
            ),
        )
        await self.tasks.transition(target.id, TaskStatus.DONE, reason="answered")
        return None

    async def _keep_as_is(self, run: Run, target: Task, parent: Task) -> bool:
        """The user approved work that failed its check: keep it as it is. A passing
        verdict records the acceptance, so the phase can finish. Returns False when
        there is nothing to keep, and the answer is taken as guidance instead."""
        failed = await self.verdicts.latest_for_task(parent.id)
        if parent.status is not TaskStatus.ESCALATED or not can_keep(parent, failed):
            return False
        assert failed is not None  # can_keep needs a failed verdict
        blockers = "; ".join(f"{f.rule_id}: {f.message}" for f in failed.blockers)
        await self.tasks.transition(target.id, TaskStatus.DONE, reason="answered")
        await self._save_verdict(
            Verdict(
                run_id=run.id,
                scope=RuleScope.TASK,
                task_id=parent.id,
                phase=parent.phase,
                passed=True,
                summary=f"Accepted by the user despite: {blockers}"[:1000],
                model="user",
            )
        )
        await self._record(
            Decision(
                run_id=run.id,
                task_id=parent.id,
                kind=DecisionKind.ASK_USER,
                inputs={"question": target.objective, "user_task": target.id},
                choice="accept as is",
                reason="the user kept work that failed its check",
                made_by="code",
                rule="audit_question",
            )
        )
        parent = await self.tasks.get_task(parent.id)
        parent.depends_on = [d for d in parent.depends_on if d != target.id]
        await self.tasks.update_task(parent)
        await self.tasks.transition(parent.id, TaskStatus.DONE, reason="accepted by the user")
        return True

    async def _answer_audit_question(self, run: Run, target: Task, text: str) -> Reply | None:
        """A failed run audit asked the user. 'stop' stops, an approval accepts the work
        as it is, and anything else is a comment that reopens the last phase."""
        await self.tasks.transition(target.id, TaskStatus.DONE, reason="answered")
        reply = text.strip()
        if reply.lower() == "stop":
            await self.stop(run.id, "the user stopped the run after a failed audit")
            return Reply(kind="stopped", text="Stopped the run.", run_id=run.id)
        accepted = bool(APPROVE.match(reply))
        await self._record(
            Decision(
                run_id=run.id,
                task_id=target.id,
                kind=DecisionKind.ASK_USER,
                inputs={"question": target.objective, "answer": reply[:500]},
                choice="accept as is" if accepted else "revise with the user's comment",
                reason="the user answered a failed audit",
                made_by="code",
                rule="audit_question",
            )
        )
        if accepted:
            await self._run_done(run, {"accepted": True})
        else:
            await self._comment(run, target.phase, reply)
        return None

    async def _comment(self, run: Run, phase: Phase, text: str) -> None:
        """Send the user's comment to the phase's first tasks and re-run from there."""
        tasks = await self.tasks.list_tasks(run.id)
        experts = expert_tasks(tasks, phase)
        in_phase = {t.id for t in experts}
        roots = [t for t in experts if not any(d in in_phase for d in t.depends_on)]
        for t in roots:
            await self.feedback.add(run.id, self._feedback(t, FeedbackSource.USER_COMMENT, text))
            await self.reopen(run, t, origin=TaskOrigin.REVISION, reason=f"user comment: {text}")

    async def request_change(self, run_id: str, text: str, *, target: str = "prd") -> Reply:
        """Apply a change request now, from the first artifact it touches onward. User
        messages go through a confirmation first (``_propose_change``); this is the
        direct entry point once a change is agreed."""
        run = await self.tasks.get_run(run_id)
        producer = (await self._change_targets(run)).get(target)
        if producer is None:
            return Reply(
                kind="error", text=f"This run has no {target} to change yet.", run_id=run_id
            )
        await self._apply_change(run, text, producer)
        return await self.drive(run_id)

    async def _apply_change(self, run: Run, text: str, producer: Task) -> None:
        """Give the change to the producer as feedback and reopen it and its downstream;
        the run is active again, even if it had finished."""
        await self.feedback.add(
            run.id,
            self._feedback(producer, FeedbackSource.USER_COMMENT, text),
        )
        await self.reopen(
            run, producer, origin=TaskOrigin.CHANGE_REQUEST, reason=f"change request: {text}"
        )
        await self._set_status(run.id, RunStatus.ACTIVE)

    async def _change_targets(self, run: Run) -> dict[str, Task]:
        """The latest live producer of each artifact a change can start from."""
        found: dict[str, Task] = {}
        for t in await self.tasks.list_tasks(run.id):
            if (
                t.status is not TaskStatus.SUPERSEDED
                and t.output
                and t.output.kind in CHANGE_TARGETS
            ):
                found[t.output.kind] = t
        return found

    async def _changeable(self, run: Run) -> set[str]:
        """The artifact kinds a change can start from: those with a live producer, or
        with a task that is producing them now (a change to it joins that redo)."""
        return {
            t.output_kind
            for t in await self.tasks.list_tasks(run.id)
            if t.output_kind in CHANGE_TARGETS and t.status is not TaskStatus.SUPERSEDED
        }

    async def _redoing(self, run: Run, target: str) -> Task | None:
        """The task already redoing an artifact, when an earlier answer reopened it."""
        for t in reversed(await self.tasks.list_tasks(run.id)):
            if t.output_kind == target and t.status not in (
                TaskStatus.SUPERSEDED,
                TaskStatus.DONE,
            ):
                return t
        return None

    async def _fold_into_redo(self, run: Run, text: str, target: str) -> str:
        """A change to an artifact that is already being redone becomes feedback for
        that redo, so nothing extra reopens. Returns the note for the user."""
        redo = await self._redoing(run, target)
        if redo is None:
            return f"I couldn't find {plain_kind(target)} to change, so I left it as it is."
        await self.feedback.add(run.id, self._feedback(redo, FeedbackSource.USER_COMMENT, text))
        return f"Your change was added to the redo of {plain_kind(target)}, already under way."

    async def _propose_change(self, run: Run, text: str, target: str) -> str | None:
        """Ask the user to confirm a change, saying in plain words what it redoes. Nothing
        reopens until they say yes. Returns a note for the user when the artifact is
        already being redone (an earlier answer in the same message reopened it)."""
        producer = (await self._change_targets(run)).get(target)
        if producer is None:
            return await self._fold_into_redo(run, text, target)
        tasks = [
            t for t in await self.tasks.list_tasks(run.id) if t.status is not TaskStatus.SUPERSEDED
        ]
        after = downstream(tasks, producer)
        redo: list[str] = []
        for t in tasks:
            if t.id in after and t.agent is not AgentName.USER:
                name = plain_kind(t.output_kind)
                if name not in redo and t.output_kind != target:
                    redo.append(name)
        rerun = f", then re-running {', '.join(redo)}" if redo else ""
        question = (
            f"You asked: {text}. That means redoing {plain_kind(target)}{rerun}. "
            "Go ahead? (yes or no)"
        )
        was_done = (await self.tasks.get_run(run.id)).status is RunStatus.DONE
        child = await create_user_task(
            self.tasks,
            run,
            phase=producer.phase,
            question=question,
            origin=TaskOrigin.CHANGE_REQUEST,
            reason=CONFIRM_CHANGE,
        )
        await self._record(
            Decision(
                run_id=run.id,
                task_id=child.id,
                kind=DecisionKind.ASK_USER,
                inputs={"text": text, "target": target, "run_was_done": was_done},
                choice="confirm the change before redoing anything",
                reason="a change redoes finished work, so the user confirms it first",
                made_by="code",
                rule=CONFIRM_CHANGE,
            )
        )
        if was_done:
            # Reopen the conversation, not the work: the run waits on the confirmation.
            await self._set_status(run.id, RunStatus.ACTIVE)
        return None

    async def _confirm_change(self, run: Run, target: Task, text: str, notes: list[str]) -> None:
        """Yes applies the proposed change to the artifact's current producer (or to its
        redo, if something reopened it meanwhile); no drops it and redoes nothing."""
        proposal = next(
            (
                d
                for d in reversed(await self.decisions.list(run.id))
                if d.task_id == target.id and d.rule == CONFIRM_CHANGE
            ),
            None,
        )
        await self.tasks.transition(target.id, TaskStatus.DONE, reason="answered")
        if proposal is None:
            notes.append(
                "I lost the details of that change, so nothing was redone. Please ask again."
            )
            return
        agreed = bool(YES.match(text)) and not NO.match(text)
        await self._record(
            Decision(
                run_id=run.id,
                task_id=target.id,
                kind=DecisionKind.ASK_USER,
                inputs={"answer": text[:500], **proposal.inputs},
                choice="apply the change" if agreed else "drop the change",
                reason="the user answered the change confirmation",
                made_by="code",
                rule=CONFIRM_CHANGE,
            )
        )
        change, kind = str(proposal.inputs["text"]), str(proposal.inputs["target"])
        if agreed:
            producer = (await self._change_targets(run)).get(kind)
            if producer is not None:
                await self._apply_change(run, change, producer)
            else:
                notes.append(await self._fold_into_redo(run, change, kind))
        elif proposal.inputs.get("run_was_done"):
            await self._set_status(run.id, RunStatus.DONE)
