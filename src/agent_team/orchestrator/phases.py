"""The phase state machine. The order is fixed in code; the model only makes judgment
calls inside a phase (docs/DESIGN.md, "Architecture overview").

There are no phase gates: every rule runs at the end of the task that already sees its
inputs. A phase is finished once all of its tasks are done, which is plain code, so
the current phase is derived from the task store alone and a resumed run lands in the
same place. Reviews work the same way, with the user's review task.
"""

from __future__ import annotations

from collections.abc import Iterable

from agent_team.core.models import (
    PHASE_ORDER,
    AgentName,
    Phase,
    RequestKind,
    Task,
    TaskOrigin,
    TaskStatus,
)

PHASES_BY_KIND: dict[RequestKind, tuple[Phase, ...]] = {
    RequestKind.BUILD: PHASE_ORDER,
    RequestKind.CHANGE: PHASE_ORDER,
    RequestKind.RESEARCH_QUESTION: (Phase.PLANNING,),
    RequestKind.RUN_QUESTION: (),
}

_CLOSED = frozenset({TaskStatus.DONE, TaskStatus.SUPERSEDED})


def expert_tasks(tasks: Iterable[Task], phase: Phase) -> list[Task]:
    """Live tasks in ``phase`` done by agents, not the user."""
    return [
        t
        for t in tasks
        if t.phase is phase
        and t.status is not TaskStatus.SUPERSEDED
        and t.agent is not AgentName.USER
    ]


def reviewed(tasks: Iterable[Task], phase: Phase) -> bool:
    """The user approved or commented on this phase's current output."""
    items = list(tasks)
    newest = max((t.created_at for t in expert_tasks(items, phase)), default=None)
    return any(
        t.agent is AgentName.USER
        and t.origin is TaskOrigin.REVIEW
        and t.phase is phase
        and t.status is TaskStatus.DONE
        and (newest is None or t.created_at > newest)
        for t in items
    )


class PhaseMachine:
    """The phases a kind of run goes through, and where a run currently is."""

    def __init__(self, kind: RequestKind) -> None:
        """The phases come from the request kind; a research question stops after planning."""
        self.kind = kind
        self.phases = PHASES_BY_KIND[kind]

    def next(self, phase: Phase) -> Phase | None:
        """The phase after ``phase`` for this kind of run, or None after the last."""
        i = self.phases.index(phase)
        return self.phases[i + 1] if i + 1 < len(self.phases) else None

    def open_tasks(self, tasks: Iterable[Task], phase: Phase) -> list[Task]:
        """Unfinished agent and user tasks in ``phase``."""
        return [t for t in tasks if t.phase is phase and t.status not in _CLOSED]

    def finished(self, tasks: Iterable[Task], phase: Phase) -> bool:
        """``phase`` has work and all of it is done."""
        items = list(tasks)
        return bool(expert_tasks(items, phase)) and not self.open_tasks(items, phase)

    def current(self, tasks: Iterable[Task]) -> Phase | None:
        """The earliest phase that isn't finished (open work, or none created yet);
        None when every phase is finished."""
        items = list(tasks)
        for phase in self.phases:
            if not self.finished(items, phase):
                return phase
        return None
