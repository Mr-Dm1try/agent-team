"""The phase state machine. The order and the gates are fixed in code; the model only
makes judgment calls inside a phase (docs/DESIGN.md, "Architecture overview").

The current phase is derived from the task store and the gate verdicts, so a resumed
run lands in the same place.
"""

from __future__ import annotations

from collections.abc import Iterable

from agent_team.core.models import PHASE_ORDER, Phase, RequestKind, Task, TaskStatus

PHASES_BY_KIND: dict[RequestKind, tuple[Phase, ...]] = {
    RequestKind.BUILD: PHASE_ORDER,
    RequestKind.CHANGE: PHASE_ORDER,
    RequestKind.RESEARCH_QUESTION: (Phase.PLANNING,),
    RequestKind.RUN_QUESTION: (),
}

_CLOSED = frozenset({TaskStatus.DONE, TaskStatus.SUPERSEDED})


class PhaseMachine:
    """The phases a kind of run goes through, and where a run currently is."""

    def __init__(self, kind: RequestKind) -> None:
        self.kind = kind
        self.phases = PHASES_BY_KIND[kind]

    def next(self, phase: Phase) -> Phase | None:
        """The phase after ``phase`` for this kind of run, or None after the last."""
        i = self.phases.index(phase)
        return self.phases[i + 1] if i + 1 < len(self.phases) else None

    def open_tasks(self, tasks: Iterable[Task], phase: Phase) -> list[Task]:
        """Unfinished agent and user tasks in ``phase``."""
        return [t for t in tasks if t.phase is phase and t.status not in _CLOSED]

    def current(self, tasks: Iterable[Task], passed: set[Phase]) -> Phase | None:
        """The earliest phase with open work or a gate not in ``passed``; None when
        every phase is finished."""
        items = list(tasks)
        for phase in self.phases:
            if self.open_tasks(items, phase) or phase not in passed:
                return phase
        return None
