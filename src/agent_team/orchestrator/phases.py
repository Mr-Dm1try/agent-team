"""The phase state machine. The order and the gates are fixed in code; the model only
makes judgment calls inside a phase (docs/DESIGN.md, "Architecture overview").

The current phase is derived from the task store, never kept only in memory, so a
resumed run lands in the same place.
"""

from __future__ import annotations

from collections.abc import Iterable

from agent_team.core.models import PHASE_ORDER, AgentName, Phase, RequestKind, Task, TaskStatus

PHASES_BY_KIND: dict[RequestKind, tuple[Phase, ...]] = {
    RequestKind.BUILD: PHASE_ORDER,
    RequestKind.CHANGE: PHASE_ORDER,
    RequestKind.RESEARCH_QUESTION: (Phase.PLANNING,),
    RequestKind.RUN_QUESTION: (),
}

_CLOSED = frozenset({TaskStatus.DONE, TaskStatus.SUPERSEDED})


def live(tasks: Iterable[Task]) -> list[Task]:
    return [t for t in tasks if t.status is not TaskStatus.SUPERSEDED]


class PhaseMachine:
    def __init__(self, kind: RequestKind) -> None:
        self.kind = kind
        self.phases = PHASES_BY_KIND[kind]

    def index(self, phase: Phase) -> int:
        return self.phases.index(phase)

    def next(self, phase: Phase) -> Phase | None:
        i = self.index(phase)
        return self.phases[i + 1] if i + 1 < len(self.phases) else None

    def open_tasks(self, tasks: Iterable[Task], phase: Phase) -> list[Task]:
        return [t for t in tasks if t.phase is phase and t.status not in _CLOSED]

    def current(self, tasks: Iterable[Task], gates_passed: set[Phase]) -> Phase | None:
        """The earliest phase with open work or an unpassed gate; None when finished."""
        items = list(tasks)
        for phase in self.phases:
            if self.open_tasks(items, phase) or phase not in gates_passed:
                return phase
        return None

    def can_enter(self, phase: Phase, gates_passed: set[Phase]) -> bool:
        """A phase starts only after every earlier phase's gate passed."""
        return all(p in gates_passed for p in self.phases[: self.index(phase)])

    def reopen(self, phase: Phase, gates_passed: set[Phase]) -> None:
        """Reopening a phase also reopens every gate after it."""
        for p in self.phases[self.index(phase) :]:
            gates_passed.discard(p)

    @staticmethod
    def expert_tasks(tasks: Iterable[Task], phase: Phase) -> list[Task]:
        return [t for t in live(tasks) if t.phase is phase and t.agent is not AgentName.USER]
