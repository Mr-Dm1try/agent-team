"""Phase gates and reviews, derived from stored records so they survive a restart.

A gate counts as passed when the newest gate verdict for its phase passed and is newer
than every expert task in the phase. Reopening work creates new tasks, so it reopens
the gate without any bookkeeping. Reviews work the same way with the user's review
task.
"""

from __future__ import annotations

from collections.abc import Iterable

from agent_team.core.interfaces import VerdictStore
from agent_team.core.models import (
    AgentName,
    Phase,
    RuleScope,
    Run,
    Task,
    TaskOrigin,
    TaskStatus,
    Verdict,
)


def expert_tasks(tasks: Iterable[Task], phase: Phase) -> list[Task]:
    """Live tasks in ``phase`` done by agents, not the user."""
    return [
        t
        for t in tasks
        if t.phase is phase
        and t.status is not TaskStatus.SUPERSEDED
        and t.agent is not AgentName.USER
    ]


def gate_passed(tasks: Iterable[Task], phase: Phase, history: list[Verdict]) -> bool:
    """The newest gate verdict in ``history`` passed and nothing in the phase is newer."""
    if not history:
        return False
    latest = max(history, key=lambda v: v.created_at)
    if not latest.passed:
        return False
    items = list(tasks)
    newest = max((t.created_at for t in expert_tasks(items, phase)), default=None)
    return newest is None or latest.created_at > newest


async def passed_gates(verdicts: VerdictStore, run_id: str, tasks: Iterable[Task]) -> set[Phase]:
    """Every phase whose gate currently counts as passed."""
    items = list(tasks)
    return {p for p in Phase if gate_passed(items, p, await verdicts.for_gate(run_id, p))}


def as_gate_verdict(verdict: Verdict, phase: Phase) -> Verdict:
    """Pin an auditor's gate verdict to its phase so ``for_gate`` finds it."""
    return verdict.model_copy(update={"scope": RuleScope.PHASE, "phase": phase, "task_id": None})


def accepted_gate(run: Run, phase: Phase, failed: Verdict | None) -> Verdict:
    """A passing gate verdict that records the user accepting a failed gate."""
    blockers = "; ".join(f"{f.rule_id}: {f.message}" for f in failed.blockers) if failed else ""
    summary = "Accepted by the user" + (f" despite: {blockers}" if blockers else "")
    return Verdict(
        run_id=run.id,
        scope=RuleScope.PHASE,
        phase=phase,
        passed=True,
        summary=summary[:1000],
        model="user",
    )


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
