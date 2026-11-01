"""Turns a passed Plan into task rows (docs/DESIGN.md, "Task tracking").

Plan keys become task ids. Each task also depends on the tasks of the previous
phase, and QA tasks on the engineer tasks, so the order holds even when the plan
leaves a dependency out.
"""

from __future__ import annotations

from agent_team.core.artifacts import ArtifactRef, Plan
from agent_team.core.models import (
    PHASE_ORDER,
    AgentName,
    Phase,
    Run,
    Task,
    TaskOrigin,
    new_id,
)


def expand_plan(
    plan: Plan,
    plan_ref: ArtifactRef,
    run: Run,
    planning_task: Task,
    *,
    max_attempts: int = 3,
) -> list[Task]:
    ids = {p.key: new_id("task") for p in plan.tasks}
    by_phase: dict[Phase, list[str]] = {p: [] for p in PHASE_ORDER}
    for p in plan.tasks:
        by_phase[Phase(p.phase)].append(ids[p.key])
    engineers = [ids[p.key] for p in plan.tasks if p.agent == AgentName.ENGINEER.value]

    tasks: list[Task] = []
    for p in plan.tasks:
        phase = Phase(p.phase)
        deps = [ids[d] for d in p.depends_on]
        idx = PHASE_ORDER.index(phase)
        for earlier in reversed(PHASE_ORDER[:idx]):
            if by_phase[earlier]:
                deps += by_phase[earlier]
                break
        if p.agent == AgentName.QA.value:
            deps += engineers
        asks = f" Covers {', '.join(p.covers_asks)}." if p.covers_asks else ""
        tasks.append(
            Task(
                id=ids[p.key],
                run_id=run.id,
                phase=phase,
                agent=AgentName(p.agent),
                objective=p.objective,
                inputs=[plan_ref],
                output_kind=p.output_kind,
                depends_on=list(dict.fromkeys(d for d in deps if d != ids[p.key])),
                max_attempts=max_attempts,
                origin=TaskOrigin.PLAN,
                parent_task_id=planning_task.id,
                reason=f"plan key {p.key}.{asks}",
            )
        )
    return tasks
