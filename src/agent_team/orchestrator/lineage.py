"""Re-running work as revisions (docs/DESIGN.md, "Updates, not rebuilds").

``reopen`` supersedes a task and everything built on it, and creates revision copies
that point back at the originals, so every attempt stays in the trace. Work the fixed
pipeline creates by itself is not copied: a new plan rebuilds the pipeline, and a
new spec rebuilds the Engineer and QA tasks from its work packages.
"""

from __future__ import annotations

from collections.abc import Iterable

from agent_team.core.artifacts import Artifact, ArtifactRef
from agent_team.core.interfaces import ArtifactStore, DecisionLog, TaskStore
from agent_team.core.models import (
    AgentName,
    Decision,
    DecisionKind,
    Phase,
    Run,
    Task,
    TaskOrigin,
    TaskStatus,
    new_id,
)


def downstream(tasks: list[Task], root: Task) -> set[str]:
    """Ids of every task that depends on ``root``, directly or not."""
    found: set[str] = set()
    frontier = {root.id}
    while frontier:
        nxt = {
            t.id
            for t in tasks
            if t.id not in found and t.id != root.id and any(d in frontier for d in t.depends_on)
        }
        found |= nxt
        frontier = nxt
    return found


#: Origins whose parent link is not a reopening: planned work points at the planner,
#: and a research request, question or check is new work that its parent asked for,
#: even when, like a Researcher's request, it has the parent's agent and output kind.
NEW_WORK = frozenset(
    {TaskOrigin.PLAN, TaskOrigin.RESEARCH_REQUEST, TaskOrigin.QUESTION, TaskOrigin.CHECK}
)


def reopens_parent(task: Task) -> bool:
    """Whether ``task`` redoes its parent's work (a retry, revision, fix or rerun), so
    the parent's output is an earlier version of its own."""
    return task.parent_task_id is not None and task.origin not in NEW_WORK


def ancestry(task: Task, tasks: list[Task]) -> list[Task]:
    """The tasks ``task`` was reopened from, oldest first, ending with ``task``: the
    chain of ``parent_task_id`` links through tasks of the same agent and output, up to
    the first task that was new work rather than a reopening."""
    by_id = {t.id: t for t in tasks}
    chain = [task]
    seen = {task.id}
    current = task
    parent = by_id.get(task.parent_task_id or "")
    while (
        parent is not None
        and reopens_parent(current)
        and parent.id not in seen
        and parent.agent is task.agent
        and parent.output_kind == task.output_kind
    ):
        chain.append(parent)
        seen.add(parent.id)
        current = parent
        parent = by_id.get(parent.parent_task_id or "")
    return chain[::-1]


async def previous_version(
    artifacts: ArtifactStore, run_id: str, artifact: Artifact, chain: Iterable[str]
) -> Artifact | None:
    """The newest earlier version of ``artifact``'s kind that one of the ``chain`` tasks
    wrote (a task and the tasks it was reopened from, see ``ancestry``). Versions are
    numbered per kind across the run, so the version just before may be a sibling's,
    such as another research question's brief."""
    ids = set(chain)
    for version in range(artifact.version - 1, 0, -1):
        try:
            older = await artifacts.load(run_id, ArtifactRef(kind=artifact.kind, version=version))
        except KeyError:
            continue
        if older.task_id in ids:
            return older
    return None


def _rebuilt_by_pipeline(root: Task, task: Task) -> bool:
    """Tasks that code recreates anyway after the reopen, so reopen mustn't copy them too."""
    if task.id == root.id:
        return False
    if root.agent is AgentName.ORCHESTRATOR and root.output_kind == "plan":
        return True  # the new plan expands into a fresh pipeline
    return root.phase is not Phase.IMPLEMENTATION and task.phase is Phase.IMPLEMENTATION


async def reopen(
    tasks: TaskStore,
    decisions: DecisionLog,
    run: Run,
    task: Task,
    *,
    origin: TaskOrigin,
    reason: str | None = None,
    finding_ids: list[str] | None = None,
) -> Task | None:
    """Re-run ``task`` as a revision, and everything built on it after it. Returns the
    revision of ``task``, or None if it was already superseded."""
    all_tasks = await tasks.list_tasks(run.id)
    current = next((t for t in all_tasks if t.id == task.id), task)
    if current.status is TaskStatus.SUPERSEDED:
        return None
    below = downstream(all_tasks, current)
    affected = [current] + [
        t for t in all_tasks if t.id in below and t.status is not TaskStatus.SUPERSEDED
    ]
    ids = {t.id for t in affected}
    # Pending questions on affected tasks go too; nothing may keep depending on them.
    affected += [
        t
        for t in all_tasks
        if t.agent is AgentName.USER and t.parent_task_id in ids and t.status is TaskStatus.WAITING
    ]
    affected_ids = {t.id for t in affected}
    new_ids = {
        t.id: new_id("task")
        for t in affected
        if t.agent is not AgentName.USER and not _rebuilt_by_pipeline(current, t)
    }
    for t in affected:
        await tasks.transition(t.id, TaskStatus.SUPERSEDED, reason=f"reopened from {current.id}")

    root: Task | None = None
    for t in affected:
        if t.id not in new_ids:
            continue
        is_root = t.id == current.id
        copy = await tasks.create_task(
            Task(
                id=new_ids[t.id],
                run_id=run.id,
                phase=t.phase,
                agent=t.agent,
                objective=t.objective,
                inputs=[r for r in t.inputs if r.kind == "plan"],
                output_kind=t.output_kind,
                depends_on=[
                    new_ids.get(d, d) for d in t.depends_on if d not in affected_ids or d in new_ids
                ],
                max_attempts=t.max_attempts,
                origin=origin if is_root else TaskOrigin.REVISION,
                parent_task_id=t.id,
                reason=reason if is_root else f"upstream {current.id} reopened",
                finding_ids=(finding_ids or []) if is_root else [],
            )
        )
        if is_root:
            root = copy
    await decisions.record(
        Decision(
            run_id=run.id,
            task_id=current.id,
            kind=DecisionKind.REOPEN_PHASE,
            inputs={"superseded": sorted(affected_ids), "new": sorted(new_ids.values())},
            choice=f"reopen {current.phase.value} from {current.output or current.id}",
            reason=reason or origin.value,
            made_by="code",
            rule="lineage",
        )
    )
    return root
