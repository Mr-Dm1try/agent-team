"""What the orchestrator says back to the user after each message."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agent_team.core.artifacts import Artifact
from agent_team.core.interfaces import ArtifactStore
from agent_team.core.models import Run, Task, TaskOrigin, TaskStatus


@dataclass
class PendingQuestion:
    """A question or review waiting on the user, by the user task that holds it."""

    task_id: str
    question: str
    origin: TaskOrigin


@dataclass
class Reply:
    """The orchestrator's answer to one message: the run's state and anything pending."""

    kind: Literal["done", "waiting", "answer", "stopped", "error"]
    text: str
    run_id: str | None = None
    pending: list[PendingQuestion] = field(default_factory=list)
    artifact: Artifact | None = None


def waiting_reply(run: Run, pending: list[Task]) -> Reply:
    """The reply listing every question and review waiting on the user."""
    items = [PendingQuestion(task_id=t.id, question=t.objective, origin=t.origin) for t in pending]
    text = "I need your input:\n" + "\n".join(f"- {p.question}" for p in items)
    return Reply(kind="waiting", text=text, run_id=run.id, pending=items)


async def done_reply(run: Run, tasks: list[Task], artifacts: ArtifactStore) -> Reply:
    """Summary of a finished run: every live artifact, the final answer or QA report,
    and every assumption agents recorded instead of asking."""
    outputs = [t.output for t in tasks if t.status is not TaskStatus.SUPERSEDED and t.output]
    lines = [f"Run {run.id} ({run.title}) is done."]
    final: Artifact | None = None
    assumptions: list[str] = []
    for ref in outputs:
        art = await artifacts.load(run.id, ref)
        lines.append(f"- {ref}: {art.summary}")
        assumptions += [f"{ref} {q.id}: {q.assumption}" for q in art.questions if q.assumption]
        if art.kind in ("research_answer", "qa_report"):
            final = art
    if final is not None and final.kind == "research_answer":
        lines.insert(1, final.answer)
    if assumptions:
        lines.append("Assumptions to check:")
        lines += [f"- {a}" for a in assumptions]
    return Reply(kind="done", text="\n".join(lines), run_id=run.id, artifact=final)
