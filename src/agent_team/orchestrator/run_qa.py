"""Answers questions about a run from the task store, verdicts, decisions and trace,
using the orchestrator's read-only control tools."""

from __future__ import annotations

from pydantic import Field

from agent_team.agents.loop import ToolLoop
from agent_team.agents.prompts import RUN_QA
from agent_team.core.base import Contract
from agent_team.core.interfaces import (
    ArtifactStore,
    CheckpointStore,
    DecisionLog,
    EventBus,
    TaskStore,
    ToolContext,
    VerdictStore,
)
from agent_team.core.models import AgentName, Phase, Run, Task, new_id
from agent_team.llm.types import LLMClient, Message
from agent_team.orchestrator.control import read_tools

_FORMAT = '\nWhen done, reply with only JSON: {"answer": "...", "cited": ["task_..."]}'


class RunAnswer(Contract):
    """The model's answer and the ids it relied on."""

    answer: str
    cited: list[str] = Field(default_factory=list)


async def answer_run_question(
    run: Run,
    question: str,
    *,
    tasks: TaskStore,
    artifacts: ArtifactStore,
    decisions: DecisionLog,
    verdicts: VerdictStore,
    checkpoints: CheckpointStore,
    bus: EventBus,
    llm: LLMClient,
) -> tuple[str | None, str | None]:
    """Returns (answer text with citations, None) or (None, why it failed)."""
    probe = Task(
        id=new_id("ask"),
        run_id=run.id,
        phase=Phase.PLANNING,
        agent=AgentName.ORCHESTRATOR,
        objective=question,
        output_kind="run_answer",
    )
    loop: ToolLoop[RunAnswer] = ToolLoop(
        agent=AgentName.ORCHESTRATOR,
        system=RUN_QA + _FORMAT,
        output_model=RunAnswer,
        llm=llm,
        bus=bus,
        checkpoints=checkpoints,
        tools=read_tools(tasks, artifacts, decisions, verdicts),
        tool_ctx=ToolContext(run=run, task=probe, bus=bus),
        run=run,
        task_id=probe.id,
        attempt=1,
    )
    outcome = await loop.run_loop(
        Message.user(f"Run {run.id}: {run.title} ({run.status.value}).\nQuestion: {question}")
    )
    if outcome.output is None:
        return None, str(outcome.error)
    cited = f"\n(from {', '.join(outcome.output.cited)})" if outcome.output.cited else ""
    return outcome.output.answer + cited, None
