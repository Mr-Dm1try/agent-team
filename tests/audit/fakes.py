"""Fakes for the stores and the LLM that other steps provide, plus builders."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from agent_team.audit.evidence import ref_of
from agent_team.audit.toollog import ToolCall
from agent_team.core.artifacts import Artifact, ArtifactRef
from agent_team.core.models import (
    AgentName,
    Decision,
    Phase,
    Run,
    Task,
    TaskStatus,
    Usage,
    Verdict,
)
from agent_team.llm.types import LLMRequest, LLMResponse, StreamChunk, TextBlock


class FakeLLM:
    """Replays scripted replies; a reply may be an exception to raise."""

    def __init__(self, replies: list[Any] | None = None) -> None:
        self.replies: list[Any] = list(replies or [])
        self.requests: list[LLMRequest] = []

    def push(self, *replies: Any) -> None:
        self.replies.extend(replies)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(
            content=[TextBlock(text=text)],
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
            provider="fake",
            model="fake-auditor",
        )

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError

    def context_budget(self, agent: AgentName) -> int:
        return 90_000


class FakeTaskStore:
    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self.tasks: dict[str, Task] = {}

    async def create_run(self, run: Run) -> Run:
        self.runs[run.id] = run
        return run

    async def get_run(self, run_id: str) -> Run:
        return self.runs[run_id]

    async def update_run(self, run: Run) -> Run:
        self.runs[run.id] = run
        return run

    async def list_runs(self) -> list[Run]:
        return list(self.runs.values())

    async def create_task(self, task: Task) -> Task:
        self.tasks[task.id] = task
        return task

    async def get_task(self, task_id: str) -> Task:
        return self.tasks[task_id]

    async def update_task(self, task: Task) -> Task:
        self.tasks[task.id] = task
        return task

    async def transition(self, task_id: str, to: TaskStatus, *, reason: str) -> Task:
        task = self.tasks[task_id].model_copy(update={"status": to})
        self.tasks[task_id] = task
        return task

    async def list_tasks(
        self, run_id: str, *, status: TaskStatus | None = None, phase: Phase | None = None
    ) -> list[Task]:
        return [
            t
            for t in self.tasks.values()
            if t.run_id == run_id
            and (status is None or t.status is status)
            and (phase is None or t.phase is phase)
        ]

    async def ready_tasks(self, run_id: str) -> list[Task]:
        return []


class FakeArtifactStore:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str, int], Artifact] = {}

    async def save(self, artifact: Artifact) -> ArtifactRef:
        self.items[(artifact.run_id, artifact.kind, artifact.version)] = artifact
        return ref_of(artifact)

    async def load(self, run_id: str, ref: ArtifactRef) -> Artifact:
        return self.items[(run_id, ref.kind, ref.version)]

    async def latest(self, run_id: str, kind: str) -> Artifact | None:
        found = [a for (r, k, _), a in self.items.items() if r == run_id and k == kind]
        return max(found, key=lambda a: a.version) if found else None

    async def dependents(self, run_id: str, ref: ArtifactRef) -> list[ArtifactRef]:
        return []


class FakeVerdictStore:
    def __init__(self) -> None:
        self.items: list[Verdict] = []

    async def save(self, verdict: Verdict) -> None:
        self.items.append(verdict)

    async def latest_for_task(self, task_id: str) -> Verdict | None:
        found = [v for v in self.items if v.task_id == task_id]
        return found[-1] if found else None

    async def for_task(self, task_id: str) -> list[Verdict]:
        return [v for v in self.items if v.task_id == task_id]


class FakeDecisionLog:
    def __init__(self) -> None:
        self.items: list[Decision] = []

    async def record(self, decision: Decision) -> None:
        self.items.append(decision)

    async def list(self, run_id: str) -> list[Decision]:
        return [d for d in self.items if d.run_id == run_id]


# --- builders ------------------------------------------------------------------------


def make_run(workspace: Path, **kw: Any) -> Run:
    workspace.mkdir(parents=True, exist_ok=True)
    return Run(title="t", request="Build a books API", workspace=str(workspace), **kw)


def make_task(run: Run, agent: AgentName, phase: Phase, output_kind: str, **kw: Any) -> Task:
    return Task(
        run_id=run.id, phase=phase, agent=agent, objective="do it", output_kind=output_kind, **kw
    )


def call(run: Run, task: Task | None, tool: str, result: str = "", **args: Any) -> ToolCall:
    return ToolCall(
        run_id=run.id,
        task_id=task.id if task else None,
        agent=task.agent.value if task else None,
        tool=tool,
        args=args,
        result=result,
    )
