from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.fakes import (
    FakeArtifactStore,
    FakeBus,
    FakeCheckpoints,
    FakeDecisionLog,
    FakeFeedback,
    FakeLLM,
    FakeTaskStore,
    FakeVerdicts,
    PassAuditor,
)
from agents.scenario import team_script

from agent_team.agents import default_agents
from agent_team.core.interfaces import VerdictPolicy
from agent_team.core.models import AgentName, ReviewMode
from agent_team.orchestrator import Dispatcher, Orchestrator, OrchestratorConfig
from agent_team.tools import file_tools


@dataclass
class Team:
    orch: Orchestrator
    dispatcher: Dispatcher
    tasks: FakeTaskStore
    artifacts: FakeArtifactStore
    checkpoints: FakeCheckpoints
    decisions: FakeDecisionLog
    feedback: FakeFeedback
    verdicts: FakeVerdicts
    bus: FakeBus
    llm: FakeLLM
    auditor: PassAuditor


def make_team(
    tmp_path: Path,
    *,
    kind: str = "build",
    script: dict[AgentName, Any] | None = None,
    review_mode: ReviewMode = ReviewMode.AUTO_ASSUMPTIONS,
    policy: VerdictPolicy | None = None,
) -> Team:
    tasks, artifacts = FakeTaskStore(), FakeArtifactStore()
    checkpoints, decisions = FakeCheckpoints(), FakeDecisionLog()
    feedback, verdicts, bus = FakeFeedback(), FakeVerdicts(), FakeBus()
    llm = FakeLLM(script if script is not None else team_script(kind))
    auditor = PassAuditor()
    dispatcher = Dispatcher(
        tasks=tasks,
        artifacts=artifacts,
        checkpoints=checkpoints,
        bus=bus,
        llm=llm,
        agents=default_agents(),
        tools=file_tools(),
        feedback=feedback,
        verdicts=verdicts,
        timeout_s=5,
    )
    orch = Orchestrator(
        tasks=tasks,
        artifacts=artifacts,
        checkpoints=checkpoints,
        decisions=decisions,
        feedback=feedback,
        verdicts=verdicts,
        bus=bus,
        llm=llm,
        auditor=auditor,
        dispatcher=dispatcher,
        config=OrchestratorConfig(runs_dir=tmp_path / "runs", review_mode=review_mode),
        policy=policy,
    )
    return Team(
        orch,
        dispatcher,
        tasks,
        artifacts,
        checkpoints,
        decisions,
        feedback,
        verdicts,
        bus,
        llm,
        auditor,
    )
