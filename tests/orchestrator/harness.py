"""The team as tests build it: the composition root over fake stores, a scripted model
and an auditor that passes unless told otherwise."""

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
from sandbox.fakes import FakeRunner

from agent_team.app import App, Stores, build_app
from agent_team.core.config import load_config
from agent_team.core.interfaces import VerdictPolicy
from agent_team.core.models import AgentName, ReviewMode
from agent_team.orchestrator import Dispatcher, Orchestrator
from agent_team.orchestrator.policy import SimplePolicy


@dataclass
class Team:
    """The built app with its fakes, typed, so tests can script and inspect them."""

    app: App
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


async def make_team(
    tmp_path: Path,
    *,
    kind: str = "build",
    script: dict[AgentName, Any] | None = None,
    review_mode: ReviewMode = ReviewMode.AUTO_ASSUMPTIONS,
    policy: VerdictPolicy | None = None,
    code_policy: bool = False,
) -> Team:
    """Build the app over fakes. ``policy`` decides on verdicts; without one,
    ``code_policy`` keeps the audit package's policy and otherwise the orchestrator's
    simple fallback decides."""
    tasks, artifacts = FakeTaskStore(), FakeArtifactStore()
    checkpoints, decisions = FakeCheckpoints(), FakeDecisionLog()
    feedback, verdicts, bus = FakeFeedback(), FakeVerdicts(), FakeBus()
    llm = FakeLLM(script if script is not None else team_script(kind))
    auditor = PassAuditor()
    config = load_config(
        overrides={
            "paths": {"runs_dir": str(tmp_path / "runs"), "db": str(tmp_path / "team.db")},
            "review": review_mode.value,
            "limits": {"task_timeout_s": 5},
            "hooks": [],
        }
    )
    app = await build_app(
        config,
        stores=Stores(tasks, artifacts, checkpoints, decisions, feedback, verdicts),
        bus=bus,
        llm=llm,
        auditor=auditor,
        policy=policy or (None if code_policy else SimplePolicy()),
        runner=FakeRunner(),
    )
    return Team(
        app,
        app.orchestrator,
        app.dispatcher,
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
