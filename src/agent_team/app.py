"""The composition root: every store, hook and service built from one Config.

The CLI and the end-to-end tests both start here. Any collaborator can be swapped
(scripted providers, in-memory stores, a scripted auditor) and the rest is still built
the same way, so no caller keeps its own copy of the wiring.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from agent_team.agents import default_agents
from agent_team.audit.auditor import LLMAuditor
from agent_team.audit.catalog import load_catalog
from agent_team.audit.policy import CodeVerdictPolicy, PolicyLimits
from agent_team.audit.precheck import make_precheck_hook
from agent_team.audit.rule_store import SqliteRuleStore
from agent_team.audit.toollog import EventFileToolCallLog
from agent_team.core.config import Config, load_config
from agent_team.core.control import KillSwitch
from agent_team.core.db import Database
from agent_team.core.events import InMemoryEventBus, load_hooks
from agent_team.core.interfaces import (
    Agent,
    ArtifactStore,
    Auditor,
    CheckpointStore,
    DecisionLog,
    EventBus,
    FeedbackStore,
    TaskStore,
    Tool,
    VerdictPolicy,
    VerdictStore,
)
from agent_team.core.models import AgentName, EventName
from agent_team.core.stores import (
    SqliteArtifactStore,
    SqliteCheckpointStore,
    SqliteDecisionLog,
    SqliteFeedbackStore,
    SqliteTaskStore,
    SqliteVerdictStore,
)
from agent_team.core.tracing import JsonlTracer, Tracer
from agent_team.llm.chaos import Chaos
from agent_team.llm.factory import build_gateway
from agent_team.llm.types import LLMClient, Provider
from agent_team.orchestrator import Dispatcher, Orchestrator, OrchestratorConfig
from agent_team.tools import file_tools


@dataclass
class Stores:
    """Everything a run keeps: runs and tasks, artifacts, checkpoints, decisions,
    feedback and verdicts."""

    tasks: TaskStore
    artifacts: ArtifactStore
    checkpoints: CheckpointStore
    decisions: DecisionLog
    feedback: FeedbackStore
    verdicts: VerdictStore


def sqlite_stores(db: Database, runs_dir: Path, bus: EventBus) -> Stores:
    """The SQLite stores over one database; files and blobs go under ``runs_dir``."""
    return Stores(
        tasks=SqliteTaskStore(db, bus),
        artifacts=SqliteArtifactStore(db, runs_dir),
        checkpoints=SqliteCheckpointStore(db, runs_dir),
        decisions=SqliteDecisionLog(db, bus),
        feedback=SqliteFeedbackStore(db),
        verdicts=SqliteVerdictStore(db),
    )


@dataclass
class App:
    """A built team: the orchestrator and everything behind it."""

    config: Config
    bus: EventBus
    stores: Stores
    llm: LLMClient
    auditor: Auditor
    kill_switch: KillSwitch
    dispatcher: Dispatcher
    orchestrator: Orchestrator
    _closers: list[Callable[[], None]] = field(default_factory=list)

    @property
    def runs_dir(self) -> Path:
        """Where each run's workspace, logs and artifacts live."""
        return Path(self.config.paths.runs_dir)

    def close(self) -> None:
        """Close the database connections this app opened."""
        while self._closers:
            self._closers.pop()()


def _make_dirs(config: Config) -> Path:
    """Create the runs folder and the database's folder; return the runs folder."""
    runs_dir = Path(config.paths.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    Path(config.paths.db).parent.mkdir(parents=True, exist_ok=True)
    return runs_dir


async def build_app(
    config: Config | None = None,
    *,
    stores: Stores | None = None,
    bus: EventBus | None = None,
    providers: dict[str, Provider] | None = None,
    llm: LLMClient | None = None,
    auditor: Auditor | None = None,
    agents: Mapping[AgentName, Agent] | None = None,
    tools: dict[str, Tool] | None = None,
    policy: VerdictPolicy | None = None,
) -> App:
    """Build the team from ``config`` (the default config file when None). Each keyword
    replaces one collaborator; the rest is built from the config: SQLite stores in WAL
    mode, the event bus with the JSONL log and the configured hooks, the gateway over
    the configured (or given) providers, the auditor over the seeded rule catalog with
    its pre-check on every tool call, the agents, the dispatcher and the orchestrator."""
    config = config or load_config()
    runs_dir = _make_dirs(config)
    closers: list[Callable[[], None]] = []
    bus = bus if bus is not None else InMemoryEventBus()
    JsonlTracer(runs_dir).attach(bus)
    load_hooks(config.hooks, bus)
    if stores is None:
        db = Database(config.paths.db)
        closers.append(db.close)
        stores = sqlite_stores(db, runs_dir, bus)
    kill_switch = KillSwitch(runs_dir)
    if llm is None:
        llm = build_gateway(
            config,
            providers=providers,
            bus=bus,
            tracer=Tracer(bus),
            kill_switch=kill_switch,
            chaos=Chaos(config.chaos),
            runs=stores.tasks,
        )
    if auditor is None:
        rules = SqliteRuleStore(config.paths.db)
        closers.append(rules.close)
        await rules.seed(load_catalog())
        auditor = LLMAuditor(
            llm=llm,
            rules=rules,
            tasks=stores.tasks,
            artifacts=stores.artifacts,
            verdicts=stores.verdicts,
            decisions=stores.decisions,
            tool_log=EventFileToolCallLog(runs_dir),
        )
    bus.on_before(EventName.TOOL_CALL_BEFORE, make_precheck_hook(auditor, stores.tasks))
    dispatcher = Dispatcher(
        tasks=stores.tasks,
        artifacts=stores.artifacts,
        checkpoints=stores.checkpoints,
        bus=bus,
        llm=llm,
        agents=agents if agents is not None else default_agents(),
        tools=tools if tools is not None else file_tools(),
        feedback=stores.feedback,
        verdicts=stores.verdicts,
        timeout_s=config.limits.task_timeout_s,
        auditor=auditor,
    )
    if policy is None:
        stronger = config.stronger_model
        limits = PolicyLimits(
            max_upstream_reruns=config.limits.max_upstream_reruns,
            stronger_model=stronger.model if stronger else PolicyLimits.stronger_model,
        )
        policy = CodeVerdictPolicy(
            tasks=stores.tasks,
            artifacts=stores.artifacts,
            decisions=stores.decisions,
            verdicts=stores.verdicts,
            limits=limits,
        )
    orchestrator = Orchestrator(
        tasks=stores.tasks,
        artifacts=stores.artifacts,
        checkpoints=stores.checkpoints,
        decisions=stores.decisions,
        feedback=stores.feedback,
        verdicts=stores.verdicts,
        bus=bus,
        llm=llm,
        dispatcher=dispatcher,
        config=OrchestratorConfig(
            runs_dir=runs_dir,
            review_mode=config.review,
            max_attempts=config.limits.max_attempts,
            max_check_attempts=config.limits.max_check_attempts,
            max_upstream_reruns=config.limits.max_upstream_reruns,
        ),
        policy=policy,
    )
    return App(config, bus, stores, llm, auditor, kill_switch, dispatcher, orchestrator, closers)
