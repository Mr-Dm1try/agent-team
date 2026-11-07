"""Config loading. ``config/default.yaml`` holds routing, budgets, limits and review
mode; ``AGENT_TEAM_CONFIG`` or an explicit path overrides it, and ``overrides`` are
deep-merged on top (used by CLI flags such as ``--chaos``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_team.core.models import AgentName, Effort, MemoryKind, ReviewMode
from agent_team.llm.tokens import CHARS_PER_TOKEN

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
ENV_VAR = "AGENT_TEAM_CONFIG"


class _Cfg(BaseModel):
    """Base for config sections; unknown keys are an error, so typos fail at load."""

    model_config = ConfigDict(extra="forbid")


class Route(_Cfg):
    """A provider and model, with an optional prompt-token budget and effort. Effort
    sets how hard the model reasons: Claude's effort, or Ollama's think level for
    the models listed in ``providers.ollama.reasoning_models``."""

    provider: str
    model: str
    context_budget: int | None = None
    effort: Effort | None = None


class DefaultRoute(Route):
    """The default route; always has a context budget."""

    context_budget: int = 90_000


class BudgetConfig(_Cfg):
    """Per-run caps on cost, time and tokens; None means no cap."""

    max_usd_per_run: float | None = None
    max_minutes_per_run: float | None = None
    max_tokens_per_run: int | None = None


class LimitsConfig(_Cfg):
    """Attempt, rerun and concurrency limits."""

    max_attempts: int = 3
    max_upstream_reruns: int = 2
    max_concurrent_calls: int = 4
    #: Attempts of each check (auditor task) before it fails closed.
    max_check_attempts: int = 3
    #: Wall-clock limit on one attempt of a task or a check.
    task_timeout_s: float = 600.0
    #: Research tasks agents may ask for in one run, on top of the planned ones.
    max_research_tasks: int = Field(default=6, ge=0)
    #: How deep requests may chain: at 1, a task made from a request can't ask for more.
    max_research_depth: int = Field(default=1, ge=0)


class RetryConfig(_Cfg):
    #: Waits before each retry of a transient error on the same provider.
    """Backoff for transient model errors."""

    delays_s: list[float] = Field(default_factory=lambda: [1.0, 2.0, 4.0])
    #: How long a provider that refused a connection is skipped before it is tried again.
    provider_down_s: float = 60.0


class ProviderLimit(_Cfg):
    """Rate limits for one provider."""

    max_concurrent: int | None = None
    requests_per_minute: int | None = None


class Price(_Cfg):
    """USD per million input and output tokens."""

    input_per_mtok: float
    output_per_mtok: float


class AnthropicConfig(_Cfg):
    """Anthropic provider settings."""

    timeout_s: float = 120.0
    #: Calls that may write more than this many tokens are streamed, so a long reply
    #: isn't cut by ``timeout_s``: on a stream the timeout is per read, not per call.
    stream_above_tokens: int = 4096
    #: Sent as ``thinking`` on every call; adaptive lets the model decide how much to
    #: think, and the route's effort sets how hard it tries.
    thinking: dict[str, Any] = Field(default_factory=lambda: {"type": "adaptive"})
    #: What the API does with a thinking block whose earlier history was edited
    #: (compaction, trimmed tool results): "drop_block" drops it and the call goes on,
    #: "error" fails the call with a 400. None sends neither and takes the API default,
    #: which is a 400 on accounts where the check is enforced.
    on_history_edit: Literal["drop_block", "error"] | None = "drop_block"

    @field_validator("thinking")
    @classmethod
    def _one_setting(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Reject the old per-model map here, at load, rather than as a 400 per call."""
        if not isinstance(value.get("type"), str):
            raise ValueError("thinking must be one setting with a type, e.g. {type: adaptive}")
        return value


class OllamaConfig(_Cfg):
    """Ollama provider settings."""

    #: The server; a trailing OpenAI-compatible ``/v1`` is ignored.
    base_url: str = "http://localhost:11434"
    #: The window asked for. Ollama's native API loads the model with it; a model whose
    #: own limit, or loaded context, is smaller gets the smaller window.
    num_ctx: int = 32_768
    timeout_s: float = 300.0
    #: The slowest generation speed planned for. A call's timeout grows past
    #: ``timeout_s`` when its ``max_tokens`` would take longer than that at this speed.
    min_tokens_per_s: float = 20.0
    #: Models (by name before the ":" tag) that take a ``think`` level. Ollama
    #: rejects it for models that can't think, so others get no effort at all.
    reasoning_models: list[str] = Field(default_factory=lambda: ["gpt-oss"])
    #: Least room left for the reply. A route's prompt budget plus this must fit
    #: ``num_ctx``, and a call whose prompt leaves less is refused rather than sent. A
    #: thinking model such as gpt-oss reasons before it writes the JSON, and both count
    #: against the reply, so a smaller room ends in "the reply hit the token limit".
    min_output_tokens: int = 4096
    #: Characters per token when estimating a prompt. Low on purpose: running over
    #: the window is silent truncation, while a high estimate only costs a fallback.
    chars_per_token: int = 3
    #: Agent tasks that run at once on Ollama. A local model answers one call at a
    #: time, so a task beyond this waits for its turn before it starts, instead of
    #: spending its timeout waiting for the model.
    max_concurrent_tasks: int = Field(default=1, ge=1)


class ProvidersConfig(_Cfg):
    """Settings for each provider."""

    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)


class ChaosSettings(_Cfg):
    """Fault injection settings; see ``agent_team.llm.chaos``."""

    rate: float = 0.0
    faults: list[str] = Field(default_factory=list)
    seed: int | None = None
    script: list[str] = Field(default_factory=list)


class SandboxConfig(_Cfg):
    """The Docker sandbox that runs model-written code, and its limits."""

    #: Off means no run_command or install_deps tools; R6 then fails every build.
    enabled: bool = True
    image: str = "python:3.12-slim"
    cpus: float = 1.0
    memory: str = "1g"
    pids_limit: int = 256
    #: Default and longest wall-clock time of one run_command call.
    timeout_s: float = 120.0
    max_timeout_s: float = 600.0
    #: Wall-clock time of one install_deps call, the only step with the network on.
    install_timeout_s: float = 300.0
    #: Characters of stdout and of stderr kept from the end of each command's output.
    output_tail_chars: int = 4000
    #: Commit the workspace as each Engineer version passes its check.
    commit_versions: bool = True


class WebConfig(_Cfg):
    """Limits for the Researcher's web_search and web_fetch tools."""

    enabled: bool = True
    timeout_s: float = 20.0
    #: Bytes read from one response before the rest is dropped.
    max_bytes: int = 2_000_000
    #: Characters of readable text one web_fetch call returns; offset reads on. Never
    #: more than fits inline in the tool loop (tools/web.py, PAGE_CHARS).
    max_chars: int = 10_000
    max_results: int = 8
    max_redirects: int = 5
    #: web_search and web_fetch calls one attempt may make, together, before the tools
    #: refuse and tell the agent to write its output with what it has.
    max_calls_per_attempt: int = Field(default=30, ge=1)
    user_agent: str = "agent-team/0.1 (research tool)"


class EmbedderConfig(_Cfg):
    """The embedding model for memory: nomic-embed-text through Ollama's ``/api/embed``,
    called once when a memory is saved and once per search."""

    model: str = "nomic-embed-text"
    #: None uses ``providers.ollama.base_url``.
    base_url: str | None = None
    #: The most tokens one text may take; the model's own window from ``/api/show``
    #: wins when smaller. Longer texts are cut at a safe character count first, so
    #: Ollama never truncates them itself.
    max_input_tokens: int = 2_048
    timeout_s: float = 10.0
    #: After a failed call, search uses BM25 alone for this long before trying again.
    retry_after_s: float = 60.0
    #: nomic-embed-text wants a task prefix on each text.
    query_prefix: str = "search_query: "
    document_prefix: str = "search_document: "


class MemoryWeights(_Cfg):
    """How a memory's score is made: meaning (cosine) and keyword (BM25) match, how
    recent it is and how often it helped."""

    cosine: float = Field(default=0.5, ge=0)
    bm25: float = Field(default=0.3, ge=0)
    recency: float = Field(default=0.1, ge=0)
    usefulness: float = Field(default=0.1, ge=0)

    @model_validator(mode="after")
    def _some_text_weight(self) -> MemoryWeights:
        """Text match must count, or every memory would match every query."""
        if self.cosine + self.bm25 <= 0:
            raise ValueError("memory.weights: cosine + bm25 must be above 0")
        return self


def _default_memory_kinds() -> dict[AgentName, list[MemoryKind]]:
    """The memory kinds each agent is given before its tasks."""
    k = MemoryKind
    return {
        AgentName.ORCHESTRATOR: [k.RUN_SUMMARY, k.PREFERENCE, k.LESSON],
        AgentName.RESEARCHER: [k.RESEARCH_FACT, k.PREFERENCE, k.LESSON],
        AgentName.PRODUCT: [k.PREFERENCE, k.DECISION, k.RESEARCH_FACT, k.RUN_SUMMARY],
        AgentName.ARCHITECT: [k.DECISION, k.PREFERENCE, k.LESSON, k.RESEARCH_FACT],
        AgentName.ENGINEER: [k.LESSON, k.DECISION, k.PREFERENCE],
        AgentName.QA: [k.LESSON, k.PREFERENCE],
    }


class MemoryConfig(_Cfg):
    """Long-term memory (docs/DESIGN.md, "Memory"): what each agent is given before a
    task, how matches are ranked, and when two memories count as one."""

    enabled: bool = True
    #: Caps on the "Relevant memory" block of one task.
    max_items: int = Field(default=10, ge=0)
    max_tokens: int = Field(default=500, ge=0)
    #: Text match (0 to 1) below which a memory is too weak to give.
    min_relevance: float = Field(default=0.3, ge=0, le=1)
    #: Text match a decision or research fact from another run needs to be given,
    #: counting words in common rather than BM25 (see memory/rank.py).
    strong_match: float = Field(default=0.6, ge=0, le=1)
    #: Cosine at or above which a new memory merges into an existing one.
    merge_similarity: float = Field(default=0.92, ge=0, le=1)
    #: A memory's recency score halves every this many days.
    recency_half_life_days: float = Field(default=30.0, gt=0)
    #: Matches taken from each of the keyword and vector searches before ranking.
    candidates: int = Field(default=50, ge=1)
    weights: MemoryWeights = Field(default_factory=MemoryWeights)
    kinds: dict[AgentName, list[MemoryKind]] = Field(default_factory=_default_memory_kinds)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)


class PathsConfig(_Cfg):
    """Where runs and the database live."""

    runs_dir: str = "runs"
    db: str = "runs/agent_team.db"

    def db_file(self) -> Path:
        """The database as an absolute path; a relative one is read from the current
        folder, so messages name this to show where a command looked."""
        return Path(self.db).resolve()


class Config(_Cfg):
    """The whole routing and limits file."""

    defaults: DefaultRoute
    agents: dict[AgentName, Route] = Field(default_factory=dict)
    stronger_model: Route | None = None
    fallbacks: list[str] = Field(default_factory=list)
    #: Model to use on each provider when it serves as a fallback.
    fallback_models: dict[str, str] = Field(default_factory=dict)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    rate_limits: dict[str, ProviderLimit] = Field(default_factory=dict)
    prices: dict[str, Price] = Field(default_factory=dict)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    chaos: ChaosSettings = Field(default_factory=ChaosSettings)
    review: ReviewMode = ReviewMode.AUTO_ASSUMPTIONS
    hooks: list[str] = Field(default_factory=list)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    @model_validator(mode="after")
    def _ollama_routes_fit(self) -> Config:
        """Check agents routed to Ollama (or its stronger model) leave reply room in num_ctx."""
        # Before any usage is observed, compaction estimates at the shared starting ratio
        # (llm.tokens) and Ollama at its own, which may be more cautious; scale to
        # Ollama's so the first call of an attempt at the budget still leaves reply room.
        ollama = self.providers.ollama
        stronger = self.stronger_model
        for agent in AgentName:
            budgets = []
            if self.route_for(agent).provider == "ollama":
                budgets.append(self.context_budget(agent))
            if stronger is not None and stronger.provider == "ollama":
                budgets.append(stronger.context_budget or self.context_budget(agent))
            for budget in budgets:
                estimated = -(-budget * CHARS_PER_TOKEN // ollama.chars_per_token)
                if estimated + ollama.min_output_tokens > ollama.num_ctx:
                    raise ValueError(
                        f"{agent.value}: context_budget {budget} (about {estimated} tokens "
                        f"as Ollama counts them) plus min_output_tokens "
                        f"{ollama.min_output_tokens} exceeds ollama num_ctx {ollama.num_ctx}"
                    )
        return self

    def route_for(self, agent: AgentName) -> Route:
        """The agent's route, or the default, with the context budget and effort
        filled in from the defaults when the agent's route leaves them out."""
        route = self.agents.get(agent)
        if route is None:
            return Route(**self.defaults.model_dump())
        fill: dict[str, Any] = {}
        if route.context_budget is None:
            fill["context_budget"] = self.defaults.context_budget
        if route.effort is None:
            fill["effort"] = self.defaults.effort
        return route.model_copy(update=fill) if fill else route

    def context_budget(self, agent: AgentName) -> int:
        """Max prompt tokens for the agent's model."""
        return self.route_for(agent).context_budget or self.defaults.context_budget

    def fallback_budget(self, provider: str) -> int:
        """Max prompt tokens when ``provider`` stands in for an agent's own route: the
        defaults' budget, cut for Ollama to what leaves reply room in its window."""
        budget = self.defaults.context_budget
        if provider == "ollama":
            ollama = self.providers.ollama
            room = ollama.num_ctx - ollama.min_output_tokens
            budget = min(budget, room * ollama.chars_per_token // CHARS_PER_TOKEN)
        return budget


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Merge ``extra`` into ``base`` recursively; ``extra`` wins."""
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    path: str | Path | None = None, *, overrides: dict[str, Any] | None = None
) -> Config:
    """Load ``path``, else ``$AGENT_TEAM_CONFIG``, else ``config/default.yaml``."""
    chosen = Path(path) if path else Path(os.environ.get(ENV_VAR) or DEFAULT_CONFIG)
    raw = yaml.safe_load(chosen.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{chosen}: config must be a mapping")
    if overrides:
        raw = deep_merge(raw, overrides)
    return Config.model_validate(raw)
