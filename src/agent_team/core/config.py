"""Config loading. ``config/default.yaml`` holds routing, budgets, limits and review
mode; ``AGENT_TEAM_CONFIG`` or an explicit path overrides it, and ``overrides`` are
deep-merged on top (used by CLI flags such as ``--chaos``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_team.core.models import AgentName, Effort, ReviewMode

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
ENV_VAR = "AGENT_TEAM_CONFIG"


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Route(_Cfg):
    """A provider and model, with an optional prompt-token budget and effort. Effort
    sets how hard the model reasons: Claude's effort, or Ollama's reasoning_effort for
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


class RetryConfig(_Cfg):
    #: Waits before each retry of a transient error on the same provider.
    """Backoff for transient model errors."""

    delays_s: list[float] = Field(default_factory=lambda: [1.0, 2.0, 4.0])


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
        # The old format mapped model names to settings; catch it here, not as a 400.
        if not isinstance(value.get("type"), str):
            raise ValueError("thinking must be one setting with a type, e.g. {type: adaptive}")
        return value


class OllamaConfig(_Cfg):
    """Ollama provider settings."""

    base_url: str = "http://localhost:11434/v1"
    num_ctx: int = 32_768
    timeout_s: float = 300.0
    #: Models (by name before the ":" tag) that take ``reasoning_effort``. Ollama
    #: rejects it for models that can't think, so others get no effort at all.
    reasoning_models: list[str] = Field(default_factory=lambda: ["gpt-oss"])
    #: Least room left for the reply. A route's prompt budget plus this must fit
    #: ``num_ctx``, and a call whose prompt leaves less is refused, because Ollama
    #: would otherwise drop the start of the prompt without saying so.
    min_output_tokens: int = 1024
    #: Characters per token when estimating a prompt. Low on purpose: running over
    #: the window is silent truncation, while a high estimate only costs a fallback.
    chars_per_token: int = 3


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


class PathsConfig(_Cfg):
    """Where runs and the database live."""

    runs_dir: str = "runs"
    db: str = "runs/agent_team.db"


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
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @model_validator(mode="after")
    def _ollama_routes_fit(self) -> Config:
        # Budgets are counted at four characters per token (as compaction does); the
        # Ollama provider estimates more cautiously, so scale before comparing.
        ollama = self.providers.ollama
        for agent in AgentName:
            route = self.route_for(agent)
            on_ollama = route.provider == "ollama" or (
                self.stronger_model is not None and self.stronger_model.provider == "ollama"
            )
            if not on_ollama:
                continue
            budget = self.context_budget(agent)
            estimated = -(-budget * 4 // ollama.chars_per_token)
            if estimated + ollama.min_output_tokens > ollama.num_ctx:
                raise ValueError(
                    f"{agent.value}: context_budget {budget} (about {estimated} tokens as "
                    f"Ollama counts them) plus min_output_tokens {ollama.min_output_tokens} "
                    f"exceeds ollama num_ctx {ollama.num_ctx}"
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
