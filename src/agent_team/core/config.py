"""Config loading. ``config/default.yaml`` holds routing, budgets, limits and review
mode; ``AGENT_TEAM_CONFIG`` or an explicit path overrides it, and ``overrides`` are
deep-merged on top (used by CLI flags such as ``--chaos``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from agent_team.core.models import AgentName, ReviewMode

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
ENV_VAR = "AGENT_TEAM_CONFIG"


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Route(_Cfg):
    """A provider and model, with an optional prompt-token budget."""

    provider: str
    model: str
    context_budget: int | None = None


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
    #: Thinking setting per model, passed through as-is. Our message format has no
    #: thinking blocks, so routes default to thinking off where the model allows it.
    thinking: dict[str, dict[str, Any]] = Field(default_factory=dict)


class OllamaConfig(_Cfg):
    """Ollama provider settings."""

    base_url: str = "http://localhost:11434/v1"
    num_ctx: int = 32_768
    timeout_s: float = 300.0


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

    def route_for(self, agent: AgentName) -> Route:
        """The agent's route, or the default, with a context budget filled in."""
        route = self.agents.get(agent)
        if route is None:
            return Route(**self.defaults.model_dump())
        if route.context_budget is None:
            route = route.model_copy(update={"context_budget": self.defaults.context_budget})
        return route

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
