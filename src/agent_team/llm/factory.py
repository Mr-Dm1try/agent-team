"""Build the gateway and providers from the config."""

from __future__ import annotations

from collections.abc import Iterable

from agent_team.core.config import Config
from agent_team.core.control import KillSwitch
from agent_team.core.interfaces import EventBus, TaskStore
from agent_team.core.tracing import Tracer
from agent_team.llm.chaos import Chaos
from agent_team.llm.gateway import Gateway
from agent_team.llm.types import Provider


def build_providers(config: Config, names: Iterable[str] | None = None) -> dict[str, Provider]:
    """Real providers for every name the routing file uses. Clients are created on
    first use, so building them needs no API key or running Ollama."""
    from agent_team.llm.anthropic_provider import AnthropicProvider
    from agent_team.llm.fake import FakeProvider
    from agent_team.llm.ollama_provider import OllamaProvider

    if names is None:
        wanted = {config.defaults.provider, *config.fallbacks}
        wanted |= {r.provider for r in config.agents.values()}
        if config.stronger_model:
            wanted.add(config.stronger_model.provider)
        names = sorted(wanted)
    out: dict[str, Provider] = {}
    for name in names:
        if name == "anthropic":
            out[name] = AnthropicProvider(config.providers.anthropic)
        elif name == "ollama":
            out[name] = OllamaProvider(config.providers.ollama)
        elif name == "fake":
            out[name] = FakeProvider()
        else:
            raise ValueError(f"unknown provider {name!r}")
    return out


def build_gateway(
    config: Config,
    *,
    providers: dict[str, Provider] | None = None,
    bus: EventBus | None = None,
    tracer: Tracer | None = None,
    kill_switch: KillSwitch | None = None,
    chaos: Chaos | None = None,
    runs: TaskStore | None = None,
) -> Gateway:
    """A Gateway over the given or configured providers."""
    return Gateway(
        config,
        providers if providers is not None else build_providers(config),
        bus=bus,
        tracer=tracer,
        kill_switch=kill_switch,
        chaos=chaos,
        runs=runs,
    )
