from typing import Any

import pytest

from agent_team.core.config import Config
from agent_team.core.events import InMemoryEventBus
from agent_team.core.models import AgentName, Event
from agent_team.llm.types import LLMRequest, Message


def make_config(**extra: Any) -> Config:
    raw: dict[str, Any] = {
        "defaults": {"provider": "fake", "model": "fake-small", "context_budget": 1000},
        "agents": {"engineer": {"provider": "fake", "model": "fake-big"}},
        "stronger_model": {"provider": "fake", "model": "fake-strong"},
        "fallbacks": ["fake", "backup"],
        "fallback_models": {"fake": "fake-small", "backup": "backup-model"},
        "retry": {"delays_s": [1, 2, 4]},
        "prices": {"fake-small": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}},
    }
    raw.update(extra)
    return Config.model_validate(raw)


def request(agent: AgentName = AgentName.PRODUCT, **kw: Any) -> LLMRequest:
    fields: dict[str, Any] = {
        "agent": agent,
        "system": "be brief",
        "messages": [Message.user("hi")],
        "run_id": "run_1",
        "task_id": "task_1",
    }
    fields.update(kw)
    return LLMRequest(**fields)


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def sleeps() -> Sleeps:
    return Sleeps()


@pytest.fixture
def events() -> list[Event]:
    return []


@pytest.fixture
def bus(events: list[Event]) -> InMemoryEventBus:
    b = InMemoryEventBus()

    async def collect(e: Event) -> None:
        events.append(e)

    b.on_after(None, collect)
    return b
