import sys
import types

import pytest

from agent_team.core.events import InMemoryEventBus, load_hooks
from agent_team.core.interfaces import EventBus
from agent_team.core.models import Event, EventName, HookOutcome


def ev(name: EventName = EventName.TOOL_CALL_BEFORE, **data: object) -> Event:
    return Event(name=name, run_id="r", data=dict(data))


async def test_after_hooks_by_name_and_wildcard() -> None:
    bus = InMemoryEventBus()
    seen: list[str] = []

    async def named(e: Event) -> None:
        seen.append(f"named:{e.name}")

    async def every(e: Event) -> None:
        seen.append(f"all:{e.name}")

    bus.on_after(EventName.RUN_STARTED, named)
    bus.on_after(None, every)
    await bus.emit(ev(EventName.RUN_STARTED))
    await bus.emit(ev(EventName.RUN_FINISHED))
    assert seen == ["named:run_started", "all:run_started", "all:run_finished"]


async def test_after_hook_errors_are_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    bus = InMemoryEventBus()
    seen: list[Event] = []

    async def broken(e: Event) -> None:
        raise RuntimeError("boom")

    async def ok(e: Event) -> None:
        seen.append(e)

    bus.on_after(None, broken)
    bus.on_after(None, ok)
    await bus.emit(ev())
    assert len(seen) == 1
    assert "boom" in caplog.text


async def test_before_hooks_first_block_wins() -> None:
    bus = InMemoryEventBus()
    calls: list[str] = []

    async def allow(e: Event) -> HookOutcome:
        calls.append("allow")
        return HookOutcome()

    async def block(e: Event) -> HookOutcome:
        calls.append("block")
        return HookOutcome(allow=False, reason="rm -rf", rule_id="S1")

    async def never(e: Event) -> HookOutcome:
        calls.append("never")
        return HookOutcome()

    for hook in (allow, block, never):
        bus.on_before(EventName.TOOL_CALL_BEFORE, hook)
    outcome = await bus.check(ev())
    assert not outcome.allow and outcome.rule_id == "S1"
    assert calls == ["allow", "block"]
    assert (await bus.check(ev(EventName.LLM_CALL_BEFORE))).allow


async def test_before_hook_rewrite_flows_to_next_hook() -> None:
    bus = InMemoryEventBus()
    seen: list[dict[str, object]] = []

    async def rewrite(e: Event) -> HookOutcome:
        return HookOutcome(replacement={**e.data, "cmd": "ls"})

    async def look(e: Event) -> HookOutcome:
        seen.append(e.data)
        return HookOutcome()

    bus.on_before(EventName.TOOL_CALL_BEFORE, rewrite)
    bus.on_before(EventName.TOOL_CALL_BEFORE, look)
    outcome = await bus.check(ev(cmd="ls -la"))
    assert outcome.allow and outcome.replacement == {"cmd": "ls"}
    assert seen == [{"cmd": "ls"}]


async def test_before_hook_error_fails_closed() -> None:
    bus = InMemoryEventBus()

    async def broken(e: Event) -> HookOutcome:
        raise ValueError("bad")

    bus.on_before(EventName.TOOL_CALL_BEFORE, broken)
    outcome = await bus.check(ev())
    assert not outcome.allow and "bad" in (outcome.reason or "")


async def test_load_hooks_from_path() -> None:
    mod = types.ModuleType("custom_hooks_for_test")
    seen: list[Event] = []

    def register(bus: EventBus) -> None:
        async def note(e: Event) -> None:
            seen.append(e)

        bus.on_after(EventName.RUN_FINISHED, note)

    mod.register = register  # type: ignore[attr-defined]
    sys.modules[mod.__name__] = mod
    bus = InMemoryEventBus()
    load_hooks(["custom_hooks_for_test:register"], bus)
    await bus.emit(ev(EventName.RUN_FINISHED))
    assert len(seen) == 1
    with pytest.raises(ValueError):
        load_hooks(["no_colon"], bus)
