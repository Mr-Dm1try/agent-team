import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agent_team.core.control import KillSwitch
from agent_team.core.errors import BudgetExceeded, RunStopped, TransientLLMError
from agent_team.core.events import InMemoryEventBus
from agent_team.core.models import AgentName, Effort, Event, EventName
from agent_team.core.tracing import JsonlTracer, Tracer, load_spans
from agent_team.llm.chaos import BAD_JSON_TEXT, Chaos, parse_chaos
from agent_team.llm.fake import FakeProvider, FakeTurn
from agent_team.llm.gateway import Gateway
from agent_team.llm.types import LLMRequest, StreamChunk

from .conftest import Sleeps, make_config, request


def gateway(
    fake: FakeProvider,
    backup: FakeProvider | None = None,
    *,
    sleeps: Sleeps,
    bus: InMemoryEventBus | None = None,
    **kw: object,
) -> Gateway:
    providers = {"fake": fake}
    if backup is not None:
        providers["backup"] = backup
    config = kw.pop("config", None) or make_config()
    return Gateway(config, providers, bus=bus, sleep=sleeps, **kw)  # type: ignore[arg-type]


async def test_routes_by_agent_and_prices_calls(
    sleeps: Sleeps, events: list[Event], bus: InMemoryEventBus
) -> None:
    fake = FakeProvider(
        [FakeTurn(text="a", input_tokens=1_000_000, output_tokens=500_000), FakeTurn(text="b")]
    )
    gw = gateway(fake, sleeps=sleeps, bus=bus)
    resp = await gw.complete(request(AgentName.PRODUCT))
    assert resp.model == "fake-small" and resp.usage.cost_usd == pytest.approx(2.0)
    await gw.complete(request(AgentName.ENGINEER))
    assert fake.models == ["fake-small", "fake-big"]
    assert gw.context_budget(AgentName.ENGINEER) == 1000

    status = gw.status("run_1")
    assert status.calls == 2 and status.total.cost_usd == pytest.approx(2.0)
    assert set(status.by_agent) == {"product", "engineer"}
    names = [e.name for e in events]
    assert names.count(EventName.LLM_CALL_AFTER) == 2
    after = next(e for e in events if e.name is EventName.LLM_CALL_AFTER)
    assert after.data["input_tokens"] == 1_000_000 and after.data["fallback"] is False


async def test_retries_transient_errors_with_backoff(sleeps: Sleeps) -> None:
    fake = FakeProvider(
        [FakeTurn(fault="rate_limit"), FakeTurn(fault="timeout"), FakeTurn(text="ok")]
    )
    gw = gateway(fake, sleeps=sleeps)
    assert (await gw.complete(request())).text == "ok"
    assert sleeps.calls == [1, 2]
    assert gw.status("run_1").retries == 2 and gw.status("run_1").fallbacks == 0


async def test_falls_back_after_retries_run_out(
    sleeps: Sleeps, events: list[Event], bus: InMemoryEventBus
) -> None:
    fake = FakeProvider([FakeTurn(fault="server_error")] * 4)
    backup = FakeProvider(["from backup"], name="backup")
    gw = gateway(fake, backup, sleeps=sleeps, bus=bus)
    resp = await gw.complete(request())
    assert resp.text == "from backup" and resp.provider == "backup"
    assert backup.models == ["backup-model"]
    assert sleeps.calls == [1, 2, 4]
    assert gw.status("run_1").fallbacks == 1
    errors = [e for e in events if e.name is EventName.ERROR]
    assert [e.data["retry"] for e in errors] == [True, True, True, False]
    after = next(e for e in events if e.name is EventName.LLM_CALL_AFTER)
    assert after.data["fallback"] is True


async def test_non_transient_error_falls_back_without_retry(sleeps: Sleeps) -> None:
    fake = FakeProvider([ValueError("400 bad request")])
    backup = FakeProvider(["ok"], name="backup")
    gw = gateway(fake, backup, sleeps=sleeps)
    assert (await gw.complete(request())).text == "ok"
    assert sleeps.calls == []


async def test_raises_last_error_when_every_route_fails(sleeps: Sleeps) -> None:
    fake = FakeProvider([FakeTurn(fault="rate_limit")] * 4)
    backup = FakeProvider([FakeTurn(fault="rate_limit")] * 4, name="backup")
    gw = gateway(fake, backup, sleeps=sleeps)
    with pytest.raises(TransientLLMError) as info:
        await gw.complete(request())
    assert info.value.provider == "backup"
    assert gw.status("run_1").failed_calls == 8


async def test_budget_cap_blocks_calls(
    sleeps: Sleeps, events: list[Event], bus: InMemoryEventBus
) -> None:
    config = make_config(budget={"max_usd_per_run": 1.0})
    fake = FakeProvider([FakeTurn(text="x", input_tokens=1_000_000, output_tokens=0)] * 2)
    gw = gateway(fake, sleeps=sleeps, bus=bus, config=config)
    await gw.complete(request())
    with pytest.raises(BudgetExceeded) as info:
        await gw.complete(request())
    assert info.value.limit == "max_usd_per_run"
    assert gw.status("run_1").blocked == 1 and fake.remaining == 1
    # other runs are unaffected
    await gw.complete(request(run_id="run_2"))


async def test_token_cap(sleeps: Sleeps) -> None:
    config = make_config(budget={"max_tokens_per_run": 10})
    fake = FakeProvider([FakeTurn(text="x", input_tokens=5, output_tokens=5)] * 2)
    gw = Gateway(config, {"fake": fake}, sleep=sleeps)
    await gw.complete(request())
    with pytest.raises(BudgetExceeded) as info:
        await gw.complete(request())
    assert info.value.limit == "max_tokens_per_run"


async def test_time_cap(sleeps: Sleeps) -> None:
    now = [0.0]
    config = make_config(budget={"max_minutes_per_run": 1})
    fake = FakeProvider(["a", "b"])
    gw = Gateway(config, {"fake": fake}, sleep=sleeps, clock=lambda: now[0])
    await gw.complete(request())
    now[0] = 61.0
    with pytest.raises(BudgetExceeded) as info:
        await gw.complete(request())
    assert info.value.limit == "max_minutes_per_run"


async def test_no_retry_or_fallback_starts_after_the_time_cap(sleeps: Sleeps) -> None:
    """A slow call that fails after the cap has tripped is not retried or handed to
    the fallback: the run stops instead of spending more."""
    clock = Clock()
    config = make_config(budget={"max_minutes_per_run": 30}, fallbacks=["fake", "backup"])

    class Slow(FakeProvider):
        """Fails after the run's 30 minutes are up, as a long Ollama call can."""

        async def complete(self, request: LLMRequest, *, model: str, effort: Effort | None = None):  # type: ignore[no-untyped-def]
            """Move the clock past the cap, then fail."""
            clock.now = 31 * 60
            raise TransientLLMError("ReadTimeout", provider="fake")

    backup = FakeProvider(["too late"], name="backup")
    gw = Gateway(config, {"fake": Slow([]), "backup": backup}, sleep=sleeps, clock=clock)
    with pytest.raises(BudgetExceeded) as info:
        await gw.complete(request())
    assert info.value.limit == "max_minutes_per_run"
    assert backup.remaining == 1 and sleeps.calls == []


async def test_custom_before_hook_can_block(sleeps: Sleeps, bus: InMemoryEventBus) -> None:
    from agent_team.core.models import HookOutcome

    async def deny(e: Event) -> HookOutcome:
        return HookOutcome(allow=False, reason="nope", rule_id="custom")

    bus.on_before(EventName.LLM_CALL_BEFORE, deny)
    gw = gateway(FakeProvider(["x"]), sleeps=sleeps, bus=bus)
    with pytest.raises(BudgetExceeded) as info:
        await gw.complete(request())
    assert info.value.limit == "custom"


async def test_kill_switch_stops_the_run(sleeps: Sleeps, tmp_path: Path) -> None:
    ks = KillSwitch(tmp_path)
    fake = FakeProvider(["a", "b"])
    gw = gateway(fake, sleeps=sleeps, kill_switch=ks)
    await gw.complete(request())
    ks.stop("run_1")
    with pytest.raises(RunStopped):
        await gw.complete(request())
    assert fake.remaining == 1


async def test_kill_switch_interrupts_retries(sleeps: Sleeps, tmp_path: Path) -> None:
    ks = KillSwitch(tmp_path)

    async def stop_on_sleep(seconds: float) -> None:
        ks.stop("run_1")

    fake = FakeProvider([FakeTurn(fault="rate_limit"), "never"])
    gw = Gateway(make_config(), {"fake": fake}, sleep=stop_on_sleep, kill_switch=ks)
    with pytest.raises(RunStopped):
        await gw.complete(request())


async def test_chaos_script_injects_in_front_of_provider(sleeps: Sleeps) -> None:
    chaos = Chaos(parse_chaos("script:rate_limit,ok,bad_json"))
    fake = FakeProvider(['{"a": 1}', '{"b": 2}'])
    gw = gateway(fake, sleeps=sleeps, chaos=chaos)
    assert (await gw.complete(request())).text == '{"a": 1}'
    assert sleeps.calls == [1]
    assert (await gw.complete(request())).text == BAD_JSON_TEXT
    assert gw.status("run_1").chaos_injected == 2
    assert [f.value for f in chaos.injected] == ["rate_limit", "bad_json"]


async def test_chaos_rate_is_seeded() -> None:
    settings = parse_chaos("timeout@1.0")
    chaos = Chaos(settings)
    assert all(chaos.next_fault() is not None for _ in range(5))
    assert Chaos(parse_chaos("0")).enabled is False


def test_parse_chaos_variants() -> None:
    assert parse_chaos("0.3").rate == 0.3 and len(parse_chaos("0.3").faults) == 4
    s = parse_chaos("rate_limit,bad_json@0.5")
    assert s.faults == ["rate_limit", "bad_json"] and s.rate == 0.5
    assert parse_chaos("timeout").rate == 0.25
    assert parse_chaos("").rate == 0
    with pytest.raises(ValueError):
        parse_chaos("explode")
    with pytest.raises(ValueError):
        parse_chaos("1.5")


async def test_stronger_model_override(sleeps: Sleeps) -> None:
    fake = FakeProvider(["a", "b", "c"])
    gw = gateway(fake, sleeps=sleeps)
    gw.use_stronger_model("task_1")
    await gw.complete(request())
    await gw.complete(request(task_id="task_2"))
    gw.set_override("task_1", None)
    await gw.complete(request())
    assert fake.models == ["fake-strong", "fake-small", "fake-small"]
    gw.set_override("task_1", {"provider": "fake", "model": "fake-custom"})
    assert gw.routes(request())[0] == ("fake", "fake-custom")
    assert gw.providers_for(AgentName.PRODUCT, "task_1")[0] == "fake"


async def test_the_stronger_model_runs_with_its_own_prompt_cap(sleeps: Sleeps) -> None:
    capped = make_config(
        stronger_model={"provider": "fake", "model": "fake-strong", "context_budget": 300}
    )
    gw = gateway(FakeProvider(), sleeps=sleeps, config=capped)
    assert gw.context_budget(AgentName.PRODUCT, "task_1") == 1000
    gw.use_stronger_model("task_1")
    assert gw.context_budget(AgentName.PRODUCT, "task_1") == 300
    assert gw.context_budget(AgentName.PRODUCT, "task_2") == 1000
    # A cap above the agent's own budget never raises it.
    loose = make_config(
        stronger_model={"provider": "fake", "model": "fake-strong", "context_budget": 5000}
    )
    wide = gateway(FakeProvider(), sleeps=sleeps, config=loose)
    wide.use_stronger_model("task_1")
    assert wide.context_budget(AgentName.PRODUCT, "task_1") == 1000
    # Without a cap of its own, it runs with the agent's budget.
    plain = gateway(FakeProvider(), sleeps=sleeps)
    plain.use_stronger_model("task_1")
    assert plain.context_budget(AgentName.PRODUCT, "task_1") == 1000


async def test_the_next_call_may_go_to_its_route_then_the_fallbacks(sleeps: Sleeps) -> None:
    gw = gateway(FakeProvider(), FakeProvider(name="backup"), sleeps=sleeps)
    assert gw.providers_for(AgentName.PRODUCT, None) == ["fake", "backup"]
    gw.set_override("task_1", {"provider": "backup", "model": "b-1"})
    assert gw.providers_for(AgentName.PRODUCT, "task_1") == ["backup", "fake"]
    assert gw.providers_for(AgentName.PRODUCT, "task_2") == ["fake", "backup"]
    # A provider that isn't configured is skipped, as the call itself skips it.
    assert gateway(FakeProvider(), sleeps=sleeps).providers_for(AgentName.PRODUCT, None) == ["fake"]


async def test_unknown_providers_are_skipped(sleeps: Sleeps) -> None:
    config = make_config(defaults={"provider": "ollama", "model": "x", "context_budget": 20_000})
    fake = FakeProvider(["ok"])
    gw = gateway(fake, sleeps=sleeps, config=config)
    assert gw.routes(request()) == [("fake", "fake-small")]
    assert (await gw.complete(request())).text == "ok"
    with pytest.raises(ValueError):
        await Gateway(config, {}, sleep=sleeps).complete(request())


async def test_rate_limit_queues_by_rpm(sleeps: Sleeps) -> None:
    now = [0.0]

    async def advance(seconds: float) -> None:
        sleeps.calls.append(seconds)
        now[0] += seconds

    config = make_config(rate_limits={"fake": {"max_concurrent": 1, "requests_per_minute": 2}})
    gw = Gateway(
        config, {"fake": FakeProvider(["a", "b", "c"])}, sleep=advance, clock=lambda: now[0]
    )
    for _ in range(3):
        await gw.complete(request())
    assert sleeps.calls == [60.0]


async def test_spans_carry_usage(sleeps: Sleeps, tmp_path: Path) -> None:
    span_bus = InMemoryEventBus()
    JsonlTracer(tmp_path).attach(span_bus)
    tracer = Tracer(span_bus)
    gw = gateway(
        FakeProvider([FakeTurn(text="x", input_tokens=7, output_tokens=3)]),
        sleeps=sleeps,
        tracer=tracer,
    )
    async with tracer.span("run_1", "attempt", "attempt", task_id="task_1"):
        await gw.complete(request())
    spans = {s.name: s for s in load_spans(tmp_path, "run_1")}
    llm = spans["llm:product"]
    assert llm.kind == "llm" and llm.parent_span_id == spans["attempt"].span_id
    assert llm.attrs["input_tokens"] == 7 and llm.attrs["model"] == "fake-small"


async def test_stream_retries_before_first_chunk(
    sleeps: Sleeps, events: list[Event], bus: InMemoryEventBus
) -> None:
    fake = FakeProvider([FakeTurn(fault="rate_limit"), FakeTurn(text="hello streaming world")])
    gw = gateway(fake, sleeps=sleeps, bus=bus)
    chunks = [c async for c in gw.stream(request())]
    assert "".join(c.text for c in chunks[:-1]) == "hello streaming world"
    final = chunks[-1]
    assert final.done and final.response is not None and final.response.usage.cost_usd > 0
    assert sleeps.calls == [1]
    assert any(e.name is EventName.LLM_CALL_AFTER for e in events)


class MidStreamFailure(FakeProvider):
    async def stream(
        self, request: LLMRequest, *, model: str, effort: Effort | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="partial")
        raise TransientLLMError("dropped", provider=self.name)


async def test_stream_error_after_first_chunk_propagates(sleeps: Sleeps) -> None:
    gw = gateway(MidStreamFailure(), FakeProvider(["unused"], name="backup"), sleeps=sleeps)
    got: list[str] = []
    with pytest.raises(TransientLLMError):
        async for chunk in gw.stream(request()):
            got.append(chunk.text)
    assert got == ["partial"] and sleeps.calls == []


async def test_stream_falls_back(sleeps: Sleeps) -> None:
    gw = gateway(
        FakeProvider([ValueError("bad")]), FakeProvider(["b"], name="backup"), sleeps=sleeps
    )
    chunks = [c async for c in gw.stream(request())]
    assert chunks[-1].response is not None and chunks[-1].response.provider == "backup"


async def test_prices_by_requested_model_when_provider_echoes_a_snapshot(sleeps: Sleeps) -> None:
    from agent_team.core.models import Usage
    from agent_team.llm.types import LLMResponse, TextBlock

    snapshot = LLMResponse(
        content=[TextBlock(text="x")],
        stop_reason="end_turn",
        usage=Usage(input_tokens=1_000_000),
        provider="fake",
        model="fake-small-20261001",
    )
    gw = gateway(FakeProvider([snapshot]), sleeps=sleeps)
    assert (await gw.complete(request())).usage.cost_usd == pytest.approx(1.0)


class NoFinalChunk(FakeProvider):
    async def stream(
        self, request: LLMRequest, *, model: str, effort: Effort | None = None
    ) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        if len(self.requests) == 1:
            return
        async for chunk in super().stream(request, model=model, effort=effort):
            yield chunk


async def test_stream_without_final_chunk_is_retried(sleeps: Sleeps) -> None:
    gw = gateway(NoFinalChunk(["second try"]), sleeps=sleeps)
    chunks = [c async for c in gw.stream(request())]
    assert chunks[-1].response is not None and chunks[-1].response.text == "second try"
    assert sleeps.calls == [1] and gw.status("run_1").failed_calls == 1


async def test_missing_primary_provider_counts_as_fallback(
    sleeps: Sleeps, events: list[Event], bus: InMemoryEventBus
) -> None:
    config = make_config(defaults={"provider": "ollama", "model": "x", "context_budget": 20_000})
    gw = gateway(FakeProvider(["ok"]), sleeps=sleeps, bus=bus, config=config)
    await gw.complete(request())
    assert gw.status("run_1").fallbacks == 1
    after = next(e for e in events if e.name is EventName.LLM_CALL_AFTER)
    assert after.data["fallback"] is True


async def test_resumed_run_over_budget_is_blocked(sleeps: Sleeps, tmp_path: Path) -> None:
    from agent_team.core.db import Database
    from agent_team.core.models import Run
    from agent_team.core.stores import SqliteTaskStore

    config = make_config(budget={"max_usd_per_run": 1.0})
    path = tmp_path / "db.sqlite"
    runs = SqliteTaskStore(Database(path))
    run = await runs.create_run(Run(id="run_1", title="t", request="r", workspace="/w"))

    # First process: two parallel calls, each $0.6 of input. Both are recorded.
    first = gateway(
        FakeProvider([FakeTurn(text="x", input_tokens=600_000, output_tokens=1)] * 2),
        sleeps=sleeps,
        config=config,
        runs=runs,
    )
    await asyncio.gather(first.complete(request()), first.complete(request()))
    stored = await runs.get_run(run.id)
    assert stored.usage.input_tokens == 1_200_000
    assert stored.usage.cost_usd == pytest.approx(1.2, abs=1e-3)

    # Second process resumes the run: it starts at $1.2, so the cap blocks at once.
    resumed = SqliteTaskStore(Database(path))
    fake = FakeProvider(["never"])
    second = gateway(fake, sleeps=sleeps, config=config, runs=resumed)
    with pytest.raises(BudgetExceeded) as info:
        await second.complete(request())
    assert info.value.limit == "max_usd_per_run" and fake.remaining == 1


async def test_run_missing_from_store_still_works(sleeps: Sleeps, tmp_path: Path) -> None:
    from agent_team.core.db import Database
    from agent_team.core.stores import SqliteTaskStore

    runs = SqliteTaskStore(Database(tmp_path / "db.sqlite"))
    gw = gateway(FakeProvider(["ok"]), sleeps=sleeps, runs=runs)
    assert (await gw.complete(request())).text == "ok"


async def test_effort_comes_from_the_route_and_survives_fallback(sleeps: Sleeps) -> None:
    config = make_config(
        defaults={
            "provider": "fake",
            "model": "fake-small",
            "context_budget": 1000,
            "effort": "low",
        },
        agents={"engineer": {"provider": "fake", "model": "fake-big", "effort": "medium"}},
    )
    down = [TransientLLMError("down", provider="fake") for _ in range(4)]
    fake = FakeProvider(["a", "b", *down])
    backup = FakeProvider(["c"], name="backup")
    gw = Gateway(config, {"fake": fake, "backup": backup}, sleep=sleeps)
    await gw.complete(request(AgentName.PRODUCT))
    await gw.complete(request(AgentName.ENGINEER))
    gw.use_stronger_model("task_1")  # the stronger route sets no effort of its own
    assert (await gw.complete(request(AgentName.ENGINEER))).provider == "backup"
    assert fake.efforts[:2] == ["low", "medium"]
    assert fake.efforts[2:] == ["medium"] * 4 and backup.efforts == ["medium"]


async def test_after_event_reports_thinking_the_api_dropped(
    sleeps: Sleeps, events: list[Event], bus: InMemoryEventBus
) -> None:
    from agent_team.core.models import Usage
    from agent_team.llm.types import LLMResponse, TextBlock

    def reply(dropped: int) -> LLMResponse:
        return LLMResponse(
            content=[TextBlock(text="x")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=1),
            provider="fake",
            model="fake-small",
            thinking_dropped=dropped,
        )

    gw = gateway(FakeProvider([reply(2), reply(0)]), sleeps=sleeps, bus=bus)
    await gw.complete(request())
    await gw.complete(request())
    after = [e.data for e in events if e.name is EventName.LLM_CALL_AFTER]
    assert after[0]["thinking_dropped"] == 2 and after[1]["thinking_dropped"] is None


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """The current time."""
        return self.now


def refused() -> TransientLLMError:
    """The error a provider gives when nothing listens on its port."""
    return TransientLLMError("APIConnectionError: Connection error.", provider="fake",
                             unreachable=True)  # fmt: skip


async def test_a_refused_connection_marks_the_provider_down_and_skips_it(
    sleeps: Sleeps, events: list[Event], bus: InMemoryEventBus
) -> None:
    clock = Clock()
    fake = FakeProvider([refused(), "back again"])
    backup = FakeProvider(["one", "two"], name="backup")
    gw = gateway(fake, backup, sleeps=sleeps, bus=bus, clock=clock)
    assert (await gw.complete(request())).text == "one"
    assert sleeps.calls == []  # no retries on a provider that is down
    # While it is down, calls go straight to the fallback.
    assert (await gw.complete(request())).text == "two"
    assert fake.remaining == 1 and gw.providers_for(AgentName.PRODUCT, "task_1") == ["backup"]
    down = [e for e in events if e.name is EventName.PROVIDER_DOWN]
    assert len(down) == 1 and down[0].data["provider"] == "fake"
    # Once the wait is over it is tried again, and comes back.
    clock.now += make_config().retry.provider_down_s
    assert (await gw.complete(request())).text == "back again"
    up = [e for e in events if e.name is EventName.PROVIDER_UP]
    assert len(up) == 1 and up[0].data["provider"] == "fake"


async def test_a_provider_still_down_after_the_wait_gives_no_second_event(
    sleeps: Sleeps, events: list[Event], bus: InMemoryEventBus
) -> None:
    clock = Clock()
    fake = FakeProvider([refused(), refused()])
    backup = FakeProvider(["one", "two"], name="backup")
    gw = gateway(fake, backup, sleeps=sleeps, bus=bus, clock=clock)
    await gw.complete(request())
    clock.now += 61
    await gw.complete(request())
    assert [e.name for e in events].count(EventName.PROVIDER_DOWN) == 1
    assert EventName.PROVIDER_UP not in [e.name for e in events]


async def test_when_every_provider_is_down_they_are_still_tried(sleeps: Sleeps) -> None:
    fake = FakeProvider([refused(), "ok"])
    backup = FakeProvider([refused()] * 4, name="backup")
    gw = gateway(fake, backup, sleeps=sleeps, clock=Clock())
    with pytest.raises(TransientLLMError):
        await gw.complete(request())  # the last route retries before it gives up
    assert sleeps.calls == [1, 2, 4]
    assert (await gw.complete(request())).text == "ok"


async def test_the_context_budget_follows_a_fallback_while_the_provider_is_down(
    sleeps: Sleeps,
) -> None:
    config = make_config(
        agents={"researcher": {"provider": "fake", "model": "fake-local", "context_budget": 200}}
    )
    fake = FakeProvider([refused()])
    backup = FakeProvider(["ok"], name="backup")
    gw = gateway(fake, backup, sleeps=sleeps, config=config, clock=Clock())
    assert gw.context_budget(AgentName.RESEARCHER, "task_1") == 200
    await gw.complete(request(AgentName.RESEARCHER))
    assert gw.context_budget(AgentName.RESEARCHER, "task_1") == 1000  # the defaults'


async def test_with_no_other_route_a_refused_connection_still_retries(sleeps: Sleeps) -> None:
    config = make_config(fallbacks=[])
    fake = FakeProvider([refused(), "ok"])
    gw = gateway(fake, sleeps=sleeps, config=config, clock=Clock())
    assert (await gw.complete(request())).text == "ok"
    assert sleeps.calls == [1]
