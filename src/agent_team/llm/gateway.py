"""The LLM gateway: the one place every model call passes through.

Per call, in order:

1. Kill switch: a stopped run raises ``RunStopped``.
2. ``llm_call_before`` before hooks, including the budget guard. A block raises
   ``BudgetExceeded`` with the hook's rule id as the limit. Hooks here can only
   block: a rewritten payload is ignored, because the event carries a summary of the
   request, not the request itself.
3. Routes: the agent's provider and model (or a per-task override such as the
   stronger model), then each fallback provider with its fallback model.
4. Per route: rate limit slot, chaos, provider call. Transient errors retry after the
   configured delays (1, 2, 4 s), then move to the next route. Other provider errors
   move to the next route at once. A provider that refuses the connection is marked
   down for ``retry.provider_down_s``: the call moves to the next route without
   retrying, and later calls skip it until then, unless every route is down. With no
   other route left, a refused connection retries like any transient error.
   ``provider_down`` is emitted when it goes down and ``provider_up`` once a call to it
   succeeds again.
5. Cost from the price table (by the requested model, then the model the provider
   reports), usage recorded in memory and, when a task store is given, added to
   ``Run.usage``. ``llm_call_after`` is emitted. A run's first call in a process
   seeds its budget from ``Run.usage``, so resuming doesn't reset spend.

Implements ``agent_team.llm.types.LLMClient``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, nullcontext
from typing import Any

from agent_team.core.config import Config, Route
from agent_team.core.control import KillSwitch
from agent_team.core.errors import BudgetExceeded, RunStopped, TransientLLMError
from agent_team.core.events import InMemoryEventBus
from agent_team.core.interfaces import EventBus, TaskStore
from agent_team.core.models import AgentName, Effort, Event, EventName, Usage
from agent_team.core.tracing import Tracer, current_span
from agent_team.llm.budget import (
    BudgetGuard,
    BudgetTracker,
    Clock,
    RateLimiter,
    RunUsage,
    Sleep,
)
from agent_team.llm.budget import cost_of as _cost_of
from agent_team.llm.chaos import TRANSIENT_FAULTS, Chaos, Fault
from agent_team.llm.types import LLMClient, LLMRequest, LLMResponse, Provider, StreamChunk

_PASS_THROUGH = (BudgetExceeded, RunStopped)


class Gateway:
    """Routes, retries, falls back, rate-limits, budgets and logs every model call."""

    def __init__(
        self,
        config: Config,
        providers: Mapping[str, Provider],
        *,
        bus: EventBus | None = None,
        tracer: Tracer | None = None,
        kill_switch: KillSwitch | None = None,
        runs: TaskStore | None = None,
        chaos: Chaos | None = None,
        sleep: Sleep = asyncio.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        """Wire up providers, hooks, limits and budgets; sleep and clock are for tests."""
        self.config = config
        self.providers = dict(providers)
        self.bus: EventBus = bus or InMemoryEventBus()
        self.tracer = tracer
        self.kill_switch = kill_switch
        self.runs = runs
        self._seeded: set[str] = set()
        self._run_locks: dict[str, asyncio.Lock] = {}
        self.chaos = chaos or Chaos(config.chaos)
        self.sleep = sleep
        self.clock = clock
        #: Providers that refused a connection, with the clock time to try them again.
        self._down: dict[str, float] = {}
        self.budget = BudgetTracker(config.budget, clock=clock)
        self.bus.on_before(EventName.LLM_CALL_BEFORE, BudgetGuard(self.budget))
        self._global = asyncio.Semaphore(config.limits.max_concurrent_calls)
        self._limiters = {
            name: RateLimiter(
                max_concurrent=lim.max_concurrent,
                requests_per_minute=lim.requests_per_minute,
                clock=clock,
                sleep=sleep,
            )
            for name, lim in config.rate_limits.items()
        }
        self._overrides: dict[str, Route] = {}

    # --- routing -----------------------------------------------------------------------

    def context_budget(self, agent: AgentName, task_id: str | None = None) -> int:
        """Max prompt tokens for the route the agent's next call tries first: the
        agent's own budget on its own route, cut to a task override's budget when that
        is smaller (such as the stronger model's cap), else the budget of the fallback
        standing in for it (when its provider is down)."""
        primary = self._route(agent, task_id)
        chain = self._chain(agent, task_id)
        if chain and chain[0] != (primary.provider, primary.model):
            return self.config.fallback_budget(chain[0][0])
        budget = self.config.context_budget(agent)
        override = self._overrides.get(task_id) if task_id else None
        if override is not None and override.context_budget is not None:
            return min(budget, override.context_budget)
        return budget

    def providers_for(self, agent: AgentName, task_id: str | None) -> list[str]:
        """The providers the next call for this agent and task may go to, in order."""
        return [provider for provider, _ in self._chain(agent, task_id)]

    def set_override(self, task_id: str, route: Route | Mapping[str, Any] | None) -> None:
        """Route one task's calls elsewhere, e.g. one attempt on the stronger model.
        Takes a Route or a ``{provider, model}`` mapping; ``None`` removes the override."""
        if route is None:
            self._overrides.pop(task_id, None)
        else:
            self._overrides[task_id] = Route.model_validate(route)

    def use_stronger_model(self, task_id: str) -> Route:
        """Route this task's calls to the configured stronger model; returns that route."""
        route = self.config.stronger_model
        if route is None:
            raise ValueError("config has no stronger_model")
        self.set_override(task_id, route)
        return route

    def _primary(self, request: LLMRequest) -> Route:
        """The task's override route, else the agent's route."""
        return self._route(request.agent, request.task_id)

    def _route(self, agent: AgentName, task_id: str | None) -> Route:
        """The task's override route, else the agent's route."""
        override = self._overrides.get(task_id) if task_id else None
        return override or self.config.route_for(agent)

    def _effort(self, request: LLMRequest, primary: Route) -> Effort | None:
        """The primary route's effort; an override without one (e.g. the stronger
        model) keeps the agent's own. Fallbacks use the same effort."""
        if primary.effort is not None:
            return primary.effort
        return self.config.route_for(request.agent).effort

    def routes(self, request: LLMRequest) -> list[tuple[str, str]]:
        """(provider, model) pairs to try, in order, limited to known providers."""
        return self._chain(request.agent, request.task_id)

    def _chain(self, agent: AgentName, task_id: str | None) -> list[tuple[str, str]]:
        """The route for this agent and task, then the fallbacks, limited to known
        providers."""
        primary = self._route(agent, task_id)
        chain = [(primary.provider, primary.model)]
        for name in self.config.fallbacks:
            if name == primary.provider:
                continue
            model = self.config.fallback_models.get(name)
            if model is None and name == self.config.defaults.provider:
                model = self.config.defaults.model
            if model is not None:
                chain.append((name, model))
        known = [(p, m) for p, m in chain if p in self.providers]
        now = self.clock()
        up = [(p, m) for p, m in known if self._down.get(p, now) <= now]
        return up or known

    async def _mark_down(self, request: LLMRequest, provider: str, exc: Exception) -> None:
        """Skip ``provider`` for a while after it refused a connection; the first time,
        emit ``provider_down``."""
        was_down = provider in self._down
        retry_s = self.config.retry.provider_down_s
        self._down[provider] = self.clock() + retry_s
        if not was_down:
            await self.bus.emit(
                self._event(
                    EventName.PROVIDER_DOWN,
                    request,
                    "warning",
                    provider=provider,
                    error=str(exc),
                    retry_after_s=retry_s,
                )
            )

    async def _mark_up(self, request: LLMRequest, provider: str) -> None:
        """A call to a provider marked down succeeded: emit ``provider_up``."""
        if self._down.pop(provider, None) is not None:
            await self.bus.emit(self._event(EventName.PROVIDER_UP, request, provider=provider))

    def status(self, run_id: str) -> RunUsage:
        """Calls, tokens, cost, retries, fallbacks and blocked calls for one run."""
        return self.budget.usage(run_id)

    async def _seed(self, run_id: str) -> None:
        """On a run's first call in this process, start its budget from ``Run.usage``
        so a resumed run doesn't start again from $0."""
        if self.runs is None or run_id in self._seeded:
            return
        self._seeded.add(run_id)
        try:
            run = await self.runs.get_run(run_id)
        except KeyError:
            return
        self.budget.restore(run_id, run.usage)

    async def _persist(self, run_id: str, usage: Usage) -> None:
        """Add one call's usage to ``Run.usage``. Serialised per run, so parallel
        calls don't lose each other's updates."""
        if self.runs is None:
            return
        lock = self._run_locks.setdefault(run_id, asyncio.Lock())
        async with lock:
            try:
                run = await self.runs.get_run(run_id)
            except KeyError:
                return
            await self.runs.update_run(run.model_copy(update={"usage": run.usage + usage}))

    # --- helpers -----------------------------------------------------------------------

    def _event(
        self, name: EventName, request: LLMRequest, level: Any = "info", **data: Any
    ) -> Event:
        """An event for this request, tagged with the current span's ids."""
        span = current_span()
        return Event(
            name=name,
            run_id=request.run_id,
            task_id=request.task_id,
            agent=request.agent,
            attempt=span.attempt if span else None,
            trace_id=span.trace_id if span else None,
            span_id=span.span_id if span else None,
            parent_span_id=span.parent_span_id if span else None,
            level=level,
            data=data,
        )

    def check_run(self, run_id: str) -> None:
        """Raise RunStopped when the run has a stop flag, or BudgetExceeded once it has
        reached a cap. Checked before every try, so a retry or fallback never starts
        after the stop, and by the tool loop before it runs a reply's tool calls."""
        if self.kill_switch is not None:
            self.kill_switch.check(run_id)
        hit = self.budget.exceeded(run_id)
        if hit is not None:
            self.budget.usage(run_id).blocked += 1
            limit, reason = hit
            raise BudgetExceeded(reason, limit=limit)

    async def _preflight(self, request: LLMRequest) -> list[tuple[str, str]]:
        """Check the kill switch, seed the run's spend, run before hooks; return the routes."""
        if self.kill_switch is not None:
            self.kill_switch.check(request.run_id)
        await self._seed(request.run_id)
        routes = self.routes(request)
        if not routes:
            raise ValueError(f"no configured provider can serve agent {request.agent.value!r}")
        outcome = await self.bus.check(
            self._event(
                EventName.LLM_CALL_BEFORE,
                request,
                provider=routes[0][0],
                model=routes[0][1],
                messages=len(request.messages),
                tools=[t.name for t in request.tools],
                max_tokens=request.max_tokens,
            )
        )
        if not outcome.allow:
            self.budget.usage(request.run_id).blocked += 1
            await self.bus.emit(
                self._event(
                    EventName.ERROR,
                    request,
                    "warning",
                    blocked=True,
                    rule_id=outcome.rule_id,
                    reason=outcome.reason,
                )
            )
            raise BudgetExceeded(
                outcome.reason or "call blocked by a before hook",
                limit=outcome.rule_id or "hook",
            )
        return routes

    def _inject(self, run_id: str, provider: str) -> Fault | None:
        """Draw a chaos fault for this try and count it; transient faults raise here."""
        fault = self.chaos.next_fault() if self.chaos.enabled else None
        if fault is not None:
            self.budget.usage(run_id).chaos_injected += 1
        if fault is not None and fault in TRANSIENT_FAULTS:
            Chaos.raise_transient(fault, provider)
        return fault

    def _finish(
        self, request: LLMRequest, model: str, response: LLMResponse, fault: Fault | None
    ) -> LLMResponse:
        """Apply a bad-JSON fault, price the response, add usage to the tracker and span."""
        if fault is Fault.BAD_JSON:
            response = Chaos.corrupt(response)
        if response.usage.cost_usd == 0:
            # Providers may echo a dated snapshot id, so price by what we asked for.
            prices = self.config.prices
            price = prices.get(model) or prices.get(response.model)
            response = response.model_copy(
                update={
                    "usage": response.usage.model_copy(
                        update={"cost_usd": _cost_of(response.usage, price)}
                    )
                }
            )
        self.budget.record(request.run_id, request.agent.value, response.provider, response.usage)
        span = current_span()
        if span is not None:
            span.add("input_tokens", response.usage.input_tokens)
            span.add("output_tokens", response.usage.output_tokens)
            span.add("cost_usd", response.usage.cost_usd)
            span.set(provider=response.provider, model=response.model)
        return response

    async def _on_error(
        self,
        request: LLMRequest,
        provider: str,
        model: str,
        exc: Exception,
        try_no: int,
        *,
        last_route: bool,
    ) -> bool:
        """Log one failed try. Returns True when the same route should retry. A refused
        connection marks the provider down, and retries only on the ``last_route``,
        since no other provider can take the call."""
        usage = self.budget.usage(request.run_id)
        transient = isinstance(exc, TransientLLMError)
        unreachable = isinstance(exc, TransientLLMError) and exc.unreachable
        cap = exc.max_retries if isinstance(exc, TransientLLMError) else None
        will_retry = (
            transient
            and not (unreachable and not last_route)
            and try_no < len(self.config.retry.delays_s)
            and (cap is None or try_no < cap)
        )
        await self.bus.emit(
            self._event(
                EventName.ERROR,
                request,
                "warning",
                provider=provider,
                model=model,
                error_type=type(exc).__name__,
                error=str(exc),
                status=exc.status if isinstance(exc, TransientLLMError) else None,
                retry=will_retry,
            )
        )
        if unreachable:
            await self._mark_down(request, provider, exc)
        if will_retry:
            self.check_run(request.run_id)  # no wait for a retry the stop would block
            usage.retries += 1
            await self.sleep(self.config.retry.delays_s[try_no])
        return will_retry

    async def _after(
        self,
        request: LLMRequest,
        response: LLMResponse,
        started: float,
        fallback: bool,
        fault: Fault | None,
    ) -> None:
        """Count fallbacks, persist usage and emit llm_call_after."""
        if fallback:
            self.budget.usage(request.run_id).fallbacks += 1
        await self._persist(request.run_id, response.usage)
        await self.bus.emit(
            self._event(
                EventName.LLM_CALL_AFTER,
                request,
                provider=response.provider,
                model=response.model,
                stop_reason=response.stop_reason,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=response.usage.cost_usd,
                duration_ms=round((time.monotonic() - started) * 1000, 1),
                fallback=fallback,
                chaos=fault.value if fault else None,
                thinking_dropped=response.thinking_dropped or None,
            )
        )

    async def _span(self, stack: AsyncExitStack, request: LLMRequest) -> None:
        """Open an llm span for the call when a tracer is set."""
        if self.tracer is not None:
            await stack.enter_async_context(
                self.tracer.span(
                    request.run_id,
                    f"llm:{request.agent.value}",
                    "llm",
                    task_id=request.task_id,
                    agent=request.agent.value,
                )
            )

    def _slot(self, provider: str) -> Any:
        """The provider's rate-limit slot, or a no-op when it has no limits."""
        limiter = self._limiters.get(provider)
        return limiter.slot() if limiter else nullcontext()

    # --- LLMClient ---------------------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """One model call with routing, retries, fallback, budgets and logging."""
        async with AsyncExitStack() as stack:
            await self._span(stack, request)
            routes = await self._preflight(request)
            primary = self._primary(request)
            effort = self._effort(request, primary)
            started = time.monotonic()
            last: Exception | None = None
            for i, (name, model) in enumerate(routes):
                fallback = (name, model) != (primary.provider, primary.model)
                last_route = i == len(routes) - 1
                provider = self.providers[name]
                try_no = 0
                while True:
                    self.check_run(request.run_id)
                    try:
                        async with self._global, self._slot(name):
                            fault = self._inject(request.run_id, name)
                            response = await provider.complete(request, model=model, effort=effort)
                    except _PASS_THROUGH:
                        raise
                    except Exception as exc:
                        last = exc
                        self.budget.usage(request.run_id).failed_calls += 1
                        if await self._on_error(
                            request, name, model, exc, try_no, last_route=last_route
                        ):
                            try_no += 1
                            continue
                        break
                    await self._mark_up(request, name)
                    response = self._finish(request, model, response, fault)
                    await self._after(request, response, started, fallback, fault)
                    return response
            assert last is not None
            raise last

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Retries and fallback apply until the first chunk arrives; after that an
        error propagates, since text already reached the caller."""
        async with AsyncExitStack() as stack:
            await self._span(stack, request)
            routes = await self._preflight(request)
            primary = self._primary(request)
            effort = self._effort(request, primary)
            started = time.monotonic()
            last: Exception | None = None
            for i, (name, model) in enumerate(routes):
                fallback = (name, model) != (primary.provider, primary.model)
                last_route = i == len(routes) - 1
                provider = self.providers[name]
                try_no = 0
                while True:
                    self.check_run(request.run_id)
                    sent = False
                    try:
                        async with self._global, self._slot(name):
                            fault = self._inject(request.run_id, name)
                            async for chunk in provider.stream(request, model=model, effort=effort):
                                if chunk.done and chunk.response is not None:
                                    await self._mark_up(request, name)
                                    response = self._finish(request, model, chunk.response, fault)
                                    await self._after(request, response, started, fallback, fault)
                                    yield StreamChunk(done=True, response=response)
                                    return
                                sent = True
                                yield chunk
                            raise TransientLLMError(
                                "stream ended without a final response", provider=name
                            )
                    except _PASS_THROUGH:
                        raise
                    except Exception as exc:
                        if sent:
                            raise
                        last = exc
                        self.budget.usage(request.run_id).failed_calls += 1
                        if await self._on_error(
                            request, name, model, exc, try_no, last_route=last_route
                        ):
                            try_no += 1
                            continue
                        break
            assert last is not None
            raise last


def _satisfies_protocol(gateway: Gateway) -> LLMClient:
    """Static check: mypy fails here if Gateway drifts from the LLMClient Protocol."""
    return gateway
