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
   move to the next route at once.
5. Cost from the price table (by the requested model, then the model the provider
   reports), usage recorded, ``llm_call_after`` emitted.

Implements ``agent_team.llm.types.LLMClient``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import Any

from agent_team.core.config import Config, Route
from agent_team.core.control import KillSwitch
from agent_team.core.errors import BudgetExceeded, RunStopped, TransientLLMError
from agent_team.core.events import InMemoryEventBus
from agent_team.core.interfaces import EventBus
from agent_team.core.models import AgentName, Event, EventName
from agent_team.core.tracing import Tracer, current_span
from agent_team.llm.budget import (
    BudgetGuard,
    BudgetTracker,
    Clock,
    RateLimiter,
    RunUsage,
    Sleep,
    usage_from_events,
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
        chaos: Chaos | None = None,
        sleep: Sleep = asyncio.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        self.config = config
        self.providers = dict(providers)
        self.bus: EventBus = bus or InMemoryEventBus()
        self.tracer = tracer
        self.kill_switch = kill_switch
        self.chaos = chaos or Chaos(config.chaos)
        self.sleep = sleep
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

    def context_budget(self, agent: AgentName) -> int:
        """Max prompt tokens for the agent's model, from the routing file."""
        return self.config.context_budget(agent)

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
        override = self._overrides.get(request.task_id) if request.task_id else None
        return override or self.config.route_for(request.agent)

    def routes(self, request: LLMRequest) -> list[tuple[str, str]]:
        """(provider, model) pairs to try, in order, limited to known providers."""
        primary = self._primary(request)
        chain = [(primary.provider, primary.model)]
        for name in self.config.fallbacks:
            if name == primary.provider:
                continue
            model = self.config.fallback_models.get(name)
            if model is None and name == self.config.defaults.provider:
                model = self.config.defaults.model
            if model is not None:
                chain.append((name, model))
        return [(p, m) for p, m in chain if p in self.providers]

    def status(self, run_id: str) -> RunUsage:
        """Calls, tokens, cost, retries, fallbacks and blocked calls for one run."""
        return self.budget.usage(run_id)

    def restore_run(self, run_id: str, events_path: str | Path) -> RunUsage:
        """Seed a resumed run's budget from its ``events.jsonl``, so a restart doesn't
        reset spend and elapsed time to zero."""
        total, elapsed_s = usage_from_events(events_path)
        return self.budget.restore(run_id, total, elapsed_s)

    # --- helpers -----------------------------------------------------------------------

    def _event(
        self, name: EventName, request: LLMRequest, level: Any = "info", **data: Any
    ) -> Event:
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

    async def _preflight(self, request: LLMRequest) -> list[tuple[str, str]]:
        if self.kill_switch is not None:
            self.kill_switch.check(request.run_id)
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
        fault = self.chaos.next_fault() if self.chaos.enabled else None
        if fault is not None:
            self.budget.usage(run_id).chaos_injected += 1
        if fault is not None and fault in TRANSIENT_FAULTS:
            Chaos.raise_transient(fault, provider)
        return fault

    def _finish(
        self, request: LLMRequest, model: str, response: LLMResponse, fault: Fault | None
    ) -> LLMResponse:
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
        self, request: LLMRequest, provider: str, model: str, exc: Exception, try_no: int
    ) -> bool:
        """Log one failed try. Returns True when the same route should retry."""
        usage = self.budget.usage(request.run_id)
        transient = isinstance(exc, TransientLLMError)
        will_retry = transient and try_no < len(self.config.retry.delays_s)
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
        if will_retry:
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
        if fallback:
            self.budget.usage(request.run_id).fallbacks += 1
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
            )
        )

    async def _span(self, stack: AsyncExitStack, request: LLMRequest) -> None:
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
        limiter = self._limiters.get(provider)
        return limiter.slot() if limiter else nullcontext()

    # --- LLMClient ---------------------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """One model call with routing, retries, fallback, budgets and logging."""
        async with AsyncExitStack() as stack:
            await self._span(stack, request)
            routes = await self._preflight(request)
            primary = self._primary(request)
            started = time.monotonic()
            last: Exception | None = None
            for name, model in routes:
                fallback = (name, model) != (primary.provider, primary.model)
                provider = self.providers[name]
                try_no = 0
                while True:
                    if self.kill_switch is not None:
                        self.kill_switch.check(request.run_id)
                    try:
                        async with self._global, self._slot(name):
                            fault = self._inject(request.run_id, name)
                            response = await provider.complete(request, model=model)
                    except _PASS_THROUGH:
                        raise
                    except Exception as exc:
                        last = exc
                        self.budget.usage(request.run_id).failed_calls += 1
                        if await self._on_error(request, name, model, exc, try_no):
                            try_no += 1
                            continue
                        break
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
            started = time.monotonic()
            last: Exception | None = None
            for name, model in routes:
                fallback = (name, model) != (primary.provider, primary.model)
                provider = self.providers[name]
                try_no = 0
                while True:
                    if self.kill_switch is not None:
                        self.kill_switch.check(request.run_id)
                    sent = False
                    try:
                        async with self._global, self._slot(name):
                            fault = self._inject(request.run_id, name)
                            async for chunk in provider.stream(request, model=model):
                                if chunk.done and chunk.response is not None:
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
                        if await self._on_error(request, name, model, exc, try_no):
                            try_no += 1
                            continue
                        break
            assert last is not None
            raise last


def _satisfies_protocol(gateway: Gateway) -> LLMClient:
    """Static check: mypy fails here if Gateway drifts from the LLMClient Protocol."""
    return gateway
