"""Usage tracking, budgets and rate limits for the gateway."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agent_team.core.config import BudgetConfig, Price
from agent_team.core.models import Event, EventName, HookOutcome, Usage

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


def cost_of(usage: Usage, price: Price | None) -> float:
    """USD for one call; 0 when the model has no price (e.g. a local model)."""
    if price is None:
        return 0.0
    return (
        usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000


class RunUsage(BaseModel):
    """What ``agent-team gateway status`` reports for one run."""

    model_config = ConfigDict(extra="forbid")
    run_id: str
    calls: int = 0
    failed_calls: int = 0
    retries: int = 0
    fallbacks: int = 0
    blocked: int = 0
    chaos_injected: int = 0
    total: Usage = Field(default_factory=Usage)
    by_agent: dict[str, Usage] = Field(default_factory=dict)
    by_provider: dict[str, Usage] = Field(default_factory=dict)
    started_at: float = 0.0  # monotonic seconds; not meaningful across processes


class BudgetTracker:
    """Per-run spend and elapsed time, checked against the configured caps."""

    def __init__(self, config: BudgetConfig, *, clock: Clock = time.monotonic) -> None:
        self.config = config
        self.clock = clock
        self._runs: dict[str, RunUsage] = {}

    def usage(self, run_id: str) -> RunUsage:
        """The run's counters, created (and its clock started) on first use."""
        if run_id not in self._runs:
            self._runs[run_id] = RunUsage(run_id=run_id, started_at=self.clock())
        return self._runs[run_id]

    def restore(self, run_id: str, total: Usage, elapsed_s: float) -> RunUsage:
        """Seed a run resumed in a new process with what it already spent."""
        u = self.usage(run_id)
        u.total = u.total + total
        u.started_at -= elapsed_s
        return u

    def record(self, run_id: str, agent: str, provider: str, usage: Usage) -> None:
        """Add one call's usage to the run, its agent and its provider."""
        u = self.usage(run_id)
        u.calls += 1
        u.total = u.total + usage
        u.by_agent[agent] = u.by_agent.get(agent, Usage()) + usage
        u.by_provider[provider] = u.by_provider.get(provider, Usage()) + usage

    def exceeded(self, run_id: str) -> tuple[str, str] | None:
        """(limit name, reason) for the first cap the run has reached, else None."""
        u = self.usage(run_id)
        c = self.config
        if c.max_usd_per_run is not None and u.total.cost_usd >= c.max_usd_per_run:
            return "max_usd_per_run", (
                f"run cost ${u.total.cost_usd:.4f} reached the ${c.max_usd_per_run:.2f} cap"
            )
        tokens = u.total.input_tokens + u.total.output_tokens
        if c.max_tokens_per_run is not None and tokens >= c.max_tokens_per_run:
            return "max_tokens_per_run", (
                f"run used {tokens} tokens, cap is {c.max_tokens_per_run}"
            )
        minutes = (self.clock() - u.started_at) / 60
        if c.max_minutes_per_run is not None and minutes >= c.max_minutes_per_run:
            return "max_minutes_per_run", (
                f"run has been calling models for {minutes:.1f} min, "
                f"cap is {c.max_minutes_per_run:g}"
            )
        return None


class BudgetGuard:
    """Before hook on ``llm_call_before`` that blocks calls past a cap."""

    def __init__(self, tracker: BudgetTracker) -> None:
        self.tracker = tracker

    async def __call__(self, event: Event) -> HookOutcome:
        hit = self.tracker.exceeded(event.run_id)
        if hit is None:
            return HookOutcome()
        limit, reason = hit
        return HookOutcome(allow=False, reason=reason, rule_id=limit)


class RateLimiter:
    """Caps concurrent calls and requests per minute; extra calls queue."""

    def __init__(
        self,
        *,
        max_concurrent: int | None = None,
        requests_per_minute: int | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None
        self.rpm = requests_per_minute
        self.clock = clock
        self.sleep = sleep
        self._sent: deque[float] = deque()
        self._rpm_lock = asyncio.Lock()

    async def _wait_rpm(self) -> None:
        if not self.rpm:
            return
        async with self._rpm_lock:
            while True:
                now = self.clock()
                while self._sent and now - self._sent[0] >= 60:
                    self._sent.popleft()
                if len(self._sent) < self.rpm:
                    self._sent.append(now)
                    return
                await self.sleep(60 - (now - self._sent[0]))

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one concurrent slot and one request of the per-minute allowance."""
        if self._sem is None:
            await self._wait_rpm()
            yield
            return
        async with self._sem:
            await self._wait_rpm()
            yield


def usage_from_events(path: str | Path) -> tuple[Usage, float]:
    """Total usage and elapsed seconds recorded in a run's ``events.jsonl``: the sum
    of ``llm_call_after`` usage and the time between the first and last event.
    Unreadable lines (e.g. one cut off by a crash) are skipped."""
    p = Path(path)
    if not p.exists():
        return Usage(), 0.0
    total = Usage()
    first: datetime | None = None
    last: datetime | None = None
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            event = Event.model_validate_json(line)
        except ValueError:
            continue
        first = first or event.ts
        last = event.ts
        if event.name is EventName.LLM_CALL_AFTER:
            d = event.data
            total = total + Usage(
                input_tokens=int(d.get("input_tokens", 0)),
                output_tokens=int(d.get("output_tokens", 0)),
                cost_usd=float(d.get("cost_usd", 0.0)),
            )
    elapsed = (last - first).total_seconds() if first and last else 0.0
    return total, elapsed
