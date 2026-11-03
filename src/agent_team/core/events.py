"""In-process event bus with before hooks (can block or rewrite) and after hooks
(observe only). Implements ``agent_team.core.interfaces.EventBus``."""

from __future__ import annotations

import importlib
import logging
from collections import defaultdict
from collections.abc import Callable

from agent_team.core.interfaces import AfterHook, BeforeHook, EventBus
from agent_team.core.models import Event, EventName, HookOutcome

log = logging.getLogger(__name__)


class InMemoryEventBus:
    """In-process event bus. Implements ``EventBus``."""

    def __init__(self) -> None:
        """Start with no hooks; before hooks are per event, after hooks may listen to all."""
        self._before: dict[EventName, list[BeforeHook]] = defaultdict(list)
        self._after: dict[EventName | None, list[AfterHook]] = defaultdict(list)

    def on_before(self, name: EventName, hook: BeforeHook) -> None:
        """Register a before hook for one event name."""
        self._before[name].append(hook)

    def on_after(self, name: EventName | None, hook: AfterHook) -> None:
        """Register an after hook; ``None`` means every event."""
        self._after[name].append(hook)

    async def check(self, event: Event) -> HookOutcome:
        """Run before hooks in registration order. The first block wins. A hook that
        rewrites the payload passes the rewritten event to the hooks after it. A hook
        that raises blocks the action (fail closed)."""
        replacement: dict[str, object] | None = None
        for hook in self._before.get(event.name, []):
            try:
                outcome = await hook(event)
            except Exception as exc:
                log.exception("before hook %r failed on %s", hook, event.name)
                return HookOutcome(allow=False, reason=f"before hook failed: {exc!r}")
            if not outcome.allow:
                return outcome
            if outcome.replacement is not None:
                replacement = outcome.replacement
                event = event.model_copy(update={"data": outcome.replacement})
        return HookOutcome(allow=True, replacement=replacement)

    async def emit(self, event: Event) -> None:
        """Deliver to every matching after hook. Hook errors are logged, never raised."""
        hooks = [*self._after.get(event.name, []), *self._after.get(None, [])]
        for hook in hooks:
            try:
                await hook(event)
            except Exception:
                log.exception("after hook %r failed on %s", hook, event.name)


def load_hooks(paths: list[str], bus: EventBus) -> None:
    """Load custom hooks from ``module:function`` paths in the config. Each function
    takes the bus and registers whatever hooks it wants."""
    for path in paths:
        module_name, _, attr = path.partition(":")
        if not attr:
            raise ValueError(f"hook path must look like 'module:function', got {path!r}")
        register: Callable[[EventBus], None] = getattr(importlib.import_module(module_name), attr)
        register(bus)
