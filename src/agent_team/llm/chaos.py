"""Fault injection for ``--chaos``. Sits in the gateway in front of every provider,
real or fake, so retries, fallback and the repair loop can be shown on demand.

Spec strings accepted by ``parse_chaos``:

- ``"0.3"``: every fault kind, 30% of calls
- ``"rate_limit,timeout"``: those kinds, at the default rate (25%)
- ``"rate_limit,bad_json@0.5"``: those kinds, 50% of calls
- ``"script:rate_limit,ok,bad_json"``: exactly that sequence, one entry per call
"""

from __future__ import annotations

import random
from enum import StrEnum

from agent_team.core.config import ChaosSettings
from agent_team.core.errors import TransientLLMError
from agent_team.llm.types import LLMResponse, TextBlock

DEFAULT_RATE = 0.25


class Fault(StrEnum):
    """The faults ``--chaos`` can inject."""

    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    BAD_JSON = "bad_json"


TRANSIENT_FAULTS = frozenset({Fault.RATE_LIMIT, Fault.TIMEOUT, Fault.SERVER_ERROR})
BAD_JSON_TEXT = '{"chaos": "this response was cut off on purpose'


def parse_chaos(spec: str) -> ChaosSettings:
    """Read a ``--chaos`` value (see the module docstring for the forms)."""
    spec = spec.strip()
    if not spec:
        return ChaosSettings()
    if spec.startswith("script:"):
        items = [s.strip() for s in spec[len("script:") :].split(",") if s.strip()]
        for item in items:
            if item != "ok":
                Fault(item)
        return ChaosSettings(script=items)
    try:
        return ChaosSettings(rate=_rate(spec), faults=[f.value for f in Fault])
    except ValueError:
        pass
    kinds, _, rate = spec.partition("@")
    faults = [Fault(k.strip()).value for k in kinds.split(",") if k.strip()]
    return ChaosSettings(rate=_rate(rate) if rate else DEFAULT_RATE, faults=faults)


def _rate(text: str) -> float:
    """Parse a fault rate and check it is between 0 and 1."""
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"chaos rate must be between 0 and 1, got {value}")
    return value


class Chaos:
    """Decides, per call, whether and which fault to inject."""

    def __init__(self, settings: ChaosSettings | None = None) -> None:
        """Seed the random source and load any scripted faults."""
        self.settings = settings or ChaosSettings()
        self._rng = random.Random(self.settings.seed)
        self._script = list(self.settings.script)
        self._faults = [Fault(f) for f in self.settings.faults] or list(Fault)
        self.injected: list[Fault] = []

    @property
    def enabled(self) -> bool:
        """Whether any fault can be injected."""
        return bool(self._script) or self.settings.rate > 0

    def next_fault(self) -> Fault | None:
        """Decide the fault for the next call (None for a clean call)."""
        fault: Fault | None
        if self._script:
            item = self._script.pop(0)
            fault = None if item == "ok" else Fault(item)
        elif self.settings.rate > 0 and self._rng.random() < self.settings.rate:
            fault = self._rng.choice(self._faults)
        else:
            fault = None
        if fault is not None:
            self.injected.append(fault)
        return fault

    @staticmethod
    def raise_transient(fault: Fault, provider: str) -> None:
        """Raise the TransientLLMError a real provider would for this fault."""
        status = {Fault.RATE_LIMIT: 429, Fault.SERVER_ERROR: 503, Fault.TIMEOUT: None}[fault]
        raise TransientLLMError(f"chaos: injected {fault.value}", provider=provider, status=status)

    @staticmethod
    def corrupt(response: LLMResponse) -> LLMResponse:
        """Replace the content with JSON that doesn't parse."""
        return response.model_copy(
            update={"content": [TextBlock(text=BAD_JSON_TEXT)], "stop_reason": "end_turn"}
        )
