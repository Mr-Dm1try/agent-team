"""Token counts for everything that has to know a prompt's size before sending it.

The provider's own count is the best evidence, so callers anchor on it: the last
response's ``usage.input_tokens`` covers everything sent up to then, and only what was
appended since is estimated (``Anchor``). A count that is implausibly low for the
characters sent (a provider that leaves cached tokens out, say) is neither anchored on
nor learned from. The estimate counts characters of the JSON that goes out, with non-ASCII kept
as is, at a chars-per-token ratio that ``Calibration`` learns per model from observed
usage, starting from ``CHARS_PER_TOKEN``. It errs high on purpose: running over a
context window is silent truncation or a rejected call.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

#: Characters per token before any usage is observed. Low, so the estimate comes out high.
CHARS_PER_TOKEN = 3
#: Plausible chars per token. A lower ratio is clamped up to MIN_RATIO. A higher one means
#: the count left part of the prompt out (a provider that counts only the tokens it
#: didn't have cached, say), so it teaches nothing and is never anchored on.
MIN_RATIO, MAX_RATIO = 1.0, 6.0
#: A learned ratio is used at this share, since appended text (code, JSON, non-Latin
#: scripts) can pack more tokens per character than the prompts it was learned from.
SAFETY = 0.85


def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """A high estimate of the tokens in ``text``."""
    return math.ceil(len(text) / chars_per_token)


def to_json(*parts: Any) -> str:
    """The parts as they go out: strings as is, everything else as JSON with non-ASCII
    text kept as is."""
    return "".join(p if isinstance(p, str) else json.dumps(p, ensure_ascii=False) for p in parts)


def estimate_json_tokens(*parts: Any, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """A high estimate over each part serialized as JSON, non-ASCII text kept as is."""
    return estimate_tokens(to_json(*parts), chars_per_token)


def fingerprint(*parts: Any) -> str:
    """A short hash of the parts as JSON, to tell whether a sent prefix was rewritten."""
    return hashlib.sha256(to_json(*parts).encode()).hexdigest()[:16]


class Calibration:
    """Chars per token for each model, learned from the provider's reported prompt
    tokens. It keeps the last few observations and uses the lowest, times ``SAFETY``,
    so estimates stay on the high side; a model not seen yet uses ``start``."""

    def __init__(self, start: float = CHARS_PER_TOKEN, *, keep: int = 8) -> None:
        """``start`` is the ratio before any usage is seen; ``keep`` observations count."""
        self.start = start
        self.keep = keep
        self._seen: dict[str, deque[float]] = {}

    def observe(self, model: str, chars: int, tokens: int) -> bool:
        """Record that ``chars`` characters of prompt came to ``tokens`` tokens. Returns
        whether the count was plausible enough to learn from (and to anchor on): no
        usage reported (zero tokens) or a partial count teaches nothing."""
        if chars <= 0 or tokens <= 0 or chars / tokens > MAX_RATIO:
            return False
        ratio = max(chars / tokens, MIN_RATIO)
        self._seen.setdefault(model, deque(maxlen=self.keep)).append(ratio)
        return True

    def ratio(self, model: str | None) -> float:
        """The ratio to estimate ``model``'s prompts with; with no model, the lowest
        over every model seen (for a caller that can't tell which model is next)."""
        seen = self._seen.get(model) if model else [r for s in self._seen.values() for r in s]
        return max(min(seen) * SAFETY, MIN_RATIO) if seen else self.start


@dataclass(frozen=True)
class Anchor:
    """What the provider counted for one call: the first ``sent`` messages (with the
    system prompt and tools) came to ``tokens`` prompt tokens. ``prefix`` optionally
    fingerprints those messages, for a caller that can't tell whether the history was
    rewritten since (a provider, unlike the tool loop that rewrites it itself)."""

    sent: int
    tokens: int
    prefix: str = ""
