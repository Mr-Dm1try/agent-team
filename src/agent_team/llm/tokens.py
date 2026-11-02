"""One token estimate for everything that has to guess a prompt's size before sending.

It counts characters of the JSON that goes out, with non-ASCII kept as is, and errs
high on purpose: running over a context window is silent truncation or a rejected call.
"""

from __future__ import annotations

import json
from typing import Any

#: Characters per token for the estimate. Low, so the estimate comes out high.
CHARS_PER_TOKEN = 3


def estimate_tokens(text: str, chars_per_token: int = CHARS_PER_TOKEN) -> int:
    """A high estimate of the tokens in ``text``."""
    return -(-len(text) // chars_per_token)


def estimate_json_tokens(*parts: Any, chars_per_token: int = CHARS_PER_TOKEN) -> int:
    """A high estimate over each part serialized as JSON, non-ASCII text kept as is."""
    sent = "".join(p if isinstance(p, str) else json.dumps(p, ensure_ascii=False) for p in parts)
    return estimate_tokens(sent, chars_per_token)
