"""Parse and validate structured model output. We validate every response ourselves,
whatever the provider promises; a failure raises OutputValidationError with readable
errors that the agent's repair loop sends back to the model."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from agent_team.core.errors import OutputValidationError

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Find the JSON value in a reply: a fenced block, the whole text, or the span
    from the first ``{`` or ``[`` to the matching last bracket."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)]
    candidates.append(text)
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    last: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate.strip())
        except json.JSONDecodeError as exc:
            last = exc
    raise OutputValidationError(
        "response is not valid JSON",
        errors=[f"invalid JSON: {last.msg} at line {last.lineno} column {last.colno}"]
        if last
        else ["no JSON found"],
    )


def parse_output[T: BaseModel](text: str, schema: type[T]) -> T:
    """Parse ``text`` into ``schema``; raises OutputValidationError with readable errors."""
    data = extract_json(text)
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        errors = [
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        ]
        raise OutputValidationError(
            f"response doesn't match {schema.__name__}", errors=errors
        ) from exc
