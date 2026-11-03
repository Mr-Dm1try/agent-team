"""Pull a JSON object out of a model reply and validate it, with cheap local repairs.

Every response is validated here, whatever the provider promises. Models wrap JSON in
prose or code fences, leave trailing commas, or use smart quotes; those are fixed
locally first, so a repair round trip to the model is only spent on real schema
errors. The agents' loop and the auditor both send the errors back to the model.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)
# A greedy fence survives backticks inside string values.
_FENCE_GREEDY = re.compile(r"```(?:json|JSON)?\s*\n?(.*)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_SMART = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})


def _outermost_object(text: str) -> str | None:
    """The first balanced {...} span, ignoring braces inside strings."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _candidates(text: str) -> list[str]:
    """Strings that might hold the JSON, most likely first: the whole reply, each
    fenced block, the greedy fence, and the outermost object."""
    out = [text.strip()]
    out += [m.group(1).strip() for m in _FENCE.finditer(text)]
    greedy = _FENCE_GREEDY.search(text)
    if greedy:
        out.append(greedy.group(1).strip())
    obj = _outermost_object(text)
    if obj:
        out.append(obj)
    return list(dict.fromkeys(c for c in out if c))


def _parsed(text: str) -> tuple[list[Any], str]:
    """Every value the candidates parse to, in order, and the last parse error."""
    values: list[Any] = []
    last_error = "no JSON object found in the reply"
    for raw in _candidates(text):
        for attempt in dict.fromkeys((raw, _TRAILING_COMMA.sub(r"\1", raw.translate(_SMART)))):
            try:
                values.append(json.loads(attempt))
                break
            except json.JSONDecodeError as exc:
                last_error = f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
    return values, last_error


def extract_json(text: str) -> Any:
    """Return the first JSON value found in ``text``, or raise ValueError saying what
    went wrong."""
    if not text.strip():
        raise ValueError("the reply was empty; reply with one JSON object")
    values, error = _parsed(text)
    if not values:
        raise ValueError(error)
    return values[0]


def format_validation_errors(exc: ValidationError) -> list[str]:
    """Pydantic errors as short ``field: message`` lines for a repair prompt."""
    errors = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        errors.append(f"{loc}: {err['msg']}")
    return errors


def parse_output[M: BaseModel](text: str, model: type[M]) -> tuple[M | None, list[str]]:
    """Parse ``text`` into ``model``: the first candidate object that validates wins.
    Returns (value, []) or (None, the errors of the first object found)."""
    if not text.strip():
        return None, ["the reply was empty; reply with one JSON object"]
    values, error = _parsed(text)
    objects = [v for v in values if isinstance(v, dict)]
    if not objects:
        if values:
            return None, ["the reply must be one JSON object, not a list or a scalar"]
        return None, [error]
    first: list[str] | None = None
    for data in objects:
        try:
            return model.model_validate(data), []
        except ValidationError as exc:
            first = first or format_validation_errors(exc)
    return None, first or [error]
