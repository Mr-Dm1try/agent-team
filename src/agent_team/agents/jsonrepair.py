"""Pull a JSON object out of a model reply and validate it, with cheap local repairs.

Models wrap JSON in prose or code fences, leave trailing commas, or use smart quotes.
We fix those locally first, so a repair round trip to the model is only spent on real
schema errors (docs/DESIGN.md, OutputValidationError: up to 2 repairs).
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)
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
    out = [text.strip()]
    out += [m.group(1).strip() for m in _FENCE.finditer(text)]
    obj = _outermost_object(text)
    if obj:
        out.append(obj)
    return out


def extract_json(text: str) -> Any:
    """Return the parsed JSON value, or raise ValueError saying what went wrong."""
    if not text.strip():
        raise ValueError("the reply was empty; reply with one JSON object")
    last_error = "no JSON object found in the reply"
    for raw in _candidates(text):
        for attempt in (raw, _TRAILING_COMMA.sub(r"\1", raw.translate(_SMART))):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError as exc:
                last_error = f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
    raise ValueError(last_error)


def format_validation_errors(exc: ValidationError) -> list[str]:
    """Pydantic errors as short ``field: message`` lines for a repair prompt."""
    errors = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        errors.append(f"{loc}: {err['msg']}")
    return errors


def parse_output[M: BaseModel](text: str, model: type[M]) -> tuple[M | None, list[str]]:
    """Parse ``text`` into ``model``. Returns (value, []) or (None, errors)."""
    try:
        data = extract_json(text)
    except ValueError as exc:
        return None, [str(exc)]
    if not isinstance(data, dict):
        return None, ["the reply must be one JSON object, not a list or a scalar"]
    try:
        return model.model_validate(data), []
    except ValidationError as exc:
        return None, format_validation_errors(exc)
