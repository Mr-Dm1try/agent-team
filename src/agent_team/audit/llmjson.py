"""Ask a model for JSON matching a Pydantic schema, with the repair loop.

The validation errors go back to the model up to ``repairs`` times. If no valid
object comes back, ``None`` is returned and the caller fails closed.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from agent_team.core.models import AgentName, Usage
from agent_team.llm.types import LLMClient, LLMRequest, Message, TextBlock

_FENCE_LAZY = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_FENCE_GREEDY = re.compile(r"```(?:json)?\s*(.*)```", re.DOTALL)


def json_candidates(text: str) -> list[str]:
    """Places the JSON object in a reply may be, most likely first: a code fence (the
    greedy match survives backticks inside string values), then first ``{`` to last ``}``."""
    out: list[str] = []
    for pattern in (_FENCE_LAZY, _FENCE_GREEDY):
        m = pattern.search(text)
        if m:
            out.append(m.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        out.append(text[start : end + 1])
    out.append(text.strip())
    return list(dict.fromkeys(out))


def parse_reply[T: BaseModel](text: str, schema: type[T]) -> T:
    """The first candidate that validates; raises the first candidate's error otherwise."""
    first: Exception | None = None
    for candidate in json_candidates(text):
        try:
            return schema.model_validate(json.loads(candidate))
        except (ValueError, ValidationError) as exc:
            first = first or exc
    raise first or ValueError("empty reply")


class Reply[T: BaseModel]:
    """The parsed value (None when every attempt failed), the model that answered, the
    usage across attempts and the last parse error."""

    def __init__(self, value: T | None, model: str | None, usage: Usage, error: str | None):
        self.value = value
        self.model = model
        self.usage = usage
        self.error = error


async def ask_json[T: BaseModel](
    llm: LLMClient,
    schema: type[T],
    *,
    system: str,
    prompt: str,
    run_id: str,
    task_id: str | None,
    repairs: int = 2,
    max_tokens: int = 4096,
) -> Reply[T]:
    """Ask for one JSON object matching ``schema``, sending parse errors back up to
    ``repairs`` times. Model errors propagate; the caller decides how to fail closed."""
    messages = [Message.user(prompt)]
    usage = Usage()
    model: str | None = None
    error: str | None = None
    for _ in range(repairs + 1):
        response = await llm.complete(
            LLMRequest(
                agent=AgentName.AUDITOR,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                run_id=run_id,
                task_id=task_id,
            )
        )
        usage = usage + response.usage
        model = response.model
        text = response.text
        try:
            return Reply(parse_reply(text, schema), model, usage, None)
        except (ValueError, ValidationError) as exc:
            error = str(exc)[:2000]
        messages = [
            *messages,
            Message(role="assistant", content=[TextBlock(text=text or "(empty)")]),
            Message.user(
                "That reply is not valid JSON for the required schema. Errors:\n"
                f"{error}\nReply again with only the corrected JSON object."
            ),
        ]
    return Reply(None, model, usage, error)
