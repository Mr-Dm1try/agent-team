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

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> str:
    """The JSON object in a reply: inside a code fence, or from the first ``{``."""
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text.strip()


class Reply[T: BaseModel]:
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
            return Reply(schema.model_validate(json.loads(extract_json(text))), model, usage, None)
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
