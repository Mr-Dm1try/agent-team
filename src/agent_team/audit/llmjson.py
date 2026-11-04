"""Ask a model for JSON matching a Pydantic schema, with the repair loop.

The validation errors go back to the model up to ``repairs`` times. If no valid
object comes back, ``None`` is returned and the caller fails closed.
"""

from __future__ import annotations

from pydantic import BaseModel

from agent_team.core.models import AgentName, Usage
from agent_team.llm.structured import parse_output
from agent_team.llm.types import LLMClient, LLMRequest, Message, TextBlock


def parse_reply[T: BaseModel](text: str, schema: type[T]) -> T:
    """The reply parsed into ``schema``; raises ValueError with the parse errors."""
    value, errors = parse_output(text, schema)
    if value is None:
        raise ValueError("; ".join(errors))
    return value


class Reply[T: BaseModel]:
    """The parsed value (None when every attempt failed), the model that answered, the
    usage across attempts and the last parse error."""

    def __init__(self, value: T | None, model: str | None, usage: Usage, error: str | None):
        """Hold one ask_json result."""
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
) -> Reply[T]:
    """Ask for one JSON object matching ``schema``, sending parse errors back up to
    ``repairs`` times. Model errors propagate; the caller decides how to fail closed.
    ``max_tokens`` stays at the request default, since it also covers thinking; only
    text blocks of the reply are parsed, so thinking blocks are ignored."""
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
                run_id=run_id,
                task_id=task_id,
            )
        )
        usage = usage + response.usage
        model = response.model
        text = response.text
        try:
            return Reply(parse_reply(text, schema), model, usage, None)
        except ValueError as exc:
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
