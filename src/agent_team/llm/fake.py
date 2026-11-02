"""Scripted provider for tests and the fake-model demo run.

Turns are consumed in order. A turn with ``agent`` set only answers that agent's
requests, so one script can drive several agents running in parallel. A turn may
carry a ``fault`` instead of a response (rate_limit, timeout, server_error, bad_json).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from agent_team.core.config import Effort
from agent_team.core.errors import TransientLLMError
from agent_team.core.models import AgentName, Usage
from agent_team.llm.chaos import BAD_JSON_TEXT, Fault
from agent_team.llm.types import (
    ContentBlock,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    TextBlock,
    ToolUseBlock,
)


class FakeScriptExhausted(RuntimeError):
    """No scripted turn matched the request. Always a test bug."""


class FakeToolCall(BaseModel):
    """A tool call in a scripted turn."""

    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class FakeTurn(BaseModel):
    """One scripted model response, or a fault."""

    model_config = ConfigDict(extra="forbid")
    agent: AgentName | None = None
    text: str = ""
    #: Convenience: serialised to JSON and used as ``text``.
    json_output: Any = None
    tool_calls: list[FakeToolCall] = Field(default_factory=list)
    fault: Fault | None = None
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "other"] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


ScriptItem = FakeTurn | LLMResponse | BaseException | str
Responder = Callable[[LLMRequest], "FakeTurn | LLMResponse | str"]


def estimate_tokens(text: str) -> int:
    """Rough token count: four characters per token."""
    return max(1, len(text) // 4)


def _request_chars(request: LLMRequest) -> int:
    return len(request.system) + len(request.model_dump_json(include={"messages", "tools"}))


class FakeProvider:
    """Provider that replays a script; records every request."""

    def __init__(
        self,
        turns: Sequence[ScriptItem] = (),
        *,
        name: str = "fake",
        responder: Responder | None = None,
    ) -> None:
        self.name = name
        self._turns: list[ScriptItem] = list(turns)
        self.responder = responder
        self.requests: list[LLMRequest] = []
        self.models: list[str] = []
        self.efforts: list[Effort | None] = []
        self._calls = 0

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> FakeProvider:
        """Load a script from YAML or JSON: a list of FakeTurn mappings."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
        return cls([FakeTurn.model_validate(t) for t in raw], **kwargs)

    def add(self, *items: ScriptItem) -> None:
        """Append items to the script."""
        self._turns.extend(items)

    @property
    def remaining(self) -> int:
        """How many scripted items are left."""
        return len(self._turns)

    def _take(self, request: LLMRequest) -> ScriptItem:
        for i, item in enumerate(self._turns):
            agent = item.agent if isinstance(item, FakeTurn) else None
            if agent is None or agent is request.agent:
                return self._turns.pop(i)
        if self.responder is not None:
            return self.responder(request)
        raise FakeScriptExhausted(
            f"fake provider has no scripted turn for agent {request.agent.value!r}"
        )

    def _build(self, item: ScriptItem, request: LLMRequest, model: str) -> LLMResponse:
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, LLMResponse):
            return item
        turn = FakeTurn(text=item) if isinstance(item, str) else item
        if turn.fault is not None and turn.fault is not Fault.BAD_JSON:
            status = {Fault.RATE_LIMIT: 429, Fault.SERVER_ERROR: 503}.get(turn.fault)
            raise TransientLLMError(f"fake: {turn.fault.value}", provider=self.name, status=status)
        text = turn.text
        if turn.json_output is not None:
            text = json.dumps(turn.json_output)
        if turn.fault is Fault.BAD_JSON:
            text = BAD_JSON_TEXT
        content: list[ContentBlock] = []
        if text:
            content.append(TextBlock(text=text))
        for call in turn.tool_calls:
            self._calls += 1
            content.append(
                ToolUseBlock(
                    id=call.id or f"call_{self._calls}", name=call.name, arguments=call.arguments
                )
            )
        stop = turn.stop_reason or ("tool_use" if turn.tool_calls else "end_turn")
        usage = Usage(
            input_tokens=turn.input_tokens or estimate_tokens("x" * _request_chars(request)),
            output_tokens=turn.output_tokens or estimate_tokens(text or "x"),
        )
        return LLMResponse(
            content=content,
            stop_reason=stop,
            usage=usage,
            provider=self.name,
            model=model,
        )

    async def complete(
        self, request: LLMRequest, *, model: str, effort: Effort | None = None
    ) -> LLMResponse:
        """Answer with the next matching scripted turn. The fake never writes
        thinking blocks and takes no action on those in the request."""
        self.requests.append(request)
        self.models.append(model)
        self.efforts.append(effort)
        return self._build(self._take(request), request, model)

    async def stream(
        self, request: LLMRequest, *, model: str, effort: Effort | None = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream the next matching turn in small chunks, then the final response."""
        response = await self.complete(request, model=model, effort=effort)
        text = response.text
        for i in range(0, len(text), 16):
            yield StreamChunk(text=text[i : i + 16])
        yield StreamChunk(done=True, response=response)
