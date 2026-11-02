"""Claude models through the official anthropic SDK.

The SDK's own retries are off (``max_retries=0``): the gateway owns retries, backoff
and fallback, so every attempt shows up in our logs. We don't send temperature.
Thinking is set per model from the config; our message format has no thinking
blocks, so the default config turns it off where the model allows that.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anthropic

from agent_team.core.config import AnthropicConfig
from agent_team.core.errors import TransientLLMError
from agent_team.core.models import Usage
from agent_team.llm.types import (
    ContentBlock,
    LLMRequest,
    LLMResponse,
    Message,
    StreamChunk,
    TextBlock,
    ToolUseBlock,
)

_STOP: dict[str, Any] = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
}


def to_wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Our messages in the Messages API format."""
    out: list[dict[str, Any]] = []
    for m in messages:
        blocks: list[dict[str, Any]] = []
        for b in m.content:
            if isinstance(b, TextBlock):
                blocks.append({"type": "text", "text": b.text})
            elif isinstance(b, ToolUseBlock):
                blocks.append(
                    {"type": "tool_use", "id": b.id, "name": b.name, "input": b.arguments}
                )
            else:
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b.tool_use_id,
                        "content": b.content,
                        "is_error": b.is_error,
                    }
                )
        out.append({"role": m.role, "content": blocks})
    return out


def from_wire_message(msg: Any, *, provider: str) -> LLMResponse:
    """Translate an SDK ``Message``. Blocks other than text and tool_use (e.g. empty
    thinking blocks) are dropped."""
    content: list[ContentBlock] = []
    for block in msg.content:
        if block.type == "text":
            content.append(TextBlock(text=block.text))
        elif block.type == "tool_use":
            content.append(ToolUseBlock(id=block.id, name=block.name, arguments=dict(block.input)))
    return LLMResponse(
        content=content,
        stop_reason=_STOP.get(msg.stop_reason or "", "other"),
        usage=Usage(input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens),
        provider=provider,
        model=msg.model,
    )


def translate_error(exc: Exception, provider: str) -> Exception:
    """Map SDK errors: rate limits, timeouts, connection errors and 5xx become
    TransientLLMError; everything else is returned unchanged."""
    if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
        return TransientLLMError(f"{type(exc).__name__}: {exc}", provider=provider)
    if isinstance(exc, anthropic.APIStatusError) and (
        exc.status_code in (408, 409, 429) or exc.status_code >= 500
    ):
        return TransientLLMError(
            f"{type(exc).__name__}: {exc.message}", provider=provider, status=exc.status_code
        )
    return exc


class AnthropicProvider:
    """Claude models via the anthropic SDK."""

    name = "anthropic"

    def __init__(self, config: AnthropicConfig | None = None, *, client: Any = None) -> None:
        self.config = config or AnthropicConfig()
        self._client = client

    @property
    def client(self) -> Any:
        """The SDK client, created on first use."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(max_retries=0, timeout=self.config.timeout_s)
        return self._client

    def _params(self, request: LLMRequest, model: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": to_wire_messages(request.messages),
        }
        if request.system:
            params["system"] = request.system
        if request.tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in request.tools
            ]
        thinking = self.config.thinking.get(model)
        if thinking is not None:
            params["extra_body"] = {"thinking": thinking}
        return params

    async def complete(self, request: LLMRequest, *, model: str) -> LLMResponse:
        """One Messages API call."""
        try:
            msg = await self.client.messages.create(**self._params(request, model))
        except anthropic.AnthropicError as exc:
            raise translate_error(exc, self.name) from exc
        return from_wire_message(msg, provider=self.name)

    async def stream(self, request: LLMRequest, *, model: str) -> AsyncIterator[StreamChunk]:
        """One streamed Messages API call."""
        try:
            async with self.client.messages.stream(**self._params(request, model)) as s:
                async for text in s.text_stream:
                    yield StreamChunk(text=text)
                final = await s.get_final_message()
        except anthropic.AnthropicError as exc:
            raise translate_error(exc, self.name) from exc
        yield StreamChunk(done=True, response=from_wire_message(final, provider=self.name))
