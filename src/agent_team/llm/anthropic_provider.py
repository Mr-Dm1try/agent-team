"""Claude models through the official anthropic SDK.

The SDK's own retries are off (``max_retries=0``): the gateway owns retries, backoff
and fallback, so every attempt shows up in our logs. We don't send temperature.

Thinking is adaptive and the route's effort goes in ``output_config``. Thinking and
redacted_thinking blocks come back as ``ThinkingBlock`` and are sent back unchanged on
later calls, which a tool loop needs. That includes blocks another Claude model wrote
(after a fallback or STRONGER_MODEL): the API drops, unbilled, any block the current
model can't read, and the docs warn that stripping blocks ourselves can cause ordering
and signature errors. Blocks from other providers are left out.

With ``on_history_edit`` set, calls go through the beta Messages API with the
``thinking-binding-controls-2026-08-01`` header, so a block whose earlier history was
edited (compaction, a trimmed tool result) is handled as configured instead of the
API's default.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anthropic
import httpx2
from anthropic.types.beta import BetaMessage, BetaThinkingDroppedInputTransformation

from agent_team.core.config import AnthropicConfig
from agent_team.core.errors import TransientLLMError
from agent_team.core.models import Effort, Usage
from agent_team.llm.types import (
    ContentBlock,
    LLMRequest,
    LLMResponse,
    Message,
    StreamChunk,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)

#: Beta that lets a request choose what happens to a thinking block whose history
#: was edited.
BINDING_BETA = "thinking-binding-controls-2026-08-01"

_STOP: dict[str, Any] = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
}


def to_wire_messages(messages: list[Message], *, provider: str) -> list[dict[str, Any]]:
    """Our messages in the Messages API format. Thinking blocks written by
    ``provider`` go back exactly as received; any others are left out."""
    out: list[dict[str, Any]] = []
    for m in messages:
        blocks: list[dict[str, Any]] = []
        for b in m.content:
            if isinstance(b, ThinkingBlock):
                if b.provider == provider:
                    blocks.append(b.raw)
            elif isinstance(b, TextBlock):
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
        if blocks:
            out.append({"role": m.role, "content": blocks})
    return out


def from_wire_message(
    msg: anthropic.types.Message | anthropic.types.beta.BetaMessage, *, provider: str
) -> LLMResponse:
    """Translate an SDK message (plain or beta). Thinking and redacted_thinking blocks
    are kept whole, even when their text is empty (the default display), since the
    signature is what the API checks. Other block types are dropped. Only the beta
    message carries ``input_transformations``, the blocks the API dropped."""
    content: list[ContentBlock] = []
    for block in msg.content:
        if block.type in ("thinking", "redacted_thinking"):
            content.append(
                ThinkingBlock(provider=provider, model=msg.model, raw=block.to_dict(mode="json"))
            )
        elif block.type == "text":
            content.append(TextBlock(text=block.text))
        elif block.type == "tool_use":
            content.append(ToolUseBlock(id=block.id, name=block.name, arguments=dict(block.input)))
    dropped: list[BetaThinkingDroppedInputTransformation] = []
    if isinstance(msg, BetaMessage):
        dropped = [
            t
            for t in msg.input_transformations or []
            if isinstance(t, BetaThinkingDroppedInputTransformation)
        ]
    return LLMResponse(
        content=content,
        stop_reason=_STOP.get(msg.stop_reason or "", "other"),
        usage=Usage(input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens),
        provider=provider,
        model=msg.model,
        thinking_dropped=len(dropped),
    )


def translate_error(exc: Exception, provider: str) -> Exception:
    """Map SDK errors: rate limits, timeouts, connection errors and 5xx become
    TransientLLMError; everything else is returned unchanged. A stream that stalls or
    drops mid-reply raises the HTTP client's own error, which is transient too."""
    if isinstance(exc, anthropic.APIConnectionError | httpx2.TransportError):
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
        """Use ``config``; ``client`` replaces the SDK client in tests."""
        self.config = config or AnthropicConfig()
        self._client = client

    @property
    def client(self) -> Any:
        """The SDK client, created on first use."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(max_retries=0, timeout=self.config.timeout_s)
        return self._client

    @property
    def _binding(self) -> bool:
        """Whether to send the history-edit setting; it only applies when thinking is on."""
        return bool(self.config.on_history_edit) and self.config.thinking["type"] in (
            "adaptive",
            "enabled",
        )

    @property
    def _api(self) -> Any:
        """The beta Messages API when the binding beta is needed, else the plain one."""
        return self.client.beta.messages if self._binding else self.client.messages

    def _params(self, request: LLMRequest, model: str, effort: Effort | None) -> dict[str, Any]:
        """Keyword arguments for one Messages API call."""
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": to_wire_messages(request.messages, provider=self.name),
        }
        if request.system:
            params["system"] = request.system
        if request.tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in request.tools
            ]
        thinking = dict(self.config.thinking)
        if self._binding:
            thinking["block_binding"] = {"prefix_mismatch_behavior": self.config.on_history_edit}
            params["betas"] = [BINDING_BETA]
        params["thinking"] = thinking
        if effort is not None:
            params["output_config"] = {"effort": effort}
        return params

    async def complete(
        self, request: LLMRequest, *, model: str, effort: Effort | None = None
    ) -> LLMResponse:
        """One Messages API call. A call that may write a long reply is streamed and
        returns the final message, so the timeout bounds each read, not the whole reply."""
        params = self._params(request, model, effort)
        try:
            if request.max_tokens > self.config.stream_above_tokens:
                async with self._api.stream(**params) as s:
                    msg = await s.get_final_message()
            else:
                msg = await self._api.create(**params)
        except (anthropic.AnthropicError, httpx2.TransportError) as exc:
            raise translate_error(exc, self.name) from exc
        return from_wire_message(msg, provider=self.name)

    async def stream(
        self, request: LLMRequest, *, model: str, effort: Effort | None = None
    ) -> AsyncIterator[StreamChunk]:
        """One streamed Messages API call."""
        try:
            async with self._api.stream(**self._params(request, model, effort)) as s:
                async for text in s.text_stream:
                    yield StreamChunk(text=text)
                final = await s.get_final_message()
        except (anthropic.AnthropicError, httpx2.TransportError) as exc:
            raise translate_error(exc, self.name) from exc
        yield StreamChunk(done=True, response=from_wire_message(final, provider=self.name))
