"""Local models through Ollama's OpenAI-compatible endpoint, using the openai SDK.

The context window is raised with Ollama's ``options.num_ctx``. The SDK's own retries
are off; the gateway retries and falls back.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import openai

from agent_team.core.config import OllamaConfig
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

_STOP: dict[str, Any] = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}


def to_chat_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Our messages in the OpenAI chat format, system prompt first."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        texts = [b.text for b in m.content if isinstance(b, TextBlock)]
        if m.role == "assistant":
            calls = [b for b in m.content if isinstance(b, ToolUseBlock)]
            msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) or None}
            if calls:
                msg["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in calls
                ]
            out.append(msg)
            continue
        for b in m.content:
            if b.type == "tool_result":
                content = f"ERROR: {b.content}" if b.is_error else b.content
                out.append({"role": "tool", "tool_call_id": b.tool_use_id, "content": content})
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


def parse_arguments(raw: str | None) -> dict[str, Any]:
    """Tool arguments as a dict. Unparseable JSON is kept under ``_invalid_json`` so
    the agent's repair loop can see it instead of crashing here."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_invalid_json": raw}
    return value if isinstance(value, dict) else {"_value": value}


def translate_error(exc: Exception, provider: str) -> Exception:
    """Map openai SDK errors; transient ones become TransientLLMError."""
    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return TransientLLMError(f"{type(exc).__name__}: {exc}", provider=provider)
    if isinstance(exc, openai.APIStatusError) and (
        exc.status_code in (408, 409, 429) or exc.status_code >= 500
    ):
        return TransientLLMError(
            f"{type(exc).__name__}: {exc.message}", provider=provider, status=exc.status_code
        )
    return exc


class OllamaProvider:
    """Local models via Ollama's OpenAI-compatible API."""

    name = "ollama"

    def __init__(self, config: OllamaConfig | None = None, *, client: Any = None) -> None:
        self.config = config or OllamaConfig()
        self._client = client

    @property
    def client(self) -> Any:
        """The SDK client, created on first use."""
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                base_url=self.config.base_url,
                api_key="ollama",
                max_retries=0,
                timeout=self.config.timeout_s,
            )
        return self._client

    def _params(self, request: LLMRequest, model: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": to_chat_messages(request.system, request.messages),
            "extra_body": {"options": {"num_ctx": self.config.num_ctx}},
        }
        if request.tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in request.tools
            ]
        return params

    async def complete(self, request: LLMRequest, *, model: str) -> LLMResponse:
        """One chat completion call."""
        try:
            resp = await self.client.chat.completions.create(**self._params(request, model))
        except openai.OpenAIError as exc:
            raise translate_error(exc, self.name) from exc
        choice = resp.choices[0]
        content: list[ContentBlock] = []
        if choice.message.content:
            content.append(TextBlock(text=choice.message.content))
        for call in choice.message.tool_calls or []:
            content.append(
                ToolUseBlock(
                    id=call.id,
                    name=call.function.name,
                    arguments=parse_arguments(call.function.arguments),
                )
            )
        usage = resp.usage
        return LLMResponse(
            content=content,
            stop_reason=_STOP.get(choice.finish_reason or "", "other"),
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
            provider=self.name,
            model=resp.model or model,
        )

    async def stream(self, request: LLMRequest, *, model: str) -> AsyncIterator[StreamChunk]:
        """One streamed chat completion; tool call deltas are joined."""
        params = self._params(request, model)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        text: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish: str | None = None
        usage = Usage()
        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.usage is not None:
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish = choice.finish_reason or finish
                delta = choice.delta
                if delta.content:
                    text.append(delta.content)
                    yield StreamChunk(text=delta.content)
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
        except openai.OpenAIError as exc:
            raise translate_error(exc, self.name) from exc
        content: list[ContentBlock] = []
        if text:
            content.append(TextBlock(text="".join(text)))
        for i in sorted(calls):
            c = calls[i]
            content.append(
                ToolUseBlock(
                    id=c["id"] or f"call_{i}",
                    name=c["name"],
                    arguments=parse_arguments(c["arguments"]),
                )
            )
        yield StreamChunk(
            done=True,
            response=LLMResponse(
                content=content,
                stop_reason=_STOP.get(finish or "", "other"),
                usage=usage,
                provider=self.name,
                model=model,
            ),
        )
