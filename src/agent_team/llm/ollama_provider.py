"""Local models through Ollama's native ``/api/chat``, over httpx.

The native API is used because it honours ``options.num_ctx``; the OpenAI-compatible
endpoint ignores it and runs every model at the server's default window (4,096 tokens
unless ``OLLAMA_CONTEXT_LENGTH`` is set). The harness also learns the window the model
really has: ``/api/show`` gives the model's own limit, and ``/api/ps`` the context it
was loaded with after the first call. The window used is the smallest of those and
``num_ctx``. ``truncate`` and ``shift`` are off, so a prompt that overflows is an error
rather than a silent cut, and that error is ``ContextTooLong``.

The route's effort is sent as ``think`` (low, medium or high; xhigh and max become high)
to the models in ``reasoning_models``, such as gpt-oss; other models get none. Their
reasoning comes back apart from the answer and is not kept: Ollama has no way to take
it back in the history, so it thinks afresh on every call, and Claude's thinking blocks
are dropped too. The gateway retries and falls back; this client doesn't retry.

A prompt is still capped before sending, so the reply has room. Its size is anchored on
what Ollama reported for the same task's previous call, with only the messages appended
since estimated, at a chars-per-token ratio learned per model. A count too low for the
characters sent is not used. A reported prompt within 2% of the window and past the room
the cap leaves is logged as a warning and not learned from, and the loaded window is
checked again.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

import httpx

from agent_team.core.config import OllamaConfig
from agent_team.core.errors import ContextTooLong, ProviderError, TransientLLMError
from agent_team.core.models import Effort, Usage
from agent_team.llm.ollama_api import api_root, error_text, loaded_context, model_context
from agent_team.llm.tokens import (
    CHARS_PER_TOKEN,
    Anchor,
    Calibration,
    estimate_json_tokens,
    fingerprint,
    to_json,
)
from agent_team.llm.types import (
    ContentBlock,
    LLMRequest,
    LLMResponse,
    Message,
    StreamChunk,
    TextBlock,
    ToolUseBlock,
)

log = logging.getLogger(__name__)

#: A reported prompt this close to the window was probably cut by Ollama.
NEAR_FULL = 0.98
#: Anchors kept, one per task, model and agent; the oldest go first.
MAX_ANCHORS = 256
#: Words in an Ollama error that mean the prompt didn't fit the window.
_OVERFLOW = ("context length", "context window", "exceeds the context")
#: Ollama's 500 when the model wrote a tool call it couldn't parse. A retry sometimes
#: gets a clean call; more than one rarely does, so it then falls back.
_BAD_TOOL_CALL = "error parsing tool call"

_REASONING_EFFORT: dict[Effort, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def to_chat_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Our messages in Ollama's chat format, system prompt first. Every message has
    string content (Ollama rejects null), tool results carry the tool's name, and
    thinking blocks are left out."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    names: dict[str, str] = {}
    for m in messages:
        texts = [b.text for b in m.content if isinstance(b, TextBlock)]
        if m.role == "assistant":
            calls = [b for b in m.content if isinstance(b, ToolUseBlock)]
            msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
            if calls:
                names.update((c.id, c.name) for c in calls)
                msg["tool_calls"] = [
                    {"id": c.id, "function": {"name": c.name, "arguments": c.arguments}}
                    for c in calls
                ]
            out.append(msg)
            continue
        for b in m.content:
            if b.type == "tool_result":
                content = f"ERROR: {b.content}" if b.is_error else b.content
                result: dict[str, Any] = {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": b.tool_use_id,
                }
                if b.tool_use_id in names:
                    result["tool_name"] = names[b.tool_use_id]
                out.append(result)
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


def prompt_tokens(body: dict[str, Any], chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """A high estimate of the prompt's tokens over the JSON that is sent, with
    non-ASCII text kept as is. High on purpose: running over is a refused call."""
    return estimate_json_tokens(
        body["messages"], body.get("tools", []), chars_per_token=chars_per_token
    )


def as_arguments(value: Any) -> dict[str, Any]:
    """Tool arguments as a dict. A string that isn't a JSON object is kept under
    ``_invalid_json`` so the agent's repair loop can see it instead of crashing here."""
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"_invalid_json": value}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return {"_value": value}


def status_error(status: int, message: str, provider: str) -> Exception:
    """The error for a failed call with HTTP ``status`` and Ollama's ``message``: an
    overflow is ContextTooLong, a garbled tool call is retried once, a 408, 409, 429 or
    5xx is transient, and any other refusal is a ProviderError."""
    text = f"HTTP {status}: {message}"
    lowered = message.lower()
    if any(word in lowered for word in _OVERFLOW):
        return ContextTooLong(text, provider=provider)
    if _BAD_TOOL_CALL in lowered:
        return TransientLLMError(text, provider=provider, status=status, max_retries=1)
    if status in (408, 409, 429) or status >= 500:
        return TransientLLMError(text, provider=provider, status=status)
    return ProviderError(text, provider=provider, status=status)


def translate_error(exc: Exception, provider: str) -> Exception:
    """Map httpx errors: a refused connection is transient and marks Ollama down (it
    isn't running), any other transport failure or timeout is a transient blip."""
    if isinstance(exc, httpx.ConnectError):
        return TransientLLMError(
            f"{type(exc).__name__}: {exc}", provider=provider, unreachable=True
        )
    if isinstance(exc, httpx.TransportError):  # timeouts, resets, dropped streams
        return TransientLLMError(f"{type(exc).__name__}: {exc}", provider=provider)
    return exc


def _usage(data: dict[str, Any]) -> Usage:
    """Token counts from a final reply; ``prompt_eval_count`` covers the whole prompt,
    cached tokens included."""
    return Usage(
        input_tokens=int(data.get("prompt_eval_count") or 0),
        output_tokens=int(data.get("eval_count") or 0),
    )


def _tool_calls(message: dict[str, Any]) -> list[ToolUseBlock]:
    """The tool calls in a reply message, each with an id (Ollama may leave it out)."""
    out: list[ToolUseBlock] = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        out.append(
            ToolUseBlock(
                id=call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                name=str(fn.get("name") or ""),
                arguments=as_arguments(fn.get("arguments")),
            )
        )
    return out


def _stop(done_reason: str | None, calls: list[ToolUseBlock]) -> Any:
    """Our stop reason: a tool call wins, ``length`` is the token cap, ``stop`` an end."""
    if calls:
        return "tool_use"
    return {"stop": "end_turn", "length": "max_tokens"}.get(done_reason or "", "other")


class OllamaProvider:
    """Local models via Ollama's native chat API."""

    name = "ollama"

    def __init__(
        self,
        config: OllamaConfig | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Use ``config``; ``transport`` replaces the network in tests."""
        self.config = config or OllamaConfig()
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self.calibration = Calibration(self.config.chars_per_token)
        self._anchors: OrderedDict[str, Anchor] = OrderedDict()
        #: The window each model really has, once Ollama has said.
        self._windows: dict[str, int] = {}
        #: Models whose loaded context has been checked with ``/api/ps``.
        self._checked: set[str] = set()

    @property
    def client(self) -> httpx.AsyncClient:
        """The HTTP client, created on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=api_root(self.config.base_url),
                timeout=self.config.timeout_s,
                transport=self._transport,
            )
        return self._client

    async def window(self, model: str) -> int:
        """The context window to plan ``model``'s calls for: ``num_ctx``, cut to the
        model's own limit from ``/api/show`` and to the context it is loaded with when
        ``/api/ps`` has said. Until Ollama answers, ``num_ctx`` alone."""
        known = self._windows.get(model)
        if known is not None:
            return known
        limit = await model_context(self.client, model)
        if limit is None:
            return self.config.num_ctx
        if limit < self.config.num_ctx:
            log.warning(
                "ollama %s: its window is %d tokens, less than num_ctx %d",
                model,
                limit,
                self.config.num_ctx,
            )
        self._windows[model] = min(limit, self.config.num_ctx)
        return self._windows[model]

    async def _check_loaded(self, model: str) -> None:
        """Cut the window to the context ``model`` is loaded with, if smaller, and
        warn: Ollama would cut prompts that fit the planned window."""
        loaded = await loaded_context(self.client, model)
        if loaded is None:
            return
        self._checked.add(model)
        planned = self._windows.get(model, self.config.num_ctx)
        if loaded < planned:
            log.warning(
                "ollama %s is loaded with a %d-token context, not %d; prompts are capped to fit it",
                model,
                loaded,
                planned,
            )
            self._windows[model] = loaded

    async def _body(
        self, request: LLMRequest, model: str, effort: Effort | None, *, stream: bool
    ) -> tuple[dict[str, Any], float]:
        """The JSON body for one chat call, and its timeout."""
        window = await self.window(model)
        body: dict[str, Any] = {
            "model": model,
            "messages": to_chat_messages(request.system, request.messages),
            "stream": stream,
            "truncate": False,
            "shift": False,
        }
        if effort is not None and model.split(":", 1)[0] in self.config.reasoning_models:
            body["think"] = _REASONING_EFFORT[effort]
        if request.tools:
            body["tools"] = [
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
        prompt = self._prompt_tokens(request, model, body)
        max_tokens = self._max_tokens(prompt, request.max_tokens, window)
        body["options"] = {"num_ctx": window, "num_predict": max_tokens}
        # A long reply at local speeds can outlast the plain timeout.
        timeout = max(self.config.timeout_s, max_tokens / self.config.min_tokens_per_s)
        return body, timeout

    @staticmethod
    def _key(request: LLMRequest, model: str) -> str:
        """Whose anchor a request uses: one per task (or run), agent and model."""
        return f"{request.run_id}:{request.task_id}:{request.agent.value}:{model}"

    @staticmethod
    def _prefix(request: LLMRequest, body: dict[str, Any], sent: int) -> str:
        """The fingerprint of the system prompt, tools and first ``sent`` messages."""
        head = [m.model_dump(mode="json") for m in request.messages[:sent]]
        return fingerprint(request.system, body.get("tools", []), head)

    def _prompt_tokens(self, request: LLMRequest, model: str, body: dict[str, Any]) -> int:
        """The prompt's size: Ollama's count for the same task's last call plus an
        estimate of the messages appended since, or a full estimate when there is no
        anchor or the history it counted was rewritten (compaction)."""
        ratio = self.calibration.ratio(model)
        anchor = self._anchors.get(self._key(request, model))
        if (
            anchor is not None
            and anchor.sent <= len(request.messages)
            and anchor.prefix == self._prefix(request, body, anchor.sent)
        ):
            appended = to_chat_messages("", request.messages[anchor.sent :])
            return anchor.tokens + estimate_json_tokens(appended, chars_per_token=ratio)
        return prompt_tokens(body, ratio)

    async def _observe(
        self, request: LLMRequest, model: str, body: dict[str, Any], usage: Usage
    ) -> None:
        """Learn from Ollama's count of a prompt. A prompt that filled the window past
        the cap's room was probably cut: warn, learn nothing and check the loaded
        context again. Otherwise, if the count is plausible, calibrate the model's
        ratio and anchor the task's next call on it."""
        tokens = usage.input_tokens
        key = self._key(request, model)
        window = int(body["options"]["num_ctx"])
        if model not in self._checked:
            await self._check_loaded(model)
        if tokens <= 0:
            return
        room = window - self.config.min_output_tokens
        if tokens >= NEAR_FULL * window and tokens > room:
            self._anchors.pop(key, None)  # a cut prompt's count is no measure of it
            log.warning(
                "ollama %s: prompt of %d tokens is within %d%% of the %d-token window; "
                "its start was probably cut",
                model,
                tokens,
                round((1 - NEAR_FULL) * 100),
                window,
            )
            self._checked.discard(model)
            return
        chars = len(to_json(body["messages"], body.get("tools", [])))
        if not self.calibration.observe(model, chars, tokens):
            self._anchors.pop(key, None)  # a partial count: estimate the next call in full
            return
        self._anchors[key] = Anchor(
            sent=len(request.messages),
            tokens=tokens,
            prefix=self._prefix(request, body, len(request.messages)),
        )
        self._anchors.move_to_end(key)
        while len(self._anchors) > MAX_ANCHORS:
            self._anchors.popitem(last=False)

    def _max_tokens(self, prompt: int, wanted: int, window: int) -> int:
        """``wanted`` cut down so a prompt of ``prompt`` tokens plus the reply fit the
        ``window``. Raises ContextTooLong when even the minimum reply won't fit."""
        room = window - prompt
        if room < self.config.min_output_tokens:
            raise ContextTooLong(
                f"prompt of about {prompt} tokens leaves {room} of the {window}-token "
                f"window; at least {self.config.min_output_tokens} needed",
                provider=self.name,
            )
        return min(wanted, room)

    async def complete(
        self, request: LLMRequest, *, model: str, effort: Effort | None = None
    ) -> LLMResponse:
        """One chat call, answered whole."""
        body, timeout = await self._body(request, model, effort, stream=False)
        try:
            reply = await self.client.post("/api/chat", json=body, timeout=timeout)
        except httpx.HTTPError as exc:
            raise translate_error(exc, self.name) from exc
        if reply.status_code >= 400:
            raise status_error(reply.status_code, error_text(reply), self.name)
        data = reply.json()
        if data.get("error"):
            raise status_error(500, str(data["error"]), self.name)
        message = data.get("message") or {}
        content: list[ContentBlock] = []
        if message.get("content"):
            content.append(TextBlock(text=message["content"]))
        calls = _tool_calls(message)
        content.extend(calls)
        usage = _usage(data)
        await self._observe(request, model, body, usage)
        return LLMResponse(
            content=content,
            stop_reason=_stop(data.get("done_reason"), calls),
            usage=usage,
            provider=self.name,
            model=data.get("model") or model,
        )

    async def stream(
        self, request: LLMRequest, *, model: str, effort: Effort | None = None
    ) -> AsyncIterator[StreamChunk]:
        """One streamed chat call: Ollama sends a JSON object per line, text in
        pieces and each tool call whole, then a final line with the counts."""
        body, timeout = await self._body(request, model, effort, stream=True)
        text: list[str] = []
        calls: list[ToolUseBlock] = []
        final: dict[str, Any] = {}
        try:
            async with self.client.stream("POST", "/api/chat", json=body, timeout=timeout) as reply:
                if reply.status_code >= 400:
                    await reply.aread()
                    raise status_error(reply.status_code, error_text(reply), self.name)
                async for line in reply.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise status_error(500, str(data["error"]), self.name)
                    message = data.get("message") or {}
                    if message.get("content"):
                        text.append(message["content"])
                        yield StreamChunk(text=message["content"])
                    calls.extend(_tool_calls(message))
                    if data.get("done"):
                        final = data
                        break
        except httpx.HTTPError as exc:
            raise translate_error(exc, self.name) from exc
        except json.JSONDecodeError as exc:
            raise TransientLLMError(f"unreadable stream line: {exc}", provider=self.name) from exc
        if not final:
            raise TransientLLMError("stream ended without a final line", provider=self.name)
        content: list[ContentBlock] = []
        if text:
            content.append(TextBlock(text="".join(text)))
        content.extend(calls)
        usage = _usage(final)
        await self._observe(request, model, body, usage)
        yield StreamChunk(
            done=True,
            response=LLMResponse(
                content=content,
                stop_reason=_stop(final.get("done_reason"), calls),
                usage=usage,
                provider=self.name,
                model=final.get("model") or model,
            ),
        )
