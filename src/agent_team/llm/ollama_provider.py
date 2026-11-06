"""Local models through Ollama's OpenAI-compatible endpoint, using the openai SDK.

The context window is raised with Ollama's ``options.num_ctx``. The route's effort is
sent as ``reasoning_effort`` (low, medium or high; xhigh and max become high) to the
models in ``reasoning_models``, such as gpt-oss; Ollama rejects it for other models.
Claude's thinking blocks are dropped: Ollama can't read them and has no way to take
reasoning back in the history, so it thinks afresh on every call. The SDK's own retries
are off; the gateway retries and falls back.

Ollama cuts the start of a prompt that overflows ``num_ctx`` without an error, so the
reply is capped before sending to leave the prompt room. The prompt's size is anchored
on what Ollama reported for the same task's previous call, with only the messages
appended since estimated, at a chars-per-token ratio learned per model. A count too
low for the characters sent (one that left out what Ollama had cached) is not used. A
reported prompt within 2% of ``num_ctx`` and past the room the cap leaves is logged as
a warning (it was probably cut) and not learned from.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

import openai

from agent_team.core.config import OllamaConfig
from agent_team.core.errors import ContextTooLong, TransientLLMError
from agent_team.core.models import Effort, Usage
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

#: A reported prompt this close to ``num_ctx`` was probably cut by Ollama.
NEAR_FULL = 0.98
#: Anchors kept, one per task, model and agent; the oldest go first.
MAX_ANCHORS = 256

_STOP: dict[str, Any] = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
_REASONING_EFFORT: dict[Effort, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def to_chat_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Our messages in the OpenAI chat format, system prompt first. Thinking blocks
    are left out."""
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


def prompt_tokens(params: dict[str, Any], chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """A high estimate of the prompt's tokens over the JSON that is sent, with
    non-ASCII text kept as is. High on purpose: running over is silent truncation."""
    return estimate_json_tokens(
        params["messages"], params.get("tools", []), chars_per_token=chars_per_token
    )


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
        return TransientLLMError(
            f"{type(exc).__name__}: {exc}",
            provider=provider,
            unreachable=not isinstance(exc, openai.APITimeoutError),
        )
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
        """Use ``config``; ``client`` replaces the SDK client in tests."""
        self.config = config or OllamaConfig()
        self._client = client
        self.calibration = Calibration(self.config.chars_per_token)
        self._anchors: OrderedDict[str, Anchor] = OrderedDict()

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

    def _params(self, request: LLMRequest, model: str, effort: Effort | None) -> dict[str, Any]:
        """Keyword arguments for one chat completion call."""
        params: dict[str, Any] = {
            "model": model,
            "messages": to_chat_messages(request.system, request.messages),
            "extra_body": {"options": {"num_ctx": self.config.num_ctx}},
        }
        if effort is not None and model.split(":", 1)[0] in self.config.reasoning_models:
            params["reasoning_effort"] = _REASONING_EFFORT[effort]
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
        prompt = self._prompt_tokens(request, model, params)
        params["max_tokens"] = self._max_tokens(prompt, request.max_tokens)
        # A long reply at local speeds can outlast the plain timeout.
        params["timeout"] = max(
            self.config.timeout_s, params["max_tokens"] / self.config.min_tokens_per_s
        )
        return params

    @staticmethod
    def _key(request: LLMRequest, model: str) -> str:
        """Whose anchor a request uses: one per task (or run), agent and model."""
        return f"{request.run_id}:{request.task_id}:{request.agent.value}:{model}"

    @staticmethod
    def _prefix(request: LLMRequest, params: dict[str, Any], sent: int) -> str:
        """The fingerprint of the system prompt, tools and first ``sent`` messages."""
        head = [m.model_dump(mode="json") for m in request.messages[:sent]]
        return fingerprint(request.system, params.get("tools", []), head)

    def _prompt_tokens(self, request: LLMRequest, model: str, params: dict[str, Any]) -> int:
        """The prompt's size: Ollama's count for the same task's last call plus an
        estimate of the messages appended since, or a full estimate when there is no
        anchor or the history it counted was rewritten (compaction)."""
        ratio = self.calibration.ratio(model)
        anchor = self._anchors.get(self._key(request, model))
        if (
            anchor is not None
            and anchor.sent <= len(request.messages)
            and anchor.prefix == self._prefix(request, params, anchor.sent)
        ):
            appended = to_chat_messages("", request.messages[anchor.sent :])
            return anchor.tokens + estimate_json_tokens(appended, chars_per_token=ratio)
        return prompt_tokens(params, ratio)

    def _observe(
        self, request: LLMRequest, model: str, params: dict[str, Any], usage: Usage
    ) -> None:
        """Learn from Ollama's count of a prompt: warn (and learn nothing) when the
        prompt filled num_ctx past the cap's room, since it was probably cut; otherwise,
        if the count is plausible, calibrate the model's ratio and anchor the task's
        next call on it."""
        tokens = usage.input_tokens
        key = self._key(request, model)
        if tokens <= 0:
            return
        room = self.config.num_ctx - self.config.min_output_tokens
        if tokens >= NEAR_FULL * self.config.num_ctx and tokens > room:
            self._anchors.pop(key, None)  # a cut prompt's count is no measure of it
            log.warning(
                "ollama %s: prompt of %d tokens is within %d%% of num_ctx %d; "
                "its start was probably cut",
                model,
                tokens,
                round((1 - NEAR_FULL) * 100),
                self.config.num_ctx,
            )
            return
        chars = len(to_json(params["messages"], params.get("tools", [])))
        if not self.calibration.observe(model, chars, tokens):
            self._anchors.pop(key, None)  # a partial count: estimate the next call in full
            return
        self._anchors[key] = Anchor(
            sent=len(request.messages),
            tokens=tokens,
            prefix=self._prefix(request, params, len(request.messages)),
        )
        self._anchors.move_to_end(key)
        while len(self._anchors) > MAX_ANCHORS:
            self._anchors.popitem(last=False)

    def _max_tokens(self, prompt: int, wanted: int) -> int:
        """``wanted`` cut down so a prompt of ``prompt`` tokens plus the reply fit
        ``num_ctx``. Ollama would otherwise drop the start of the prompt (the system
        prompt and the task) without an error. Raises ContextTooLong when even the
        minimum reply won't fit."""
        room = self.config.num_ctx - prompt
        if room < self.config.min_output_tokens:
            raise ContextTooLong(
                f"prompt of about {prompt} tokens leaves {room} of num_ctx "
                f"{self.config.num_ctx}; at least {self.config.min_output_tokens} needed",
                provider=self.name,
            )
        return min(wanted, room)

    async def complete(
        self, request: LLMRequest, *, model: str, effort: Effort | None = None
    ) -> LLMResponse:
        """One chat completion call."""
        params = self._params(request, model, effort)
        try:
            resp = await self.client.chat.completions.create(**params)
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
        usage = Usage(
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
        self._observe(request, model, params, usage)
        return LLMResponse(
            content=content,
            stop_reason=_STOP.get(choice.finish_reason or "", "other"),
            usage=usage,
            provider=self.name,
            model=resp.model or model,
        )

    async def stream(
        self, request: LLMRequest, *, model: str, effort: Effort | None = None
    ) -> AsyncIterator[StreamChunk]:
        """One streamed chat completion; tool call deltas are joined."""
        params = self._params(request, model, effort)
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
        self._observe(request, model, params, usage)
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
