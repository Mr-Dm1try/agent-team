"""Provider translation tests against stub SDK clients. No network."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import openai
import pytest

from agent_team.core.config import AnthropicConfig
from agent_team.core.errors import TransientLLMError
from agent_team.llm import anthropic_provider as ap
from agent_team.llm import ollama_provider as op
from agent_team.llm.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)

from .conftest import Sleeps, make_config, request

HISTORY = [
    Message.user("read it"),
    Message(
        role="assistant",
        content=[TextBlock(text="ok"), ToolUseBlock(id="c1", name="read", arguments={"p": "a"})],
    ),
    Message(role="user", content=[ToolResultBlock(tool_use_id="c1", content="body")]),
]
TOOLS = [ToolSpec(name="read", description="read a file", input_schema={"type": "object"})]


def sdk_message(**kw: Any) -> anthropic.types.Message:
    data: dict[str, Any] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [
            {"type": "text", "text": "hi"},
            {"type": "tool_use", "id": "tu_1", "name": "read", "input": {"p": "b"}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }
    data.update(kw)
    return anthropic.types.Message.model_validate(data)


class StubAnthropicStream:
    def __init__(self, final: anthropic.types.Message) -> None:
        self.final = final

    async def __aenter__(self) -> "StubAnthropicStream":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    def text_stream(self) -> AsyncIterator[str]:
        async def gen() -> AsyncIterator[str]:
            for t in ("h", "i"):
                yield t

        return gen()

    async def get_final_message(self) -> anthropic.types.Message:
        return self.final


class StubAnthropicMessages:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kw: Any) -> anthropic.types.Message:
        self.calls.append(kw)
        if isinstance(self.result, Exception):
            raise self.result
        assert isinstance(self.result, anthropic.types.Message)
        return self.result

    def stream(self, **kw: Any) -> StubAnthropicStream:
        self.calls.append(kw)
        return StubAnthropicStream(self.result)


def _response(status: int) -> httpx2.Response:
    return httpx2.Response(status, request=httpx2.Request("POST", "https://example.invalid"))


def anthropic_client(messages: "StubAnthropicMessages") -> SimpleNamespace:
    """A stub client; the beta namespace is what the provider uses by default."""
    return SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages))


async def test_anthropic_translates_request_and_response() -> None:
    messages = StubAnthropicMessages(sdk_message())
    provider = ap.AnthropicProvider(client=anthropic_client(messages))
    resp = await provider.complete(
        request(messages=HISTORY, tools=TOOLS), model="claude-haiku-5-5", effort="low"
    )
    sent = messages.calls[0]
    assert sent["model"] == "claude-haiku-5-5" and sent["system"] == "be brief"
    assert "temperature" not in sent
    assert sent["thinking"] == {
        "type": "adaptive",
        "block_binding": {"prefix_mismatch_behavior": "drop_block"},
    }
    assert sent["betas"] == [ap.BINDING_BETA]
    assert sent["output_config"] == {"effort": "low"}
    assert sent["tools"][0]["input_schema"] == {"type": "object"}
    assert sent["messages"][1]["content"][1] == {
        "type": "tool_use",
        "id": "c1",
        "name": "read",
        "input": {"p": "a"},
    }
    assert sent["messages"][2]["content"][0]["type"] == "tool_result"
    assert resp.stop_reason == "tool_use" and resp.text == "hi"
    assert resp.tool_calls[0].arguments == {"p": "b"}
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (11, 7)


async def test_anthropic_without_binding_uses_the_plain_api_and_no_effort() -> None:
    plain = StubAnthropicMessages(sdk_message())
    provider = ap.AnthropicProvider(
        AnthropicConfig(on_history_edit=None),
        client=SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=None)),
    )
    await provider.complete(request(), model="claude-haiku-5-5")
    sent = plain.calls[0]
    assert sent["thinking"] == {"type": "adaptive"}
    assert "betas" not in sent and "output_config" not in sent


async def test_anthropic_sends_no_block_binding_when_thinking_is_off() -> None:
    plain = StubAnthropicMessages(sdk_message())
    provider = ap.AnthropicProvider(
        AnthropicConfig(thinking={"type": "disabled"}),
        client=SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=None)),
    )
    await provider.complete(request(), model="claude-haiku-5-5")
    assert plain.calls[0]["thinking"] == {"type": "disabled"} and "betas" not in plain.calls[0]


def test_anthropic_counts_thinking_blocks_the_api_dropped() -> None:
    msg = anthropic.types.beta.BetaMessage.model_validate(
        {
            **sdk_message().model_dump(),
            "input_transformations": [
                {
                    "type": "thinking_dropped",
                    "path": "messages.1.content.0",
                    "reason": "prefix_binding_mismatch",
                },
                {
                    "type": "thinking_mismatch_allowed",
                    "path": "messages.3.content.0",
                    "reason": "prefix_binding_mismatch",
                },
            ],
        }
    )
    assert ap.from_wire_message(msg, provider="anthropic").thinking_dropped == 1
    assert ap.from_wire_message(sdk_message(), provider="anthropic").thinking_dropped == 0


THINKING = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
REDACTED = {"type": "redacted_thinking", "data": "opaque=="}


async def test_anthropic_keeps_thinking_blocks_and_sends_them_back_unchanged() -> None:
    first = sdk_message(
        content=[
            THINKING,
            {"type": "tool_use", "id": "tu_1", "name": "read", "input": {"p": "b"}},
        ],
    )
    messages = StubAnthropicMessages(first)
    provider = ap.AnthropicProvider(client=anthropic_client(messages))
    resp = await provider.complete(request(), model="claude-haiku-5-5")
    thinking = resp.content[0]
    assert isinstance(thinking, ThinkingBlock)
    assert (thinking.provider, thinking.model, thinking.raw) == (
        "anthropic",
        "claude-haiku-5-5",
        THINKING,
    )
    assert resp.text == "" and [c.id for c in resp.tool_calls] == ["tu_1"]

    history = [
        Message.user("read it"),
        Message(role="assistant", content=resp.content),
        Message(role="user", content=[ToolResultBlock(tool_use_id="tu_1", content="body")]),
    ]
    await provider.complete(request(messages=history), model="claude-haiku-5-5")
    assert messages.calls[1]["messages"][1]["content"] == [
        THINKING,
        {"type": "tool_use", "id": "tu_1", "name": "read", "input": {"p": "b"}},
    ]


async def test_anthropic_keeps_redacted_thinking_and_maps_stop() -> None:
    msg = sdk_message(content=[REDACTED, {"type": "text", "text": "done"}], stop_reason="end_turn")
    resp = ap.from_wire_message(msg, provider="anthropic")
    assert [b.type for b in resp.content] == ["thinking", "text"] and resp.text == "done"
    block = resp.content[0]
    assert isinstance(block, ThinkingBlock) and block.raw == REDACTED
    msg = sdk_message(content=[{"type": "text", "text": "x"}], stop_reason="stop_sequence")
    assert ap.from_wire_message(msg, provider="anthropic").stop_reason == "end_turn"


def test_thinking_from_another_claude_model_is_sent_another_providers_is_not() -> None:
    sonnet = ThinkingBlock(provider="anthropic", model="claude-sonnet-5-5", raw=THINKING)
    elsewhere = ThinkingBlock(provider="other", model="m", raw={"type": "thinking"})
    wire = ap.to_wire_messages(
        [
            Message.user("go"),
            Message(role="assistant", content=[sonnet, elsewhere, TextBlock(text="ok")]),
            Message(role="assistant", content=[elsewhere]),
        ],
        provider="anthropic",
    )
    # A haiku call after STRONGER_MODEL ran on sonnet: the sonnet block goes back
    # unchanged and the API drops it if haiku can't read it. A message left with no
    # blocks is not sent at all.
    assert wire[1]["content"] == [THINKING, {"type": "text", "text": "ok"}]
    assert len(wire) == 2


async def test_ollama_drops_thinking_and_sends_reasoning_effort() -> None:
    claude = ThinkingBlock(provider="anthropic", model="claude-haiku-5-5", raw=THINKING)
    history = [
        Message.user("read it"),
        Message(role="assistant", content=[claude, TextBlock(text="ok")]),
    ]
    provider, comps = ollama(chat_completion())
    await provider.complete(request(messages=history), model="gpt-oss:20b", effort="medium")
    await provider.complete(request(), model="gpt-oss:20b", effort="max")
    await provider.complete(request(), model="gpt-oss:20b")
    await provider.complete(request(), model="llama3.2:3b", effort="low")
    sent = comps.calls[0]
    assert sent["messages"][2] == {"role": "assistant", "content": "ok"}
    assert "sig-abc" not in str(sent["messages"])
    assert sent["reasoning_effort"] == "medium"
    assert comps.calls[1]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in comps.calls[2]
    # Ollama rejects reasoning_effort for models that can't think.
    assert "reasoning_effort" not in comps.calls[3]


async def test_fallback_to_ollama_drops_claude_thinking(sleeps: Sleeps) -> None:
    from agent_team.llm.gateway import Gateway

    block = ThinkingBlock(provider="anthropic", model="claude-haiku-5-5", raw=THINKING)
    history = [
        Message.user("read it"),
        Message(role="assistant", content=[block, TextBlock(text="ok")]),
        Message.user("go on"),
    ]
    down = anthropic.InternalServerError("oops", response=_response(529), body=None)
    claude_api = StubAnthropicMessages(down)
    provider, comps = ollama(chat_completion())
    config = make_config(
        defaults={"provider": "anthropic", "model": "claude-haiku-5-5", "effort": "low"},
        fallbacks=["anthropic", "ollama"],
        fallback_models={"ollama": "gpt-oss:20b"},
        retry={"delays_s": []},
    )
    gw = Gateway(
        config,
        {
            "anthropic": ap.AnthropicProvider(client=anthropic_client(claude_api)),
            "ollama": provider,
        },
        sleep=sleeps,
    )
    resp = await gw.complete(request(messages=history))
    assert resp.provider == "ollama"
    assert claude_api.calls[0]["messages"][1]["content"][0] == THINKING
    assert "sig-abc" not in str(comps.calls[0]["messages"])
    assert comps.calls[0]["reasoning_effort"] == "low"


async def test_anthropic_stream() -> None:
    provider = ap.AnthropicProvider(
        client=anthropic_client(
            StubAnthropicMessages(
                sdk_message(content=[{"type": "text", "text": "hi"}], stop_reason="end_turn")
            )
        )
    )
    chunks = [c async for c in provider.stream(request(), model="claude-haiku-5-5")]
    assert [c.text for c in chunks[:-1]] == ["h", "i"]
    assert chunks[-1].response is not None and chunks[-1].response.text == "hi"


@pytest.mark.parametrize(
    ("exc", "transient", "status"),
    [
        (anthropic.RateLimitError("slow", response=_response(429), body=None), True, 429),
        (anthropic.InternalServerError("oops", response=_response(529), body=None), True, 529),
        (anthropic.APITimeoutError(request=httpx2.Request("POST", "https://x")), True, None),
        (anthropic.BadRequestError("bad", response=_response(400), body=None), False, None),
    ],
)
async def test_anthropic_error_mapping(exc: Exception, transient: bool, status: int | None) -> None:
    provider = ap.AnthropicProvider(client=anthropic_client(StubAnthropicMessages(exc)))
    with pytest.raises(Exception) as info:
        await provider.complete(request(), model="m")
    assert isinstance(info.value, TransientLLMError) is transient
    if transient:
        assert info.value.status == status  # type: ignore[attr-defined]


# --- ollama ------------------------------------------------------------------------


def chat_completion(**kw: Any) -> openai.types.chat.ChatCompletion:
    data: dict[str, Any] = {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-oss:20b",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "thinking aloud",
                    "tool_calls": [
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"p": "x"}'},
                        },
                        {
                            "id": "t2",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{broken"},
                        },
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }
    data.update(kw)
    return openai.types.chat.ChatCompletion.model_validate(data)


def stream_chunks() -> list[openai.types.chat.ChatCompletionChunk]:
    def chunk(
        delta: dict[str, Any], finish: str | None = None, usage: dict[str, int] | None = None
    ) -> openai.types.chat.ChatCompletionChunk:
        return openai.types.chat.ChatCompletionChunk.model_validate(
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}],
                "usage": usage,
            }
        )

    return [
        chunk({"content": "he"}),
        chunk({"content": "llo"}),
        chunk(
            {
                "tool_calls": [
                    {"index": 0, "id": "t1", "function": {"name": "read", "arguments": '{"p":'}}
                ]
            }
        ),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' "y"}'}}]}, "tool_calls"),
        chunk({}, usage={"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7}),
    ]


class StubCompletions:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if isinstance(self.result, Exception):
            raise self.result
        if kw.get("stream"):

            async def gen() -> AsyncIterator[Any]:
                for c in self.result:
                    yield c

            return gen()
        return self.result


def ollama(result: Any) -> tuple[op.OllamaProvider, StubCompletions]:
    comps = StubCompletions(result)
    client = SimpleNamespace(chat=SimpleNamespace(completions=comps))
    return op.OllamaProvider(client=client), comps


async def test_ollama_translates_request_and_response() -> None:
    provider, comps = ollama(chat_completion())
    resp = await provider.complete(request(messages=HISTORY, tools=TOOLS), model="gpt-oss:20b")
    sent = comps.calls[0]
    assert sent["messages"][0] == {"role": "system", "content": "be brief"}
    assert sent["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"p": "a"}'
    assert sent["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "body"}
    assert sent["extra_body"] == {"options": {"num_ctx": 32768}}
    assert sent["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert resp.stop_reason == "tool_use" and resp.text == "thinking aloud"
    assert resp.tool_calls[0].arguments == {"p": "x"}
    assert resp.tool_calls[1].arguments == {"_invalid_json": "{broken"}
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (3, 4)


async def test_ollama_stream_accumulates_tool_calls() -> None:
    provider, _ = ollama(stream_chunks())
    chunks = [c async for c in provider.stream(request(), model="m")]
    assert [c.text for c in chunks[:-1]] == ["he", "llo"]
    final = chunks[-1].response
    assert final is not None and final.text == "hello"
    assert final.tool_calls[0].arguments == {"p": "y"} and final.stop_reason == "tool_use"
    assert final.usage.output_tokens == 5


async def test_ollama_error_mapping() -> None:
    provider, _ = ollama(openai.APIConnectionError(request=httpx2.Request("POST", "http://l")))
    with pytest.raises(TransientLLMError):
        await provider.complete(request(), model="m")
    bad = openai.BadRequestError("bad", response=_response(400), body=None)
    provider, _ = ollama(bad)
    with pytest.raises(openai.BadRequestError):
        await provider.complete(request(), model="m")


def test_build_providers_needs_no_credentials() -> None:
    from agent_team.core.config import load_config
    from agent_team.llm.factory import build_gateway

    gw = build_gateway(load_config())
    assert set(gw.providers) == {"anthropic", "ollama"}
