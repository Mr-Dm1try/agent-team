"""Provider translation tests against stub SDK clients. No network."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from agent_team.core.config import AnthropicConfig
from agent_team.core.errors import TransientLLMError
from agent_team.llm import anthropic_provider as ap
from agent_team.llm.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)

from .conftest import request

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
    def __init__(self, final: Any) -> None:
        self.final = final

    async def __aenter__(self) -> "StubAnthropicStream":
        if isinstance(self.final, Exception):
            raise self.final
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
        self.streamed: list[bool] = []

    async def create(self, **kw: Any) -> anthropic.types.Message:
        self.calls.append(kw)
        self.streamed.append(False)
        if isinstance(self.result, Exception):
            raise self.result
        assert isinstance(self.result, anthropic.types.Message)
        return self.result

    def stream(self, **kw: Any) -> StubAnthropicStream:
        self.calls.append(kw)
        self.streamed.append(True)
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


async def test_anthropic_sends_tool_choice_none_for_a_reply_only_call() -> None:
    messages = StubAnthropicMessages(sdk_message())
    provider = ap.AnthropicProvider(client=anthropic_client(messages))
    await provider.complete(request(messages=HISTORY, tools=TOOLS), model="claude-haiku-5-5")
    await provider.complete(
        request(messages=HISTORY, tools=TOOLS, tool_choice="none"), model="claude-haiku-5-5"
    )
    auto, none = messages.calls
    assert "tool_choice" not in auto
    # The tools still go, since the history holds tool calls; the model may only reply.
    assert none["tool_choice"] == {"type": "none"} and none["tools"] == auto["tools"]


async def test_anthropic_streams_calls_that_may_write_a_long_reply() -> None:
    messages = StubAnthropicMessages(sdk_message())
    provider = ap.AnthropicProvider(client=anthropic_client(messages))
    long = await provider.complete(request(max_tokens=16_000), model="claude-haiku-5-5")
    await provider.complete(request(max_tokens=1_000), model="claude-haiku-5-5")
    assert messages.streamed == [True, False]  # the timeout bounds each read, not the reply
    assert long.text == "hi" and messages.calls[0]["max_tokens"] == 16_000


@pytest.mark.parametrize(
    "dropped", [httpx2.ReadTimeout("stalled"), httpx2.RemoteProtocolError("peer closed")]
)
async def test_a_stream_that_stalls_or_drops_is_transient(dropped: Exception) -> None:
    provider = ap.AnthropicProvider(client=anthropic_client(StubAnthropicMessages(dropped)))
    with pytest.raises(TransientLLMError):
        await provider.complete(request(max_tokens=16_000), model="claude-haiku-5-5")


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


def test_only_a_failed_connection_counts_as_anthropic_being_down() -> None:
    from agent_team.llm import anthropic_provider

    req = httpx2.Request("POST", "http://l")
    refused = anthropic.APIConnectionError(request=req)
    refused.__cause__ = httpx2.ConnectError("refused")
    blip = anthropic.APIConnectionError(request=req)
    blip.__cause__ = httpx2.ReadError("reset")
    assert anthropic_provider.translate_error(refused, "anthropic").unreachable  # type: ignore[attr-defined]
    assert not anthropic_provider.translate_error(blip, "anthropic").unreachable  # type: ignore[attr-defined]


def test_build_providers_needs_no_credentials() -> None:
    from agent_team.core.config import load_config
    from agent_team.llm.factory import build_gateway

    gw = build_gateway(load_config())
    assert set(gw.providers) == {"anthropic", "ollama"}
