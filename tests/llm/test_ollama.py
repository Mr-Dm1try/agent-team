"""Ollama provider and embedder tests against a fake Ollama server (httpx MockTransport).
No network."""

import json
from typing import Any

import anthropic
import httpx
import pytest

from agent_team.core.config import EmbedderConfig, OllamaConfig
from agent_team.core.errors import (
    ContextTooLong,
    EmbedderUnavailable,
    ProviderError,
    TransientLLMError,
)
from agent_team.llm import anthropic_provider as ap
from agent_team.llm import ollama_provider as op
from agent_team.llm.fake import FakeProvider
from agent_team.llm.gateway import Gateway
from agent_team.llm.ollama_api import api_root
from agent_team.llm.tokens import SAFETY, estimate_json_tokens, to_json
from agent_team.llm.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from agent_team.memory.embed import OllamaEmbedder

from .conftest import Sleeps, make_config, request
from .test_providers import StubAnthropicMessages, _response, anthropic_client

HISTORY = [
    Message.user("read it"),
    Message(
        role="assistant",
        content=[TextBlock(text="ok"), ToolUseBlock(id="c1", name="read", arguments={"p": "a"})],
    ),
    Message(role="user", content=[ToolResultBlock(tool_use_id="c1", content="body")]),
]
TOOLS = [ToolSpec(name="read", description="read a file", input_schema={"type": "object"})]
THINKING = {"type": "thinking", "thinking": "", "signature": "sig-abc"}


def chat_reply(**kw: Any) -> dict[str, Any]:
    """A whole /api/chat reply with two tool calls, one of them garbled."""
    data: dict[str, Any] = {
        "model": "gpt-oss:20b",
        "message": {
            "role": "assistant",
            "content": "thinking aloud",
            "thinking": "hidden reasoning",
            "tool_calls": [
                {"id": "t1", "function": {"name": "read", "arguments": {"p": "x"}}},
                {"function": {"name": "read", "arguments": "{broken"}},
            ],
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 3,
        "eval_count": 4,
    }
    data.update(kw)
    return data


def counted(prompt_tokens: int) -> dict[str, Any]:
    """A plain text reply whose usage reports ``prompt_tokens``."""
    return {
        "model": "gpt-oss:20b",
        "message": {"role": "assistant", "content": "ok"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": prompt_tokens,
        "eval_count": 4,
    }


STREAM = [
    {"message": {"role": "assistant", "content": "", "thinking": "hmm"}, "done": False},
    {"message": {"role": "assistant", "content": "he"}, "done": False},
    {"message": {"role": "assistant", "content": "llo"}, "done": False},
    {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "read", "arguments": {"p": "y"}}}],
        },
        "done": False,
    },
    {
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 2,
        "eval_count": 5,
    },
]


class FakeOllama:
    """A fake Ollama server: ``/api/show`` reports ``trained``, ``/api/ps`` reports
    ``loaded`` (None: not loaded), and ``/api/chat`` answers with the next of ``replies``
    (the last one repeats). A reply is a dict, a list of stream lines, an
    ``httpx.Response`` or an exception to raise."""

    def __init__(
        self,
        *replies: Any,
        trained: int | None = 131_072,
        loaded: int | None = None,
        model: str = "gpt-oss:20b",
    ) -> None:
        """Serve ``replies`` for ``model``."""
        self.replies = list(replies) or [chat_reply()]
        self.trained = trained
        self.loaded = loaded
        self.model = model
        self.chats: list[dict[str, Any]] = []
        self.paths: list[str] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        """Answer one request."""
        path = req.url.path
        self.paths.append(path)
        if path == "/api/show":
            if self.trained is None:
                return httpx.Response(404, json={"error": "model not found"})
            return httpx.Response(200, json={"model_info": {"gptoss.context_length": self.trained}})
        if path == "/api/ps":
            models = (
                [] if self.loaded is None else [{"name": self.model, "context_length": self.loaded}]
            )
            return httpx.Response(200, json={"models": models})
        assert path == "/api/chat", path
        body = json.loads(req.content)
        self.chats.append(body)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        if isinstance(reply, list):
            return httpx.Response(200, content="\n".join(json.dumps(x) for x in reply) + "\n")
        return httpx.Response(200, json=reply)

    @property
    def transport(self) -> httpx.MockTransport:
        """The transport to hand the provider."""
        return httpx.MockTransport(self)


def ollama(
    *replies: Any, config: OllamaConfig | None = None, **server: Any
) -> tuple[op.OllamaProvider, FakeOllama]:
    """A provider wired to a fake server."""
    fake = FakeOllama(*replies, **server)
    return op.OllamaProvider(config, transport=fake.transport), fake


def walk_messages(messages: list[dict[str, Any]]) -> list[Any]:
    """Every ``content`` value in a list of chat messages."""
    return [m.get("content", "MISSING") for m in messages]


# --- request and reply translation -------------------------------------------------


async def test_ollama_translates_request_and_response() -> None:
    provider, fake = ollama()
    resp = await provider.complete(request(messages=HISTORY, tools=TOOLS), model="gpt-oss:20b")
    sent = fake.chats[0]
    assert sent["messages"][0] == {"role": "system", "content": "be brief"}
    assert sent["messages"][2]["tool_calls"][0] == {
        "id": "c1",
        "function": {"name": "read", "arguments": {"p": "a"}},
    }
    assert sent["messages"][3] == {
        "role": "tool",
        "content": "body",
        "tool_call_id": "c1",
        "tool_name": "read",
    }
    assert sent["options"]["num_ctx"] == 32_768
    assert sent["stream"] is False and sent["truncate"] is False and sent["shift"] is False
    assert sent["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert resp.stop_reason == "tool_use" and resp.text == "thinking aloud"
    assert "hidden reasoning" not in resp.text
    assert resp.tool_calls[0].arguments == {"p": "x"} and resp.tool_calls[0].id == "t1"
    assert resp.tool_calls[1].arguments == {"_invalid_json": "{broken"}
    assert resp.tool_calls[1].id.startswith("call_")
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (3, 4)


async def test_no_message_sent_to_ollama_has_null_content() -> None:
    """Ollama rejects ``content: null`` with a 400, so a tool call with no text, an
    assistant turn that held only Claude's thinking and an empty tool result all go
    out as strings."""
    claude = ThinkingBlock(provider="anthropic", model="claude-haiku-5-5", raw=THINKING)
    history = [
        Message.user("go"),
        Message(role="assistant", content=[ToolUseBlock(id="c1", name="read", arguments={})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="c1", content="")]),
        Message(role="assistant", content=[claude]),
        Message.user("again"),
    ]
    provider, fake = ollama()
    await provider.complete(request(messages=history, tools=TOOLS), model="gpt-oss:20b")
    contents = walk_messages(fake.chats[0]["messages"])
    assert all(isinstance(c, str) for c in contents), contents
    assert fake.chats[0]["messages"][2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": {}}}],
    }


async def test_ollama_drops_thinking_and_sends_the_think_level() -> None:
    claude = ThinkingBlock(provider="anthropic", model="claude-haiku-5-5", raw=THINKING)
    history = [
        Message.user("read it"),
        Message(role="assistant", content=[claude, TextBlock(text="ok")]),
    ]
    provider, fake = ollama()
    await provider.complete(request(messages=history), model="gpt-oss:20b", effort="medium")
    await provider.complete(request(), model="gpt-oss:20b", effort="max")
    await provider.complete(request(), model="gpt-oss:20b")
    await provider.complete(request(), model="llama3.2:3b", effort="low")
    sent = fake.chats[0]
    assert sent["messages"][2] == {"role": "assistant", "content": "ok"}
    assert "sig-abc" not in str(sent["messages"])
    assert sent["think"] == "medium"
    assert fake.chats[1]["think"] == "high"
    assert "think" not in fake.chats[2]
    # Ollama rejects a think level for models that can't think.
    assert "think" not in fake.chats[3]


async def test_fallback_to_ollama_drops_claude_thinking(sleeps: Sleeps) -> None:
    block = ThinkingBlock(provider="anthropic", model="claude-haiku-5-5", raw=THINKING)
    history = [
        Message.user("read it"),
        Message(role="assistant", content=[block, TextBlock(text="ok")]),
        Message.user("go on"),
    ]
    down = anthropic.InternalServerError("oops", response=_response(529), body=None)
    claude_api = StubAnthropicMessages(down)
    provider, fake = ollama()
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
    assert "sig-abc" not in str(fake.chats[0]["messages"])
    assert fake.chats[0]["think"] == "low"


async def test_ollama_stream_joins_text_and_keeps_tool_calls() -> None:
    provider, fake = ollama(STREAM)
    chunks = [c async for c in provider.stream(request(), model="gpt-oss:20b")]
    assert fake.chats[0]["stream"] is True
    assert [c.text for c in chunks[:-1]] == ["he", "llo"]
    final = chunks[-1].response
    assert final is not None and final.text == "hello"
    assert final.tool_calls[0].arguments == {"p": "y"} and final.stop_reason == "tool_use"
    assert final.usage.output_tokens == 5


async def test_a_stream_without_a_final_line_is_transient() -> None:
    provider, _ = ollama(STREAM[:3])
    with pytest.raises(TransientLLMError, match="final line"):
        [c async for c in provider.stream(request(), model="gpt-oss:20b")]


async def test_ollama_maps_the_token_cap_to_max_tokens() -> None:
    reply = counted(10) | {"done_reason": "length"}
    provider, _ = ollama(reply)
    resp = await provider.complete(request(), model="gpt-oss:20b")
    assert resp.stop_reason == "max_tokens"


# --- errors ------------------------------------------------------------------------


def error(status: int, message: str) -> httpx.Response:
    """An Ollama error response."""
    return httpx.Response(status, json={"error": message})


async def test_ollama_error_mapping() -> None:
    provider, _ = ollama(httpx.ConnectError("refused"))
    with pytest.raises(TransientLLMError) as down:
        await provider.complete(request(), model="gpt-oss:20b")
    assert down.value.unreachable
    provider, _ = ollama(httpx.ReadError("reset"))
    with pytest.raises(TransientLLMError) as blip:
        await provider.complete(request(), model="gpt-oss:20b")
    assert not blip.value.unreachable
    provider, _ = ollama(error(503, "busy"))
    with pytest.raises(TransientLLMError) as busy:
        await provider.complete(request(), model="gpt-oss:20b")
    assert busy.value.status == 503 and busy.value.max_retries is None
    provider, _ = ollama(error(400, "invalid message content type: <nil>"))
    with pytest.raises(ProviderError) as bad:
        await provider.complete(request(), model="gpt-oss:20b")
    assert bad.value.status == 400
    # Ollama's overflow error, with truncation off, is the same as the harness's own.
    provider, _ = ollama(error(500, "the input length exceeds the context length"))
    with pytest.raises(ContextTooLong):
        await provider.complete(request(), model="gpt-oss:20b")


async def test_a_garbled_tool_call_is_retried_once_then_falls_back(sleeps: Sleeps) -> None:
    garbled = error(500, 'error parsing tool call: raw=\'{"path":"/Users/mrdmit\'')
    provider, fake = ollama(garbled)
    config = make_config(
        defaults={"provider": "ollama", "model": "gpt-oss:20b", "context_budget": 20_000},
        fallbacks=["ollama", "fake"],
        fallback_models={"fake": "fake-small"},
    )
    gw = Gateway(config, {"ollama": provider, "fake": FakeProvider(["ok"])}, sleep=sleeps)
    resp = await gw.complete(request())
    assert resp.provider == "fake"
    assert len(fake.chats) == 2  # the first try and one retry, not the usual three
    assert sleeps.calls == [1]


async def test_a_stream_error_line_is_mapped() -> None:
    lines = [STREAM[0], {"error": "error parsing tool call: raw=''"}]
    provider, _ = ollama(lines)
    with pytest.raises(TransientLLMError) as exc:
        [c async for c in provider.stream(request(), model="gpt-oss:20b")]
    assert exc.value.max_retries == 1


async def test_a_stream_http_error_is_mapped() -> None:
    provider, _ = ollama(error(404, "model 'nope' not found"))
    with pytest.raises(ProviderError, match="not found"):
        [c async for c in provider.stream(request(), model="nope")]


# --- the context window ------------------------------------------------------------


async def test_a_prompt_over_the_real_window_is_refused_not_cut() -> None:
    """Ollama reports a 4,096-token window although 32,768 is configured: the cap and
    the guard use 4,096, so a prompt that won't fit raises instead of being cut."""
    provider, fake = ollama(trained=4_096, config=OllamaConfig(min_output_tokens=1_000))
    assert await provider.window("gpt-oss:20b") == 4_096
    long = request(messages=[Message.user("x" * 12_000)])
    with pytest.raises(ContextTooLong, match="4096-token window"):
        await provider.complete(long, model="gpt-oss:20b")
    with pytest.raises(ContextTooLong):
        [c async for c in provider.stream(long, model="gpt-oss:20b")]
    assert fake.chats == []
    await provider.complete(request(), model="gpt-oss:20b")
    assert fake.chats[0]["options"]["num_ctx"] == 4_096


async def test_the_loaded_context_cuts_the_window_after_the_first_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, fake = ollama(loaded=4_096)
    with caplog.at_level("WARNING", logger=op.__name__):
        await provider.complete(request(), model="gpt-oss:20b")
    assert fake.chats[0]["options"]["num_ctx"] == 32_768
    assert "loaded with a 4096-token context" in caplog.text
    assert await provider.window("gpt-oss:20b") == 4_096
    with pytest.raises(ContextTooLong):
        await provider.complete(request(messages=[Message.user("x" * 12_000)]), model="gpt-oss:20b")
    assert fake.paths.count("/api/ps") == 1  # checked once, not on every call


async def test_the_configured_window_is_used_until_ollama_answers() -> None:
    provider, fake = ollama(trained=None)
    await provider.complete(request(), model="gpt-oss:20b")
    await provider.complete(request(), model="gpt-oss:20b")
    assert [c["options"]["num_ctx"] for c in fake.chats] == [32_768, 32_768]
    assert fake.paths.count("/api/show") == 2  # asked again, since it didn't say


async def test_ollama_timeout_grows_with_the_reply_it_may_write() -> None:
    provider, _ = ollama(counted(0))
    body, short = await provider._body(request(max_tokens=1_000), "gpt-oss:20b", None, stream=False)
    body, long = await provider._body(request(max_tokens=16_000), "gpt-oss:20b", None, stream=False)
    assert short == provider.config.timeout_s  # the plain timeout covers a short reply
    assert long == body["options"]["num_predict"] / provider.config.min_tokens_per_s > short


async def test_ollama_fits_prompt_and_reply_into_the_window() -> None:
    # No usage reported, so nothing is anchored or learned: the plain estimate caps.
    provider, fake = ollama(counted(0), config=OllamaConfig(num_ctx=4_000, min_output_tokens=500))
    await provider.complete(request(max_tokens=16_000), model="gpt-oss:20b")
    sent = fake.chats[0]
    prompt = op.prompt_tokens(sent)
    assert sent["options"]["num_predict"] == 4_000 - prompt
    # Non-ASCII text counts once per character, not as \\u escapes.
    assert op.prompt_tokens({"messages": [{"content": "привет" * 100}]}) < 250
    await provider.complete(request(max_tokens=200), model="gpt-oss:20b")
    assert fake.chats[1]["options"]["num_predict"] == 200

    long = request(messages=[Message.user("x" * 11_000)])
    with pytest.raises(ContextTooLong):
        await provider.complete(long, model="gpt-oss:20b")
    assert len(fake.chats) == 2


async def test_gateway_moves_on_when_ollama_cannot_fit_the_prompt(sleeps: Sleeps) -> None:
    provider, fake = ollama()
    config = make_config(
        defaults={"provider": "ollama", "model": "gpt-oss:20b", "context_budget": 20_000},
        fallbacks=["ollama", "fake"],
        fallback_models={"fake": "fake-small"},
    )
    gw = Gateway(config, {"ollama": provider, "fake": FakeProvider(["ok"])}, sleep=sleeps)
    resp = await gw.complete(request(messages=[Message.user("x" * 120_000)]))
    assert resp.provider == "fake" and fake.chats == [] and sleeps.calls == []


async def test_ollama_anchors_the_cap_on_the_last_reported_prompt() -> None:
    provider, fake = ollama(counted(100), config=OllamaConfig(num_ctx=8_000, min_output_tokens=500))
    first = [Message.user("hello " * 50)]
    await provider.complete(request(messages=first), model="gpt-oss:20b")
    sent = fake.chats[0]
    chars = len(to_json(sent["messages"], sent.get("tools", [])))
    assert provider.calibration.ratio("gpt-oss:20b") == chars / 100 * SAFETY

    # The next call extends the same history: Ollama's 100 plus the new messages only.
    later = [*first, Message(role="assistant", content=[TextBlock(text="ok")]), Message.user("go")]
    await provider.complete(request(messages=later, max_tokens=16_000), model="gpt-oss:20b")
    ratio = provider.calibration.ratio("gpt-oss:20b")
    appended = estimate_json_tokens(op.to_chat_messages("", later[1:]), chars_per_token=ratio)
    assert fake.chats[1]["options"]["num_predict"] == 8_000 - (100 + appended)

    # A rewritten history (compaction) isn't anchored on: it is estimated in full.
    rewritten = [Message.user("summary"), *later[1:]]
    ratio = provider.calibration.ratio("gpt-oss:20b")
    await provider.complete(request(messages=rewritten, max_tokens=16_000), model="gpt-oss:20b")
    full = op.prompt_tokens(fake.chats[2], ratio)
    assert fake.chats[2]["options"]["num_predict"] == 8_000 - full


async def test_ollama_warns_when_the_prompt_nearly_filled_the_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, fake = ollama(
        counted(3_950), config=OllamaConfig(num_ctx=4_000, min_output_tokens=500)
    )
    with caplog.at_level("WARNING", logger=op.__name__):
        await provider.complete(request(), model="gpt-oss:20b")
    assert "within 2% of the 4000-token window" in caplog.text
    # A count from a prompt that was probably cut teaches nothing.
    assert provider.calibration.ratio("gpt-oss:20b") == provider.config.chars_per_token
    assert not provider._anchors
    caplog.clear()
    fake.replies = [counted(3_000)]
    with caplog.at_level("WARNING", logger=op.__name__):
        await provider.complete(request(messages=[Message.user("other")]), model="gpt-oss:20b")
    assert not caplog.text
    assert fake.paths.count("/api/ps") == 2  # a near-full prompt checks the loaded context again


async def test_ollama_does_not_warn_about_a_full_prompt_the_cap_left_room_for(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Within 2% of the window, but inside the room the cap leaves: nothing was cut.
    provider, _ = ollama(
        counted(98_500),
        config=OllamaConfig(num_ctx=100_000, min_output_tokens=1_024),
        trained=None,
    )
    with caplog.at_level("WARNING", logger=op.__name__):
        await provider.complete(request(), model="gpt-oss:20b")
    assert not caplog.text


def test_the_native_api_root_ignores_an_openai_v1_suffix() -> None:
    assert api_root("http://localhost:11434/v1") == "http://localhost:11434"
    assert api_root("http://localhost:11434/") == "http://localhost:11434"


# --- the embedder ------------------------------------------------------------------


class FakeEmbedServer:
    """A fake ``/api/show`` and ``/api/embed`` that records what was sent."""

    def __init__(self, trained: int | None = 2_048) -> None:
        """Report ``trained`` as the embedder's window (None: unknown)."""
        self.trained = trained
        self.inputs: list[list[str]] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        """Answer one request."""
        if req.url.path == "/api/show":
            if self.trained is None:
                return httpx.Response(404, json={"error": "not found"})
            return httpx.Response(
                200, json={"model_info": {"nomic-bert.context_length": self.trained}}
            )
        body = json.loads(req.content)
        self.inputs.append(body["input"])
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2] for _ in body["input"]]})


async def test_the_embedder_cuts_text_to_the_models_window() -> None:
    server = FakeEmbedServer(trained=512)
    embedder = OllamaEmbedder(
        EmbedderConfig(), "http://localhost:11434/v1", transport=httpx.MockTransport(server)
    )
    vectors = await embedder.embed(["short", "y" * 20_000], query=False)
    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    sent = server.inputs[0]
    assert sent[0] == "search_document: short"
    assert len(sent[1]) == await embedder.max_chars() < 512 * 3
    # The configured cap wins when it is the smaller one.
    wide = OllamaEmbedder(
        EmbedderConfig(max_input_tokens=100),
        "http://localhost:11434",
        transport=httpx.MockTransport(FakeEmbedServer(trained=8_192)),
    )
    assert await wide.max_chars() < 100 * 3


async def test_an_embed_error_is_embedder_unavailable() -> None:
    def refuse(req: httpx.Request) -> httpx.Response:
        """Fail every call the way a missing model does."""
        return httpx.Response(404, json={"error": "model 'nomic-embed-text' not found"})

    embedder = OllamaEmbedder(
        EmbedderConfig(), "http://localhost:11434", transport=httpx.MockTransport(refuse)
    )
    with pytest.raises(EmbedderUnavailable, match="not found"):
        await embedder.embed(["x"], query=True)
