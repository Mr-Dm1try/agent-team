import json
from pathlib import Path

import pytest

from agent_team.core.errors import TransientLLMError
from agent_team.core.models import AgentName, Usage
from agent_team.llm.chaos import BAD_JSON_TEXT
from agent_team.llm.fake import FakeProvider, FakeScriptExhausted, FakeToolCall, FakeTurn
from agent_team.llm.types import LLMResponse, Message, TextBlock, ToolResultBlock

from .conftest import request


async def test_replays_a_scripted_tool_conversation() -> None:
    fake = FakeProvider(
        [
            FakeTurn(tool_calls=[FakeToolCall(name="read_file", arguments={"path": "a.py"})]),
            FakeTurn(json_output={"summary": "done"}),
        ]
    )
    first = await fake.complete(request(), model="m")
    assert first.stop_reason == "tool_use"
    call = first.tool_calls[0]
    assert (call.name, call.arguments) == ("read_file", {"path": "a.py"})

    history = [
        Message.user("hi"),
        Message(role="assistant", content=first.content),
        Message(role="user", content=[ToolResultBlock(tool_use_id=call.id, content="print(1)")]),
    ]
    second = await fake.complete(request(messages=history), model="m")
    assert second.stop_reason == "end_turn"
    assert json.loads(second.text) == {"summary": "done"}
    assert second.usage.input_tokens > 0 and second.provider == "fake"
    assert len(fake.requests) == 2 and fake.requests[1].messages == history
    assert fake.remaining == 0
    with pytest.raises(FakeScriptExhausted):
        await fake.complete(request(), model="m")


async def test_turns_scoped_to_agents() -> None:
    fake = FakeProvider(
        [FakeTurn(agent=AgentName.ENGINEER, text="code"), FakeTurn(text="any"), "plain"]
    )
    assert (await fake.complete(request(AgentName.PRODUCT), model="m")).text == "any"
    assert (await fake.complete(request(AgentName.ENGINEER), model="m")).text == "code"
    assert (await fake.complete(request(AgentName.QA), model="m")).text == "plain"


async def test_faults_and_raw_items() -> None:
    resp = LLMResponse(
        content=[TextBlock(text="x")],
        stop_reason="end_turn",
        usage=Usage(),
        provider="p",
        model="m",
    )
    fake = FakeProvider(
        [FakeTurn(fault="rate_limit"), FakeTurn(fault="bad_json"), ValueError("bug"), resp]
    )
    with pytest.raises(TransientLLMError) as info:
        await fake.complete(request(), model="m")
    assert info.value.status == 429
    assert (await fake.complete(request(), model="m")).text == BAD_JSON_TEXT
    with pytest.raises(ValueError):
        await fake.complete(request(), model="m")
    assert await fake.complete(request(), model="m") is resp


async def test_responder_and_stream() -> None:
    fake = FakeProvider(responder=lambda req: f"echo {req.agent.value} " * 5)
    chunks = [c async for c in fake.stream(request(), model="m")]
    assert chunks[-1].done and chunks[-1].response is not None
    assert "".join(c.text for c in chunks[:-1]) == chunks[-1].response.text


async def test_from_file(tmp_path: Path) -> None:
    path = tmp_path / "script.yaml"
    path.write_text(
        "- {agent: product, json_output: {title: Inventory}}\n"
        "- {tool_calls: [{name: write_file, arguments: {path: x}}]}\n"
    )
    fake = FakeProvider.from_file(path)
    assert json.loads((await fake.complete(request(), model="m")).text) == {"title": "Inventory"}
    assert (await fake.complete(request(), model="m")).tool_calls[0].name == "write_file"
