"""Provider-neutral LLM types. Checkpoints store these, so an attempt can resume on
the fallback provider. Providers translate to and from their own wire formats."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agent_team.core.models import AgentName, Usage


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextBlock(_Strict):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(_Strict):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResultBlock(_Strict):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False
    stored_at: str | None = None  # path of the full result when it was too large to inline


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


class Message(_Strict):
    role: Literal["user", "assistant"]
    content: list[ContentBlock]

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=[TextBlock(text=text)])


class ToolSpec(_Strict):
    name: str
    description: str
    input_schema: dict[str, Any]


class LLMRequest(_Strict):
    agent: AgentName
    system: str
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    max_tokens: int = 4096
    run_id: str
    task_id: str | None = None


class LLMResponse(_Strict):
    content: list[ContentBlock]
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "other"]
    usage: Usage
    provider: str
    model: str

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


class StreamChunk(_Strict):
    text: str = ""
    done: bool = False
    response: LLMResponse | None = None  # set on the final chunk


class Provider(Protocol):
    """One model backend: anthropic, ollama or fake."""

    name: str

    async def complete(self, request: LLMRequest, *, model: str) -> LLMResponse: ...

    def stream(self, request: LLMRequest, *, model: str) -> AsyncIterator[StreamChunk]: ...


class LLMClient(Protocol):
    """What agents call. The gateway implements it: routing, fallback, budgets, chaos."""

    async def complete(self, request: LLMRequest) -> LLMResponse: ...

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]: ...

    def context_budget(self, agent: AgentName) -> int:
        """Max prompt tokens for this agent's model, from the routing file."""
        ...
