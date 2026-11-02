"""Provider-neutral LLM types. Checkpoints store these, so an attempt can resume on
the fallback provider. Providers translate to and from their own wire formats."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol

from pydantic import Field

from agent_team.core.base import Contract
from agent_team.core.models import AgentName, Usage


class TextBlock(Contract):
    """Plain text in a message."""

    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(Contract):
    """A model's request to call a tool."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResultBlock(Contract):
    """A tool's result sent back to the model."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False
    stored_at: str | None = None  # path of the full result when it was too large to inline


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


class Message(Contract):
    """One turn of a conversation, as a list of content blocks."""

    role: Literal["user", "assistant"]
    content: list[ContentBlock]

    @classmethod
    def user(cls, text: str) -> Message:
        """A user message holding one text block."""
        return cls(role="user", content=[TextBlock(text=text)])


class ToolSpec(Contract):
    """A tool as the model sees it: name, description and JSON input schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


class LLMRequest(Contract):
    """One model call from an agent, tagged with the run and task for routing,
    budgets and tracing."""

    agent: AgentName
    system: str
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    max_tokens: int = 4096
    run_id: str
    task_id: str | None = None


class LLMResponse(Contract):
    """A model's reply: content, why it stopped, usage, and which provider and
    model answered."""

    content: list[ContentBlock]
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "other"]
    usage: Usage
    provider: str
    model: str

    @property
    def text(self) -> str:
        """All text blocks joined together."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> list[ToolUseBlock]:
        """The tool calls the model made, in order."""
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


class StreamChunk(Contract):
    """A piece of a streamed reply. The last one has ``done`` set and carries the
    full response."""

    text: str = ""
    done: bool = False
    response: LLMResponse | None = None  # set on the final chunk


class Provider(Protocol):
    """One model backend: anthropic, ollama or fake."""

    name: str

    async def complete(self, request: LLMRequest, *, model: str) -> LLMResponse:
        """One call to ``model``, returning the whole reply."""
        ...

    def stream(self, request: LLMRequest, *, model: str) -> AsyncIterator[StreamChunk]:
        """One call to ``model``, streamed as chunks."""
        ...


class LLMClient(Protocol):
    """What agents call. The gateway implements it: routing, fallback, budgets, chaos."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Route, call and account for one request, falling back and retrying as
        configured. Raises BudgetExceeded or RunStopped when the run must stop."""
        ...

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Like ``complete``, streamed. Usage is counted when the final chunk arrives."""
        ...

    def context_budget(self, agent: AgentName) -> int:
        """Max prompt tokens for this agent's model, from the routing file."""
        ...

    def use_stronger_model(self, task_id: str) -> object:
        """Route this task's next calls to the routing file's stronger model
        (the STRONGER_MODEL decision). Returns the route now in force."""
        ...

    def set_override(self, task_id: str, route: Any) -> None:
        """Set or, with ``None``, clear a per-task route override."""
        ...
