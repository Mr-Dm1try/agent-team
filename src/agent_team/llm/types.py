"""Provider-neutral LLM types. Checkpoints store these, so an attempt can resume on
the fallback provider. Providers translate to and from their own wire formats."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol

from pydantic import Field

from agent_team.core.base import Contract
from agent_team.core.models import AgentName, Effort, Usage


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


class ThinkingBlock(Contract):
    """A model's reasoning, kept so a tool loop can hand it back on the next call.

    ``raw`` is the provider's block exactly as returned (a ``thinking`` block with its
    signature, or a ``redacted_thinking`` block) and is never edited. Only the
    provider in ``provider`` sends it back; every other provider drops it. ``model``
    is the model that wrote it. Text and tool calls never include it."""

    type: Literal["thinking"] = "thinking"
    provider: str
    model: str
    raw: dict[str, Any]


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock


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
    #: Covers thinking as well as the answer, so it is set well above a typical reply.
    max_tokens: int = 16_000
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
    #: Thinking blocks the provider's API dropped from the request because their
    #: history had been edited (e.g. by compaction); 0 when none were.
    thinking_dropped: int = 0

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

    async def complete(
        self, request: LLMRequest, *, model: str, effort: Effort | None = None
    ) -> LLMResponse:
        """One call to ``model`` at the route's ``effort``, returning the whole reply."""
        ...

    def stream(
        self, request: LLMRequest, *, model: str, effort: Effort | None = None
    ) -> AsyncIterator[StreamChunk]:
        """One call to ``model`` at the route's ``effort``, streamed as chunks."""
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

    def context_budget(self, agent: AgentName, task_id: str | None = None) -> int:
        """Max prompt tokens for the model this agent's next call tries first: its own
        route's budget from the routing file, or a fallback's while its provider is down."""
        ...

    def providers_for(self, agent: AgentName, task_id: str | None) -> list[str]:
        """The providers the next call for this agent and task may go to, in the
        order they are tried: its route, then the fallbacks."""
        ...

    def use_stronger_model(self, task_id: str) -> object:
        """Route this task's next calls to the routing file's stronger model
        (the STRONGER_MODEL decision). Returns the route now in force."""
        ...

    def set_override(self, task_id: str, route: Any) -> None:
        """Set or, with ``None``, clear a per-task route override."""
        ...

    def check_run(self, run_id: str) -> None:
        """Raise RunStopped or BudgetExceeded when the run must stop; else nothing."""
        ...
