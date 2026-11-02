"""The tool-calling loop every model-driven agent runs.

One loop, whatever the provider: call the model, run the tools it asks for, save a
checkpoint after every model step and every tool step, compact the context when it
nears its budget, and finish when the model replies with JSON that validates. Bad
JSON goes back to the model with the errors, up to ``max_repairs`` times.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from agent_team.agents.compaction import Compactor
from agent_team.agents.jsonrepair import parse_output
from agent_team.core.errors import OutputValidationError, ToolError
from agent_team.core.interfaces import CheckpointStore, EventBus, Tool, ToolContext
from agent_team.core.models import AgentName, Event, EventName, Run, ToolResult, Usage
from agent_team.llm.types import (
    ContentBlock,
    LLMClient,
    LLMRequest,
    LLMResponse,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from agent_team.tools.base import MAX_INLINE_CHARS, store_large

Level = Literal["debug", "info", "warning", "error"]

#: Extra checks beyond the schema, such as "every planned file exists". Returns errors.
Check = Callable[[Any], Awaitable[list[str]]]

INTERRUPTED_NOTE = (
    "This tool call was running when the process stopped. It was not repeated because "
    "it is not safe to repeat. Check the current state before trying it again."
)


@dataclass
class LoopSettings:
    """Limits for one tool loop: steps, repair rounds and tokens per model call."""

    max_steps: int = 80  # checkpointed steps: one per model call and one per tool round
    max_repairs: int = 2
    max_tokens: int = 4096


@dataclass
class LoopOutcome[M: BaseModel]:
    """What a tool loop produced: the parsed output or the error, usage and the transcript."""

    output: M | None
    usage: Usage = field(default_factory=Usage)
    model: str | None = None
    error: Exception | None = None
    messages: list[Message] = field(default_factory=list)
    steps: int = 0


@dataclass
class ToolLoop[M: BaseModel]:
    """Runs one attempt for one agent. Build a fresh one per attempt."""

    agent: AgentName
    system: str
    output_model: type[M]
    llm: LLMClient
    bus: EventBus
    checkpoints: CheckpointStore
    tools: dict[str, Tool]
    tool_ctx: ToolContext
    run: Run
    task_id: str
    attempt: int
    settings: LoopSettings = field(default_factory=LoopSettings)
    compactor: Compactor = field(default_factory=Compactor)
    check: Check | None = None
    trace_id: str | None = None
    span_id: str | None = None

    _step: int = 0
    _usage: Usage = field(default_factory=Usage)
    _model: str | None = None

    # --- checkpoints ---------------------------------------------------------------

    async def _save(self, messages: list[Message]) -> None:
        self._step += 1
        await self.checkpoints.save_step(self.task_id, self.attempt, self._step, messages)
        await self._emit(
            EventName.CHECKPOINT_SAVED,
            {"step": self._step, "messages": len(messages)},
            level="debug",
        )

    async def _emit(self, name: EventName, data: dict[str, Any], *, level: Level = "info") -> None:
        await self.bus.emit(self._event(name, data, level=level))

    def _event(self, name: EventName, data: dict[str, Any], *, level: Level = "info") -> Event:
        return Event(
            name=name,
            run_id=self.run.id,
            task_id=self.task_id,
            attempt=self.attempt,
            agent=self.agent,
            trace_id=self.trace_id,
            span_id=self.span_id,
            level=level,
            data=data,
        )

    # --- tools ---------------------------------------------------------------------

    async def _call_tool(self, call: ToolUseBlock, *, resumed: bool = False) -> ToolResultBlock:
        tool = self.tools.get(call.name)
        if tool is None:
            known = ", ".join(sorted(self.tools)) or "none"
            return ToolResultBlock(
                tool_use_id=call.id,
                content=f"unknown tool {call.name!r}; available tools: {known}",
                is_error=True,
            )
        if resumed and not tool.safe_to_repeat:
            return ToolResultBlock(tool_use_id=call.id, content=INTERRUPTED_NOTE, is_error=True)

        args = call.arguments
        before = self._event(
            EventName.TOOL_CALL_BEFORE, {"tool": call.name, "args": args, "call_id": call.id}
        )
        outcome = await self.bus.check(before)
        if not outcome.allow:
            rule = f" by rule {outcome.rule_id}" if outcome.rule_id else ""
            content = f"blocked{rule}: {outcome.reason or 'not allowed'}"
            await self._emit(
                EventName.TOOL_CALL_AFTER,
                {
                    "tool": call.name,
                    "call_id": call.id,
                    "blocked": True,
                    "rule_id": outcome.rule_id,
                },
                level="warning",
            )
            return ToolResultBlock(tool_use_id=call.id, content=content, is_error=True)
        if outcome.replacement and isinstance(outcome.replacement.get("args"), dict):
            args = outcome.replacement["args"]

        started = time.monotonic()
        try:
            result = await tool(args, self.tool_ctx)
        except ToolError as exc:
            result = ToolResult(content=f"{call.name} failed: {exc}", is_error=True)
        except OSError as exc:
            # A filesystem error is the tool failing, not a bug: the model can adapt.
            result = ToolResult(content=f"{call.name} failed: {exc}", is_error=True)
        duration = round(time.monotonic() - started, 3)

        content = result.content
        stored_at: str | None = None
        if len(content) > MAX_INLINE_CHARS:
            stored_at = store_large(content, self.run, label=call.name)
            content = (
                content[: MAX_INLINE_CHARS // 4]
                + f"\n[output truncated; {len(result.content)} chars saved at {stored_at}; "
                "read it with read_file and an offset]"
            )
        await self._emit(
            EventName.TOOL_CALL_AFTER,
            {
                "tool": call.name,
                "call_id": call.id,
                "args": args,
                "result": content,
                "is_error": result.is_error,
                "blocked": False,
                "duration_s": duration,
                "chars": len(result.content),
                "stored_at": stored_at,
                "data": result.data,
            },
            level="warning" if result.is_error else "info",
        )
        return ToolResultBlock(
            tool_use_id=call.id, content=content, is_error=result.is_error, stored_at=stored_at
        )

    async def _run_tools(self, calls: list[ToolUseBlock], *, resumed: bool = False) -> Message:
        results: list[ContentBlock] = [await self._call_tool(c, resumed=resumed) for c in calls]
        return Message(role="user", content=results)

    # --- the loop ------------------------------------------------------------------

    async def _complete(self, messages: list[Message]) -> LLMResponse:
        request = LLMRequest(
            agent=self.agent,
            system=self.system,
            messages=messages,
            tools=[t.spec for t in self.tools.values()],
            max_tokens=self.settings.max_tokens,
            run_id=self.run.id,
            task_id=self.task_id,
        )
        response = await self.llm.complete(request)
        self._usage = self._usage + response.usage
        self._model = response.model
        return response

    async def _validate(self, text: str) -> tuple[M | None, list[str]]:
        value, errors = parse_output(text, self.output_model)
        if value is not None and self.check is not None:
            errors = await self.check(value)
            if errors:
                value = None
        return value, errors

    async def _maybe_compact(self, messages: list[Message]) -> list[Message]:
        budget = self.llm.context_budget(self.agent)
        if not self.compactor.needs_compaction(self.system, messages, budget):
            return messages
        result = await self.compactor.compact(
            system=self.system,
            messages=messages,
            budget=budget,
            llm=self.llm,
            bus=self.bus,
            run=self.run,
            agent=self.agent,
            task_id=self.task_id,
            attempt=self.attempt,
        )
        self._usage = self._usage + result.usage
        if result.steps:
            # The checkpoint keeps the compacted history, so a resume continues from it.
            await self._save(result.messages)
        return result.messages

    async def run_loop(self, first_message: Message, *, resume: bool = False) -> LoopOutcome[M]:
        """Call the model and its tools until it returns a valid output, checkpointing each step."""
        messages: list[Message] = [first_message]
        repairs = 0
        pending_calls: list[ToolUseBlock] = []
        final_text: str | None = None

        saved = await self.checkpoints.load_latest(self.task_id, self.attempt) if resume else None
        if saved is None:
            await self._save(messages)
        else:
            self._step, messages = saved
            last = messages[-1]
            if last.role == "assistant":
                calls = [b for b in last.content if isinstance(b, ToolUseBlock)]
                if calls:
                    pending_calls = calls
                else:
                    final_text = "".join(b.text for b in last.content if isinstance(b, TextBlock))

        if pending_calls:
            messages.append(await self._run_tools(pending_calls, resumed=True))
            await self._save(messages)

        while True:
            if final_text is not None:
                value, errors = await self._validate(final_text)
                if value is not None:
                    return self._outcome(value, messages)
                if repairs >= self.settings.max_repairs:
                    err = OutputValidationError(
                        f"output did not validate after {repairs} repairs", errors=errors
                    )
                    return self._outcome(None, messages, error=err)
                repairs += 1
                messages.append(Message.user(_repair_prompt(errors, self.output_model)))
                await self._save(messages)
                final_text = None

            if self._step >= self.settings.max_steps:
                err = OutputValidationError(
                    f"no valid output within {self.settings.max_steps} steps",
                    errors=["step limit reached"],
                )
                return self._outcome(None, messages, error=err)

            messages = await self._maybe_compact(messages)
            response = await self._complete(messages)
            messages.append(Message(role="assistant", content=response.content))
            await self._save(messages)

            if response.tool_calls:
                messages.append(await self._run_tools(response.tool_calls))
                await self._save(messages)
                continue
            if response.stop_reason == "max_tokens":
                final_text = response.text or "(cut off)"
                # Count it as a failed parse so the model is told to be shorter.
                value, errors = await self._validate(final_text)
                if value is not None:
                    return self._outcome(value, messages)
                errors = ["the reply hit the token limit; reply with shorter JSON", *errors]
                if repairs >= self.settings.max_repairs:
                    err = OutputValidationError("output was cut off", errors=errors)
                    return self._outcome(None, messages, error=err)
                repairs += 1
                messages.append(Message.user(_repair_prompt(errors, self.output_model)))
                await self._save(messages)
                final_text = None
                continue
            final_text = response.text

    def _outcome(
        self, value: M | None, messages: list[Message], *, error: Exception | None = None
    ) -> LoopOutcome[M]:
        return LoopOutcome(
            output=value,
            usage=self._usage,
            model=self._model,
            error=error,
            messages=messages,
            steps=self._step,
        )


def _repair_prompt(errors: list[str], model: type[BaseModel]) -> str:
    listed = "\n".join(f"- {e}" for e in errors[:20])
    return (
        f"Your final answer did not validate as {model.__name__}:\n{listed}\n"
        "Reply again with only the corrected JSON object, no prose and no code fences."
    )
