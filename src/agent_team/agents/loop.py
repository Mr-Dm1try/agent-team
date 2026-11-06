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

from agent_team.agents.compaction import Compactor, estimate_tokens, sent_json
from agent_team.core.errors import OutputValidationError, ToolError
from agent_team.core.interfaces import CheckpointStore, EventBus, Tool, ToolContext
from agent_team.core.models import AgentName, Event, EventName, Run, ToolResult, Usage
from agent_team.core.tracing import Tracer
from agent_team.llm.structured import parse_output
from agent_team.llm.tokens import Anchor, Calibration
from agent_team.llm.types import (
    ContentBlock,
    LLMClient,
    LLMRequest,
    LLMResponse,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from agent_team.tools.base import MAX_INLINE_CHARS, store_large

Level = Literal["debug", "info", "warning", "error"]

#: Extra checks beyond the schema, such as "every planned file exists". Returns errors.
Check = Callable[[Any], Awaitable[list[str]]]

#: The result of a tool call the model was still writing when its reply hit the
#: output token limit.
CUT_OFF_NOTE = (
    "Not run: your reply hit the output token limit while writing this call, so its "
    "arguments may be cut off. Make the call again with less in it: one file per "
    "write_file call, and split a long file into smaller modules."
)
#: The first repair error for a final reply cut off at the output token limit.
CUT_OFF_REPLY = (
    "the reply hit the output token limit; reply with shorter JSON that reports what "
    "you did, such as the paths of files you wrote, without repeating their contents"
)
INTERRUPTED_NOTE = (
    "This tool call was running when the process stopped. It was not repeated because "
    "it is not safe to repeat. Check the current state before trying it again."
)


@dataclass
class LoopSettings:
    """Limits for one tool loop: steps, repair rounds and, optionally, output tokens
    per model call. Leave ``max_tokens`` unset to keep the request's default, which
    leaves room for thinking."""

    max_steps: int = 80  # checkpointed steps: one per model call and one per tool round
    max_repairs: int = 2
    max_tokens: int | None = None


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
    #: Chars per token per model, learned from this loop's calls (or shared, if given).
    calibration: Calibration = field(default_factory=Calibration)
    check: Check | None = None
    trace_id: str | None = None
    span_id: str | None = None

    _step: int = 0
    _usage: Usage = field(default_factory=Usage)
    _model: str | None = None
    #: The provider's prompt count for the last call, which compaction anchors on.
    _anchor: Anchor | None = None
    #: The provider and model that served the last call.
    _served: tuple[str, str] | None = None

    # --- checkpoints ---------------------------------------------------------------

    async def _save(self, messages: list[Message]) -> None:
        """Checkpoint the history as the next step, so a crash resumes from here."""
        self._step += 1
        await self.checkpoints.save_step(self.task_id, self.attempt, self._step, messages)
        await self._emit(
            EventName.CHECKPOINT_SAVED,
            {"step": self._step, "messages": len(messages)},
            level="debug",
        )

    async def _emit(self, name: EventName, data: dict[str, Any], *, level: Level = "info") -> None:
        """Emit a loop event tagged with this task, attempt and trace."""
        await self.bus.emit(self._event(name, data, level=level))

    def _event(self, name: EventName, data: dict[str, Any], *, level: Level = "info") -> Event:
        """Build an event tagged with this task, attempt and trace."""
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
        """Run one tool call in its own ``tool`` span, under the attempt's."""
        async with Tracer(self.bus).span(self.run.id, call.name, "tool", call_id=call.id) as span:
            result = await self._tool(call, resumed=resumed)
            span.set(is_error=result.is_error)
            return result

    async def _tool(self, call: ToolUseBlock, *, resumed: bool) -> ToolResultBlock:
        """Run one tool call through the pre-call hooks; every failure becomes an error result for
        the model."""
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

    async def _run_tools(
        self, calls: list[ToolUseBlock], *, resumed: bool = False, cut_off: bool = False
    ) -> Message:
        """Run a turn's tool calls in order and return their results as one user turn.
        When the reply hit the output token limit (``cut_off``), its last call may be
        missing the end of its arguments, so it isn't run and the model is told why."""
        run, last = (calls[:-1], calls[-1]) if cut_off else (calls, None)
        results: list[ContentBlock] = [await self._call_tool(c, resumed=resumed) for c in run]
        if last is not None:
            results.append(
                ToolResultBlock(tool_use_id=last.id, content=CUT_OFF_NOTE, is_error=True)
            )
        return Message(role="user", content=results)

    # --- the loop ------------------------------------------------------------------

    async def _complete(self, messages: list[Message]) -> LLMResponse:
        """One model call with the loop's system prompt, tools and output cap."""
        request = LLMRequest(
            agent=self.agent,
            system=self.system,
            messages=messages,
            tools=self._specs(),
            run_id=self.run.id,
            task_id=self.task_id,
        )
        if self.settings.max_tokens is not None:
            request = request.model_copy(update={"max_tokens": self.settings.max_tokens})
        response = await self.llm.complete(request)
        self._usage = self._usage + response.usage
        self._model = response.model
        self._learn(request, response)
        return response

    def _learn(self, request: LLMRequest, response: LLMResponse) -> None:
        """Anchor on the provider's count of this call's prompt and calibrate the model's
        chars per token from it, measured without thinking blocks (which the provider
        doesn't count as stored). A response with no plausible count leaves no anchor,
        and so does one from another provider or model than the call before (a
        fallback): the next call may go back, and the two count differently, so it is
        estimated in full."""
        tokens = response.usage.input_tokens
        served = (response.provider, response.model)
        switched = self._served is not None and served != self._served
        self._served = served
        if tokens <= 0:
            self._anchor = None
            return
        chars = len(sent_json(request.system, request.messages, request.tools, thinking=False))
        plausible = self.calibration.observe(response.model, chars, tokens)
        if switched or not plausible:
            self._anchor = None
            return
        self._anchor = Anchor(sent=len(request.messages), tokens=tokens)

    async def _validate(self, text: str) -> tuple[M | None, list[str]]:
        """Parse the reply into the output model, then run the agent's own checks on it."""
        value, errors = parse_output(text, self.output_model)
        if value is not None and self.check is not None:
            errors = await self.check(value)
            if errors:
                value = None
        return value, errors

    def _specs(self) -> list[ToolSpec]:
        """The tool specs sent with every call, also counted by compaction."""
        return [t.spec for t in self.tools.values()]

    async def _maybe_compact(self, messages: list[Message]) -> list[Message]:
        """Compact the history when it nears the budget, and checkpoint the compacted version."""
        budget = self.llm.context_budget(self.agent, self.task_id)
        specs = self._specs()
        # The lowest ratio over the models seen: after a fallback the next call may go
        # to either, so estimate for the one that counts more tokens.
        ratio = self.calibration.ratio(None)
        # A provider sends back only its own thinking blocks, and the next call may go
        # to any provider on the route, so estimate for the one that would send most.
        providers: list[str | None] = [*self.llm.providers_for(self.agent, self.task_id)]
        provider = max(
            providers or [None],
            key=lambda p: estimate_tokens(self.system, messages, specs, ratio, provider=p),
        )
        if not self.compactor.needs_compaction(
            self.system,
            messages,
            budget,
            specs,
            anchor=self._anchor,
            chars_per_token=ratio,
            provider=provider,
        ):
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
            tools=specs,
            anchor=self._anchor,
            chars_per_token=ratio,
            provider=provider,
        )
        self._usage = self._usage + result.usage
        if result.steps:
            self._anchor = None  # the provider never counted the rewritten history
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
            if response.tool_calls:
                # A reply that arrived after the run stopped or ran out of budget starts
                # no tool turn. It isn't kept either, so the resumed attempt asks again
                # instead of finding tool calls that look interrupted.
                self.llm.check_run(self.run.id)
            messages.append(Message(role="assistant", content=response.content))
            await self._save(messages)

            if response.tool_calls:
                cut_off = response.stop_reason == "max_tokens"
                messages.append(await self._run_tools(response.tool_calls, cut_off=cut_off))
                await self._save(messages)
                continue
            if response.stop_reason == "max_tokens":
                final_text = response.text or "(cut off)"
                # Count it as a failed parse so the model is told to be shorter.
                value, errors = await self._validate(final_text)
                if value is not None:
                    return self._outcome(value, messages)
                errors = [CUT_OFF_REPLY, *errors]
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
        """Wrap the loop's result with its usage, model and transcript."""
        return LoopOutcome(
            output=value,
            usage=self._usage,
            model=self._model,
            error=error,
            messages=messages,
            steps=self._step,
        )


def _repair_prompt(errors: list[str], model: type[BaseModel]) -> str:
    """The message that sends validation errors back to the model for another try."""
    listed = "\n".join(f"- {e}" for e in errors[:20])
    return (
        f"Your final answer did not validate as {model.__name__}:\n{listed}\n"
        "If fixing it needs a tool call first, make that call. "
        "Then reply with only the corrected JSON object, no prose and no code fences."
    )
