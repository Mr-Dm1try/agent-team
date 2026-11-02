"""Context compaction for long attempts (docs/DESIGN.md, "Context compaction").

When an attempt nears its context budget, three steps run in order and stop once it
fits: trim large tool results, stub old tool results, then summarize older turns with
one model call. The first message (the task envelope) and the latest turns are never
compacted. Each compaction emits ``context_compacted`` with token counts.

Assistant turns are never edited, so a thinking block always goes back exactly as the
provider returned it: trim and stub only rewrite tool results, which sit in user turns,
and summarize drops whole turns before the kept tail, thinking included. Any of these
still rewrites history the model has seen, so the API drops thinking blocks after the
rewritten point (docs/DESIGN.md, "Context compaction") and reports it as
``thinking_dropped``. The progress note is what carries the work forward.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from agent_team.agents.jsonrepair import parse_output
from agent_team.core.interfaces import EventBus
from agent_team.core.models import AgentName, Event, EventName, Run, Usage
from agent_team.llm.tokens import estimate_json_tokens
from agent_team.llm.types import (
    ContentBlock,
    LLMClient,
    LLMRequest,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from agent_team.tools.base import store_large

#: Compaction starts when the prompt passes this share of the budget.
TRIGGER_RATIO = 0.85
#: Tool results longer than this are trimmed to a preview in the trim stage.
TRIM_CHARS = 2_000
PREVIEW_CHARS = 400


def estimate_tokens(system: str, messages: list[Message], tools: Sequence[ToolSpec] = ()) -> int:
    """The shared high estimate over everything a call sends: the system prompt, the
    messages and the tool specs as JSON. Thinking blocks count at their full raw size."""
    return estimate_json_tokens(
        system,
        [m.model_dump(mode="json") for m in messages],
        [t.model_dump(mode="json") for t in tools],
    )


class ProgressNote(BaseModel):
    """What the summarize step must return."""

    model_config = ConfigDict(extra="forbid")
    done: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    remaining: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """The note as the plain-text message that replaces summarized turns."""

        def section(title: str, items: list[str]) -> str:
            body = "\n".join(f"- {i}" for i in items) or "- (none)"
            return f"{title}:\n{body}"

        return "\n".join(
            [
                "Progress note (earlier turns were compacted; files are still in the workspace)",
                section("Done", self.done),
                section("Decisions", self.decisions),
                section("Left to do", self.remaining),
            ]
        )


SUMMARY_SYSTEM = (
    "You compress an agent's working history into a progress note. Reply with one JSON "
    'object: {"done": [...], "decisions": [...], "remaining": [...]}. Keep file paths, '
    "ids such as REQ-1 and AC-1.1, and anything the agent must not redo."
)


@dataclass
class CompactionResult:
    """The compacted messages, token estimates before and after, and the stages used."""

    messages: list[Message]
    before: int
    after: int
    steps: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


def _tool_names(messages: list[Message]) -> dict[str, ToolUseBlock]:
    return {b.id: b for m in messages for b in m.content if isinstance(b, ToolUseBlock)}


def _stub_line(result: ToolResultBlock, call: ToolUseBlock | None) -> str:
    first = result.content.strip().splitlines()[0][:100] if result.content.strip() else ""
    if call is None:
        label = "tool result"
    else:
        arg = next((v for v in call.arguments.values() if isinstance(v, str)), "")
        label = f"{call.name} {arg[:80]}".strip()
    where = f"; full result at {result.stored_at}" if result.stored_at else ""
    status = "error" if result.is_error else "ok"
    return f"[compacted {label} ({status}): {first}{where}]"


class Compactor:
    """Keeps a tool loop's context under budget: trim, then stub, then summarize."""

    def __init__(self, *, keep_last: int = 4, trigger_ratio: float = TRIGGER_RATIO) -> None:
        self.keep_last = keep_last
        self.trigger_ratio = trigger_ratio

    def needs_compaction(
        self,
        system: str,
        messages: list[Message],
        budget: int,
        tools: Sequence[ToolSpec] = (),
    ) -> bool:
        """True when the estimated context is over the trigger share of ``budget``."""
        return estimate_tokens(system, messages, tools) > int(budget * self.trigger_ratio)

    def _protected_tail_start(self, messages: list[Message]) -> int:
        """Index of the first message in the protected tail. It always starts on an
        assistant turn, so the tail never begins with an orphan tool result."""
        start = max(1, len(messages) - self.keep_last)
        while start < len(messages) and messages[start].role != "assistant":
            start += 1
        return start

    async def compact(
        self,
        *,
        system: str,
        messages: list[Message],
        budget: int,
        llm: LLMClient,
        bus: EventBus,
        run: Run,
        agent: AgentName,
        task_id: str,
        attempt: int,
        tools: Sequence[ToolSpec] = (),
    ) -> CompactionResult:
        """Shrink ``messages`` stage by stage until they fit; emits a compaction event."""
        limit = int(budget * self.trigger_ratio)
        before = estimate_tokens(system, messages, tools)
        result = CompactionResult(messages=list(messages), before=before, after=before)
        tail = self._protected_tail_start(messages)

        def fits() -> bool:
            result.after = estimate_tokens(system, result.messages, tools)
            return result.after <= limit

        # 1. Trim large tool results outside the tail.
        trimmed = False
        new: list[Message] = []
        for i, msg in enumerate(result.messages):
            if i == 0 or i >= tail or msg.role != "user":
                new.append(msg)
                continue
            trim_blocks: list[ContentBlock] = []
            for b in msg.content:
                if isinstance(b, ToolResultBlock) and len(b.content) > TRIM_CHARS:
                    path = b.stored_at or store_large(b.content, run, label="compacted")
                    preview = b.content[:PREVIEW_CHARS]
                    trim_blocks.append(
                        b.model_copy(
                            update={
                                "content": f"{preview}\n[trimmed; full result at {path}]",
                                "stored_at": path,
                            }
                        )
                    )
                    trimmed = True
                else:
                    trim_blocks.append(b)
            new.append(msg.model_copy(update={"content": trim_blocks}))
        result.messages = new
        if trimmed:
            result.steps.append("trim")
        if not fits():
            # 2. Stub every tool result outside the tail.
            calls = _tool_names(result.messages)
            stubbed: list[Message] = []
            changed = False
            for i, msg in enumerate(result.messages):
                if i == 0 or i >= tail or msg.role != "user":
                    stubbed.append(msg)
                    continue
                blocks: list[ContentBlock] = []
                for b in msg.content:
                    if isinstance(b, ToolResultBlock) and not b.content.startswith("[compacted "):
                        stub = _stub_line(b, calls.get(b.tool_use_id))
                        blocks.append(b.model_copy(update={"content": stub}))
                        changed = True
                    else:
                        blocks.append(b)
                stubbed.append(msg.model_copy(update={"content": blocks}))
            result.messages = stubbed
            if changed:
                result.steps.append("stub")
        if not fits() and tail > 1:
            # 3. Summarize the middle into a progress note on the first message. An
            # earlier note is folded into the new one, so the head doesn't keep growing.
            first = result.messages[0]
            envelope, old_notes = first.content[:1], first.content[1:]
            middle = [Message(role="user", content=old_notes)] if old_notes else []
            middle += result.messages[1:tail]
            note, usage = await self._summarize(middle, llm, run, agent, task_id)
            result.usage = usage
            head = first.model_copy(update={"content": [*envelope, TextBlock(text=note.render())]})
            result.messages = [head, *result.messages[tail:]]
            result.steps.append("summarize")
            fits()

        if result.steps:
            await bus.emit(
                Event(
                    name=EventName.CONTEXT_COMPACTED,
                    run_id=run.id,
                    task_id=task_id,
                    attempt=attempt,
                    agent=agent,
                    data={
                        "tokens_before": result.before,
                        "tokens_after": result.after,
                        "steps": result.steps,
                        "budget": budget,
                    },
                )
            )
        return result

    async def _summarize(
        self,
        middle: list[Message],
        llm: LLMClient,
        run: Run,
        agent: AgentName,
        task_id: str,
    ) -> tuple[ProgressNote, Usage]:
        lines: list[str] = []
        for msg in middle:
            for b in msg.content:
                if isinstance(b, TextBlock):
                    lines.append(f"{msg.role}: {b.text}")
                elif isinstance(b, ToolUseBlock):
                    lines.append(f"tool call {b.name} {json.dumps(b.arguments)[:300]}")
                elif isinstance(b, ToolResultBlock):
                    lines.append(f"tool result: {b.content[:300]}")
                # Thinking is left out: it is bound to its turn, which is being dropped.
        request = LLMRequest(
            agent=agent,
            system=SUMMARY_SYSTEM,
            messages=[Message.user("\n".join(lines))],
            run_id=run.id,
            task_id=task_id,
        )
        response = await llm.complete(request)
        note, _ = parse_output(response.text, ProgressNote)
        if note is None:
            # Fall back to a mechanical note rather than failing the attempt.
            note = ProgressNote(done=[line[:160] for line in lines if line.startswith("tool")])
        return note, response.usage
