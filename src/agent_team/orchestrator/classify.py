"""Sorts each user message: a new build, a change to the current run, a question
about the run or a research question (docs/DESIGN.md, "Questions as well as builds").

One model call returns the kind and a reason; code then checks the answer makes sense
(a change needs a run to change) and falls back to a keyword rule if the model's reply
doesn't parse, so classification never blocks a session.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from agent_team.agents.jsonrepair import parse_output
from agent_team.core.models import AgentName, RequestKind, Run, Usage
from agent_team.llm.types import LLMClient, LLMRequest, Message

SYSTEM = """\
You sort one message from the user of a software-building agent team. Reply with one
JSON object: {"kind": ..., "reason": "...", "target": ..., "title": "..."}.

kind is one of:
- "build": a new piece of software to build
- "change": a change to the run in progress, such as a new endpoint or a rename
- "run_question": a question about the run in progress, such as why a task retried,
  what the PRD says or which files were written
- "research_question": a question to research and answer with sources, no code

target, only for "change": the first artifact the change touches: "prd" for new or
changed behaviour, "spec" for a design change, "code_change" for a code-only change.
title: a short name for a new run, only for "build" and "research_question".
"""

_QUESTION_START = re.compile(
    r"^\s*(what|why|how|which|who|when|where|is|are|does|do|can|should|explain|compare)\b",
    re.IGNORECASE,
)
_RUN_WORDS = re.compile(
    r"\b(run|task|retr(y|ied)|prd|spec|audit|verdict|phase|files?|trace)\b", re.IGNORECASE
)


class Classification(BaseModel):
    """The kind of a user message and why."""

    model_config = ConfigDict(extra="forbid")
    kind: RequestKind
    reason: str
    target: str | None = None
    title: str | None = None


def heuristic(text: str, has_run: bool) -> Classification:
    """Keyword fallback for when the model can't classify the message."""
    is_question = text.strip().endswith("?") or bool(_QUESTION_START.match(text))
    if has_run and is_question and _RUN_WORDS.search(text):
        return Classification(kind=RequestKind.RUN_QUESTION, reason="question about the run")
    if is_question:
        return Classification(kind=RequestKind.RESEARCH_QUESTION, reason="reads as a question")
    if has_run:
        return Classification(kind=RequestKind.CHANGE, reason="instruction while a run is open")
    return Classification(kind=RequestKind.BUILD, reason="instruction with no open run")


def _sanitize(c: Classification, has_run: bool) -> Classification:
    if not has_run and c.kind is RequestKind.CHANGE:
        return c.model_copy(
            update={"kind": RequestKind.BUILD, "reason": c.reason + " (no run to change)"}
        )
    if not has_run and c.kind is RequestKind.RUN_QUESTION:
        return c.model_copy(
            update={
                "kind": RequestKind.RESEARCH_QUESTION,
                "reason": c.reason + " (no run to ask about)",
            }
        )
    if c.kind is RequestKind.CHANGE and c.target not in ("prd", "spec", "code_change"):
        return c.model_copy(update={"target": "prd"})
    return c


async def classify(llm: LLMClient, text: str, run: Run | None) -> tuple[Classification, str, Usage]:
    """Returns (classification, made_by, usage); made_by is "model" or "code"."""
    context = (
        f"A run is in progress: {run.title!r} ({run.status.value}). Its request was: {run.request}"
        if run
        else "No run is in progress."
    )
    messages = [Message.user(f"{context}\n\nThe user's message:\n{text}")]
    usage = Usage()
    for _ in range(2):
        response = await llm.complete(
            LLMRequest(
                agent=AgentName.ORCHESTRATOR,
                system=SYSTEM,
                messages=messages,
                run_id=run.id if run else "session",
            )
        )
        usage = usage + response.usage
        value, errors = parse_output(response.text, Classification)
        if value is not None:
            return _sanitize(value, run is not None), "model", usage
        messages = [
            *messages,
            Message(role="assistant", content=response.content),
            Message.user("That didn't validate: " + "; ".join(errors) + ". Reply with JSON only."),
        ]
    return heuristic(text, run is not None), "code", usage
