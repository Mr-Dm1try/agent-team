"""A scripted team: canned replies per agent that make a whole run go through."""

from __future__ import annotations

import json
from typing import Any

from agent_team.core.models import AgentName
from agent_team.llm.types import ContentBlock, LLMRequest, ToolResultBlock
from agents.fakes import reply, tool_call

PLAN_BUILD: dict[str, Any] = {
    "summary": "Research the database, then PRD, spec, code and QA.",
    "request_kind": "build",
    "asks": [{"id": "ASK-1", "text": "A REST API that lists books"}],
    "research_questions": ["Which database?"],
    "tasks": [
        {
            "key": "research-1",
            "agent": "researcher",
            "phase": "planning",
            "objective": "Find which database to use",
            "output_kind": "research_brief",
            "covers_asks": ["ASK-1"],
        },
        {
            "key": "prd",
            "agent": "product",
            "phase": "prd",
            "objective": "Write the PRD",
            "output_kind": "prd",
            "covers_asks": ["ASK-1"],
            "depends_on": ["research-1"],
        },
        {
            "key": "spec",
            "agent": "architect",
            "phase": "spec",
            "objective": "Write the spec",
            "output_kind": "spec",
            "covers_asks": ["ASK-1"],
        },
        {
            "key": "code",
            "agent": "engineer",
            "phase": "implementation",
            "objective": "Implement the API",
            "output_kind": "code_change",
            "covers_asks": ["ASK-1"],
        },
        {
            "key": "qa",
            "agent": "qa",
            "phase": "implementation",
            "objective": "Review and run it",
            "output_kind": "qa_report",
            "covers_asks": ["ASK-1"],
        },
    ],
}

PLAN_RESEARCH: dict[str, Any] = {
    "summary": "One researcher task answers the question.",
    "request_kind": "research_question",
    "asks": [{"id": "ASK-1", "text": "Which Python web framework is fastest?"}],
    "tasks": [
        {
            "key": "answer",
            "agent": "researcher",
            "phase": "planning",
            "objective": "Answer the question with sources",
            "output_kind": "research_answer",
            "covers_asks": ["ASK-1"],
        }
    ],
}

BRIEF = {
    "summary": "SQLite fits.",
    "answers": [
        {"question": "Which database?", "answer": "SQLite", "sources": [{"ref": "notes.md"}]}
    ],
    "findings": [{"claim": "SQLite needs no server", "sources": [{"ref": "notes.md"}]}],
}
ANSWER = {
    "summary": "Starlette-based frameworks lead.",
    "question": "Which Python web framework is fastest?",
    "answer": "Starlette-based frameworks lead the benchmarks.",
    "sources": [{"ref": "https://example.com/bench"}],
}
PRD = {
    "summary": "One requirement.",
    "title": "Books API",
    "requirements": [
        {
            "id": "REQ-1",
            "text": "List books",
            "asks": ["ASK-1"],
            "acceptance_criteria": [{"id": "AC-1.1", "text": "GET /books returns 200"}],
        }
    ],
}
SPEC = {
    "summary": "One module.",
    "components": [
        {"name": "api", "responsibility": "routes", "requirements": ["REQ-1"], "files": ["app.py"]}
    ],
    "checks": [{"id": "CHK-1", "criteria": ["AC-1.1"], "kind": "unit_test", "description": "test"}],
    "dependencies": ["fastapi==0.115.0"],
}
CODE = {"summary": "Wrote app.py.", "files_written": ["app.py"]}
QA = {
    "summary": "All good.",
    "results": [{"criterion": "AC-1.1", "result": "pass", "evidence": "app.py:1"}],
    "ran_end_to_end": False,
}


def _last_is_tool_result(request: LLMRequest) -> bool:
    last = request.messages[-1]
    return any(isinstance(b, ToolResultBlock) for b in last.content)


def orchestrator_responder(
    *, kind: str = "build", title: str = "Books API", plan: dict[str, Any] | None = None
) -> Any:
    def respond(request: LLMRequest) -> list[ContentBlock] | str:
        if request.system.startswith("You sort one message"):
            return json.dumps({"kind": kind, "reason": "scripted", "title": title})
        if "Orchestrator's planner" in request.system:
            chosen = plan or (PLAN_RESEARCH if kind == "research_question" else PLAN_BUILD)
            return json.dumps(chosen)
        if "questions about a run" in request.system:
            if not _last_is_tool_result(request):
                return tool_call("list_tasks", {})
            return json.dumps({"answer": "The PRD retried once.", "cited": ["prd@v2"]})
        if request.system.startswith("You compress"):
            return json.dumps({"done": ["things"], "decisions": [], "remaining": []})
        raise AssertionError("unexpected orchestrator call")

    return respond


def engineer_responder(request: LLMRequest) -> list[ContentBlock] | str:
    if not _last_is_tool_result(request):
        return tool_call("write_file", {"path": "app.py", "content": "print('books')\n"})
    return json.dumps(CODE)


def researcher_responder(request: LLMRequest) -> list[ContentBlock] | str:
    first = request.messages[0].content[0]
    text = getattr(first, "text", "")
    if "Produce: research_answer" in text:
        return json.dumps(ANSWER)
    return json.dumps(BRIEF)


def team_script(kind: str = "build") -> dict[AgentName, Any]:
    return {
        AgentName.ORCHESTRATOR: orchestrator_responder(kind=kind),
        AgentName.RESEARCHER: researcher_responder,
        AgentName.PRODUCT: lambda r: json.dumps(PRD),
        AgentName.ARCHITECT: lambda r: json.dumps(SPEC),
        AgentName.ENGINEER: engineer_responder,
        AgentName.QA: lambda r: json.dumps(QA),
    }


__all__ = ["reply"]
