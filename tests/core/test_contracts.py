import pytest
from pydantic import TypeAdapter, ValidationError

from agent_team.core.artifacts import PRD, Artifact, ArtifactRef, ResearchAnswer, Source
from agent_team.core.models import (
    ALLOWED_TRANSITIONS,
    AgentName,
    Phase,
    Task,
    TaskStatus,
    can_transition,
)


def test_artifact_ref_round_trip() -> None:
    ref = ArtifactRef.parse("prd@v2")
    assert ref == ArtifactRef(kind="prd", version=2)
    assert str(ref) == "prd@v2"
    with pytest.raises(ValueError):
        ArtifactRef.parse("prd-2")


def test_every_status_has_transition_entry() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(TaskStatus)


def test_done_only_goes_to_superseded() -> None:
    assert can_transition(TaskStatus.DONE, TaskStatus.SUPERSEDED)
    assert not can_transition(TaskStatus.DONE, TaskStatus.READY)
    assert not ALLOWED_TRANSITIONS[TaskStatus.SUPERSEDED]


def test_artifact_union_discriminates_on_kind() -> None:
    data = {
        "kind": "prd",
        "summary": "s",
        "title": "Inventory API",
        "requirements": [
            {
                "id": "REQ-1",
                "text": "List books",
                "asks": ["ASK-1"],
                "acceptance_criteria": [{"id": "AC-1.1", "text": "GET /books returns 200"}],
            }
        ],
    }
    artifact = TypeAdapter(Artifact).validate_python(data)
    assert isinstance(artifact, PRD)


def test_requirement_must_cite_an_ask() -> None:
    with pytest.raises(ValidationError):
        PRD.model_validate(
            {
                "summary": "s",
                "title": "t",
                "requirements": [
                    {"id": "REQ-1", "text": "x", "asks": [], "acceptance_criteria": []}
                ],
            }
        )


def test_task_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(
            {
                "run_id": "run_1",
                "phase": Phase.PRD,
                "agent": AgentName.PRODUCT,
                "objective": "o",
                "output_kind": "prd",
                "surprise": 1,
            }
        )


def test_plan_holds_asks_and_questions_but_no_tasks() -> None:
    data = {
        "kind": "plan",
        "summary": "s",
        "request_kind": "build",
        "asks": [{"id": "ASK-1", "text": "An inventory API"}],
        "research_questions": [{"question": "Which database?", "ask": "ASK-1"}],
    }
    plan = TypeAdapter(Artifact).validate_python(data)
    assert plan.kind == "plan"
    with pytest.raises(ValidationError):
        TypeAdapter(Artifact).validate_python({**data, "tasks": []})
    with pytest.raises(ValidationError):
        TypeAdapter(Artifact).validate_python({**data, "asks": []})


def test_contract_guards() -> None:
    from agent_team.core.artifacts import ARTIFACT_TYPES, Plan
    from agent_team.core.models import Finding, RuleScope, Severity, Verdict

    assert set(ARTIFACT_TYPES) == {
        "plan", "research_brief", "prd", "spec", "code_change", "qa_report", "research_answer"
    }  # fmt: skip
    with pytest.raises(ValidationError):
        Plan(summary="s", request_kind="banana", asks=[{"id": "ASK-1", "text": "x"}])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ArtifactRef.parse("prd@v2\n")
    with pytest.raises(ValidationError):
        ArtifactRef(kind="PRD", version=1)
    blocker = Finding(rule_id="R1", severity=Severity.BLOCKER, message="bad")
    with pytest.raises(ValidationError):
        Verdict(run_id="r", scope=RuleScope.TASK, passed=True, findings=[blocker])
    assert not Verdict(run_id="r", scope=RuleScope.TASK, passed=False, findings=[blocker]).passed
    assert can_transition(TaskStatus.RUNNING, TaskStatus.SUPERSEDED)
    assert can_transition(TaskStatus.AUDITING, TaskStatus.SUPERSEDED)


@pytest.mark.parametrize(
    ("title", "clean", "read"),
    [
        (
            "RFC 9110 (opened for its header only; §15.5.19 not reached)",
            "RFC 9110",
            "opened for its header only; §15.5.19 not reached",
        ),
        (
            "Write-Ahead Logging, wal.html (first 10,000 of 27,246 characters)",
            "Write-Ahead Logging, wal.html",
            "first 10,000 of 27,246 characters",
        ),
        ("HTTP 418 - MDN, opened at offset 20000", "HTTP 418 - MDN", "opened at offset 20000"),
        # Titles that only look like notes keep every word.
        ("Write-Ahead Logging (WAL)", "Write-Ahead Logging (WAL)", None),
        ("Python - Unicode characters", "Python - Unicode characters", None),
        ("Opened files, fetched by offset 3", "Opened files", "fetched by offset 3"),
    ],
)
def test_source_titles_carry_no_notes_on_how_they_were_read(
    title: str, clean: str, read: str | None
) -> None:
    source = Source(ref="https://example.org", title=title)
    assert (source.title, source.read) == (clean, read)


def test_a_read_note_moved_from_the_title_joins_the_read_field() -> None:
    source = Source.model_validate(
        {"ref": "a.md", "title": "Notes (opened at offset 4000)", "read": "lines 1-80"}
    )
    assert source.title == "Notes"
    assert source.read == "lines 1-80; opened at offset 4000"
    # A title that is nothing but a note isn't emptied: its first part stays.
    assert Source(ref="a.md", title="opened at offset 0").title == "opened at offset 0"


def test_a_research_answer_keeps_coverage_notes_in_gaps() -> None:
    answer = ResearchAnswer(
        summary="s",
        question="q",
        answer="WAL lets readers run while one writer appends.",
        sources=[Source(ref="https://sqlite.org/wal.html")],
        gaps=["read the first 10,000 of 27,246 characters of wal.html"],
    )
    assert answer.gaps and "10,000" not in answer.answer
    assert ResearchAnswer.model_validate(answer.model_dump()).gaps == answer.gaps
