import pytest
from pydantic import TypeAdapter, ValidationError

from agent_team.core.artifacts import PRD, Artifact, ArtifactRef
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
