"""Section diffs between two versions of an artifact: fields compared one by one, list
items matched by id, and version bookkeeping left out."""

from __future__ import annotations

from typing import Any

from agent_team.core.artifacts import PRD, ArtifactRef, Spec
from agent_team.core.models import AgentName, Phase, Run, Task
from agent_team.ui.diffs import change_report, section_diff


def _req(req_id: str, text: str) -> dict[str, Any]:
    """One requirement with one criterion."""
    criteria = [{"id": f"AC-{req_id}", "text": f"{text} works"}]
    return {"id": req_id, "text": text, "asks": ["ASK-1"], "acceptance_criteria": criteria}


def _prd(version: int, *reqs: dict[str, Any], summary: str = "Books.") -> PRD:
    """A PRD at ``version`` with the given requirements."""
    return PRD(
        version=version,
        task_id=f"task_{version}",
        summary=summary,
        title="Books API",
        requirements=list(reqs),
    )


def test_a_reordered_list_with_one_new_item_shows_only_the_new_item() -> None:
    old = _prd(1, _req("REQ-1", "List books"), _req("REQ-2", "Show a book"))
    new = _prd(
        2, _req("REQ-2", "Show a book"), _req("REQ-1", "List books"), _req("REQ-3", "Paginate")
    )
    lines = section_diff(old, new)
    assert lines[0] == "+ added requirements[REQ-3]"
    assert any(line.startswith("+") and "Paginate" in line for line in lines)
    assert not any("REQ-1" in line or "REQ-2" in line for line in lines)
    assert not any("version" in line or "task_id" in line for line in lines)


def test_a_changed_item_and_field_show_as_unified_diffs() -> None:
    old = _prd(1, _req("REQ-1", "List books"), _req("REQ-2", "Show a book"))
    new = _prd(2, _req("REQ-1", "List books by title"), summary="Books, sorted.")
    lines = section_diff(old, new)
    assert "- removed requirements[REQ-2]" in lines
    assert "--- requirements[REQ-1] (before)" in lines
    assert '-  "text": "List books"' in lines and '+  "text": "List books by title"' in lines
    assert "-Books." in lines and "+Books, sorted." in lines


def test_versions_with_the_same_content_have_no_diff() -> None:
    assert section_diff(_prd(1, _req("REQ-1", "x")), _prd(2, _req("REQ-1", "x"))) == []


def _spec(*names: str) -> Spec:
    """A spec with one component per name, each covering REQ-1."""
    components = [
        {"name": n, "responsibility": f"part {i}", "requirements": ["REQ-1"], "files": []}
        for i, n in enumerate(names)
    ]
    return Spec(summary="s", components=components, checks=[])


def test_items_that_share_a_key_are_matched_by_place() -> None:
    lines = section_diff(_spec("api", "api"), _spec("api", "api", "db"))
    assert "+ added components[#3]" in lines
    assert not any("#1" in line or "#2" in line for line in lines)


class _Tasks:
    """Just enough of a task store: the run's tasks."""

    def __init__(self, tasks: list[Task]) -> None:
        """Hold the tasks."""
        self.tasks = tasks

    async def list_tasks(self, run_id: str) -> list[Task]:
        """Every task of the run."""
        return self.tasks


def _output(kind: str, version: int) -> Task:
    """A task that wrote ``kind`` at ``version``."""
    return Task(
        run_id="run_1",
        phase=Phase.IMPLEMENTATION,
        agent=AgentName.ENGINEER,
        objective="o",
        output_kind=kind,
        output=ArtifactRef(kind=kind, version=version),
    )


async def test_kinds_first_written_since_the_last_review_are_named() -> None:
    run = Run(id="run_1", title="t", request="r", workspace="/w")
    tasks = _Tasks([_output("qa_report", 1), _output("code_change", 1)])
    # Nothing is loaded: there is no earlier version to compare with.
    lines = await change_report(run, {}, None, tasks)  # type: ignore[arg-type]
    assert lines == ["qa_report v1: new", "code: code_change v1: new"]
    seen = {"qa_report": 1, "code_change": 1}
    same = await change_report(run, seen, None, tasks)  # type: ignore[arg-type]
    assert same == ["Nothing changed since the last version you saw."]
