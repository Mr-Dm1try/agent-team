"""Section diffs between two versions of an artifact: fields compared one by one, list
items matched by id, and version bookkeeping left out."""

from __future__ import annotations

from typing import Any

import pytest

from agent_team.core.artifacts import PRD, ArtifactRef, CriterionResult, QAReport, Spec
from agent_team.core.models import AgentName, Phase, Run, Task
from agent_team.ui import diffs
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
    assert lines == ["code: code_change v1: new", "qa_report v1: new"]
    seen = {"qa_report": 1, "code_change": 1}
    same = await change_report(run, seen, None, tasks)  # type: ignore[arg-type]
    assert same == ["Nothing changed since the last version you saw."]


class _Artifacts:
    """Just enough of an artifact store: load by ref."""

    def __init__(self, *items: Any) -> None:
        """Hold the artifacts by their ref."""
        self.items = {str(ArtifactRef(kind=a.kind, version=a.version)): a for a in items}

    async def load(self, run_id: str, ref: ArtifactRef) -> Any:
        """One artifact version."""
        return self.items[str(ref)]


def _qa(version: int, evidence: str) -> QAReport:
    """A QA report with 300 criteria, each with ``evidence``."""
    results = [
        CriterionResult(criterion=f"AC-{i}", result="pass", evidence=f"{evidence} {i}")
        for i in range(300)
    ]
    return QAReport(version=version, summary="s", results=results, ran_end_to_end=True)


async def test_the_code_diff_comes_first_and_survives_a_long_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Run(id="run_1", title="t", request="r", workspace="/w")
    code = [f"+line {i}" for i in range(119)]

    async def fake_code_diff(run: Run, old: int, new: int) -> list[str]:
        """The workspace diff between two code versions."""
        return code

    monkeypatch.setattr(diffs, "code_diff", fake_code_diff)
    tasks = _Tasks([_output("qa_report", 2), _output("code_change", 2)])
    store = _Artifacts(_qa(1, "ran curl"), _qa(2, "called the endpoint"))
    before = {"qa_report": 1, "code_change": 1}
    lines = await change_report(run, before, store, tasks)  # type: ignore[arg-type]
    assert lines[0] == "code: code_change v1 → v2:"
    assert lines[1 : 1 + len(code)] == code  # every code line, before any artifact
    rest = lines[1 + len(code) :]
    assert rest[0] == "qa_report v1 → v2:"
    assert len(rest) == 1 + diffs.MAX_SECTION_LINES + 1
    assert rest[-1].startswith("[... ") and rest[-1].endswith(" more lines of the qa_report diff]")


async def test_a_long_code_diff_is_cut_on_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Run(id="run_1", title="t", request="r", workspace="/w")

    async def fake_code_diff(run: Run, old: int, new: int) -> list[str]:
        """A code diff longer than the cap."""
        return [f"+line {i}" for i in range(diffs.MAX_CODE_DIFF_LINES + 5)]

    monkeypatch.setattr(diffs, "code_diff", fake_code_diff)
    tasks = _Tasks([_output("code_change", 2)])
    lines = await change_report(run, {"code_change": 1}, None, tasks)  # type: ignore[arg-type]
    assert lines[-1] == "[... 5 more lines of the code diff]"
