"""The end-of-run summary's lines: a research answer's sources, audit findings that
name the task they are about, and memory from earlier runs."""

from __future__ import annotations

from agent_team.core.artifacts import Source
from agent_team.core.models import AgentName, Event, EventName, Finding, Severity
from agent_team.ui.summary import RunSummary, named_tasks, remembered, remembered_count, source_line

TASK = "task_34606c70506b4d94"


def test_a_finding_names_the_task_it_is_about() -> None:
    finding = Finding(
        rule_id="R8",
        severity=Severity.WARNING,
        message=f"{TASK} is still 'running' with no end time",
        item=TASK,
    )
    named = named_tasks(finding, {TASK: "Check research brief from the researcher"})
    assert named.message == (
        f"{TASK} (\"Check research brief from the researcher\") is still 'running' with no end time"
    )
    assert named.item == f'{TASK} ("Check research brief from the researcher")'
    # An id the run doesn't have stays as it is.
    assert named_tasks(finding, {}).message == finding.message


def test_sources_print_before_the_audit_report() -> None:
    summary = RunSummary(
        audit_passed=True,
        sources=[
            source_line(Source(ref="https://pypi.org/pypi/mcp/json", title="mcp on  PyPI")),
            source_line(Source(ref="notes.md")),
        ],
    )
    assert summary.lines()[:5] == [
        "",
        "Sources:",
        "  - mcp on PyPI: https://pypi.org/pypi/mcp/json",
        "  - notes.md",
        "",
    ]
    assert summary.lines()[5] == "Audit report:"
    assert RunSummary(audit_passed=True).lines()[:2] == ["", "Audit report:"]


def test_a_source_line_shows_the_page_title_without_notes_on_how_it_was_read() -> None:
    rfc = Source(
        ref="https://www.rfc-editor.org/rfc/rfc9110",
        title="RFC 9110: HTTP Semantics (opened in this attempt at offset 361000)",
    )
    assert (
        source_line(rfc) == "  - RFC 9110: HTTP Semantics: https://www.rfc-editor.org/rfc/rfc9110"
    )
    assert rfc.read == "opened in this attempt at offset 361000"
    read = Source(
        ref="https://sqlite.org/wal.html", title="WAL", read="characters 0-10000 of 27246"
    )
    assert "10000" not in source_line(read)


def _fetched(agent: AgentName, *memories: dict[str, str]) -> Event:
    """A memory_fetched event for ``agent`` with these memories from other runs."""
    return Event(
        name=EventName.MEMORY_FETCHED,
        run_id="run_1",
        agent=agent,
        data={"ids": [m["id"] for m in memories], "from_other_runs": list(memories)},
    )


def test_memory_from_earlier_runs_is_named_once_for_the_builders_only() -> None:
    a = {"id": "mem_a", "kind": "decision", "run_id": "run_0", "text": "paginate, default 50"}
    b = {"id": "mem_b", "kind": "decision", "run_id": "run_0", "text": "use SQLite"}
    c = {"id": "mem_c", "kind": "lesson", "run_id": "run_0", "text": "run the tests"}
    events = [
        _fetched(AgentName.PRODUCT, a),
        _fetched(AgentName.ARCHITECT, a, b),  # a again: named once
        _fetched(AgentName.RESEARCHER, c),  # research isn't the build
        _fetched(AgentName.ENGINEER),
        Event(name=EventName.MEMORY_FETCHED, run_id="run_1", agent=AgentName.ENGINEER, data={}),
    ]
    found = remembered(events)
    assert [m["id"] for m in found] == ["mem_a", "mem_b"]
    assert remembered_count(found) == "2 decisions"
    assert remembered_count([a, c]) == "1 decision and 1 lesson"
    summaries = [{"id": f"mem_{i}", "kind": "run_summary"} for i in range(2)]
    assert remembered_count([*summaries, a]) == "2 run summaries and 1 decision"
    lines = RunSummary(remembered=found).lines()
    assert "From memory: 2 decisions from earlier runs" in lines
    assert "  - decision mem_a from run_0: 'paginate, default 50'" in lines
