"""The end-of-run summary's lines: a research answer's sources, and audit findings that
name the task they are about."""

from __future__ import annotations

from agent_team.core.artifacts import Source
from agent_team.core.models import Finding, Severity
from agent_team.ui.summary import RunSummary, named_tasks, source_line

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
