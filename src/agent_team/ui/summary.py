"""The end-of-run summary: the sources of a research answer, what the audit found,
QA's evidence for each acceptance criterion, the test results and every assumption.

The orchestrator's done reply already lists each artifact's summary and the
assumptions; this adds the parts that make the run reviewable, read from the store
and from the run's audit verdicts in events.jsonl.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path

from agent_team.core.artifacts import CodeChange, QAReport, ResearchAnswer, Source
from agent_team.core.interfaces import ArtifactStore, TaskStore
from agent_team.core.models import EventName, Finding, TaskStatus
from agent_team.core.tracing import read_events
from agent_team.ui.view import short

RESULT_MARK = {"pass": "✓", "fail": "✗", "not_checked": "-"}


@dataclass
class RunSummary:
    """The reviewable parts of a finished run."""

    audit_passed: bool | None = None  # None when the run had no run audit
    findings: list[Finding] = field(default_factory=list)
    criteria: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    ran_end_to_end: bool | None = None
    sources: list[str] = field(default_factory=list)  # a research answer's, one line each

    def lines(self) -> list[str]:
        """The summary as plain lines, section by section."""
        out = ["", "Sources:", *self.sources] if self.sources else []
        out += ["", "Audit report:"]
        if self.audit_passed is None:
            out.append("  no run audit was recorded")
        else:
            out.append(f"  {'passed' if self.audit_passed else 'failed'}")
        out += [
            f"  - {f.severity} {f.rule_id}: {f.message}" + (f" ({f.item})" if f.item else "")
            for f in self.findings
        ]
        if self.criteria or self.issues:
            out += ["", "Acceptance criteria (QA):", *self.criteria]
            if self.ran_end_to_end is not None:
                ran = "yes" if self.ran_end_to_end else "no"
                out.append(f"  ran end to end: {ran}")
            out += self.issues
        if self.tests:
            out += ["", "Tests:", *self.tests]
        return out


async def summarize(
    run_id: str, tasks: TaskStore, artifacts: ArtifactStore, runs_dir: str | Path
) -> RunSummary:
    """Gather the summary of one run from its live outputs and its last run audit."""
    summary = RunSummary()
    every = await tasks.list_tasks(run_id)
    live = [t for t in every if t.status is not TaskStatus.SUPERSEDED and t.output is not None]
    reports: list[QAReport] = []
    answers: list[ResearchAnswer] = []
    for task in live:
        assert task.output is not None
        art = await artifacts.load(run_id, task.output)
        if isinstance(art, QAReport):
            reports.append(art)
        elif isinstance(art, ResearchAnswer):
            answers.append(art)
        elif isinstance(art, CodeChange) and art.tests is not None:
            t = art.tests
            summary.tests.append(
                f"  {task.objective[:60]}: `{t.command}` {t.passed} passed, "
                f"{t.failed} failed, exit {t.exit_code}"
            )
    if answers:
        answer = max(answers, key=lambda a: a.version)
        summary.sources = [source_line(s) for s in answer.sources]
    if reports:
        report = max(reports, key=lambda r: r.version)
        summary.criteria = [
            f"  {RESULT_MARK.get(r.result, '?')} {r.criterion} {r.result}: {r.evidence}"
            for r in report.results
        ]
        summary.issues = [f"  issue ({i.severity}): {i.text}" for i in report.issues]
        summary.ran_end_to_end = report.ran_end_to_end
    events = await asyncio.to_thread(read_events, runs_dir, run_id)
    audits = [
        e for e in events if e.name is EventName.AUDIT_VERDICT and e.data.get("scope") == "run"
    ]
    if audits:
        last = audits[-1].data
        summary.audit_passed = bool(last.get("passed"))
        objectives = {t.id: t.objective for t in every}
        summary.findings = [
            named_tasks(Finding.model_validate(f), objectives) for f in last.get("findings") or []
        ]
    return summary


def source_line(source: Source) -> str:
    """One source as a summary line: its title, if it has one, then its URL or path."""
    title = " ".join((source.title or "").split())
    return f"  - {title}: {source.ref}" if title else f"  - {source.ref}"


def named_tasks(finding: Finding, objectives: dict[str, str]) -> Finding:
    """The finding with each task id in its message and item followed by that task's
    objective, so a reader sees which step it means."""

    def name(text: str | None) -> str | None:
        """``text`` with every known task id named."""
        if not text:
            return text
        return _TASK_ID.sub(
            lambda m: f'{m[0]} ("{short(objectives[m[0]])}")' if m[0] in objectives else m[0],
            text,
        )

    return finding.model_copy(update={"message": name(finding.message), "item": name(finding.item)})


#: A task id as the store writes it.
_TASK_ID = re.compile(r"\btask_[0-9a-f]{12,}\b")
