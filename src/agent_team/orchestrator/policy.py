"""A minimal verdict policy, the fallback when no full policy from agent_team.audit is
passed in. It implements the VerdictPolicy Protocol:

- pass -> no decisions (the orchestrator marks the task done)
- fail with attempts left -> RETRY
- fail with none left, or a check that failed closed -> ESCALATE. A check that could
  not be completed already used its own attempts; it never costs the task one.
"""

from __future__ import annotations

from agent_team.core.models import AUDIT_RULE, Decision, DecisionKind, Run, Task, Verdict


class SimplePolicy:
    """Fallback verdict policy: retry until attempts run out, then escalate or ask the user."""

    async def on_task_verdict(
        self, run: Run, task: Task, verdict: Verdict, history: list[Verdict]
    ) -> list[Decision]:
        """Retry a failed task while attempts remain, then escalate it."""
        if verdict.passed:
            return []
        rules = sorted({f.rule_id for f in verdict.blockers}) or ["(none)"]
        unchecked = AUDIT_RULE in rules
        if task.attempt < task.max_attempts and not unchecked:
            return [
                Decision(
                    run_id=run.id,
                    task_id=task.id,
                    kind=DecisionKind.RETRY,
                    inputs={"task_id": task.id, "verdict_id": verdict.id},
                    choice="retry as a revision with the findings",
                    reason=f"failed {', '.join(rules)} with attempts left",
                    made_by="code",
                    rule="simple_policy.retry",
                )
            ]
        return [
            Decision(
                run_id=run.id,
                task_id=task.id,
                kind=DecisionKind.ESCALATE,
                inputs={"task_id": task.id, "verdict_id": verdict.id},
                choice="escalate",
                reason="the check could not be completed"
                if unchecked
                else f"failed {', '.join(rules)} with no attempts left",
                made_by="code",
                rule="simple_policy.escalate",
            )
        ]
