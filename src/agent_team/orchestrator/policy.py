"""A minimal verdict policy, the fallback when no full policy from agent_team.audit is
passed in. It implements the VerdictPolicy Protocol:

- pass -> no decisions (the orchestrator marks the task done)
- fail with attempts left -> RETRY
- fail with none left -> ESCALATE
- a failed gate -> ASK_USER with the blockers
"""

from __future__ import annotations

from agent_team.core.models import Decision, DecisionKind, Phase, Run, Task, Verdict


class SimplePolicy:
    async def on_task_verdict(
        self, run: Run, task: Task, verdict: Verdict, history: list[Verdict]
    ) -> list[Decision]:
        if verdict.passed:
            return []
        rules = sorted({f.rule_id for f in verdict.blockers}) or ["(none)"]
        if task.attempt < task.max_attempts:
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
                reason=f"failed {', '.join(rules)} with no attempts left",
                made_by="code",
                rule="simple_policy.escalate",
            )
        ]

    async def on_gate_verdict(self, run: Run, phase: Phase, verdict: Verdict) -> list[Decision]:
        if verdict.passed:
            return []
        lines = "; ".join(f"{f.rule_id} {f.item or ''}: {f.message}" for f in verdict.blockers)
        return [
            Decision(
                run_id=run.id,
                kind=DecisionKind.ASK_USER,
                inputs={"question": f"The {phase.value} gate failed: {lines}. How to proceed?"},
                choice="ask the user",
                reason="gate failed and this policy has no automatic fix routing",
                made_by="code",
                rule="simple_policy.gate",
            )
        ]
