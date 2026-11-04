"""Typed errors. Every failure in the harness maps to one of these.

The dispatcher catches everything at the agent boundary, stores the traceback on the
task record, and applies the policy in docs/DESIGN.md ("Logging, tracing and error
handling"). Anything that isn't an AgentTeamError is treated as a bug in our code.
"""

from __future__ import annotations


class AgentTeamError(Exception):
    """Base class for every error the harness raises on purpose."""

    #: Whether retrying the same operation can reasonably succeed.
    retryable: bool = False


class TransientLLMError(AgentTeamError):
    """Rate limit, timeout or 5xx from a model provider. Retried with backoff, then fallback."""

    retryable = True

    def __init__(self, message: str, *, provider: str, status: int | None = None) -> None:
        """Keep which provider failed and its HTTP status, if any."""
        super().__init__(message)
        self.provider = provider
        self.status = status


class ContextTooLong(AgentTeamError):
    """The prompt plus room for the reply doesn't fit the model's context window.
    Raised instead of sending, since a local model would silently cut the start of
    the prompt. Not retried on the same provider; the gateway moves to the next one."""

    def __init__(self, message: str, *, provider: str) -> None:
        """Keep which provider refused the prompt."""
        super().__init__(message)
        self.provider = provider


class OutputValidationError(AgentTeamError):
    """A model response didn't parse into the agent's output schema."""

    retryable = True

    def __init__(self, message: str, *, errors: list[str]) -> None:
        """Keep the validation errors to send back to the model."""
        super().__init__(message)
        self.errors = errors


class ToolError(AgentTeamError):
    """A tool failed, e.g. file not found or a command exited non-zero.

    Returned to the agent as a tool result so it can adapt; never kills the attempt.
    """

    def __init__(self, message: str, *, tool: str) -> None:
        """Keep the name of the tool that failed."""
        super().__init__(message)
        self.tool = tool


class ToolBlocked(ToolError):
    """The auditor's pre-check refused a tool call."""

    def __init__(self, message: str, *, tool: str, rule_id: str) -> None:
        """Keep the tool and the rule that blocked it."""
        super().__init__(message, tool=tool)
        self.rule_id = rule_id


class SandboxError(AgentTeamError):
    """Docker missing or a container failure. Stops the app at start when Docker is
    missing; during a command the tool turns it into an error the model can retry."""


class AuditFailure(AgentTeamError):
    """A blocker rule failed. Carries the verdict id so the orchestrator can route it."""

    def __init__(self, message: str, *, verdict_id: str) -> None:
        """Keep the id of the verdict that failed."""
        super().__init__(message)
        self.verdict_id = verdict_id


class BudgetExceeded(AgentTeamError):
    """A run passed its token, cost or time cap. The run stops cleanly and can resume."""

    def __init__(self, message: str, *, limit: str) -> None:
        """Keep the name of the cap that was hit."""
        super().__init__(message)
        self.limit = limit


class RunStopped(AgentTeamError):
    """The kill switch (`agent-team stop`) halted the run at a safe point."""


class InvalidTransition(AgentTeamError):
    """A task state change that ALLOWED_TRANSITIONS doesn't permit. Always a bug."""
