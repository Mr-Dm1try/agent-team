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

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status: int | None = None,
        unreachable: bool = False,
        max_retries: int | None = None,
    ) -> None:
        """Keep which provider failed, its HTTP status, if any, whether the connection
        itself failed (the provider is down, not busy) and, for a failure that rarely
        clears, how many retries it is worth before falling back (None: the usual)."""
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.unreachable = unreachable
        self.max_retries = max_retries


class ProviderError(AgentTeamError):
    """A provider refused a call as sent (a 4xx other than a rate limit). Not retried;
    the gateway moves to the next route."""

    def __init__(self, message: str, *, provider: str, status: int | None = None) -> None:
        """Keep which provider refused the call and its HTTP status."""
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


#: Error types that mean the attempt's own output was at fault, like a failed check:
#: they count toward no progress and the stronger model, and against the first pass.
OWN_OUTPUT_ERRORS = frozenset({OutputValidationError.__name__})


def error_type(error: str | None) -> str | None:
    """The type name at the start of a task's recorded error ("Type: message")."""
    if not error or ":" not in error:
        return None
    return error.split(":", 1)[0].strip() or None


def is_own_output_error(error: str | None) -> bool:
    """Whether a recorded error means the attempt's own output was at fault."""
    return error_type(error) in OWN_OUTPUT_ERRORS


def repeats_error(error: str | None, previous: str | None) -> bool:
    """Whether an attempt failed its output the same way as the attempt before it:
    no progress, as when a retry fails the same rules on the same items."""
    if error is None or not previous or not is_own_output_error(error):
        return False
    return " ".join(error.split()).lower() == " ".join(previous.split()).lower()


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


class EmbedderUnavailable(AgentTeamError):
    """The embedding model can't be reached or gave no vectors; memory search falls
    back to BM25 alone."""

    retryable = True


class InvalidTransition(AgentTeamError):
    """A task state change that ALLOWED_TRANSITIONS doesn't permit. Always a bug."""
