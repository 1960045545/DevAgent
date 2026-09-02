class AgentRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        error_code: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.error_code = error_code
        self.retry_after = retry_after


class PromptTooLongError(AgentRequestError):
    """The provider rejected the request because its prompt is too large."""


class RateLimitError(AgentRequestError):
    """A provider returned HTTP 429 after the recovery budget was exhausted."""


class ProviderOverloadedError(AgentRequestError):
    """A provider returned HTTP 529 after its retry budget was exhausted."""
