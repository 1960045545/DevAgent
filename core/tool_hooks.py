from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(slots=True)
class UserPromptSubmitContext:
    """Mutable request context shared by UserPromptSubmit hooks."""

    user_message: str
    injected_messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    submitted_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def inject_message(self, content: str, *, role: str = "system") -> None:
        value = content.strip()
        if value:
            self.injected_messages.append({"role": role, "content": value})


@dataclass(frozen=True, slots=True)
class ToolCallContext:
    """Stable metadata passed to every tool lifecycle hook."""

    call_id: str
    name: str
    category: str
    arguments: dict[str, Any]
    round_index: int


@dataclass(frozen=True, slots=True)
class ToolHookDecision:
    """A PreToolUse decision; allowed calls continue to the handler."""

    blocked: bool = False
    reason: str = ""
    result: Any = None

    @classmethod
    def block(
        cls,
        reason: str,
        *,
        result: Any = None,
    ) -> "ToolHookDecision":
        return cls(blocked=True, reason=reason, result=result)


@dataclass(frozen=True, slots=True)
class ToolPostprocessResult:
    """Explicitly replaces a tool result from a PostToolUse hook."""

    result: Any


@dataclass(frozen=True, slots=True)
class StopContext:
    """Information emitted when the agent loop is about to return."""

    response: Any
    messages: tuple[dict[str, Any], ...]
    round_index: int
    reason: str


class ToolHook(Protocol):
    """Comprehensive agent lifecycle hook protocol.

    Hook methods are discovered dynamically, so implementations may provide
    only the phases they need. The three legacy tool methods remain supported.
    """

    def on_user_prompt_submit(
        self,
        context: UserPromptSubmitContext,
    ) -> None:
        ...

    def pre_tool_use(
        self,
        context: ToolCallContext,
    ) -> ToolHookDecision | None:
        ...

    def post_tool_use(
        self,
        context: ToolCallContext,
        result: Any,
    ) -> ToolPostprocessResult | None:
        ...

    def on_stop(self, context: StopContext) -> None:
        ...

    def before_tool_call(self, context: ToolCallContext) -> None:
        ...

    def after_tool_call(
        self,
        context: ToolCallContext,
        result: Any,
    ) -> None:
        ...

    def on_tool_error(
        self,
        context: ToolCallContext,
        error: Exception,
    ) -> None:
        ...
