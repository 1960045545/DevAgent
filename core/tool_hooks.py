from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ToolCallContext:
    """Stable metadata passed to every tool lifecycle hook."""

    call_id: str
    name: str
    category: str
    arguments: dict[str, Any]
    round_index: int


class ToolHook(Protocol):
    """Lifecycle hooks for all tool invocations."""

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
