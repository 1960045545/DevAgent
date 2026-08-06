from dataclasses import dataclass
from typing import Any, Literal
from tool_space import ToolSpec

@dataclass
class InvokeOptions:
    tools: list["ToolSpec"] | None = None
    tool_choice: dict[str, Any] | str | None = None
    response_format: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    previous_response_id: str | None = None
    metadata: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None
    extra_headers: dict[str, str] | None = None
    timeout: float | None = None