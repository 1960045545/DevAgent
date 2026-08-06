from pydantic import BaseModel
from tool_call import ToolCall
from token_usage import TokenUsage

class ModelResponse(BaseModel):
    id: str | None = None
    model: str | None = None
    text: str = ""
    raw: dict | None = None
    tool_calls: list[ToolCall] = []
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    previous_response_id: str | None = None