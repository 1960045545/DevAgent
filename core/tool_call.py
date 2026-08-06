from pydantic import BaseModel

class ToolCall(BaseModel):
    id: str | None = None
    name: str
    arguments: dict
    raw: dict | None = None