from dataclasses import dataclass

from .event import ChatEvent

@dataclass
class ChatResult:
    task_id: str | None
    text: str
    success: bool
    events: list[ChatEvent]
    model: str | None
    token_usage: int
    error: str | None = None
