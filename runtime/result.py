from dataclasses import dataclass

from .event import ChatEvent

@dataclass
class ChatResult:
    task_id: str | None
    text: str
    success: bool
    events: list[ChatEvent]
    model: str | None
    provider_name: str | None = None
    token_usage: int | None = None
    error: str | None = None
