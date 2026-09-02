from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

class EventType(str, Enum):
    STARTED = "started"
    DELTA = "delta"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    TOOL_FAILED = "tool_failed"
    TODO_CREATED = "todo_created"
    TODO_UPDATED = "todo_updated"
    SUBTASK_STARTED = "subtask_started"
    SUBTASK_FINISHED = "subtask_finished"
    SUBTASK_FAILED = "subtask_failed"
    SUBTASK_DEFERRED = "subtask_deferred"
    BACKGROUND_QUEUED = "background_queued"
    BACKGROUND_STARTED = "background_started"
    BACKGROUND_COMPLETED = "background_completed"
    BACKGROUND_FAILED = "background_failed"

@dataclass
class ChatEvent:
    event_id: str = field(default_factory=lambda: str(uuid4()))
    task_id: str | None = None
    event_type: EventType = EventType.STARTED
    timestamp: datetime = field(default_factory=datetime.now)
    seq: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
