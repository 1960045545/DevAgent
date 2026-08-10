from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4

from .event import ChatEvent


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskState:
    task_id: str = field(default_factory=lambda: str(uuid4()))
    user_input: str = ""
    assistant_output: str = ""
    status: TaskStatus = TaskStatus.PENDING
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    error_msg: str | None = None
    events: list[ChatEvent] = field(default_factory=list)
    model_id: str | None = None
    provider_name: str | None = None