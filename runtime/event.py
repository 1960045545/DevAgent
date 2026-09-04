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
    TOOL_BLOCKED = "tool_blocked"
    TOOL_OUTPUT_LARGE = "tool_output_large"
    USER_PROMPT_SUBMITTED = "user_prompt_submitted"
    BACKGROUND_NOTIFICATION_INJECTED = "background_notification_injected"
    CONTEXT_COMPACTED = "context_compacted"
    AGENT_STOP = "agent_stop"
    TODO_CREATED = "todo_created"
    TODO_UPDATED = "todo_updated"
    SUBTASK_STARTED = "subtask_started"
    SUBTASK_FINISHED = "subtask_finished"
    SUBTASK_FAILED = "subtask_failed"
    SUBTASK_DEFERRED = "subtask_deferred"
    AUTONOMOUS_TASK_CLAIMED = "autonomous_task_claimed"
    AUTONOMOUS_INBOX_MESSAGE = "autonomous_inbox_message"
    AUTONOMOUS_SHUTDOWN = "autonomous_shutdown"
    AUTONOMOUS_FAILED = "autonomous_failed"
    SUBAGENT_COMMUNICATION_READY = "subagent_communication_ready"
    BACKGROUND_QUEUED = "background_queued"
    BACKGROUND_STARTED = "background_started"
    BACKGROUND_COMPLETED = "background_completed"
    BACKGROUND_FAILED = "background_failed"
    WORKTREE_CREATED = "worktree_created"
    WORKTREE_BOUND = "worktree_bound"
    WORKTREE_CLEANUP_PENDING = "worktree_cleanup_pending"
    WORKTREE_KEPT = "worktree_kept"
    WORKTREE_REMOVED = "worktree_removed"
    WORKTREE_FAILED = "worktree_failed"

@dataclass
class ChatEvent:
    event_id: str = field(default_factory=lambda: str(uuid4()))
    task_id: str | None = None
    event_type: EventType = EventType.STARTED
    timestamp: datetime = field(default_factory=datetime.now)
    seq: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
