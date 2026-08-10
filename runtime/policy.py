from dataclasses import dataclass

@dataclass
class RuntimePolicy:
    stream: bool = False
    collect_events: bool = True
    raise_on_error: bool = False
    persist_task_state: bool = False
    timeout_seconds: float | None = None