from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

from core.tool_space import ToolSpec


TodoStatus = Literal["pending", "in_progress", "completed", "blocked"]
TodoObserver = Callable[[str, dict[str, Any]], None]


TODO_CREATE_SPEC = ToolSpec(
    name="todo_create",
    description=(
        "Create the execution plan for a complex task. Call this before "
        "using any other tool when a task has multiple steps."
    ),
    parameters={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 20,
                "description": "Ordered, actionable tasks to complete.",
            },
        },
        "required": ["items"],
        "additionalProperties": False,
    },
    category="planning",
)

TODO_UPDATE_SPEC = ToolSpec(
    name="todo_update",
    description=(
        "Update one task item. Mark work in_progress before doing it and "
        "completed only after it is actually finished; use blocked when it "
        "cannot be completed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "item_id": {
                "type": "string",
                "description": "The id returned by todo_create.",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "blocked"],
            },
            "note": {
                "type": "string",
                "description": "Short progress or blocking note.",
                "default": "",
            },
        },
        "required": ["item_id", "status"],
        "additionalProperties": False,
    },
    category="planning",
)

TODO_LIST_SPEC = ToolSpec(
    name="todo_list",
    description="Read the current task plan and progress.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    category="planning",
)


@dataclass(slots=True)
class TodoItem:
    item_id: str
    title: str
    status: TodoStatus = "pending"
    note: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class TodoList:
    """Per-request task plan owned by the host, not by model text."""

    def __init__(self, observer: TodoObserver | None = None) -> None:
        self.items: list[TodoItem] = []
        self.observer = observer

    @property
    def created(self) -> bool:
        return bool(self.items)

    @property
    def is_terminal(self) -> bool:
        return bool(self.items) and all(
            item.status in {"completed", "blocked"}
            for item in self.items
        )

    def create(self, items: list[str]) -> dict[str, Any]:
        if self.created:
            raise ValueError("todo list has already been created")
        normalized = [item.strip() for item in items if item.strip()]
        if not normalized:
            raise ValueError("todo list must contain at least one item")
        if len(normalized) > 20:
            raise ValueError("todo list cannot contain more than 20 items")

        self.items = [
            TodoItem(
                item_id=f"todo-{index}",
                title=title,
            )
            for index, title in enumerate(normalized, start=1)
        ]
        snapshot = self.snapshot()
        self._emit("todo_created", snapshot)
        return snapshot

    def update(
        self,
        item_id: str,
        status: TodoStatus,
        note: str = "",
    ) -> dict[str, Any]:
        if status not in {"pending", "in_progress", "completed", "blocked"}:
            raise ValueError(f"unsupported todo status: {status}")
        for item in self.items:
            if item.item_id != item_id:
                continue
            self._validate_transition(item, status)
            item.status = status
            item.note = note.strip()
            snapshot = self.snapshot()
            self._emit(
                "todo_updated",
                {
                    **snapshot,
                    "updated_item_id": item_id,
                },
            )
            return snapshot
        raise ValueError(f"unknown todo item: {item_id}")

    def _validate_transition(
        self,
        item: TodoItem,
        next_status: TodoStatus,
    ) -> None:
        current_status = item.status
        if current_status in {"completed", "blocked"}:
            if next_status != current_status:
                raise ValueError(
                    f"terminal todo item cannot transition from "
                    f"{current_status} to {next_status}",
                )
            return

        if current_status == "pending":
            allowed = {"pending", "in_progress", "blocked"}
        else:
            allowed = {"in_progress", "completed", "blocked"}
        if next_status not in allowed:
            raise ValueError(
                f"invalid todo transition from "
                f"{current_status} to {next_status}",
            )

        if next_status == "in_progress":
            another_in_progress = any(
                other.item_id != item.item_id
                and other.status == "in_progress"
                for other in self.items
            )
            if another_in_progress:
                raise ValueError(
                    "only one todo item may be in_progress at a time",
                )

    def snapshot(self) -> dict[str, Any]:
        return {
            "todos": [item.to_dict() for item in self.items],
            "completed_count": sum(
                item.status == "completed"
                for item in self.items
            ),
            "total_count": len(self.items),
            "is_terminal": self.is_terminal,
        }

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        if self.observer is not None:
            self.observer(event_type, data)


class TodoToolset:
    """Request-scoped Todo tools and state."""

    def __init__(self, observer: TodoObserver | None = None) -> None:
        self.todo_list = TodoList(observer=observer)

    @property
    def specs(self) -> list[ToolSpec]:
        return [TODO_CREATE_SPEC, TODO_UPDATE_SPEC, TODO_LIST_SPEC]

    @property
    def handlers(self) -> dict[str, Callable[..., Any]]:
        return {
            "todo_create": self.todo_list.create,
            "todo_update": self.todo_list.update,
            "todo_list": lambda: self.todo_list.snapshot(),
        }


def is_complex_task(message: str) -> bool:
    """Conservative local heuristic; the model still owns the task details."""
    text = " ".join(message.split())
    if len(text) >= 180:
        return True
    if text.count("?") + text.count("？") >= 2:
        return True

    markers = (
        "并且",
        "然后",
        "同时",
        "分别",
        "步骤",
        "先",
        "最后",
        "分析并",
        "实现并",
        "检查并",
        "修改并",
        "部署并",
        "and",
        "then",
        "also",
        "step",
    )
    marker_count = sum(marker in text.lower() for marker in markers)
    return marker_count >= 2 or text.count("。") + text.count(".") >= 3
