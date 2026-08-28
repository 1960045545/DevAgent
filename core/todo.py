from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any, Callable, Literal

from core.tool_space import ToolSpec


TodoStatus = Literal[
    "pending",
    "in_progress",
    "completed",
    "blocked",
    "failed",
]
TodoObserver = Callable[[str, dict[str, Any]], None]
TodoCreateItem = str | dict[str, Any]


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
                "items": {
                    "oneOf": [
                        {"type": "string", "minLength": 1},
                        {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "minLength": 1},
                                "dependencies": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["title"],
                            "additionalProperties": False,
                        },
                    ],
                },
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
                "enum": [
                    "pending",
                    "in_progress",
                    "completed",
                    "blocked",
                    "failed",
                ],
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
    dependencies: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.dependencies is None:
            value.pop("dependencies", None)
        return value


class TodoList:
    """Per-request task plan owned by the host, not by model text."""

    def __init__(self, observer: TodoObserver | None = None) -> None:
        self.items: list[TodoItem] = []
        self.observer = observer
        self._lock = RLock()

    @property
    def created(self) -> bool:
        with self._lock:
            return bool(self.items)

    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return bool(self.items) and all(
                item.status in {"completed", "blocked", "failed"}
                for item in self.items
            )

    def create(self, items: list[TodoCreateItem]) -> dict[str, Any]:
        with self._lock:
            if self.created:
                raise ValueError("todo list has already been created")

            normalized: list[tuple[str, list[str]]] = []
            for item in items:
                if isinstance(item, str):
                    title = item.strip()
                    dependencies: list[str] = []
                elif isinstance(item, dict):
                    title = str(item.get("title", "")).strip()
                    raw_dependencies = item.get("dependencies", [])
                    dependencies = [
                        dependency.strip()
                        for dependency in raw_dependencies
                        if isinstance(dependency, str) and dependency.strip()
                    ]
                else:
                    continue
                if title:
                    normalized.append((title, dependencies))

            if not normalized:
                raise ValueError("todo list must contain at least one item")
            if len(normalized) > 20:
                raise ValueError("todo list cannot contain more than 20 items")

            candidate_items = [
                TodoItem(
                    item_id=f"todo-{index}",
                    title=title,
                    dependencies=dependencies or None,
                )
                for index, (title, dependencies) in enumerate(
                    normalized,
                    start=1,
                )
            ]
            self._validate_dependencies(candidate_items)
            self.items = candidate_items
            snapshot = self.snapshot()
        self._emit("todo_created", snapshot)
        return snapshot

    def update(
        self,
        item_id: str,
        status: TodoStatus,
        note: str = "",
    ) -> dict[str, Any]:
        if status not in {
            "pending",
            "in_progress",
            "completed",
            "blocked",
            "failed",
        }:
            raise ValueError(f"unsupported todo status: {status}")
        with self._lock:
            for item in self.items:
                if item.item_id != item_id:
                    continue
                self._validate_transition(item, status)
                item.status = status
                item.note = note.strip()
                snapshot = self.snapshot()
                event_data = {
                    **snapshot,
                    "updated_item_id": item_id,
                }
                break
            else:
                raise ValueError(f"unknown todo item: {item_id}")
        self._emit("todo_updated", event_data)
        return snapshot

    def get(self, item_id: str) -> TodoItem:
        with self._lock:
            for item in self.items:
                if item.item_id == item_id:
                    return item
        raise ValueError(f"unknown todo item: {item_id}")

    def is_ready(self, item_id: str) -> bool:
        with self._lock:
            item = self.get(item_id)
            if item.status != "pending":
                return False
            item_ids = {candidate.item_id for candidate in self.items}
            missing = [
                dependency
                for dependency in item.dependencies or []
                if dependency not in item_ids
            ]
            if missing:
                raise ValueError(
                    f"todo item {item_id} has unknown dependencies: {missing}"
                )
            return all(
                self.get(dependency).status == "completed"
                for dependency in item.dependencies or []
            )

    def claim(self, item_id: str, note: str = "") -> dict[str, Any]:
        """Atomically reserve a ready pending item for one executor."""
        with self._lock:
            item = self.get(item_id)
            if not self.is_ready(item_id):
                raise ValueError(
                    f"todo item {item_id} is not ready for execution"
                )
            item.status = "in_progress"
            item.note = note.strip()
            snapshot = self.snapshot()
            event_data = {
                **snapshot,
                "updated_item_id": item_id,
            }
        self._emit("todo_updated", event_data)
        return snapshot

    def _validate_transition(
        self,
        item: TodoItem,
        next_status: TodoStatus,
    ) -> None:
        current_status = item.status
        if current_status in {"completed", "blocked", "failed"}:
            if next_status != current_status:
                raise ValueError(
                    f"terminal todo item cannot transition from "
                    f"{current_status} to {next_status}",
                )
            return

        if current_status == "pending":
            allowed = {"pending", "in_progress", "blocked", "failed"}
        else:
            allowed = {"in_progress", "completed", "blocked", "failed"}
        if next_status not in allowed:
            raise ValueError(
                f"invalid todo transition from "
                f"{current_status} to {next_status}",
            )

    def _validate_dependencies(
        self,
        items: list[TodoItem] | None = None,
    ) -> None:
        items = self.items if items is None else items
        item_ids = {item.item_id for item in items}
        dependencies_by_id = {
            item.item_id: set(item.dependencies or [])
            for item in items
        }
        for item in items:
            unknown = [
                dependency
                for dependency in item.dependencies or []
                if dependency not in item_ids
            ]
            if unknown:
                raise ValueError(
                    f"todo item {item.item_id} has unknown dependencies: {unknown}"
                )
            if item.item_id in (item.dependencies or []):
                raise ValueError(
                    f"todo item {item.item_id} cannot depend on itself"
                )

        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(item_id: str) -> None:
            if item_id in visiting:
                raise ValueError(
                    f"todo dependency cycle detected at {item_id}"
                )
            if item_id in visited:
                return

            visiting.add(item_id)
            for dependency in dependencies_by_id[item_id]:
                visit(dependency)
            visiting.remove(item_id)
            visited.add(item_id)

        for item_id in item_ids:
            visit(item_id)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "todos": [item.to_dict() for item in self.items],
                "completed_count": sum(
                    item.status == "completed"
                    for item in self.items
                ),
                "total_count": len(self.items),
                "is_terminal": all(
                    item.status in {"completed", "blocked", "failed"}
                    for item in self.items
                ) if self.items else False,
            }

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        if self.observer is not None:
            self.observer(event_type, data)


class TodoToolset:
    """Request-scoped Todo tools and state."""

    def __init__(
        self,
        observer: TodoObserver | None = None,
        delegate_handler: Callable[..., Any] | None = None,
    ) -> None:
        self.todo_list = TodoList(observer=observer)
        self.delegate_handler = delegate_handler

    @property
    def specs(self) -> list[ToolSpec]:
        specs = [TODO_CREATE_SPEC, TODO_UPDATE_SPEC, TODO_LIST_SPEC]
        if self.delegate_handler is not None:
            specs.append(TODO_DELEGATE_SPEC)
        return specs

    @property
    def handlers(self) -> dict[str, Callable[..., Any]]:
        handlers = {
            "todo_create": self.todo_list.create,
            "todo_update": self.todo_list.update,
            "todo_list": lambda: self.todo_list.snapshot(),
        }
        if self.delegate_handler is not None:
            handlers["todo_delegate"] = self.delegate_handler
        return handlers


TODO_DELEGATE_SPEC = ToolSpec(
    name="todo_delegate",
    description=(
        "Delegate one independent todo item to a fresh sub-agent. Pass only "
        "the item id and the precise instructions needed for that item."
    ),
    parameters={
        "type": "object",
        "properties": {
            "item_id": {"type": "string"},
            "instructions": {
                "type": "string",
                "minLength": 1,
                "description": "Complete, self-contained instructions for the sub-agent.",
            },
            "tool_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional minimum set of registered tools needed by this item.",
            },
        },
        "required": ["item_id", "instructions"],
        "additionalProperties": False,
    },
    category="planning",
)


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
