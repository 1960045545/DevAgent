from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Literal

from core.tool_space import ToolSpec


# ``failed`` and ``in_progress`` remain accepted only for compatibility with
# older callers. New state written by TodoList is always ``in_process``.
TaskStatus = Literal[
    "pending",
    "in_process",
    "completed",
    "blocked",
    "failed",
]
TodoStatus = Literal[
    "pending",
    "in_process",
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
        "Create a directed acyclic execution graph for a complex task. "
        "Each item must state one concrete task and may include an initial "
        "summary and task dependencies. Call this before other tools."
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
                                "task_id": {"type": "string", "minLength": 1},
                                "task": {"type": "string", "minLength": 1},
                                # ``title`` is the old input name.
                                "title": {"type": "string", "minLength": 1},
                                "summary": {"type": "string"},
                                "note": {"type": "string"},
                                "dependencies": {
                                    "type": "array",
                                    "items": {"type": "string", "minLength": 1},
                                },
                            },
                            "anyOf": [
                                {"required": ["task"]},
                                {"required": ["title"]},
                            ],
                            "additionalProperties": False,
                        },
                    ],
                },
                "minItems": 1,
                "maxItems": 20,
                "description": "Concrete tasks and their dependency edges.",
            },
        },
        "required": ["items"],
        "additionalProperties": False,
    },
    category="planning",
)


TODO_CLAIM_SPEC = ToolSpec(
    name="todo_claim",
    description=(
        "Claim one ready pending task before doing any work. This changes "
        "pending to in_process atomically."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task_id returned by todo_create.",
            },
            "note": {
                "type": "string",
                "description": "Optional progress note.",
                "default": "",
            },
        },
        "required": ["task_id"],
        "additionalProperties": False,
    },
    category="planning",
)


TODO_COMPLETE_SPEC = ToolSpec(
    name="todo_complete",
    description=(
        "Complete one in_process task after the work is actually finished. "
        "Always provide a concise execution summary."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task_id returned by todo_create.",
            },
            "summary": {
                "type": "string",
                "minLength": 1,
                "description": "What was done and the result.",
            },
        },
        "required": ["task_id", "summary"],
        "additionalProperties": False,
    },
    category="planning",
)


TODO_BLOCK_SPEC = ToolSpec(
    name="todo_block",
    description=(
        "Block a task that cannot proceed. Provide the blocking reason; "
        "pending dependents will be marked blocked automatically."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task_id returned by todo_create.",
            },
            "reason": {
                "type": "string",
                "minLength": 1,
                "description": "Why the task cannot proceed.",
            },
        },
        "required": ["task_id", "reason"],
        "additionalProperties": False,
    },
    category="planning",
)


TODO_UPDATE_SPEC = ToolSpec(
    name="todo_update",
    description=(
        "Legacy compatibility update for one task. Prefer todo_claim, "
        "todo_complete, and todo_block so the task action protocol is "
        "enforced."
    ),
    parameters={
        "type": "object",
        "properties": {
            "item_id": {
                "type": "string",
                "description": "The task_id returned by todo_create.",
            },
            "status": {
                "type": "string",
                "enum": [
                    "pending",
                    "in_process",
                    "in_progress",
                    "completed",
                    "blocked",
                    "failed",
                ],
            },
            "note": {
                "type": "string",
                "description": "Short execution or blocking summary.",
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
    description="Read the current Task DAG, statuses, summaries, and progress.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    category="planning",
)


@dataclass(slots=True)
class Task:
    """One executable node in a request-scoped task DAG."""

    task_id: str
    task: str
    summary: str = ""
    status: TaskStatus = "pending"
    dependencies: list[str] = field(default_factory=list)
    blocked_reason: str = ""

    # Compatibility aliases used by the previous TodoItem API.
    @property
    def item_id(self) -> str:
        return self.task_id

    @property
    def title(self) -> str:
        return self.task

    @property
    def note(self) -> str:
        return self.summary

    @note.setter
    def note(self, value: str) -> None:
        self.summary = str(value).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "summary": self.summary,
            "status": self.status,
            "dependencies": list(self.dependencies),
            "blocked_reason": self.blocked_reason,
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        result = self.to_dict()
        result.update(
            {
                "item_id": self.task_id,
                "title": self.task,
                "note": self.summary,
                "status": (
                    "in_progress"
                    if self.status == "in_process"
                    else self.status
                ),
            }
        )
        return result


class TodoItem:
    """Read/write compatibility view of a canonical :class:`Task`."""

    __slots__ = ("_task",)

    def __init__(self, task: Task) -> None:
        self._task = task

    @property
    def item_id(self) -> str:
        return self._task.task_id

    @property
    def task_id(self) -> str:
        return self._task.task_id

    @property
    def title(self) -> str:
        return self._task.task

    @title.setter
    def title(self, value: str) -> None:
        self._task.task = str(value).strip()

    @property
    def task(self) -> str:
        return self._task.task

    @task.setter
    def task(self, value: str) -> None:
        self._task.task = str(value).strip()

    @property
    def status(self) -> str:
        return (
            "in_progress"
            if self._task.status == "in_process"
            else self._task.status
        )

    @status.setter
    def status(self, value: str) -> None:
        self._task.status = _canonical_status(value)  # type: ignore[assignment]

    @property
    def note(self) -> str:
        return self._task.summary

    @note.setter
    def note(self, value: str) -> None:
        self._task.summary = str(value).strip()

    @property
    def summary(self) -> str:
        return self._task.summary

    @summary.setter
    def summary(self, value: str) -> None:
        self._task.summary = str(value).strip()

    @property
    def blocked_reason(self) -> str:
        return self._task.blocked_reason

    @property
    def dependencies(self) -> list[str]:
        return list(self._task.dependencies)

    def to_dict(self) -> dict[str, Any]:
        return self._task.to_legacy_dict()


def _canonical_status(status: str) -> TaskStatus:
    if status == "in_progress":
        return "in_process"
    if status not in {"pending", "in_process", "completed", "blocked", "failed"}:
        raise ValueError(f"unsupported todo status: {status}")
    return status  # type: ignore[return-value]


class TodoList:
    """Thread-safe, request-scoped Task DAG owned by the host agent."""

    def __init__(self, observer: TodoObserver | None = None) -> None:
        self._tasks: list[Task] = []
        self.observer = observer
        self._lock = RLock()

    @property
    def tasks(self) -> list[Task]:
        with self._lock:
            return list(self._tasks)

    @property
    def items(self) -> list[TodoItem]:
        """Deprecated TodoItem view; new code should use ``tasks``/``get``."""
        with self._lock:
            return [TodoItem(task) for task in self._tasks]

    @property
    def created(self) -> bool:
        with self._lock:
            return bool(self._tasks)

    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return bool(self._tasks) and all(
                task.status in {"completed", "blocked", "failed"}
                for task in self._tasks
            )

    def create(self, items: list[TodoCreateItem]) -> dict[str, Any]:
        with self._lock:
            if self.created:
                raise ValueError("todo list has already been created")

            normalized: list[tuple[str | None, str, str, list[str]]] = []
            for item in items:
                if isinstance(item, str):
                    task = item.strip()
                    task_id = None
                    summary = ""
                    dependencies: list[str] = []
                elif isinstance(item, dict):
                    task_value = item.get("task")
                    if not isinstance(task_value, str) or not task_value.strip():
                        task_value = item.get("title")
                    task = str(task_value or "").strip()
                    raw_id = item.get("task_id", item.get("item_id"))
                    task_id = str(raw_id).strip() if raw_id else None
                    raw_summary = item.get("summary", item.get("note", ""))
                    summary = str(raw_summary or "").strip()
                    raw_dependencies = item.get("dependencies", [])
                    if raw_dependencies is None:
                        raw_dependencies = []
                    if not isinstance(raw_dependencies, list):
                        raise ValueError(
                            "task dependencies must be an array of task ids"
                        )
                    dependencies = []
                    for dependency in raw_dependencies:
                        if not isinstance(dependency, str) or not dependency.strip():
                            continue
                        dependency = dependency.strip()
                        if dependency not in dependencies:
                            dependencies.append(dependency)
                else:
                    continue
                if task:
                    normalized.append((task_id, task, summary, dependencies))

            if not normalized:
                raise ValueError("todo list must contain at least one task")
            if len(normalized) > 20:
                raise ValueError("todo list cannot contain more than 20 tasks")

            task_ids = {
                task_id or f"todo-{index}"
                for index, (task_id, _task, _summary, _dependencies) in enumerate(
                    normalized,
                    start=1,
                )
            }
            if len(task_ids) != len(normalized):
                raise ValueError("task ids must be unique")

            candidate_tasks = [
                Task(
                    task_id=task_id or f"todo-{index}",
                    task=task,
                    summary=summary,
                    dependencies=dependencies,
                )
                for index, (task_id, task, summary, dependencies) in enumerate(
                    normalized,
                    start=1,
                )
            ]
            self._validate_dependencies(candidate_tasks)
            self._tasks = candidate_tasks
            snapshot = self.snapshot()
        self._emit("todo_created", snapshot)
        return snapshot

    def update(
        self,
        item_id: str,
        status: TodoStatus,
        note: str = "",
    ) -> dict[str, Any]:
        """Compatibility updater using the canonical transition rules.

        Model-facing code should prefer the explicit action methods. The old
        ``pending -> completed`` shortcut is available only through
        ``update_legacy`` for integrations that still require it.
        """
        return self._update(item_id, status, note, allow_legacy_shortcut=False)

    def update_legacy(
        self,
        item_id: str,
        status: TodoStatus,
        note: str = "",
    ) -> dict[str, Any]:
        """Deprecated updater kept for clients using the old Todo API."""
        return self._update(item_id, status, note, allow_legacy_shortcut=True)

    def _update(
        self,
        item_id: str,
        status: TodoStatus,
        note: str,
        *,
        allow_legacy_shortcut: bool,
    ) -> dict[str, Any]:
        next_status = _canonical_status(status)
        with self._lock:
            task = self._get_locked(item_id)
            self._validate_compat_transition(
                task,
                next_status,
                allow_legacy_shortcut=allow_legacy_shortcut,
            )
            if (
                task.status == "pending"
                and next_status == "in_process"
                and not self._is_ready_locked(task)
            ):
                reason = self._dependency_block_reason_locked(task)
                raise ValueError(
                    f"task {item_id} is not ready for execution"
                    + (f": {reason}" if reason else "")
                )
            task.status = next_status
            if note.strip():
                task.summary = note.strip()
            if next_status == "blocked":
                task.blocked_reason = (
                    note.strip()
                    or task.blocked_reason
                    or "task cannot proceed"
                )
            propagated = self._propagate_blocked_locked(task.task_id)
            snapshot = self.snapshot()
            event_data = self._task_event_data(
                snapshot,
                task.task_id,
                propagated,
                action="update",
            )
        self._emit("todo_updated", event_data)
        return snapshot

    def claim(self, task_id: str, note: str = "") -> dict[str, Any]:
        """Atomically perform ``pending -> claim -> in_process``."""
        with self._lock:
            task = self._get_locked(task_id)
            if task.status != "pending":
                raise ValueError(
                    f"task {task_id} can only be claimed from pending, "
                    f"current status is {task.status}"
                )
            if not self._is_ready_locked(task):
                reason = self._dependency_block_reason_locked(task)
                raise ValueError(
                    f"task {task_id} is not ready for execution"
                    + (f": {reason}" if reason else "")
                )
            task.status = "in_process"
            if note.strip():
                task.summary = note.strip()
            snapshot = self.snapshot()
            event_data = self._task_event_data(
                snapshot,
                task_id,
                action="claim",
            )
        self._emit("todo_updated", event_data)
        return snapshot

    def complete(self, task_id: str, summary: str = "") -> dict[str, Any]:
        """Atomically perform ``in_process -> complete -> completed``."""
        with self._lock:
            task = self._get_locked(task_id)
            if task.status != "in_process":
                raise ValueError(
                    f"task {task_id} can only be completed from in_process, "
                    f"current status is {task.status}"
                )
            if summary.strip():
                task.summary = summary.strip()
            task.status = "completed"
            snapshot = self.snapshot()
            event_data = self._task_event_data(
                snapshot,
                task_id,
                action="complete",
            )
        self._emit("todo_updated", event_data)
        return snapshot

    def block(self, task_id: str, reason: str = "") -> dict[str, Any]:
        """Mark a pending/in-process task blocked and propagate to dependents."""
        with self._lock:
            task = self._get_locked(task_id)
            if task.status not in {"pending", "in_process"}:
                raise ValueError(
                    f"task {task_id} can only be blocked from pending or "
                    f"in_process, current status is {task.status}"
                )
            task.status = "blocked"
            task.blocked_reason = reason.strip() or "task cannot proceed"
            propagated = self._propagate_blocked_locked(task_id)
            snapshot = self.snapshot()
            event_data = self._task_event_data(
                snapshot,
                task_id,
                propagated,
                action="block",
            )
        self._emit("todo_updated", event_data)
        return snapshot

    def get(self, task_id: str) -> Task:
        with self._lock:
            return self._get_locked(task_id)

    def is_ready(self, task_id: str) -> bool:
        with self._lock:
            task = self._get_locked(task_id)
            return self._is_ready_locked(task)

    def get_ready_tasks(self) -> list[Task]:
        with self._lock:
            return [task for task in self._tasks if self._is_ready_locked(task)]

    def get_blocked_tasks(self) -> list[Task]:
        with self._lock:
            return [task for task in self._tasks if task.status == "blocked"]

    def get_dependents(self, task_id: str) -> list[str]:
        with self._lock:
            self._get_locked(task_id)
            return self._dependents_locked(task_id)

    def graph_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._graph_snapshot_locked()

    def _get_locked(self, task_id: str) -> Task:
        for task in self._tasks:
            if task.task_id == task_id:
                return task
        raise ValueError(f"unknown todo task: {task_id}")

    def _is_ready_locked(self, task: Task) -> bool:
        if task.status != "pending":
            return False
        return all(
            self._get_locked(dependency).status == "completed"
            for dependency in task.dependencies
        )

    def _dependency_block_reason_locked(self, task: Task) -> str:
        for dependency_id in task.dependencies:
            dependency = self._get_locked(dependency_id)
            if dependency.status in {"blocked", "failed"}:
                return (
                    f"dependency {dependency_id} is {dependency.status}"
                    + (
                        f": {dependency.blocked_reason}"
                        if dependency.blocked_reason
                        else ""
                    )
                )
            if dependency.status != "completed":
                return f"waiting for dependency {dependency_id}"
        return ""

    def _validate_compat_transition(
        self,
        task: Task,
        next_status: TaskStatus,
        *,
        allow_legacy_shortcut: bool = False,
    ) -> None:
        if task.status in {"completed", "blocked", "failed"}:
            if next_status != task.status:
                raise ValueError(
                    f"terminal todo task cannot transition from "
                    f"{task.status} to {next_status}"
                )
            return
        if (
            allow_legacy_shortcut
            and task.status == "pending"
            and next_status == "completed"
        ):
            # This is the one legacy shortcut kept for existing integrations.
            return
        allowed = {
            "pending": {"pending", "in_process", "blocked", "failed"},
            "in_process": {"in_process", "completed", "blocked", "failed"},
        }[task.status]
        if next_status not in allowed:
            raise ValueError(
                f"invalid todo transition from {task.status} to {next_status}"
            )

    def _validate_dependencies(self, tasks: list[Task]) -> None:
        task_ids = {task.task_id for task in tasks}
        dependencies_by_id = {
            task.task_id: task.dependencies for task in tasks
        }
        for task in tasks:
            unknown = [
                dependency
                for dependency in task.dependencies
                if dependency not in task_ids
            ]
            if unknown:
                raise ValueError(
                    f"todo task {task.task_id} has unknown dependencies: {unknown}"
                )
            if task.task_id in task.dependencies:
                raise ValueError(
                    f"todo task {task.task_id} cannot depend on itself"
                )

        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError(f"todo dependency cycle detected at {task_id}")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in dependencies_by_id[task_id]:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task in tasks:
            visit(task.task_id)

    def _propagate_blocked_locked(self, task_id: str) -> list[str]:
        changed: list[str] = []
        queue = [task_id]
        while queue:
            dependency_id = queue.pop(0)
            dependency = self._get_locked(dependency_id)
            for dependent_id in self._dependents_locked(dependency_id):
                dependent = self._get_locked(dependent_id)
                if dependent.status != "pending":
                    continue
                dependent.status = "blocked"
                dependent.blocked_reason = (
                    f"dependency {dependency_id} is {dependency.status}"
                    + (
                        f": {dependency.blocked_reason}"
                        if dependency.blocked_reason
                        else ""
                    )
                )
                changed.append(dependent_id)
                queue.append(dependent_id)
        return changed

    def _dependents_locked(self, task_id: str) -> list[str]:
        return [
            task.task_id
            for task in self._tasks
            if task_id in task.dependencies
        ]

    def _graph_snapshot_locked(self) -> dict[str, Any]:
        edges = [
            {"from": dependency, "to": task.task_id}
            for task in self._tasks
            for dependency in task.dependencies
        ]
        return {
            "nodes": [task.to_dict() for task in self._tasks],
            "edges": edges,
            "topological_order": self._topological_order_locked(),
        }

    def _topological_order_locked(self) -> list[str]:
        order: list[str] = []
        visited: set[str] = set()

        def visit(task: Task) -> None:
            if task.task_id in visited:
                return
            for dependency_id in task.dependencies:
                visit(self._get_locked(dependency_id))
            visited.add(task.task_id)
            order.append(task.task_id)

        for task in self._tasks:
            visit(task)
        return order

    @staticmethod
    def _task_event_data(
        snapshot: dict[str, Any],
        task_id: str,
        propagated: list[str] | None = None,
        *,
        action: str = "update",
    ) -> dict[str, Any]:
        updated_task = next(
            (
                task
                for task in snapshot.get("tasks", [])
                if task.get("task_id") == task_id
            ),
            None,
        )
        return {
            **snapshot,
            "action": action,
            "updated_task_id": task_id,
            "updated_item_id": task_id,
            "updated_task": updated_task,
            "propagated_blocked_task_ids": list(propagated or []),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            graph = self._graph_snapshot_locked()
            ready_task_ids = [
                task.task_id
                for task in self._tasks
                if self._is_ready_locked(task)
            ]
            blocked_task_ids = [
                task.task_id
                for task in self._tasks
                if task.status == "blocked"
            ]
            return {
                "tasks": [task.to_dict() for task in self._tasks],
                # Kept for clients written against the original TodoList API.
                "todos": [task.to_legacy_dict() for task in self._tasks],
                "edges": graph["edges"],
                "graph": graph,
                "ready_task_ids": ready_task_ids,
                "blocked_task_ids": blocked_task_ids,
                "pending_task_ids": [
                    task.task_id
                    for task in self._tasks
                    if task.status == "pending"
                ],
                "in_process_task_ids": [
                    task.task_id
                    for task in self._tasks
                    if task.status == "in_process"
                ],
                "completed_count": sum(
                    task.status == "completed" for task in self._tasks
                ),
                "total_count": len(self._tasks),
                "is_terminal": bool(self._tasks)
                and all(
                    task.status in {"completed", "blocked", "failed"}
                    for task in self._tasks
                ),
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
        specs = [
            TODO_CREATE_SPEC,
            TODO_CLAIM_SPEC,
            TODO_COMPLETE_SPEC,
            TODO_BLOCK_SPEC,
            TODO_UPDATE_SPEC,
            TODO_LIST_SPEC,
        ]
        if self.delegate_handler is not None:
            specs.append(TODO_DELEGATE_SPEC)
        return specs

    @property
    def handlers(self) -> dict[str, Callable[..., Any]]:
        handlers = {
            "todo_create": self.todo_list.create,
            "todo_claim": self.todo_list.claim,
            "todo_complete": self.todo_list.complete,
            "todo_block": self.todo_list.block,
            "todo_update": self.todo_list.update_legacy,
            "todo_list": lambda: self.todo_list.snapshot(),
        }
        if self.delegate_handler is not None:
            handlers["todo_delegate"] = self.delegate_handler
        return handlers


TODO_DELEGATE_SPEC = ToolSpec(
    name="todo_delegate",
    description=(
        "Delegate one independent task to a fresh sub-agent. Pass only the "
        "task id and the precise instructions needed for that task."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "item_id": {"type": "string"},
            "instructions": {
                "type": "string",
                "minLength": 1,
                "description": "Complete, self-contained instructions for the sub-agent.",
            },
            "tool_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional minimum set of registered tools needed by this task.",
            },
        },
        "required": ["instructions"],
        "anyOf": [{"required": ["task_id"]}, {"required": ["item_id"]}],
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
