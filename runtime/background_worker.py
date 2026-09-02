from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from threading import Event, RLock
from typing import TYPE_CHECKING, Any, Callable, Literal
from uuid import uuid4

from core.tool_registry import ToolRegistry
from core.tool_space import ToolSpec

if TYPE_CHECKING:
    from core.todo import TodoList


logger = logging.getLogger(__name__)


BackgroundStatus = Literal["queued", "running", "completed", "failed"]
BackgroundObserver = Callable[[str, dict[str, Any]], None]
BackgroundOperation = Callable[[], Any]
BackgroundCompletion = Callable[[dict[str, Any]], None]


BACKGROUND_LIST_SPEC = ToolSpec(
    name="background_list",
    description=(
        "List background jobs started by this Agent, including queued, "
        "running, completed, and failed jobs."
    ),
    parameters={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["queued", "running", "completed", "failed"],
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    category="background",
)


BACKGROUND_GET_SPEC = ToolSpec(
    name="background_get",
    description="Read one background job and its bounded command output.",
    parameters={
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
        "additionalProperties": False,
    },
    category="background",
)


BACKGROUND_NOTIFICATIONS_SPEC = ToolSpec(
    name="background_notifications",
    description=(
        "Read background completion notifications. By default notifications "
        "are marked delivered after this call."
    ),
    parameters={
        "type": "object",
        "properties": {
            "consume": {"type": "boolean", "default": True},
        },
        "required": [],
        "additionalProperties": False,
    },
    category="background",
)


@dataclass(slots=True)
class BackgroundTask:
    job_id: str
    name: str
    status: BackgroundStatus = "queued"
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: Any = None
    error: str | None = None
    callback_error: str | None = None
    notification_delivered: bool = False

    def to_dict(self, *, include_output: bool = False) -> dict[str, Any]:
        result = self.result
        if isinstance(result, dict):
            result = dict(result)
            if not include_output and "output" in result:
                output = str(result.pop("output") or "")
                result["output_chars"] = len(output)
        value: dict[str, Any] = {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status,
            "task_id": self.task_id,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "started_at": (
                self.started_at.isoformat() if self.started_at else None
            ),
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at else None
            ),
            "result": result,
            "error": self.error,
        }
        if self.callback_error:
            value["callback_error"] = self.callback_error
        return value


class BackgroundTaskManager:
    """Agent-scoped background execution with durable in-memory notices."""

    def __init__(self, max_workers: int | None = None) -> None:
        configured = max_workers or _positive_int(
            os.getenv("AGENT_BACKGROUND_MAX_WORKERS"),
            4,
        )
        self.max_workers = min(max(configured, 1), 16)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="agent-background",
        )
        self._tasks: dict[str, BackgroundTask] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._lock = RLock()

    def submit(
        self,
        *,
        name: str,
        operation: BackgroundOperation,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        observer: BackgroundObserver | None = None,
        on_complete: BackgroundCompletion | None = None,
    ) -> dict[str, Any]:
        job = BackgroundTask(
            job_id=f"bg-{uuid4().hex[:12]}",
            name=name.strip() or "background task",
            task_id=task_id,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._tasks[job.job_id] = job
        self._emit(observer, "background_queued", job.to_dict())

        try:
            future = self._executor.submit(
                self._run,
                job.job_id,
                operation,
                observer,
                on_complete,
            )
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.finished_at = datetime.now()
                job.error = str(exc)
            self._emit(observer, "background_failed", job.to_dict())
            raise

        with self._lock:
            self._futures[job.job_id] = future
        future.add_done_callback(
            lambda _future: self._discard_future(job.job_id)
        )
        return job.to_dict()

    def _run(
        self,
        job_id: str,
        operation: BackgroundOperation,
        observer: BackgroundObserver | None,
        on_complete: BackgroundCompletion | None,
    ) -> None:
        with self._lock:
            job = self._get_locked(job_id)
            job.status = "running"
            job.started_at = datetime.now()
            started = job.to_dict()
        self._emit(observer, "background_started", started)

        try:
            result = operation()
            error = self._result_error(result)
        except Exception as exc:
            result = None
            error = str(exc)

        with self._lock:
            job = self._get_locked(job_id)
            job.result = result
            job.error = error
            job.status = "failed" if error else "completed"
            job.finished_at = datetime.now()
            terminal = job.to_dict(include_output=True)

        if on_complete is not None:
            try:
                on_complete(terminal)
            except Exception as exc:
                with self._lock:
                    job.callback_error = str(exc)
                    job.error = f"completion callback failed: {exc}"
                    job.status = "failed"
                    terminal = job.to_dict(include_output=True)

        with self._lock:
            event_data = self._get_locked(job_id).to_dict()
        event_type = (
            "background_completed"
            if terminal["status"] == "completed"
            else "background_failed"
        )
        self._emit(observer, event_type, event_data)

    @staticmethod
    def _result_error(result: Any) -> str | None:
        if not isinstance(result, dict):
            return None
        status = str(result.get("status", "")).strip()
        if status in {"approval_required", "denied", "blocked"}:
            return str(
                result.get("reason")
                or result.get("error")
                or f"command status is {status}"
            )
        if result.get("timed_out"):
            return "background command timed out"
        return_code = result.get("return_code")
        if isinstance(return_code, int) and return_code != 0:
            return f"background command exited with code {return_code}"
        return None

    def list_jobs(
        self,
        status: BackgroundStatus | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            jobs = [
                task.to_dict()
                for task in self._tasks.values()
                if status is None or task.status == status
            ]
        return {"jobs": jobs, "count": len(jobs)}

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._get_locked(job_id).to_dict(include_output=True)

    def notifications(self, consume: bool = True) -> dict[str, Any]:
        with self._lock:
            tasks = [
                task
                for task in self._tasks.values()
                if task.status in {"completed", "failed"}
                and not task.notification_delivered
            ]
            values = [task.to_dict() for task in tasks]
            if consume:
                for task in tasks:
                    task.notification_delivered = True
        return {"notifications": values, "count": len(values)}

    def register(self, registry: ToolRegistry) -> None:
        registry.register(BACKGROUND_LIST_SPEC, self.list_jobs)
        registry.register(BACKGROUND_GET_SPEC, self.get_job)
        registry.register(
            BACKGROUND_NOTIFICATIONS_SPEC,
            self.notifications,
        )

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _get_locked(self, job_id: str) -> BackgroundTask:
        try:
            return self._tasks[job_id]
        except KeyError as exc:
            raise ValueError(f"unknown background job: {job_id}") from exc

    def _discard_future(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)

    @staticmethod
    def _emit(
        observer: BackgroundObserver | None,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        if observer is not None:
            try:
                observer(event_type, data)
            except Exception:
                logger.exception(
                    "background observer failed event_type=%s",
                    event_type,
                )


class TodoBackgroundExecutor:
    """Connect one background shell job to a Todo Task lifecycle."""

    def __init__(
        self,
        manager: BackgroundTaskManager,
        todo_list: TodoList,
        shell_runner: Callable[..., dict[str, Any]],
        *,
        shell_preflight: Callable[..., Any],
        progress_callback: BackgroundObserver | None = None,
    ) -> None:
        self.manager = manager
        self.todo_list = todo_list
        self.shell_runner = shell_runner
        self.shell_preflight = shell_preflight
        self.progress_callback = progress_callback

    def run(
        self,
        task_id: str,
        command: str,
        working_directory: str = ".",
        timeout_seconds: int = 1800,
        max_output_chars: int = 30000,
    ) -> dict[str, Any]:
        self.shell_preflight(command, working_directory)
        submitted: dict[str, Any] = {}
        release = Event()

        def operation() -> dict[str, Any]:
            release.wait()
            return self.shell_runner(
                command,
                working_directory=working_directory,
                timeout_seconds=timeout_seconds,
                max_output_chars=max_output_chars,
                _background=True,
            )

        def finish(job: dict[str, Any]) -> None:
            current = self.todo_list.get(task_id)
            if (
                current.status != "in_process"
                or current.background_job_id != job["job_id"]
            ):
                return
            if job["status"] == "completed":
                self.todo_list.complete(
                    task_id,
                    summary=(
                        f"Background command completed successfully "
                        f"(job {job['job_id']}): {command}"
                    ),
                    background_job_id=str(job["job_id"]),
                )
                return
            self.todo_list.block(
                task_id,
                reason=(
                    str(job.get("error") or "background command failed")
                    + f" (job {job['job_id']})"
                ),
                background_job_id=str(job["job_id"]),
            )

        def submit() -> str:
            job = self.manager.submit(
                name=command,
                operation=operation,
                task_id=task_id,
                metadata={
                    "command": command,
                    "working_directory": working_directory,
                },
                observer=self.progress_callback,
                on_complete=finish,
            )
            submitted.update(job)
            return str(job["job_id"])

        try:
            todo_snapshot = self.todo_list.start_background(
                task_id,
                submit,
                summary=f"Running in background: {command}",
            )
        finally:
            release.set()
        job_id = self.todo_list.get(task_id).background_job_id
        job = (
            self.manager.get_job(job_id)
            if job_id is not None
            else submitted
        )
        return {
            "status": "background_started",
            "job": job,
            "todo": todo_snapshot,
        }


def _positive_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("background worker count must be greater than zero")
    return parsed
