from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

from core.tool_registry import ToolRegistry
from core.tool_space import ToolSpec


WORKTREE_CREATE_SPEC = ToolSpec(
    name="worktree_create",
    description=(
        "Create an isolated Git worktree under .worktrees and optionally "
        "bind it to a Todo task. The worktree is retained until explicitly "
        "kept or removed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "description": "A safe worktree name, without path separators.",
            },
            "task_id": {
                "type": "string",
                "default": "",
                "description": "Optional Todo task id to bind.",
            },
            "base_ref": {
                "type": "string",
                "default": "HEAD",
                "description": "Git ref used as the worktree starting point.",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    },
    category="git",
)

WORKTREE_BIND_SPEC = ToolSpec(
    name="worktree_bind",
    description="Bind a Todo task to an existing isolated Git worktree.",
    parameters={
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "minLength": 1},
            "worktree": {"type": "string", "minLength": 1},
        },
        "required": ["task_id", "worktree"],
        "additionalProperties": False,
    },
    category="git",
)

WORKTREE_LIST_SPEC = ToolSpec(
    name="worktree_list",
    description="List isolated Git worktrees and their dirty/commit status.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    category="git",
)

WORKTREE_KEEP_SPEC = ToolSpec(
    name="worktree_keep",
    description="Keep a completed worktree and branch for review or later merging.",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string", "minLength": 1}},
        "required": ["name"],
        "additionalProperties": False,
    },
    category="git",
)

WORKTREE_REMOVE_SPEC = ToolSpec(
    name="worktree_remove",
    description=(
        "Remove an isolated worktree and its branch. Dirty worktrees are "
        "refused unless discard_changes is explicitly true."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "discard_changes": {
                "type": "boolean",
                "default": False,
                "description": "Explicitly discard uncommitted changes.",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    },
    category="git",
)


class WorktreeError(RuntimeError):
    """Raised when a worktree lifecycle operation cannot be completed."""


_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_ROOT_LOCKS: dict[str, RLock] = {}
_ROOT_LOCKS_GUARD = RLock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class WorktreeManager:
    """Persistent, task-bound Git worktree lifecycle management."""

    def __init__(
        self,
        workspace_root: str | Path,
        scope_id: str | None = None,
        observer: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        if not self.workspace_root.exists() or not self.workspace_root.is_dir():
            raise ValueError(f"workspace root is not a directory: {self.workspace_root}")
        self.worktrees_root = self.workspace_root / ".worktrees"
        self.index_path = self.worktrees_root / "index.json"
        self.events_path = self.worktrees_root / "events.jsonl"
        self.scope_id = str(scope_id or uuid4()).strip()
        self.observer = observer
        key = str(self.workspace_root).lower() if os.name == "nt" else str(self.workspace_root)
        with _ROOT_LOCKS_GUARD:
            self._lock = _ROOT_LOCKS.setdefault(key, RLock())

    @staticmethod
    def validate_worktree_name(name: str) -> str:
        value = str(name or "").strip()
        if not _NAME_PATTERN.fullmatch(value) or value in {".", ".."}:
            raise ValueError(
                "worktree name must match [A-Za-z0-9._-]{1,64} and cannot be . or .."
            )
        return value

    def task_worktree_name(self, task_id: str) -> str:
        raw_scope = re.sub(r"[^A-Za-z0-9._-]+", "-", self.scope_id).strip("-._")
        raw_task = re.sub(r"[^A-Za-z0-9._-]+", "-", str(task_id)).strip("-._")
        prefix = f"task-{raw_scope[:16]}-{raw_task[:32]}".strip("-")
        return self.validate_worktree_name(prefix[:64])

    def create_worktree(
        self,
        name: str,
        task_id: str = "",
        base_ref: str = "HEAD",
    ) -> dict[str, Any]:
        name = self.validate_worktree_name(name)
        task_id = str(task_id or "").strip()
        base_ref = str(base_ref or "HEAD").strip()
        if (
            not base_ref
            or base_ref.startswith("-")
            or any(character.isspace() or ord(character) < 32 for character in base_ref)
        ):
            raise ValueError("base_ref must be a single Git ref")
        with self._lock:
            records = self._load_records_locked()
            existing = records.get(name)
            if existing is not None:
                path = self.resolve_worktree(name)
                if existing.get("task_id") not in {"", task_id}:
                    raise WorktreeError(
                        f"worktree {name} is already bound to task "
                        f"{existing.get('task_id')}"
                    )
                return self.inspect_worktree(name)

            path = self._worktree_path(name)
            if path.exists():
                raise WorktreeError(f"worktree path already exists: {path}")
            branch = f"wt/{name}"
            if self._git("branch", "--list", branch):
                raise WorktreeError(f"Git branch already exists: {branch}")
            base_commit = self._git("rev-parse", base_ref)
            self.worktrees_root.mkdir(parents=True, exist_ok=True)
            self._git(
                "worktree",
                "add",
                "-b",
                branch,
                str(path),
                base_ref,
            )
            now = _utc_now()
            record = {
                "name": name,
                "path": path.relative_to(self.workspace_root).as_posix(),
                "branch": branch,
                "task_id": task_id,
                "status": "active",
                "base_ref": base_ref,
                "base_commit": base_commit,
                "created_at": now,
                "updated_at": now,
            }
            records[name] = record
            self._save_records_locked(records)
            self._append_event_locked("create", record)
            result = self.inspect_worktree(name)
            self._notify("worktree_created", result)
            return result

    def bind_task_to_worktree(
        self,
        task_id: str,
        worktree_name: str,
    ) -> dict[str, Any]:
        task_id = str(task_id or "").strip()
        if not task_id:
            raise ValueError("task id cannot be empty")
        worktree_name = self.validate_worktree_name(worktree_name)
        with self._lock:
            records = self._load_records_locked()
            record = records.get(worktree_name)
            if record is None:
                raise WorktreeError(f"unknown worktree: {worktree_name}")
            current_task = str(record.get("task_id") or "")
            if current_task and current_task != task_id:
                raise WorktreeError(
                    f"worktree {worktree_name} is already bound to task {current_task}"
                )
            record["task_id"] = task_id
            record["updated_at"] = _utc_now()
            records[worktree_name] = record
            self._save_records_locked(records)
            self._append_event_locked("bind", record)
            result = self.inspect_worktree(worktree_name)
            self._notify("worktree_bound", result)
            return result

    def resolve_worktree(self, worktree_name: str) -> Path:
        name = self.validate_worktree_name(worktree_name)
        path = self._worktree_path(name)
        resolved_root = self.worktrees_root.resolve()
        resolved_path = path.resolve()
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise PermissionError("worktree path escapes .worktrees") from exc
        if not resolved_path.exists() or not resolved_path.is_dir():
            raise WorktreeError(f"worktree does not exist: {name}")
        return resolved_path

    def get_record(self, worktree_name: str) -> dict[str, Any]:
        name = self.validate_worktree_name(worktree_name)
        with self._lock:
            record = self._load_records_locked().get(name)
            if record is None:
                raise WorktreeError(f"unknown worktree: {name}")
            return dict(record)

    def list_worktrees(self) -> dict[str, Any]:
        with self._lock:
            records = self._load_records_locked()
            result = []
            for name in sorted(records):
                result.append(self._inspect_locked(name, records[name]))
            return {"worktrees": result, "root": str(self.workspace_root)}

    def inspect_worktree(self, worktree_name: str) -> dict[str, Any]:
        name = self.validate_worktree_name(worktree_name)
        with self._lock:
            records = self._load_records_locked()
            record = records.get(name)
            if record is None:
                raise WorktreeError(f"unknown worktree: {name}")
            return self._inspect_locked(name, record)

    def keep_worktree(self, worktree_name: str) -> dict[str, Any]:
        name = self.validate_worktree_name(worktree_name)
        with self._lock:
            records = self._load_records_locked()
            record = records.get(name)
            if record is None:
                raise WorktreeError(f"unknown worktree: {name}")
            self.resolve_worktree(name)
            record["status"] = "kept"
            record["updated_at"] = _utc_now()
            records[name] = record
            self._save_records_locked(records)
            self._append_event_locked("keep", record)
            result = self._inspect_locked(name, record)
            self._notify("worktree_kept", result)
            return result

    def remove_worktree(
        self,
        worktree_name: str,
        discard_changes: bool = False,
    ) -> dict[str, Any]:
        name = self.validate_worktree_name(worktree_name)
        with self._lock:
            records = self._load_records_locked()
            record = records.get(name)
            if record is None:
                raise WorktreeError(f"unknown worktree: {name}")
            inspection = self._inspect_locked(name, record)
            if inspection["dirty"] and not discard_changes:
                raise WorktreeError(
                    f"worktree {name} has uncommitted changes; choose keep_worktree "
                    "or call remove_worktree with discard_changes=true"
                )
            path = self.resolve_worktree(name)
            command = ["worktree", "remove"]
            if discard_changes:
                command.append("--force")
            command.append(str(path))
            self._git(*command)
            try:
                self._git("branch", "-D", str(record["branch"]))
            except Exception as exc:
                record["status"] = "remove_failed"
                record["updated_at"] = _utc_now()
                records[name] = record
                self._save_records_locked(records)
                raise WorktreeError(
                    f"worktree removed but branch cleanup failed: {exc}"
                ) from exc
            removed = {
                **record,
                "status": "removed",
                "dirty": inspection["dirty"],
                "changed_files": inspection["changed_files"],
                "commits": inspection["commits"],
                "discard_changes": bool(discard_changes),
                "updated_at": _utc_now(),
            }
            records.pop(name, None)
            self._save_records_locked(records)
            self._append_event_locked("remove", removed)
            self._notify("worktree_removed", removed)
            return removed

    def _inspect_locked(
        self,
        name: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        path = self._worktree_path(name)
        exists = path.exists() and path.is_dir()
        changed_files = 0
        dirty = False
        commits = 0
        if exists:
            status = self._git_at(path, "status", "--porcelain", "--untracked-files=all")
            changed_files = len([line for line in status.splitlines() if line.strip()])
            dirty = changed_files > 0
            base_commit = str(record.get("base_commit") or "").strip()
            branch = str(record.get("branch") or "").strip()
            if base_commit and branch:
                try:
                    commits = int(self._git("rev-list", "--count", f"{base_commit}..{branch}"))
                except WorktreeError:
                    commits = 0
        return {
            **record,
            "path": str(path),
            "exists": exists,
            "dirty": dirty,
            "changed_files": changed_files,
            "commits": commits,
        }

    def _worktree_path(self, name: str) -> Path:
        return self.worktrees_root / self.validate_worktree_name(name)

    def _load_records_locked(self) -> dict[str, dict[str, Any]]:
        if not self.index_path.exists():
            return {}
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorktreeError(f"cannot read worktree index: {exc}") from exc
        if not isinstance(payload, dict):
            raise WorktreeError("worktree index must be a JSON object")
        records = payload.get("worktrees", payload)
        if not isinstance(records, dict):
            raise WorktreeError("worktree index has invalid worktrees")
        return {
            str(name): dict(record)
            for name, record in records.items()
            if isinstance(record, dict)
        }

    def _save_records_locked(self, records: dict[str, dict[str, Any]]) -> None:
        self.worktrees_root.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "worktrees": records}
        fd, temporary = tempfile.mkstemp(
            prefix="index.",
            suffix=".tmp",
            dir=str(self.worktrees_root),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.index_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _append_event_locked(self, event_type: str, record: dict[str, Any]) -> None:
        self.worktrees_root.mkdir(parents=True, exist_ok=True)
        event = {
            "event": event_type,
            "timestamp": _utc_now(),
            "worktree": dict(record),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            json.dump(event, handle, ensure_ascii=False)
            handle.write("\n")

    def _notify(self, event_type: str, data: dict[str, Any]) -> None:
        if self.observer is not None:
            self.observer(event_type, dict(data))

    def _git(self, *arguments: str) -> str:
        return self._run_git(["git", "-C", str(self.workspace_root), *arguments])

    @staticmethod
    def _git_at(path: Path, *arguments: str) -> str:
        return WorktreeManager._run_git(["git", "-C", str(path), *arguments])

    @staticmethod
    def _run_git(command: list[str]) -> str:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise WorktreeError(
                f"Git command failed ({completed.returncode}): "
                f"{' '.join(command)}{': ' + detail if detail else ''}"
            )
        return completed.stdout.strip()


class WorktreeToolset:
    """Model-facing lifecycle tools for the request's worktree manager."""

    def __init__(self, manager: WorktreeManager, todo_list: Any) -> None:
        self.manager = manager
        self.todo_list = todo_list

    @property
    def specs(self) -> list[ToolSpec]:
        return [
            WORKTREE_CREATE_SPEC,
            WORKTREE_BIND_SPEC,
            WORKTREE_LIST_SPEC,
            WORKTREE_KEEP_SPEC,
            WORKTREE_REMOVE_SPEC,
        ]

    @property
    def handlers(self) -> dict[str, Any]:
        return {
            "worktree_create": self.create,
            "worktree_bind": self.bind,
            "worktree_list": self.manager.list_worktrees,
            "worktree_keep": self.keep,
            "worktree_remove": self.remove,
        }

    def create(
        self,
        name: str,
        task_id: str = "",
        base_ref: str = "HEAD",
    ) -> dict[str, Any]:
        task_id = str(task_id or "").strip()
        result = self.manager.create_worktree(
            name,
            task_id=task_id,
            base_ref=base_ref,
        )
        if not task_id:
            return result
        try:
            self.todo_list.bind_worktree(task_id, name)
            return self.manager.bind_task_to_worktree(task_id, name)
        except Exception:
            try:
                self.manager.remove_worktree(name)
            except Exception:
                pass
            raise

    def bind(self, task_id: str, worktree: str) -> dict[str, Any]:
        if self.todo_list is None:
            raise ValueError("worktree_bind requires an active Todo list")
        task_id = str(task_id).strip()
        worktree = str(worktree).strip()
        task = self.todo_list.get(task_id)
        if task.worktree and task.worktree != worktree:
            raise ValueError(
                f"task {task_id} is already bound to worktree {task.worktree}"
            )
        result = self.manager.bind_task_to_worktree(task_id, worktree)
        self.todo_list.bind_worktree(task_id, worktree)
        return result

    def keep(self, name: str) -> dict[str, Any]:
        return self.manager.keep_worktree(name)

    def remove(
        self,
        name: str,
        discard_changes: bool = False,
    ) -> dict[str, Any]:
        return self.manager.remove_worktree(
            name,
            discard_changes=discard_changes,
        )

    def register(self, registry: ToolRegistry) -> None:
        for spec in self.specs:
            registry.register(spec, self.handlers[spec.name])
