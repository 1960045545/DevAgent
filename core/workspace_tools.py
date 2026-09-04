from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from config.settings import PROJECT_ROOT, load_project_env
from core.tool_registry import ToolRegistry
from core.tool_space import ToolSpec


@dataclass(frozen=True, slots=True)
class WorkspaceApprovalRequest:
    """A host-level approval request for a potentially destructive action."""

    action: str
    command: str
    working_directory: str
    reason: str


ApprovalCallback = Callable[[WorkspaceApprovalRequest], bool]


WORKSPACE_LIST_FILES_SPEC = ToolSpec(
    name="workspace_list_files",
    description=(
        "List files and directories inside the project workspace. "
        "Use this before reading or editing an unfamiliar path."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative directory path.",
                "default": ".",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to include descendants.",
                "default": False,
            },
            "max_entries": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "default": 200,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    category="filesystem",
)

WORKSPACE_READ_FILE_SPEC = ToolSpec(
    name="workspace_read_file",
    description=(
        "Read a UTF-8 text file inside the project workspace. "
        "Large files are truncated by line and character limits."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative file path.",
            },
            "start_line": {
                "type": "integer",
                "minimum": 1,
                "default": 1,
            },
            "max_lines": {
                "type": "integer",
                "minimum": 1,
                "maximum": 2000,
                "default": 400,
            },
            "max_chars": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100000,
                "default": 30000,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    category="filesystem",
)

WORKSPACE_WRITE_FILE_SPEC = ToolSpec(
    name="workspace_write_file",
    description=(
        "Create or overwrite a UTF-8 text file inside the project workspace. "
        "Use this only when the requested change is clear."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative file path.",
            },
            "content": {
                "type": "string",
                "description": "Complete UTF-8 file content.",
            },
            "create_parents": {
                "type": "boolean",
                "description": "Create missing parent directories.",
                "default": False,
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    category="filesystem",
)

WORKSPACE_REPLACE_TEXT_SPEC = ToolSpec(
    name="workspace_replace_text",
    description=(
        "Replace an exact text fragment in a UTF-8 file inside the workspace. "
        "By default exactly one replacement is required."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative file path.",
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to replace.",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text.",
            },
            "expected_replacements": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 1,
            },
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    },
    category="filesystem",
)

WORKSPACE_RUN_SHELL_SPEC = ToolSpec(
    name="workspace_run_shell",
    description=(
        "Run one approved shell command in the project workspace. "
        "Use it for inspection, tests, builds, git status, and diagnostics. "
        "Destructive operations and parent-directory navigation require "
        "explicit host-user approval; recursive deletion is always blocked."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "One shell command without command chaining.",
            },
            "working_directory": {
                "type": "string",
                "description": "Workspace-relative working directory.",
                "default": ".",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3600,
                "default": 120,
            },
            "max_output_chars": {
                "type": "integer",
                "minimum": 100,
                "maximum": 100000,
                "default": 30000,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    category="shell",
)

WORKSPACE_RUN_PYTHON_SPEC = ToolSpec(
    name="workspace_run_python",
    description=(
        "Run a Python script from inside the project workspace using the "
        "current interpreter. The script must remain inside the workspace; "
        "parent-directory navigation requires explicit host-user approval."
    ),
    parameters={
        "type": "object",
        "properties": {
            "script_path": {
                "type": "string",
                "description": "Workspace-relative Python script path.",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Command-line arguments for the script.",
                "default": [],
            },
            "working_directory": {
                "type": "string",
                "description": "Workspace-relative working directory.",
                "default": ".",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 600,
                "default": 120,
            },
            "max_output_chars": {
                "type": "integer",
                "minimum": 100,
                "maximum": 100000,
                "default": 30000,
            },
        },
        "required": ["script_path"],
        "additionalProperties": False,
    },
    category="python",
)


@dataclass(frozen=True, slots=True)
class WorkspaceToolSettings:
    workspace_root: Path
    max_output_chars: int = 30000
    shell_timeout_seconds: int = 120
    python_timeout_seconds: int = 120
    max_file_chars: int = 100000
    allow_shell: bool = True
    allowed_shell_commands: tuple[str, ...] = (
        "Get-ChildItem",
        "Get-Content",
        "Get-Location",
        "Select-String",
        "findstr",
        "git",
        "dir",
        "find",
        "mvn",
        "node",
        "npm",
        "pip",
        "py",
        "pytest",
        "python",
        "rg",
        "rm",
        "remove-item",
        "del",
        "erase",
        "tree",
        "type",
        "uv",
        "where",
    )

    @classmethod
    def from_env(
        cls,
        *,
        workspace_root: str | Path | None = None,
    ) -> "WorkspaceToolSettings":
        load_project_env()
        defaults = cls(workspace_root=PROJECT_ROOT)
        root_value = workspace_root or os.getenv(
            "AGENT_WORKSPACE_ROOT",
            str(PROJECT_ROOT),
        )
        root = Path(root_value).expanduser().resolve()
        return cls(
            workspace_root=root,
            max_output_chars=_positive_int(
                "AGENT_TOOL_MAX_OUTPUT_CHARS",
                defaults.max_output_chars,
            ),
            shell_timeout_seconds=_bounded_int(
                "AGENT_SHELL_TIMEOUT_SECONDS",
                defaults.shell_timeout_seconds,
                maximum=3600,
            ),
            python_timeout_seconds=_bounded_int(
                "AGENT_PYTHON_TIMEOUT_SECONDS",
                defaults.python_timeout_seconds,
                maximum=600,
            ),
            max_file_chars=_positive_int(
                "AGENT_MAX_FILE_CHARS",
                defaults.max_file_chars,
            ),
            allow_shell=_get_bool(
                "AGENT_ALLOW_SHELL",
                defaults.allow_shell,
            ),
            allowed_shell_commands=_get_command_allowlist(
                os.getenv("AGENT_ALLOWED_SHELL_COMMANDS"),
                defaults.allowed_shell_commands,
            ),
        )


class WorkspaceToolset:
    """Controlled local workspace operations exposed as Agent tools."""

    _sensitive_names = {
        ".env",
        ".env.local",
        ".env.production",
        "id_rsa",
        "id_ed25519",
    }
    _protected_directory_names = {
        ".agent_sandbox",
        ".git",
        ".ssh",
    }
    _sensitive_suffixes = {
        ".key",
        ".pem",
        ".p12",
        ".pfx",
    }
    _blocked_shell_patterns = (
        r"^\s*(?:&\s*)?(?:remove-item|clear-content|del(?:ete)?|erase|rmdir|rd)(?:\s|$)",
        r"^\s*(?:&\s*)?(?:format(?:\.com)?|shutdown|stop-computer)(?:\s|$)",
        r"^\s*(?:&\s*)?git\s+(?:reset\s+--hard|clean\s+-[^\s]*f|checkout\s+--)(?:\s|$)",
        r"^\s*(?:&\s*)?set-executionpolicy(?:\s|$)",
        r"^\s*(?:&\s*)?(?:py|python(?:\.exe)?)\s+-c(?:\s|$)",
        r"^\s*(?:&\s*)?(?:powershell|pwsh)(?:\.exe)?\s+[^\r\n]*-(?:c|command|encodedcommand)(?:\s|$)",
        r"\binvoke-expression\b",
        r"\bstart-process\b",
    )
    _chain_characters = (";", "|", "&", ">", "<", "`")
    _powershell_expression_characters = ("(", ")", "{", "}", "[", "]")
    _powershell_provider_pattern = re.compile(
        r"(?i)(?:"
        r"(?:env|variable|function|alias|registry|cert|wsman|"
        r"hklm|hkcu|hkcr|hkcc|hk Users):[\\/]"
        r"|(?:registry|filesystem|cert):{2}"
        r")",
    )

    def __init__(
        self,
        settings: WorkspaceToolSettings | None = None,
        *,
        approval_callback: ApprovalCallback | None = None,
    ) -> None:
        raw_settings = settings or WorkspaceToolSettings.from_env()
        workspace_root = (
            Path(raw_settings.workspace_root)
            .expanduser()
            .resolve()
        )
        self.settings = replace(
            raw_settings,
            workspace_root=workspace_root,
        )
        self.approval_callback = approval_callback
        if workspace_root.name.lower() in self._protected_directory_names:
            raise PermissionError(
                "a protected directory cannot be selected as workspace root",
            )
        if not workspace_root.exists():
            raise ValueError(
                f"workspace root does not exist: "
                f"{workspace_root}",
            )
        if not workspace_root.is_dir():
            raise ValueError(
                f"workspace root is not a directory: "
                f"{workspace_root}",
            )

    def for_workspace(
        self,
        workspace_root: str | Path,
    ) -> "WorkspaceToolset":
        """Create the same policy-bound toolset for another workspace root."""
        return type(self)(
            replace(self.settings, workspace_root=Path(workspace_root)),
            approval_callback=self.approval_callback,
        )

    def list_files(
        self,
        path: str = ".",
        recursive: bool = False,
        max_entries: int = 200,
    ) -> dict[str, Any]:
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")
        directory, _ = self._resolve_workspace_path(
            path,
            must_exist=True,
        )
        if not directory.is_dir():
            raise ValueError(f"not a directory: {path}")
        self._check_sensitive(directory)

        iterator: Iterable[Path]
        iterator = directory.rglob("*") if recursive else directory.iterdir()
        entries = []
        for entry in sorted(iterator, key=lambda item: str(item).lower()):
            if len(entries) >= max_entries:
                break
            try:
                relative = entry.relative_to(self.settings.workspace_root)
            except ValueError:
                continue
            if self._is_sensitive(entry):
                continue
            entries.append(
                {
                    "path": relative.as_posix(),
                    "type": "directory" if entry.is_dir() else "file",
                },
            )
        return {
            "workspace_root": str(self.settings.workspace_root),
            "path": self._relative_path(directory),
            "recursive": recursive,
            "truncated": len(entries) >= max_entries,
            "entries": entries,
        }

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        max_lines: int = 400,
        max_chars: int = 30000,
    ) -> dict[str, Any]:
        if start_line <= 0:
            raise ValueError("start_line must be greater than zero")
        if max_lines <= 0:
            raise ValueError("max_lines must be greater than zero")
        max_chars = min(
            max(max_chars, 1),
            self.settings.max_file_chars,
        )
        file_path, _ = self._resolve_workspace_path(
            path,
            must_exist=True,
        )
        if not file_path.is_file():
            raise ValueError(f"not a file: {path}")
        self._check_sensitive(file_path)

        lines = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        selected = lines[start_line - 1:start_line - 1 + max_lines]
        content = "\n".join(selected)
        truncated = (
            len(selected) < len(lines[start_line - 1:])
            or len(content) > max_chars
        )
        if len(content) > max_chars:
            content = content[:max_chars]
        return {
            "path": self._relative_path(file_path),
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1,
            "truncated": truncated,
            "content": content,
        }

    def write_file(
        self,
        path: str,
        content: str,
        create_parents: bool = False,
    ) -> dict[str, Any]:
        file_path, _ = self._resolve_workspace_path(
            path,
            must_exist=False,
        )
        self._check_sensitive(file_path)
        if file_path.exists() and not file_path.is_file():
            raise ValueError(f"not a regular file: {path}")
        if len(content) > self.settings.max_file_chars:
            raise ValueError(
                f"content exceeds max_file_chars="
                f"{self.settings.max_file_chars}",
            )
        if not file_path.parent.exists():
            if not create_parents:
                raise ValueError(
                    f"parent directory does not exist: "
                    f"{file_path.parent}",
                )
            file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text(content, encoding="utf-8")
        return {
            "path": self._relative_path(file_path),
            "bytes_written": len(content.encode("utf-8")),
        }

    def replace_text(
        self,
        path: str,
        old_text: str,
        new_text: str,
        expected_replacements: int = 1,
    ) -> dict[str, Any]:
        if not old_text:
            raise ValueError("old_text must not be empty")
        if expected_replacements <= 0:
            raise ValueError(
                "expected_replacements must be greater than zero",
            )
        file_path, _ = self._resolve_workspace_path(
            path,
            must_exist=True,
        )
        if not file_path.is_file():
            raise ValueError(f"not a file: {path}")
        self._check_sensitive(file_path)

        original = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        actual = original.count(old_text)
        if actual != expected_replacements:
            raise ValueError(
                f"expected {expected_replacements} replacements, "
                f"found {actual}",
            )
        updated = original.replace(old_text, new_text)
        file_path.write_text(updated, encoding="utf-8")
        return {
            "path": self._relative_path(file_path),
            "replacements": actual,
            "bytes_written": len(updated.encode("utf-8")),
        }

    def run_shell(
        self,
        command: str,
        working_directory: str = ".",
        timeout_seconds: int | None = None,
        max_output_chars: int | None = None,
        *,
        _background: bool = False,
    ) -> dict[str, Any]:
        if not self.settings.allow_shell:
            raise PermissionError("shell tools are disabled")
        command = command.strip()
        if not command:
            raise ValueError("command must not be empty")
        if self._is_long_running_command(command) and not _background:
            raise ValueError(
                "long-running dependency installation must use "
                "todo_run_background"
            )
        cwd, uses_parent_navigation = self._resolve_workspace_path(
            working_directory,
            must_exist=True,
        )
        if not cwd.is_dir():
            raise ValueError(f"not a directory: {working_directory}")

        approval_reasons = self._validate_shell_command(command, cwd)
        if uses_parent_navigation:
            approval_reasons.append(
                "working_directory explicitly navigates to a parent path",
            )
        if approval_reasons:
            approval_result = self._request_approval(
                action="run_shell",
                command=command,
                working_directory=cwd,
                reasons=approval_reasons,
            )
            if approval_result is not None:
                return approval_result

        timeout = timeout_seconds or self.settings.shell_timeout_seconds
        output_limit = max_output_chars or self.settings.max_output_chars
        timeout = min(max(timeout, 1), 3600)
        output_limit = min(max(output_limit, 100), 100000)
        executable, args = self._shell_command(command)

        try:
            completed = subprocess.run(
                args,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
                env=self._subprocess_env(),
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            return {
                "command": command,
                "working_directory": self._relative_path(cwd),
                "return_code": completed.returncode,
                "timed_out": False,
                "output": self._truncate(output, output_limit),
                "shell": executable,
            }
        except subprocess.TimeoutExpired as exc:
            output = self._timeout_output(exc)
            return {
                "command": command,
                "working_directory": self._relative_path(cwd),
                "return_code": None,
                "timed_out": True,
                "output": self._truncate(output, output_limit),
                "shell": executable,
            }

    def preflight_background_shell(
        self,
        command: str,
        working_directory: str = ".",
    ) -> dict[str, Any]:
        """Validate a background shell request before a worker is started."""
        if not self.settings.allow_shell:
            raise PermissionError("shell tools are disabled")
        command = command.strip()
        if not command:
            raise ValueError("command must not be empty")
        cwd, uses_parent_navigation = self._resolve_workspace_path(
            working_directory,
            must_exist=True,
        )
        if not cwd.is_dir():
            raise ValueError(f"not a directory: {working_directory}")

        approval_reasons = self._validate_shell_command(command, cwd)
        if uses_parent_navigation:
            approval_reasons.append(
                "working_directory explicitly navigates to a parent path"
            )
        if approval_reasons:
            raise PermissionError(
                "commands requiring user approval cannot run in the "
                "background; use workspace_run_shell in the foreground: "
                + "; ".join(approval_reasons)
            )
        return {
            "command": command,
            "working_directory": self._relative_path(cwd),
            "approved_for_background": True,
        }

    def run_python(
        self,
        script_path: str,
        args: list[str] | None = None,
        working_directory: str = ".",
        timeout_seconds: int | None = None,
        max_output_chars: int | None = None,
    ) -> dict[str, Any]:
        script, _ = self._resolve_workspace_path(
            script_path,
            must_exist=True,
        )
        if not script.is_file():
            raise ValueError(f"not a file: {script_path}")
        if script.suffix.lower() != ".py":
            raise ValueError("script_path must point to a .py file")
        self._check_sensitive(script)
        cwd, uses_parent_navigation = self._resolve_workspace_path(
            working_directory,
            must_exist=True,
        )
        if not cwd.is_dir():
            raise ValueError(f"not a directory: {working_directory}")
        approval_reasons: list[str] = []
        if uses_parent_navigation:
            approval_reasons.append(
                "working_directory explicitly navigates to a parent path",
            )
        if self._has_parent_navigation(script_path):
            approval_reasons.append(
                "script_path explicitly navigates to a parent path",
            )
        if approval_reasons:
            approval_result = self._request_approval(
                action="run_python",
                command=f"{script_path} {args or []}",
                working_directory=cwd,
                reasons=approval_reasons,
            )
            if approval_result is not None:
                return approval_result

        timeout = timeout_seconds or self.settings.python_timeout_seconds
        output_limit = max_output_chars or self.settings.max_output_chars
        timeout = min(max(timeout, 1), 600)
        output_limit = min(max(output_limit, 100), 100000)
        command = [sys.executable, "-I", str(script), *(args or [])]

        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
                env=self._subprocess_env(),
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            return {
                "script_path": self._relative_path(script),
                "working_directory": self._relative_path(cwd),
                "return_code": completed.returncode,
                "timed_out": False,
                "output": self._truncate(output, output_limit),
                "interpreter": sys.executable,
            }
        except subprocess.TimeoutExpired as exc:
            output = self._timeout_output(exc)
            return {
                "script_path": self._relative_path(script),
                "working_directory": self._relative_path(cwd),
                "return_code": None,
                "timed_out": True,
                "output": self._truncate(output, output_limit),
                "interpreter": sys.executable,
            }

    def register(self, registry: ToolRegistry) -> None:
        registry.register(WORKSPACE_LIST_FILES_SPEC, self.list_files)
        registry.register(WORKSPACE_READ_FILE_SPEC, self.read_file)
        registry.register(WORKSPACE_WRITE_FILE_SPEC, self.write_file)
        registry.register(WORKSPACE_REPLACE_TEXT_SPEC, self.replace_text)
        registry.register(WORKSPACE_RUN_SHELL_SPEC, self.run_shell)
        registry.register(WORKSPACE_RUN_PYTHON_SPEC, self.run_python)

    def _resolve_workspace_path(
        self,
        path: str,
        *,
        must_exist: bool,
    ) -> tuple[Path, bool]:
        if not path or not path.strip():
            path = "."
        uses_parent_navigation = self._has_parent_navigation(path)
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.settings.workspace_root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.settings.workspace_root)
        except ValueError as exc:
            raise PermissionError(
                f"path is outside selected workspace: {path}",
            ) from exc
        if must_exist and not resolved.exists():
            raise FileNotFoundError(path)
        return resolved, uses_parent_navigation

    @staticmethod
    def _has_parent_navigation(path: str) -> bool:
        normalized = path.replace("\\", "/")
        return any(part == ".." for part in normalized.split("/"))

    def _check_sensitive(self, path: Path) -> None:
        if self._is_sensitive(path):
            relative = path.relative_to(self.settings.workspace_root)
            raise PermissionError(
                f"sensitive file access is not allowed: "
                f"{relative.as_posix()}",
            )

    def _is_sensitive(self, path: Path) -> bool:
        relative = path.relative_to(self.settings.workspace_root)
        parts = {part.lower() for part in relative.parts}
        if parts.intersection(self._protected_directory_names):
            return True
        name = path.name.lower()
        if (
            name in self._sensitive_names
            or name.startswith(".env.")
            or name.endswith(tuple(self._sensitive_suffixes))
            or "secret" in name
            or "credential" in name
        ):
            return True
        return False

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self.settings.workspace_root).as_posix() or "."

    def _validate_shell_command(
        self,
        command: str,
        cwd: Path,
    ) -> list[str]:
        lowered = command.lower()
        approval_reasons: list[str] = []
        if "\n" in command or "\r" in command:
            raise PermissionError("multiline shell commands are not allowed")
        if any(
            character in command
            for character in self._powershell_expression_characters
        ):
            raise PermissionError(
                "PowerShell expressions are not allowed in workspace shell commands",
            )
        if self._powershell_provider_pattern.search(command):
            raise PermissionError(
                "PowerShell provider paths are not allowed",
            )
        if any(
            sensitive in lowered
            for sensitive in (
                ".env",
                ".git",
                ".agent_sandbox",
                ".ssh",
                "id_rsa",
                "id_ed25519",
                ".pem",
                ".p12",
                ".pfx",
                "secret",
                "credential",
            )
        ):
            raise PermissionError(
                "shell command references a protected path or secret",
            )
        if any(character in command for character in self._chain_characters):
            raise PermissionError(
                "shell command chaining and redirection are not allowed",
            )
        if any(
            token in command
            for token in ("$env:", "$(", "${", "~\\", "~/")
        ):
            raise PermissionError(
                "shell variable expansion and home-directory paths are not allowed",
            )
        if re.search(r"(?i)(?:^|\s)~(?:\s|$)", command):
            raise PermissionError(
                "shell home-directory paths are not allowed",
            )
        if re.search(r"(?i)%[A-Za-z_][A-Za-z0-9_]*%", command):
            raise PermissionError(
                "shell environment-variable expansion is not allowed",
            )
        if any(
            re.search(pattern, lowered)
            for pattern in self._blocked_shell_patterns
        ):
            raise PermissionError(
                "shell command is permanently blocked by workspace policy",
            )

        command_name = self._command_name(command)
        if self._is_absolute_command(command):
            raise PermissionError(
                "absolute executable paths are not allowed",
            )
        allowed = {
            item.lower()
            for item in self.settings.allowed_shell_commands
        }
        command_names = {
            command.lower()
            for command in (
                command_name,
                Path(command_name).stem,
            )
        }
        if not command_names.intersection(allowed):
            raise PermissionError(
                f"shell command is not allowed: {command_name}",
            )
        if command_name.lower() in {
            "rm",
        }:
            if self._rm_is_recursive(command):
                raise PermissionError(
                    "recursive deletion is permanently blocked by workspace policy",
                )
            approval_reasons.append(
                "deleting a file requires explicit user approval",
            )
        approval_reasons.extend(
            self._validate_command_paths(command, cwd),
        )
        return approval_reasons

    @staticmethod
    def _is_absolute_command(command: str) -> bool:
        first = re.split(r"\s+", command.strip(), maxsplit=1)[0]
        first = first.strip("\"'")
        return bool(
            re.match(r"^[A-Za-z]:[\\/]", first)
            or re.match(r"^[A-Za-z]:$", first)
            or first.startswith(("\\\\", "/"))
        )

    @staticmethod
    def _is_long_running_command(command: str) -> bool:
        return bool(
            re.match(
                r"(?i)^\s*(?:"
                r"npm(?:\.cmd)?\s+(?:install|i|ci)"
                r"|pip(?:3|\.exe)?\s+install"
                r"|(?:py|python)(?:\.exe)?\s+-m\s+pip\s+install"
                r"|uv(?:\.exe)?\s+sync"
                r"|mvn(?:\.cmd)?\s+install"
                r")\b",
                command,
            )
        )

    def _validate_command_paths(
        self,
        command: str,
        cwd: Path,
    ) -> list[str]:
        """Reject paths outside the selected workspace before PowerShell runs."""
        reasons: list[str] = []
        for token in self._command_tokens(command):
            value = token.strip(",;")
            if len(value) >= 2 and value[0] == value[-1]:
                value = value[1:-1]
            if value == "~" or value.startswith(("~\\", "~/")):
                raise PermissionError(
                    "shell home-directory paths are not allowed",
                )
            if not self._looks_like_path(value):
                candidate_in_cwd = cwd / value if value else cwd
                if not candidate_in_cwd.exists():
                    continue
            if self._is_absolute_path(value):
                candidate = Path(value)
            else:
                candidate = cwd / value
            try:
                resolved = candidate.resolve()
                resolved.relative_to(self.settings.workspace_root)
            except ValueError as exc:
                raise PermissionError(
                    f"shell path is outside selected workspace: {value}",
                ) from exc
            if self._has_parent_navigation(value):
                reasons.append(
                    "shell command explicitly references a parent path",
                )
        return list(dict.fromkeys(reasons))

    @staticmethod
    def _command_tokens(command: str) -> list[str]:
        return re.findall(r"\"[^\"]*\"|'[^']*'|[^\s]+", command)

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        if not value or value.startswith("-"):
            return False
        if value in {".", ".."}:
            return True
        return (
            "/" in value
            or "\\" in value
            or value.endswith(".py")
            or bool(re.match(r"^[A-Za-z]:$", value))
        )

    @staticmethod
    def _is_absolute_path(value: str) -> bool:
        return bool(
            re.match(r"^[A-Za-z]:[\\/]", value)
            or re.match(r"^[A-Za-z]:$", value)
            or value.startswith(("\\\\", "/"))
        )

    def _request_approval(
        self,
        *,
        action: str,
        command: str,
        working_directory: Path,
        reasons: list[str],
    ) -> dict[str, Any] | None:
        request = WorkspaceApprovalRequest(
            action=action,
            command=command,
            working_directory=self._relative_path(working_directory),
            reason="; ".join(dict.fromkeys(reasons)),
        )
        if self.approval_callback is None:
            return {
                "status": "approval_required",
                "action": request.action,
                "command": request.command,
                "working_directory": request.working_directory,
                "reason": request.reason,
                "message": (
                    "A host user must approve this operation. "
                    "The Agent cannot approve it itself."
                ),
            }
        if self.approval_callback(request):
            return None
        return {
            "status": "denied",
            "action": request.action,
            "command": request.command,
            "working_directory": request.working_directory,
            "reason": request.reason,
        }

    def _subprocess_env(self) -> dict[str, str]:
        """Give child processes workspace-local temp and home directories."""
        blocked_fragments = (
            "KEY",
            "TOKEN",
            "PASSWORD",
            "SECRET",
            "CREDENTIAL",
        )
        environment = {
            name: value
            for name, value in os.environ.items()
            if not any(fragment in name.upper() for fragment in blocked_fragments)
        }
        sandbox_root = self.settings.workspace_root / ".agent_sandbox"
        temp_root = sandbox_root / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        environment.update(
            {
                "AGENT_WORKSPACE_ROOT": str(self.settings.workspace_root),
                "HOME": str(sandbox_root),
                "USERPROFILE": str(sandbox_root),
                "TEMP": str(temp_root),
                "TMP": str(temp_root),
                "TMPDIR": str(temp_root),
                "PYTHONNOUSERSITE": "1",
                "PIP_CACHE_DIR": str(sandbox_root / "pip"),
                "npm_config_cache": str(sandbox_root / "npm"),
                "UV_CACHE_DIR": str(sandbox_root / "uv"),
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "NUL" if os.name == "nt" else "/dev/null",
            },
        )
        return environment

    @staticmethod
    def _rm_is_recursive(command: str) -> bool:
        tokens = WorkspaceToolset._command_tokens(command)
        for token in tokens[1:]:
            value = token.strip("\"'").lower()
            if value == "--recursive":
                return True
            if (
                value.startswith("-")
                and not value.startswith("--")
                and "r" in value[1:]
            ):
                return True
        return False

    @staticmethod
    def _command_name(command: str) -> str:
        match = re.match(
            r"^\s*(?:&\s*)?(?:\.\\)?([^\s]+)",
            command,
        )
        if match is None:
            raise ValueError("could not determine shell command")
        name = match.group(1)
        return Path(name).name

    @staticmethod
    def _shell_command(command: str) -> tuple[str, list[str]]:
        if os.name == "nt":
            executable = "powershell.exe"
            return executable, [
                executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ]
        return "/bin/sh", ["/bin/sh", "-c", command]

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return (
            value[:limit]
            + f"\n...[output truncated at {limit} characters]"
        )

    @staticmethod
    def _timeout_output(exc: subprocess.TimeoutExpired) -> str:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return f"{stdout}{stderr}\n[process timed out]"


def register_workspace_tools(
    registry: ToolRegistry,
    settings: WorkspaceToolSettings | None = None,
    *,
    approval_callback: ApprovalCallback | None = None,
) -> WorkspaceToolset:
    toolset = WorkspaceToolset(
        settings,
        approval_callback=approval_callback,
    )
    toolset.register(registry)
    return toolset


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _bounded_int(
    name: str,
    default: int,
    *,
    maximum: int,
) -> int:
    return min(_positive_int(name, default), maximum)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _get_command_allowlist(
    value: str | None,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if value is None or not value.strip():
        return default
    commands = tuple(
        item.strip()
        for item in value.split(",")
        if item.strip()
    )
    if not commands:
        raise ValueError("AGENT_ALLOWED_SHELL_COMMANDS must not be empty")
    return commands
