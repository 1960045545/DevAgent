from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from core.tool_registry import ToolRegistry
from core.workspace_tools import (
    WorkspaceApprovalRequest,
    WorkspaceToolSettings,
    WorkspaceToolset,
    register_workspace_tools,
)


class WorkspaceToolTests(unittest.TestCase):
    def make_toolset(
        self,
        root: Path,
    ) -> WorkspaceToolset:
        return WorkspaceToolset(
            WorkspaceToolSettings(
                workspace_root=root,
                max_output_chars=10000,
                shell_timeout_seconds=10,
                python_timeout_seconds=10,
            ),
        )

    def test_file_read_write_replace_and_list(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            toolset = self.make_toolset(root)

            written = toolset.write_file(
                "notes.txt",
                "first line\nsecond line\n",
            )
            read = toolset.read_file("notes.txt")
            replaced = toolset.replace_text(
                "notes.txt",
                "second line",
                "updated line",
            )
            listed = toolset.list_files()

            self.assertEqual(written["path"], "notes.txt")
            self.assertIn("first line", read["content"])
            self.assertEqual(replaced["replacements"], 1)
            self.assertTrue(
                any(
                    entry["path"] == "notes.txt"
                    for entry in listed["entries"]
                ),
            )

    def test_paths_cannot_escape_workspace(self) -> None:
        with TemporaryDirectory() as directory:
            toolset = self.make_toolset(Path(directory))

            with self.assertRaises(PermissionError):
                toolset.read_file("../outside.txt")

    def test_sensitive_files_are_protected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("SECRET=value", encoding="utf-8")
            toolset = self.make_toolset(root)

            with self.assertRaises(PermissionError):
                toolset.read_file(".env")
            with self.assertRaises(PermissionError):
                toolset.write_file(".env", "replacement")

    def test_python_script_runs_with_current_interpreter(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hello.py").write_text(
                "print('hello from script')",
                encoding="utf-8",
            )
            result = self.make_toolset(root).run_python("hello.py")

            self.assertEqual(result["return_code"], 0)
            self.assertIn("hello from script", result["output"])

    def test_shell_command_runs_in_workspace(self) -> None:
        with TemporaryDirectory() as directory:
            toolset = self.make_toolset(Path(directory))
            command = "py --version" if __import__("os").name == "nt" else "python --version"

            result = toolset.run_shell(command)

            self.assertEqual(result["return_code"], 0)
            self.assertFalse(result["timed_out"])
            self.assertTrue(result["output"].strip())

    def test_shell_policy_rejects_chaining_and_unknown_commands(self) -> None:
        with TemporaryDirectory() as directory:
            toolset = self.make_toolset(Path(directory))

            with self.assertRaises(PermissionError):
                toolset.run_shell("python --version; whoami")
            with self.assertRaises(PermissionError):
                toolset.run_shell("whoami")
            with self.assertRaises(PermissionError):
                toolset.run_shell("py -c print('not allowed')")

    def test_rm_requires_host_approval_and_cannot_self_approve(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "delete-me.txt").write_text("x", encoding="utf-8")
            toolset = self.make_toolset(root)

            result = toolset.run_shell("rm delete-me.txt")

            self.assertEqual(result["status"], "approval_required")
            self.assertTrue((root / "delete-me.txt").exists())

    def test_approval_callback_can_deny_or_allow_file_delete(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "delete-me.txt"
            target.write_text("x", encoding="utf-8")
            requests: list[WorkspaceApprovalRequest] = []

            def deny(request: WorkspaceApprovalRequest) -> bool:
                requests.append(request)
                return False

            denied = WorkspaceToolset(
                WorkspaceToolSettings(workspace_root=root),
                approval_callback=deny,
            ).run_shell("rm delete-me.txt")
            self.assertEqual(denied["status"], "denied")
            self.assertTrue(target.exists())
            self.assertEqual(requests[0].action, "run_shell")

            target.unlink()
            target.write_text("x", encoding="utf-8")
            allowed = WorkspaceToolset(
                WorkspaceToolSettings(workspace_root=root),
                approval_callback=lambda request: True,
            ).run_shell("rm delete-me.txt")
            self.assertEqual(allowed["return_code"], 0)
            self.assertFalse(target.exists())

    def test_parent_navigation_inside_workspace_requires_approval(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "subdir").mkdir()
            toolset = self.make_toolset(root)

            result = toolset.run_shell(
                "Get-Location",
                working_directory="subdir\\..",
            )

            self.assertEqual(result["status"], "approval_required")
            self.assertEqual(result["working_directory"], ".")

    def test_parent_navigation_outside_workspace_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(PermissionError):
                self.make_toolset(root).run_shell(
                    "Get-Location",
                    working_directory="..",
                )

    def test_recursive_deletion_is_permanently_blocked(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "delete-me.txt").write_text("x", encoding="utf-8")
            toolset = WorkspaceToolset(
                WorkspaceToolSettings(workspace_root=root),
                approval_callback=lambda request: True,
            )

            for command in (
                "rm -rf delete-me.txt",
                "rm -r delete-me.txt",
                "rm --recursive delete-me.txt",
            ):
                with self.assertRaises(PermissionError):
                    toolset.run_shell(command)
            self.assertTrue((root / "delete-me.txt").exists())

    def test_shell_cannot_use_power_shell_or_home_escape_hatches(self) -> None:
        with TemporaryDirectory() as directory:
            toolset = self.make_toolset(Path(directory))
            blocked_commands = (
                "Get-Content $env:USERPROFILE",
                "Get-Content HKCU:\\Software",
                "Get-Content ~\\outside.txt",
                "Get-Content C:",
                "python -c print('no')",
            )
            for command in blocked_commands:
                with self.assertRaises(PermissionError):
                    toolset.run_shell(command)

    def test_python_parent_navigation_requires_approval(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "subdir").mkdir()
            (root / "hello.py").write_text(
                "print('hello')",
                encoding="utf-8",
            )
            result = self.make_toolset(root).run_python(
                "subdir\\..\\hello.py",
            )
            self.assertEqual(result["status"], "approval_required")

    def test_tools_register_into_registry(self) -> None:
        with TemporaryDirectory() as directory:
            registry = ToolRegistry()
            register_workspace_tools(
                registry,
                WorkspaceToolSettings(workspace_root=Path(directory)),
            )

            self.assertTrue(registry.has("workspace_read_file"))
            self.assertTrue(registry.has("workspace_run_shell"))
            self.assertTrue(registry.has("workspace_run_python"))

    def test_settings_can_load_environment_defaults(self) -> None:
        with TemporaryDirectory() as directory:
            with patch.dict(
                "os.environ",
                {
                    "AGENT_WORKSPACE_ROOT": directory,
                    "AGENT_ALLOW_SHELL": "false",
                },
                clear=True,
            ):
                settings = WorkspaceToolSettings.from_env()

            self.assertEqual(settings.workspace_root, Path(directory))
            self.assertFalse(settings.allow_shell)


if __name__ == "__main__":
    unittest.main()
