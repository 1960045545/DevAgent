from __future__ import annotations

import unittest

from core.tool_space import ToolSpec
from manager.prompt_manager import PromptManager


class PromptManagerTests(unittest.TestCase):
    def test_system_prompt_has_stable_sections_and_runtime_tool_catalog(self) -> None:
        manager = PromptManager()
        tool = ToolSpec(
            name="workspace_read_file",
            description="Read a workspace file.",
            parameters={"type": "object"},
            category="filesystem",
        )

        prompt = manager.build_chat_system_prompt(
            history="old conversation",
            recent_chat_record="recent conversation",
            user_profile="prefers concise answers",
            enabled_tools=[tool],
            workspace="D:/workspace",
            skills_catalog="# Available Skills\n- reader: Read files.",
            operation_history="1. [tool] workspace_read_file\n结果：ok",
            todo_enabled=True,
            delegation_enabled=True,
            todo_state={
                "completed_count": 1,
                "total_count": 2,
            },
        )

        positions = [
            prompt.index("# Agent 身份"),
            prompt.index("# 工作方式"),
            prompt.index("# 安全边界"),
            prompt.index("# 可用工具"),
            prompt.index("# 当前工作区"),
            prompt.index("# TodoList 任务规划"),
            prompt.index("# 子 Agent 委派"),
            prompt.index("# 本地技能目录"),
            prompt.index("# 对话上下文"),
            prompt.index("# 用户画像"),
            prompt.index("# 最近工具和技能调用结果"),
        ]

        self.assertEqual(positions, sorted(positions))
        self.assertIn("[filesystem] workspace_read_file", prompt)
        self.assertIn("D:/workspace", prompt)
        self.assertIn('"completed_count": 1', prompt)

    def test_optional_sections_are_not_loaded_without_runtime_state(self) -> None:
        manager = PromptManager()

        prompt = manager.build_chat_system_prompt(
            history="",
            recent_chat_record="",
            user_profile="",
        )

        self.assertIn("# Agent 身份", prompt)
        self.assertIn("# 安全边界", prompt)
        self.assertIn("# 可用工具", prompt)
        self.assertIn("# 当前工作区", prompt)
        self.assertNotIn("# TodoList 任务规划", prompt)
        self.assertNotIn("# 子 Agent 委派", prompt)
        self.assertNotIn("# 本地技能目录", prompt)
        self.assertNotIn("# 对话上下文", prompt)
        self.assertNotIn("# 用户画像", prompt)
        self.assertNotIn("# 最近工具和技能调用结果", prompt)

    def test_skill_catalog_does_not_imply_skill_body_loading(self) -> None:
        manager = PromptManager()

        prompt = manager.build_chat_system_prompt(
            history="",
            recent_chat_record="",
            user_profile="",
            skills_catalog=(
                "# Available Skills\n"
                "- reader: Read files."
            ),
        )

        self.assertIn("reader: Read files.", prompt)
        self.assertIn("load_skill", prompt)
        self.assertNotIn("PRIVATE_SKILL_BODY", prompt)

    def test_identical_context_uses_cached_prompt(self) -> None:
        manager = PromptManager()
        context = {
            "history": "history",
            "recent_chat_record": "recent",
            "user_profile": "",
            "enabled_tools": [],
            "workspace": "D:/workspace",
            "skills_catalog": "",
            "operation_history": "",
            "todo_enabled": False,
            "delegation_enabled": False,
            "todo_state": None,
        }

        first = manager.get_system_prompt(context)
        second = manager.get_system_prompt(dict(context))

        self.assertIs(first, second)
        self.assertEqual(first, manager.assemble_system_prompt(context))


if __name__ == "__main__":
    unittest.main()
