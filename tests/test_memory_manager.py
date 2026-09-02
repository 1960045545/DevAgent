from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from core.message import Message
from core.tool_call import ToolCall
from manager.memory_manager import MemoryManager


class FakePromptManager:
    def __init__(self) -> None:
        self.history_prompt = ""

    def build_compress_history_prompt(self, *, history: str) -> str:
        self.history_prompt = history
        return f"summarize:\n{history}"


class FakeLLMManager:
    def __init__(self, text: str = "compressed context") -> None:
        self.text = text
        self.calls: list[dict] = []

    def invoke(self, message, *, model_id, purpose):
        self.calls.append(
            {
                "message": message,
                "model_id": model_id,
                "purpose": purpose,
            }
        )
        return SimpleNamespace(text=self.text)


class MemoryManagerTests(unittest.TestCase):
    def test_retains_first_three_and_latest_forty_seven_rounds(self) -> None:
        with TemporaryDirectory() as directory:
            manager = MemoryManager(
                max_tokens=100000,
                transcripts_dir=directory,
            )
            for index in range(55):
                manager.append_chat(
                    user_message=f"user-{index}",
                    assistant_message=f"assistant-{index}",
                )
            for index in range(5):
                manager.append_tool_result(
                    ToolCall(
                        id=f"call-{index}",
                        name="read_file",
                        arguments={"path": f"file-{index}"},
                    ),
                    f"result-{index}",
                )

            manager.compress_if_needed()

            self.assertEqual(len(manager.history), 100)
            self.assertIn("user-0", manager.history[0].content)
            self.assertIn("assistant-2", manager.history[5].content)
            self.assertIn("user-8", manager.history[6].content)
            self.assertIn("assistant-54", manager.history[-1].content)
            self.assertEqual(
                [operation["result"] for operation in manager.operation_history],
                ["result-2", "result-3", "result-4"],
            )
            self.assertIsNotNone(manager.last_transcript_path)

            transcript = json.loads(
                Path(manager.last_transcript_path).read_text(encoding="utf-8")
            )
            self.assertEqual(len(transcript["messages"]), 110)
            self.assertEqual(len(transcript["operations"]), 5)

    def test_over_budget_context_is_summarized_after_transcript_save(self) -> None:
        with TemporaryDirectory() as directory:
            prompt_manager = FakePromptManager()
            llm_manager = FakeLLMManager()
            manager = MemoryManager(
                max_tokens=40,
                prompt_manager=prompt_manager,
                llm_manager=llm_manager,
                history_abstract_model_id="history-model",
                transcripts_dir=directory,
            )
            manager.append_chat(
                user_message="u" * 80,
                assistant_message="a" * 80,
            )
            for index in range(4):
                manager.append_skill_result(
                    f"skill-{index}",
                    "skill result " + ("x" * 30),
                )

            manager.compress_if_needed()

            self.assertEqual(len(llm_manager.calls), 1)
            self.assertEqual(llm_manager.calls[0]["purpose"], "history")
            self.assertEqual(manager.history[0].role, "system")
            self.assertIn("compressed context", manager.history[0].content)
            self.assertIn("技能和工具调用结果", prompt_manager.history_prompt)
            self.assertIsNotNone(manager.last_transcript_path)

            transcript = json.loads(
                Path(manager.last_transcript_path).read_text(encoding="utf-8")
            )
            self.assertEqual(len(transcript["messages"]), 2)
            self.assertEqual(len(transcript["operations"]), 4)

    def test_prompt_texts_split_retained_window_into_head_and_recent(self) -> None:
        manager = MemoryManager(max_tokens=100000)
        for index in range(4):
            manager.append_chat(
                user_message=f"user-{index}",
                assistant_message=f"assistant-{index}",
            )

        head, recent = manager.get_prompt_texts()

        self.assertIn("user-0", head)
        self.assertIn("assistant-2", head)
        self.assertIn("user-3", recent)
        self.assertNotIn("user-0", recent)

    def test_summary_is_not_counted_as_a_conversation_round_on_next_compress(self) -> None:
        with TemporaryDirectory() as directory:
            prompt_manager = FakePromptManager()
            llm_manager = FakeLLMManager()
            manager = MemoryManager(
                max_tokens=40,
                prompt_manager=prompt_manager,
                llm_manager=llm_manager,
                transcripts_dir=directory,
            )
            manager.append_chat(
                user_message="u" * 80,
                assistant_message="a" * 80,
            )
            manager.compress_if_needed()
            first_transcript = manager.last_transcript_path

            for index in range(55):
                manager.append_chat(
                    user_message=f"new-user-{index}",
                    assistant_message=f"new-assistant-{index}",
                )

            manager.compress_if_needed()

            self.assertEqual(len(llm_manager.calls), 2)
            self.assertEqual(len(manager.history), 1)
            self.assertTrue(
                manager.history[0].metadata.get("context_summary")
            )
            self.assertIn("new-user-0", prompt_manager.history_prompt)
            self.assertIn("new-assistant-54", prompt_manager.history_prompt)

            second_transcript = manager.last_transcript_path
            self.assertNotEqual(first_transcript, second_transcript)
            transcript = json.loads(
                Path(second_transcript).read_text(encoding="utf-8")
            )
            self.assertEqual(
                transcript["parent_transcript"],
                str(first_transcript),
            )


if __name__ == "__main__":
    unittest.main()
