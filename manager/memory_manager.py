from __future__ import annotations

from typing import TYPE_CHECKING

from core.message import Message

if TYPE_CHECKING:
    from manager.llm_manager import LLMManager
    from manager.prompt_manager import PromptManager


class MemoryManager:
    def __init__(
        self,
        *,
        max_tokens: int = 1000,
        rounds: int = 3,
        prompt_manager: PromptManager | None = None,
        llm_manager: LLMManager | None = None,
        history_abstract_model_id: str | None = None,
    ) -> None:
        self.history: list[Message] = []
        self.max_tokens = max_tokens
        self.rounds = rounds
        self.prompt_manager = prompt_manager
        self.llm_manager = llm_manager
        self.history_abstract_model_id = history_abstract_model_id

    def append_chat(
        self,
        *,
        user_message: str,
        assistant_message: str,
    ) -> None:
        self.history.append(
            Message(role="user", content=user_message)
        )
        self.history.append(
            Message(role="assistant", content=assistant_message)
        )

    def compress_if_needed(self) -> None:
        total_tokens = sum(
            self.estimate_tokens(msg.content)
            for msg in self.history
        )
        keep_messages = self.rounds * 2

        if (
            total_tokens <= 0.8 * self.max_tokens
            or len(self.history) <= keep_messages
        ):
            return

        old_messages = self.history[:-keep_messages]
        recent_messages = self.history[-keep_messages:]

        if (
            not old_messages
            or self.prompt_manager is None
            or self.llm_manager is None
        ):
            return

        prompt = self.prompt_manager.build_compress_history_prompt(
            history=self.format_messages(old_messages),
        )

        try:
            response = self.llm_manager.invoke(
                Message(role="user", content=prompt),
                model_id=self.history_abstract_model_id,
                purpose="history",
            )
        except Exception:
            return

        summary = Message(
            role="system",
            content=f"以下是之前对话的摘要：\n{response.text}",
        )
        self.history = [summary] + recent_messages

    def get_prompt_texts(self) -> tuple[str, str]:
        keep_messages = self.rounds * 2
        history_messages = self.history[:-keep_messages]
        recent_messages = self.history[-keep_messages:]

        return (
            self.format_messages(history_messages),
            self.format_messages(recent_messages),
        )

    @staticmethod
    def estimate_tokens(text: str | None) -> int:
        if not text:
            return 0

        total = 0

        for char in text:
            if char.isspace():
                continue

            if ord(char) > 127:
                total += 2
            else:
                total += 1

        return total

    @staticmethod
    def format_messages(messages: list[Message]) -> str:
        lines = [
            f"{msg.role}: {msg.content}"
            for msg in messages
            if msg.content
        ]
        return "\n".join(lines)
