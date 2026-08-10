from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class PromptManager:
    def __init__(
        self,
        prompt_dir: str | Path | None = None,
    ) -> None:
        self.prompt_dir = (
            Path(prompt_dir)
            if prompt_dir
            else Path(__file__).resolve().parent.parent / "prompts"
        )

    def load(self, file_name: str) -> str:
        prompt_path = self.prompt_dir / file_name
        logger.debug(
            "loading prompt template file=%s",
            prompt_path,
        )
        return prompt_path.read_text(encoding="utf-8")

    def render(
        self,
        file_name: str,
        **context: Any,
    ) -> str:
        logger.debug(
            "rendering prompt file=%s keys=%s",
            file_name,
            sorted(context.keys()),
        )
        prompt = self.load(file_name)

        for key, value in context.items():
            prompt = prompt.replace(
                "{" + key + "}",
                "" if value is None else str(value),
            )

        return prompt

    def build_chat_system_prompt(
        self,
        *,
        history: str,
        recent_chat_record: str,
        user_profile: str,
    ) -> str:
        return self.render(
            "chat_prompt.md",
            history=history,
            recent_chat_record=recent_chat_record,
            user_profile=user_profile,
        )

    def build_chat_prompt(
        self,
        *,
        user_message: str,
        history: str,
        recent_chat_record: str,
        user_profile: str,
    ) -> str:
        system_prompt = self.build_chat_system_prompt(
            history=history,
            recent_chat_record=recent_chat_record,
            user_profile=user_profile,
        )

        return (
            f"{system_prompt}\n\n"
            f"用户最新问题：\n{user_message}"
        )

    def build_compress_history_prompt(
        self,
        *,
        history: str,
    ) -> str:
        return self.render(
            "compress_history_prompt.md",
            history=history,
        )

    def build_user_profile_update_prompt(
        self,
        *,
        user_profile: str,
        user_message: str,
        assistant_message: str,
    ) -> str:
        return self.render(
            "user_profile_update_prompt.md",
            user_profile=user_profile,
            user_message=user_message,
            assistant_message=assistant_message,
        )
