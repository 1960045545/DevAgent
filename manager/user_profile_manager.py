from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core.invoke_options import InvokeOptions
from core.message import Message
from core.user_profile import UserProfile
from manager.llm_manager import LLMManager
from manager.prompt_manager import PromptManager


class UserProfileManager:
    def __init__(
        self,
        *,
        prompt_manager: PromptManager,
        llm_manager: LLMManager,
        model_id: str | None = None,
        profile_path: str | Path | None = None,
        max_items: int = 20,
        enabled: bool = True,
    ) -> None:
        self.prompt_manager = prompt_manager
        self.llm_manager = llm_manager
        self.model_id = model_id or llm_manager.model_id
        self.enabled = enabled

        default_profile_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "user_profile.json"
        )
        path = (
            Path(profile_path)
            if profile_path
            else default_profile_path
        )
        self.profile = (
            UserProfile.load(
                path,
                max_items=max_items,
            )
            if enabled
            else UserProfile(max_items=max_items)
        )

    def format(self) -> str:
        if not self.enabled:
            return "暂无已记录的用户画像。"

        return self.profile.format()

    def update(
        self,
        *,
        user_message: str,
        assistant_message: str,
    ) -> None:
        if not self.enabled:
            return

        prompt = (
            self.prompt_manager.build_user_profile_update_prompt(
                user_profile=self.format(),
                user_message=user_message,
                assistant_message=assistant_message,
            )
        )

        try:
            response = self.llm_manager.invoke(
                Message(role="user", content=prompt),
                options=InvokeOptions(
                    temperature=0,
                    max_output_tokens=400,
                ),
                model_id=self.model_id,
            )
            update = self.parse_json_object(response.text)
            additions = self.as_string_list(
                update.get("add")
                or update.get("items")
                or update.get("profile")
            )
            removals = self.as_string_list(
                update.get("remove")
            )

            self.profile.merge(
                items=additions,
                remove=removals,
            )
            self.profile.save()
        except Exception:
            return

    @staticmethod
    def as_string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]

        if not isinstance(value, list):
            return []

        return [
            item
            for item in value
            if isinstance(item, str)
        ]

    @staticmethod
    def parse_json_object(text: str) -> dict[str, Any]:
        if not text:
            return {}

        cleaned = text.strip()
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start < 0 or end <= start:
                return {}

            try:
                data = json.loads(
                    cleaned[start:end + 1]
                )
            except json.JSONDecodeError:
                return {}

        return data if isinstance(data, dict) else {}
