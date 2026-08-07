from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class UserProfile:
    """保存对后续回答有帮助的长期用户特征。"""

    path: Path | None = None
    max_items: int = 20
    items: list[str] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        path: str | Path | None,
        *,
        max_items: int = 20,
    ) -> "UserProfile":
        profile = cls(
            path=Path(path) if path else None,
            max_items=max_items,
        )

        if profile.path is None or not profile.path.exists():
            return profile

        try:
            data: Any = json.loads(
                profile.path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return profile

        if isinstance(data, dict):
            items = data.get("items", [])
        else:
            items = data

        if isinstance(items, list):
            profile.merge(items)

        return profile

    def merge(
        self,
        items: list[Any] | None = None,
        remove: list[Any] | None = None,
    ) -> None:
        remove_keys = {
            self._normalise(item)
            for item in remove or []
            if isinstance(item, str)
        }

        self.items = [
            item
            for item in self.items
            if self._normalise(item) not in remove_keys
        ]

        existing_keys = {
            self._normalise(item)
            for item in self.items
        }

        for item in items or []:
            if not isinstance(item, str):
                continue

            cleaned = " ".join(item.split())

            if not cleaned:
                continue

            key = self._normalise(cleaned)

            if key not in existing_keys:
                self.items.append(cleaned)
                existing_keys.add(key)

        if self.max_items > 0:
            self.items = self.items[-self.max_items:]

    def format(self) -> str:
        if not self.items:
            return "暂无已记录的用户画像。"

        return "\n".join(
            f"- {item}"
            for item in self.items
        )

    def save(self) -> None:
        if self.path is None:
            return

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.path.write_text(
            json.dumps(
                {"items": self.items},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _normalise(value: str) -> str:
        return " ".join(value.strip().lower().split())
