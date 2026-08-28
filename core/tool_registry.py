from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core.tool_space import ToolSpec


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        spec: ToolSpec,
        handler: Callable[..., Any],
    ) -> None:
        if not spec.name:
            raise ValueError("tool name can not be empty")

        self._tools[spec.name] = RegisteredTool(
            spec=spec,
            handler=handler,
        )

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def clear(self) -> None:
        self._tools.clear()

    def has(self, name: str) -> bool:
        return name in self._tools

    def get_spec(self, name: str) -> ToolSpec | None:
        tool = self._tools.get(name)
        return None if tool is None else tool.spec

    def get_handler(
        self,
        name: str,
    ) -> Callable[..., Any] | None:
        tool = self._tools.get(name)

        if tool is None:
            return None

        return tool.handler

    def list_specs(self, category: str | None = None) -> list[ToolSpec]:
        specs = [
            tool.spec
            for tool in self._tools.values()
        ]
        if category is None:
            return specs
        normalized = category.strip().lower()
        return [spec for spec in specs if spec.category == normalized]

    def list_categories(self) -> list[str]:
        return sorted({spec.category for spec in self.list_specs()})

    def handlers(self) -> dict[str, Callable[..., Any]]:
        return {
            name: tool.handler
            for name, tool in self._tools.items()
        }
