from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping


logger = logging.getLogger(__name__)


class PromptManager:
    SYSTEM_SECTION_ORDER = (
        "identity",
        "behavior",
        "safety",
        "tools",
        "workspace",
        "planning",
        "delegation",
        "skills",
        "memory",
        "profile",
        "operations",
    )

    def __init__(
        self,
        prompt_dir: str | Path | None = None,
    ) -> None:
        self.prompt_dir = (
            Path(prompt_dir)
            if prompt_dir
            else Path(__file__).resolve().parent.parent / "prompts"
        )
        self.default_prompt_dir = Path(__file__).resolve().parent.parent / "prompts"
        self._system_prompt_cache_key: str | None = None
        self._system_prompt_cache: str | None = None

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
        enabled_tools: list[Any] | None = None,
        workspace: str | Path | None = None,
        skills_catalog: str = "",
        operation_history: str = "",
        todo_enabled: bool = False,
        delegation_enabled: bool = False,
        todo_state: Any = None,
    ) -> str:
        context = {
            "history": history,
            "recent_chat_record": recent_chat_record,
            "user_profile": user_profile,
            "enabled_tools": enabled_tools or [],
            "workspace": str(workspace or ""),
            "skills_catalog": skills_catalog,
            "operation_history": operation_history,
            "todo_enabled": todo_enabled,
            "delegation_enabled": delegation_enabled,
            "todo_state": todo_state,
        }
        return self.get_system_prompt(context)

    def assemble_system_prompt(
        self,
        context: Mapping[str, Any],
    ) -> str:
        """Assemble named sections from the current runtime state.

        The section order is deterministic. Optional sections are enabled by
        actual state supplied by the host, rather than by keywords in a user
        message.
        """
        values = dict(context)
        sections: list[str] = []

        for name in self.SYSTEM_SECTION_ORDER:
            if not self._should_include_section(name, values):
                continue
            section_context = self._section_context(name, values)
            content = self._render_system_section(
                name,
                **section_context,
            ).strip()
            if content:
                sections.append(content)

        return "\n\n".join(sections)

    def _render_system_section(
        self,
        name: str,
        **context: Any,
    ) -> str:
        """Render an overridden section, falling back to bundled defaults."""
        section_path = self.prompt_dir / "system" / f"{name}.md"
        if not section_path.is_file():
            section_path = self.default_prompt_dir / "system" / f"{name}.md"
        prompt = section_path.read_text(encoding="utf-8")
        for key, value in context.items():
            prompt = prompt.replace(
                "{" + key + "}",
                "" if value is None else str(value),
            )
        return prompt

    def get_system_prompt(self, context: Mapping[str, Any]) -> str:
        """Return a cached prompt while the runtime context is unchanged."""
        cache_key = json.dumps(
            self._json_safe(context),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if (
            cache_key == self._system_prompt_cache_key
            and self._system_prompt_cache is not None
        ):
            logger.debug("system prompt cache hit")
            return self._system_prompt_cache

        prompt = self.assemble_system_prompt(context)
        self._system_prompt_cache_key = cache_key
        self._system_prompt_cache = prompt
        logger.debug(
            "system prompt assembled sections=%s",
            [
                name
                for name in self.SYSTEM_SECTION_ORDER
                if self._should_include_section(name, dict(context))
            ],
        )
        return prompt

    @classmethod
    def _should_include_section(
        cls,
        name: str,
        context: Mapping[str, Any],
    ) -> bool:
        if name in {"identity", "behavior", "safety", "tools", "workspace"}:
            return True
        if name == "planning":
            return bool(context.get("todo_enabled"))
        if name == "delegation":
            return bool(context.get("delegation_enabled"))
        if name == "skills":
            return bool(str(context.get("skills_catalog", "")).strip())
        if name == "memory":
            return bool(
                str(context.get("history", "")).strip()
                or str(context.get("recent_chat_record", "")).strip()
            )
        if name == "profile":
            return bool(str(context.get("user_profile", "")).strip())
        if name == "operations":
            return bool(str(context.get("operation_history", "")).strip())
        return False

    @staticmethod
    def _section_context(
        name: str,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        if name == "tools":
            return {
                "tools": PromptManager._format_tools(
                    context.get("enabled_tools", [])
                )
            }
        if name == "workspace":
            return {"workspace": str(context.get("workspace", "")) or "未指定"}
        if name == "skills":
            return {"skills_catalog": str(context.get("skills_catalog", ""))}
        if name == "memory":
            return {
                "history": str(context.get("history", "")),
                "recent_chat_record": str(
                    context.get("recent_chat_record", "")
                ),
            }
        if name == "profile":
            return {"user_profile": str(context.get("user_profile", ""))}
        if name == "operations":
            return {
                "operation_history": str(
                    context.get("operation_history", "")
                )
            }
        if name == "planning":
            todo_state = context.get("todo_state")
            if todo_state is None:
                todo_state = "尚未创建 TodoList。"
            elif not isinstance(todo_state, str):
                todo_state = json.dumps(
                    PromptManager._json_safe(todo_state),
                    ensure_ascii=False,
                )
            return {"todo_state": todo_state}
        return {}

    @staticmethod
    def _format_tools(tools: Any) -> str:
        entries: list[str] = []
        for tool in tools or []:
            if isinstance(tool, Mapping):
                tool_data = tool.get("function", tool)
                if not isinstance(tool_data, Mapping):
                    tool_data = tool
                name = str(tool_data.get("name", "")).strip()
                description = str(tool_data.get("description", "")).strip()
                category = (
                    str(tool_data.get("category", tool.get("category", "general"))).strip()
                    or "general"
                )
            else:
                name = str(getattr(tool, "name", "")).strip()
                description = str(getattr(tool, "description", "")).strip()
                category = (
                    str(getattr(tool, "category", "general")).strip()
                    or "general"
                )
            if name:
                suffix = f": {description}" if description else ""
                entries.append(f"- [{category}] {name}{suffix}")
        return "\n".join(entries) or "- 本轮没有注册可调用工具。"

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): cls._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "name") and hasattr(value, "description"):
            return {
                "name": str(value.name),
                "description": str(value.description),
                "category": str(getattr(value, "category", "general")),
            }
        return str(value)

    def build_chat_prompt(
        self,
        *,
        user_message: str,
        history: str,
        recent_chat_record: str,
        user_profile: str,
        enabled_tools: list[Any] | None = None,
        workspace: str | Path | None = None,
        skills_catalog: str = "",
        operation_history: str = "",
        todo_enabled: bool = False,
        delegation_enabled: bool = False,
        todo_state: Any = None,
    ) -> str:
        system_prompt = self.build_chat_system_prompt(
            history=history,
            recent_chat_record=recent_chat_record,
            user_profile=user_profile,
            enabled_tools=enabled_tools,
            workspace=workspace,
            skills_catalog=skills_catalog,
            operation_history=operation_history,
            todo_enabled=todo_enabled,
            delegation_enabled=delegation_enabled,
            todo_state=todo_state,
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
