from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import replace
from typing import Any, Callable

from core.invoke_options import InvokeOptions
from core.response import ModelResponse
from core.tool_call import ToolCall
from core.tool_registry import ToolRegistry
from core.tool_space import ToolSpec
from error.request_error import AgentRequestError
from manager.llm_manager import LLMManager


class ToolManager:
    def __init__(
        self,
        *,
        llm_manager: LLMManager,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.llm_manager = llm_manager
        self.tool_registry = tool_registry or ToolRegistry()

    def register_tool(
        self,
        spec: ToolSpec,
        handler: Callable[..., Any],
    ) -> None:
        self.tool_registry.register(
            spec,
            handler,
        )

    def merge_options(
        self,
        options: InvokeOptions,
    ) -> InvokeOptions:
        registered_specs = self.tool_registry.list_specs()
        registered_handlers = self.tool_registry.handlers()

        if not registered_specs and not options.tools:
            return options

        specs_by_name: dict[str, Any] = {}

        for spec in registered_specs:
            specs_by_name[spec.name] = spec

        if options.tools:
            for spec in options.tools:
                name = self._tool_spec_name(spec)

                if name:
                    specs_by_name[name] = spec

        handlers = dict(registered_handlers)

        if options.tool_handlers:
            handlers.update(options.tool_handlers)

        return replace(
            options,
            tools=list(specs_by_name.values()),
            tool_handlers=handlers,
        )

    def chat_with_tools(
        self,
        *,
        prompt: str,
        user_message: str,
        options: InvokeOptions,
    ) -> ModelResponse:
        if not options.tool_handlers:
            raise AgentRequestError(
                "tools require tool_handlers",
                retryable=False,
            )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        for _ in range(options.max_tool_rounds):
            raw_response = (
                self.llm_manager.create_chat_completion(
                    messages,
                    options,
                    stream=False,
                    purpose="chat",
                )
            )
            data = self.llm_manager.response_to_dict(raw_response)
            response = self.llm_manager.parse_chat_response(data)

            if not response.tool_calls:
                return response

            messages.append(
                self.llm_manager.assistant_message_from_response(data)
            )

            for tool_call in response.tool_calls:
                if not tool_call.id:
                    raise AgentRequestError(
                        "tool_call id is required",
                        retryable=False,
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": self._execute_tool_call(
                            tool_call,
                            options.tool_handlers,
                        ),
                    }
                )

        raise AgentRequestError(
            "max tool rounds exceeded",
            retryable=False,
        )

    @staticmethod
    def _tool_spec_name(tool: Any) -> str | None:
        if hasattr(tool, "name"):
            return tool.name

        if isinstance(tool, dict):
            function = tool.get("function") or {}
            return function.get("name") or tool.get("name")

        return None

    def _execute_tool_call(
        self,
        tool_call: ToolCall,
        handlers: dict[str, Any],
    ) -> str:
        handler = handlers.get(tool_call.name)

        if handler is None:
            raise AgentRequestError(
                f"tool handler not found: {tool_call.name}",
                retryable=False,
            )

        try:
            if isinstance(tool_call.arguments, dict):
                result = handler(**tool_call.arguments)
            else:
                result = handler(tool_call.arguments)

            if inspect.isawaitable(result):
                result = asyncio.run(result)

        except Exception as exc:
            result = {
                "error": str(exc),
            }

        return self._stringify_tool_result(result)

    @staticmethod
    def _stringify_tool_result(result: Any) -> str:
        if isinstance(result, str):
            return result

        try:
            return json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            )
        except TypeError:
            return str(result)
