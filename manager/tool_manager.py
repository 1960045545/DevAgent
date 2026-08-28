from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import replace
from typing import Any, Callable

from core.invoke_options import InvokeOptions
from core.response import ModelResponse
from core.tool_call import ToolCall
from core.tool_hooks import ToolCallContext, ToolHook
from core.tool_registry import ToolRegistry
from core.tool_space import ToolSpec
from core.todo import TodoList
from error.request_error import AgentRequestError
from manager.llm_manager import LLMManager


logger = logging.getLogger(__name__)


class ToolManager:
    def __init__(
        self,
        *,
        llm_manager: LLMManager,
        tool_registry: ToolRegistry | None = None,
        hooks: list[ToolHook] | None = None,
    ) -> None:
        self.llm_manager = llm_manager
        self.tool_registry = tool_registry or ToolRegistry()
        self.hooks = list(hooks or [])

    def register_hook(self, hook: ToolHook) -> None:
        self.hooks.append(hook)

    def unregister_hook(self, hook: ToolHook) -> None:
        if hook in self.hooks:
            self.hooks.remove(hook)

    def register_tool(
        self,
        spec: ToolSpec,
        handler: Callable[..., Any],
    ) -> None:
        logger.info("register tool name=%s", spec.name)
        self.tool_registry.register(
            spec,
            handler,
        )

    def merge_options(
        self,
        options: InvokeOptions,
        *,
        extra_specs: list[ToolSpec] | None = None,
        extra_handlers: dict[str, Callable[..., Any]] | None = None,
    ) -> InvokeOptions:
        registered_specs = self.tool_registry.list_specs()
        registered_handlers = self.tool_registry.handlers()
        if extra_specs:
            registered_specs = [*registered_specs, *extra_specs]
        if extra_handlers:
            registered_handlers = {
                **registered_handlers,
                **extra_handlers,
            }

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
        todo_list: TodoList | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> ModelResponse:
        if not options.tool_handlers:
            raise AgentRequestError(
                "tools require tool_handlers",
                retryable=False,
            )

        logger.info(
            "tool chat start tool_count=%d",
            len(options.tools or []),
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

        spec_by_name = {
            spec.name: spec
            for spec in options.tools or []
        }
        for round_index in range(options.max_tool_rounds):
            round_options = options
            if todo_list is not None and not todo_list.created:
                round_options = replace(
                    options,
                    tool_choice={
                        "type": "function",
                        "function": {"name": "todo_create"},
                    },
                )

            raw_response = (
                self.llm_manager.create_chat_completion(
                    messages,
                    round_options,
                    stream=False,
                    purpose="chat",
                )
            )
            data = self.llm_manager.response_to_dict(raw_response)
            response = self.llm_manager.parse_chat_response(data)

            if not response.tool_calls:
                if todo_list is not None and not todo_list.is_terminal:
                    messages.append(
                        self.llm_manager.assistant_message_from_response(data)
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": self._todo_continue_instruction(todo_list),
                        }
                    )
                    continue
                logger.info("tool chat finished without tool call")
                return response

            if (
                todo_list is not None
                and not todo_list.created
                and any(
                    tool_call.name != "todo_create"
                    for tool_call in response.tool_calls
                )
            ):
                raise AgentRequestError(
                    "complex tasks must create a todo list before using other tools",
                    retryable=False,
                )

            messages.append(
                self.llm_manager.assistant_message_from_response(data)
            )

            for tool_call in response.tool_calls:
                if not tool_call.id:
                    raise AgentRequestError(
                        "tool_call id is required",
                        retryable=False,
                    )

                logger.info(
                    "tool call requested name=%s id=%s",
                    tool_call.name,
                    tool_call.id,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": self._execute_tool_call(
                            tool_call,
                            options.tool_handlers,
                            spec_by_name=spec_by_name,
                            round_index=round_index,
                            progress_callback=progress_callback,
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
        *,
        spec_by_name: dict[str, ToolSpec],
        round_index: int,
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> str:
        handler = handlers.get(tool_call.name)

        if handler is None:
            raise AgentRequestError(
                f"tool handler not found: {tool_call.name}",
                retryable=False,
            )

        spec = spec_by_name.get(tool_call.name)
        if spec is None:
            spec = self.tool_registry.get_spec(tool_call.name)
        category = spec.category if spec is not None else "general"
        context = ToolCallContext(
            call_id=tool_call.id or "unknown",
            name=tool_call.name,
            category=category,
            arguments=tool_call.arguments,
            round_index=round_index,
        )
        self._notify_before(context, progress_callback)

        try:
            logger.debug(
                "executing tool name=%s arguments=%s",
                tool_call.name,
                tool_call.arguments,
            )
            if isinstance(tool_call.arguments, dict):
                result = handler(**tool_call.arguments)
            else:
                result = handler(tool_call.arguments)

            if inspect.isawaitable(result):
                result = asyncio.run(result)

        except Exception as exc:
            logger.exception(
                "tool execution failed name=%s",
                tool_call.name,
            )
            self._notify_error(context, exc, progress_callback)
            result = {
                "error": str(exc),
            }
        else:
            self._notify_after(context, result, progress_callback)

        logger.debug(
            "tool execution finished name=%s",
            tool_call.name,
        )
        return self._stringify_tool_result(result)

    def _notify_before(
        self,
        context: ToolCallContext,
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> None:
        self._notify(
            "tool_started",
            {
                "call_id": context.call_id,
                "name": context.name,
                "category": context.category,
                "round_index": context.round_index,
            },
            progress_callback,
        )
        for hook in self.hooks:
            try:
                hook.before_tool_call(context)
            except Exception:
                logger.exception("tool before hook failed name=%s", context.name)

    def _notify_after(
        self,
        context: ToolCallContext,
        result: Any,
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> None:
        self._notify(
            "tool_finished",
            {
                "call_id": context.call_id,
                "name": context.name,
                "category": context.category,
                "round_index": context.round_index,
            },
            progress_callback,
        )
        for hook in self.hooks:
            try:
                hook.after_tool_call(context, result)
            except Exception:
                logger.exception("tool after hook failed name=%s", context.name)

    def _notify_error(
        self,
        context: ToolCallContext,
        error: Exception,
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> None:
        self._notify(
            "tool_failed",
            {
                "call_id": context.call_id,
                "name": context.name,
                "category": context.category,
                "round_index": context.round_index,
                "error": str(error),
            },
            progress_callback,
        )
        for hook in self.hooks:
            try:
                hook.on_tool_error(context, error)
            except Exception:
                logger.exception("tool error hook failed name=%s", context.name)

    @staticmethod
    def _notify(
        event_type: str,
        data: dict[str, Any],
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> None:
        if progress_callback is not None:
            progress_callback(event_type, data)

    @staticmethod
    def _todo_continue_instruction(todo_list: TodoList) -> str:
        return (
            "You are working on a complex task. Do not provide the final answer "
            "yet. Read the current todo state and continue the next pending "
            "step. Mark it in_progress before work and completed only after "
            "the work is actually finished. Current state: "
            f"{todo_list.snapshot()}"
        )

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
