from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
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
        prompt_builder: Callable[[], str] | None = None,
        user_message: str,
        options: InvokeOptions,
        todo_list: TodoList | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        tool_result_callback: Callable[[ToolCall, str, int], None] | None = None,
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
            if prompt_builder is not None:
                messages[0]["content"] = prompt_builder()
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
                    if (
                        todo_list.can_pause_for_background
                    ):
                        logger.info(
                            "tool chat paused while background tasks continue"
                        )
                        if not response.text.strip():
                            active_tasks = todo_list.snapshot()[
                                "active_background_task_ids"
                            ]
                            response.text = (
                                "Background tasks are still running: "
                                + ", ".join(active_tasks)
                            )
                        return response
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

            first_round_plan = (
                todo_list is not None
                and not todo_list.created
            )
            if first_round_plan and not any(
                tool_call.name == "todo_create"
                for tool_call in response.tool_calls
            ):
                raise AgentRequestError(
                    "complex tasks must create a todo list before using other tools",
                    retryable=False,
                )

            messages.append(
                self.llm_manager.assistant_message_from_response(data)
            )

            tool_calls = response.tool_calls
            deferred_tool_calls: list[ToolCall] = []
            if first_round_plan:
                executable_plan_calls = [
                    tool_call
                    for tool_call in response.tool_calls
                    if tool_call.name == "todo_create"
                ][:1]
                tool_calls = executable_plan_calls
                executable_call_ids = {
                    id(tool_call)
                    for tool_call in executable_plan_calls
                }
                deferred_tool_calls = [
                    tool_call
                    for tool_call in response.tool_calls
                    if id(tool_call) not in executable_call_ids
                ]

            delegated_calls = self._parallel_delegated_calls(
                tool_calls,
                todo_list,
                handlers=options.tool_handlers,
                spec_by_name=spec_by_name,
                round_index=round_index,
                progress_callback=progress_callback,
            )
            for tool_call in tool_calls:
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
                tool_result = delegated_calls.get(tool_call.id)
                if tool_result is None:
                    tool_result = self._execute_tool_call(
                        tool_call,
                        options.tool_handlers,
                        spec_by_name=spec_by_name,
                        round_index=round_index,
                        progress_callback=progress_callback,
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )
                if tool_result_callback is not None:
                    tool_result_callback(
                        tool_call,
                        tool_result,
                        round_index,
                    )

            # The API requires one tool result for every call in an assistant
            # message. These calls are explicitly deferred, never executed.
            for tool_call in deferred_tool_calls:
                if not tool_call.id:
                    raise AgentRequestError(
                        "tool_call id is required",
                        retryable=False,
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": (
                            "Deferred until the next round: the execution plan "
                            "was created first. Do not treat this as executed."
                        ),
                    }
                )

        raise AgentRequestError(
            "max tool rounds exceeded",
            retryable=False,
        )

    def _parallel_delegated_calls(
        self,
        tool_calls: list[ToolCall],
        todo_list: TodoList | None,
        *,
        handlers: dict[str, Any],
        spec_by_name: dict[str, ToolSpec],
        round_index: int,
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> dict[str, str]:
        if todo_list is None:
            return {}

        candidates = [
            tool_call
            for tool_call in tool_calls
            if tool_call.name == "todo_delegate"
            and tool_call.id
            and isinstance(tool_call.arguments, dict)
        ]
        if len(candidates) < 2:
            return {}

        ready: list[ToolCall] = []
        deferred: dict[str, str] = {}
        seen_items: set[str] = set()
        for tool_call in candidates:
            item_id = str(
                tool_call.arguments.get(
                    "task_id",
                    tool_call.arguments.get("item_id", ""),
                )
            )
            if item_id in seen_items:
                deferred[tool_call.id or "unknown"] = (
                    "Deferred because the same todo item was delegated more than once."
                )
                continue
            seen_items.add(item_id)
            try:
                is_ready = todo_list.is_ready(item_id)
            except Exception as exc:
                deferred[tool_call.id or "unknown"] = f"Deferred: {exc}"
                continue
            if is_ready:
                ready.append(tool_call)
            else:
                deferred[tool_call.id or "unknown"] = (
                    "Deferred until its todo dependencies are completed."
                )

        if ready:
            if len(ready) >= 2:
                with ThreadPoolExecutor(max_workers=len(ready)) as executor:
                    futures = {
                        tool_call.id: executor.submit(
                            self._execute_tool_call,
                            tool_call,
                            handlers,
                            spec_by_name=spec_by_name,
                            round_index=round_index,
                            progress_callback=progress_callback,
                        )
                        for tool_call in ready
                    }
                    for call_id, future in futures.items():
                        deferred[call_id or "unknown"] = future.result()
            else:
                tool_call = ready[0]
                deferred[tool_call.id or "unknown"] = self._execute_tool_call(
                    tool_call,
                    handlers,
                    spec_by_name=spec_by_name,
                    round_index=round_index,
                    progress_callback=progress_callback,
                )
            return deferred

        return deferred

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
            "yet. Read the current Task DAG and continue a ready pending task. "
            "First call todo_claim to move pending to in_process. After the "
            "work is actually finished call todo_complete with an execution "
            "summary. Call todo_block with a reason when a task cannot proceed. "
            "For npm install, pip install, or another long command, call "
            "todo_run_background and continue other ready tasks. If only "
            "background tasks remain, report their job ids and return; the "
            "host will notify the user when they finish. "
            "Current state: "
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
