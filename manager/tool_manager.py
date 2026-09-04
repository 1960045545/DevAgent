from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Protocol

from core.invoke_options import InvokeOptions
from core.response import ModelResponse
from core.tool_call import ToolCall
from core.tool_hooks import (
    StopContext,
    ToolCallContext,
    ToolHook,
    ToolHookDecision,
    ToolPostprocessResult,
    UserPromptSubmitContext,
)
from core.tool_registry import ToolRegistry
from core.tool_space import ToolSpec
from core.todo import TodoList
from error.request_error import AgentRequestError
from manager.llm_manager import LLMManager
from manager.message_compactor import MessageCompactionPipeline
from runtime.background_worker import BackgroundTaskManager, BackgroundToolDispatcher


logger = logging.getLogger(__name__)


class MCPToolProvider(Protocol):
    """Minimal transport-neutral contract for a connected MCP client."""

    def list_tools(self) -> Iterable[Any]:
        ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        ...


class ToolManager:
    def __init__(
        self,
        *,
        llm_manager: LLMManager,
        tool_registry: ToolRegistry | None = None,
        hooks: list[ToolHook] | None = None,
        compaction_pipeline: MessageCompactionPipeline | None = None,
        background_manager: BackgroundTaskManager | None = None,
    ) -> None:
        self.llm_manager = llm_manager
        self.tool_registry = tool_registry or ToolRegistry()
        self.hooks = list(hooks or [])
        self.compaction_pipeline = compaction_pipeline
        self.background_manager = background_manager
        self.background_dispatcher = (
            BackgroundToolDispatcher(background_manager)
            if background_manager is not None
            else None
        )
        self._mcp_specs: dict[str, ToolSpec] = {}
        self._mcp_handlers: dict[str, Callable[..., Any]] = {}
        self._mcp_servers: dict[str, set[str]] = {}

    def set_background_manager(
        self,
        manager: BackgroundTaskManager | None,
    ) -> None:
        self.background_manager = manager
        self.background_dispatcher = (
            BackgroundToolDispatcher(manager) if manager is not None else None
        )

    def set_compaction_pipeline(
        self,
        pipeline: MessageCompactionPipeline | None,
    ) -> None:
        self.compaction_pipeline = pipeline

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

    def register_mcp_tool(
        self,
        spec: ToolSpec,
        handler: Callable[..., Any],
        *,
        server_name: str | None = None,
    ) -> str:
        """Register one normalized tool from a connected MCP server."""
        name = spec.name
        if server_name and not name.startswith("mcp__"):
            name = f"mcp__{server_name}__{name}"
            spec = replace(spec, name=name, category="mcp")
        self._mcp_specs[name] = spec
        self._mcp_handlers[name] = handler
        if server_name:
            self._mcp_servers.setdefault(server_name, set()).add(name)
        return name

    def register_mcp_tools(
        self,
        specs: list[ToolSpec],
        handlers: dict[str, Callable[..., Any]],
        *,
        server_name: str | None = None,
    ) -> list[str]:
        names: list[str] = []
        for spec in specs:
            handler = handlers.get(spec.name)
            if handler is None:
                continue
            names.append(
                self.register_mcp_tool(
                    spec,
                    handler,
                    server_name=server_name,
                )
            )
        return names

    def connect_mcp(
        self,
        server_name: str,
        provider: MCPToolProvider,
    ) -> list[str]:
        """Normalize a connected MCP client's tools into this agent's pool."""
        normalized_server = server_name.strip()
        if not normalized_server:
            raise ValueError("MCP server name must not be empty")
        names: list[str] = []
        for raw_tool in provider.list_tools():
            spec = self._coerce_mcp_spec(raw_tool)
            source_name = spec.name

            def handler(
                _provider: MCPToolProvider = provider,
                _source_name: str = source_name,
                **arguments: Any,
            ) -> Any:
                return _provider.call_tool(_source_name, arguments)

            name = self.register_mcp_tool(
                spec,
                handler,
                server_name=normalized_server,
            )
            names.append(name)
        return names

    def disconnect_mcp(self, server_name: str) -> list[str]:
        """Remove tools belonging to one connected MCP server."""
        names = sorted(self._mcp_servers.pop(server_name.strip(), set()))
        for name in names:
            self._mcp_specs.pop(name, None)
            self._mcp_handlers.pop(name, None)
        return names

    @staticmethod
    def _coerce_mcp_spec(raw_tool: Any) -> ToolSpec:
        if isinstance(raw_tool, ToolSpec):
            return raw_tool
        if isinstance(raw_tool, dict):
            name = raw_tool.get("name")
            description = raw_tool.get("description", "")
            parameters = raw_tool.get("parameters") or raw_tool.get(
                "inputSchema",
                {"type": "object", "properties": {}},
            )
        else:
            name = getattr(raw_tool, "name", None)
            description = getattr(raw_tool, "description", "")
            parameters = getattr(raw_tool, "inputSchema", None) or getattr(
                raw_tool,
                "parameters",
                {"type": "object", "properties": {}},
            )
        if not isinstance(name, str) or not name.strip():
            raise ValueError("MCP tool is missing a name")
        if not isinstance(parameters, dict):
            raise ValueError(f"MCP tool parameters must be an object: {name}")
        return ToolSpec(
            name=name.strip(),
            description=str(description or "MCP tool"),
            parameters=parameters,
            category="mcp",
        )

    def assemble_tool_pool(
        self,
        options: InvokeOptions,
        *,
        extra_specs: list[ToolSpec] | None = None,
        extra_handlers: dict[str, Callable[..., Any]] | None = None,
    ) -> InvokeOptions:
        """Build the model-visible pool from built-ins, runtime, and MCP tools."""
        registered_specs = [
            *self.tool_registry.list_specs(),
            *self._mcp_specs.values(),
            *(extra_specs or []),
        ]
        registered_handlers = {
            **self.tool_registry.handlers(),
            **self._mcp_handlers,
            **(extra_handlers or {}),
        }
        if not registered_specs and not options.tools:
            return options

        specs_by_name: dict[str, Any] = {
            spec.name: spec for spec in registered_specs
        }
        for spec in options.tools or []:
            name = self._tool_spec_name(spec)
            if name:
                specs_by_name[name] = spec
        handlers = dict(registered_handlers)
        handlers.update(options.tool_handlers or {})
        return replace(
            options,
            tools=list(specs_by_name.values()),
            tool_handlers=handlers,
        )

    def merge_options(
        self,
        options: InvokeOptions,
        *,
        extra_specs: list[ToolSpec] | None = None,
        extra_handlers: dict[str, Callable[..., Any]] | None = None,
    ) -> InvokeOptions:
        return self.assemble_tool_pool(
            options,
            extra_specs=extra_specs,
            extra_handlers=extra_handlers,
        )

    def run_user_prompt_hooks(
        self,
        user_message: str,
        *,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> UserPromptSubmitContext:
        context = UserPromptSubmitContext(user_message=user_message)
        self._notify(
            "user_prompt_submitted",
            {"message_chars": len(user_message)},
            progress_callback,
        )
        for hook in self.hooks:
            callback = getattr(hook, "on_user_prompt_submit", None)
            if not callable(callback):
                continue
            try:
                callback(context)
            except Exception:
                logger.exception("UserPromptSubmit hook failed")
        return context

    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._append_background_notifications(messages, progress_callback)
        if self.compaction_pipeline is not None:
            self.compaction_pipeline.compact(
                messages,
                observer=progress_callback,
            )

    def chat_with_tools(
        self,
        *,
        prompt: str,
        prompt_builder: Callable[..., str] | None = None,
        user_message: str,
        options: InvokeOptions,
        todo_list: TodoList | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        tool_result_callback: Callable[[ToolCall, str, int], None] | None = None,
        injected_messages: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        options = self.assemble_tool_pool(options)
        if options.tools and not options.tool_handlers:
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
        if injected_messages is None:
            prompt_context = self.run_user_prompt_hooks(
                user_message,
                progress_callback=progress_callback,
            )
            injected_messages = prompt_context.injected_messages
        if injected_messages:
            messages[1:1] = [dict(message) for message in injected_messages]

        for round_index in range(options.max_tool_rounds):
            round_options = self.assemble_tool_pool(options)
            if prompt_builder is not None:
                messages[0]["content"] = self._call_prompt_builder(
                    prompt_builder,
                    round_options.tools or [],
                )
            self.prepare_messages(messages, progress_callback=progress_callback)
            spec_by_name = {
                spec.name: spec
                for spec in round_options.tools or []
            }
            if todo_list is not None and not todo_list.created:
                round_options = replace(
                    round_options,
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
                        return self._stop(
                            response,
                            messages,
                            round_index,
                            "background_pause",
                            progress_callback,
                        )
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
                return self._stop(
                    response,
                    messages,
                    round_index,
                    "no_tool_use",
                    progress_callback,
                )

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
                handlers=round_options.tool_handlers or {},
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
                        round_options.tool_handlers or {},
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

    @staticmethod
    def _call_prompt_builder(
        prompt_builder: Callable[..., str],
        tools: list[ToolSpec],
    ) -> str:
        try:
            signature = inspect.signature(prompt_builder)
        except (TypeError, ValueError):
            return prompt_builder()
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        if positional:
            return prompt_builder(tools)
        return prompt_builder()

    def _append_background_notifications(
        self,
        messages: list[dict[str, Any]],
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> None:
        if self.background_manager is None:
            return
        notifications = self.background_manager.notifications(consume=True)
        for notification in notifications["notifications"]:
            content = (
                "<task_notification>\n"
                + json.dumps(notification, ensure_ascii=False, default=str)
                + "\n</task_notification>"
            )
            messages.append({"role": "system", "content": content})
            self._notify(
                "background_notification_injected",
                notification,
                progress_callback,
            )

    def notify_stop(
        self,
        response: ModelResponse,
        messages: list[dict[str, Any]],
        *,
        round_index: int = 0,
        reason: str = "no_tool_use",
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> ModelResponse:
        return self._stop(
            response,
            messages,
            round_index,
            reason,
            progress_callback,
        )

    def _stop(
        self,
        response: ModelResponse,
        messages: list[dict[str, Any]],
        round_index: int,
        reason: str,
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> ModelResponse:
        context = StopContext(
            response=response,
            messages=tuple(messages),
            round_index=round_index,
            reason=reason,
        )
        for hook in self.hooks:
            callback = getattr(hook, "on_stop", None)
            if not callable(callback):
                continue
            try:
                callback(context)
            except Exception:
                logger.exception("Stop hook failed")
        self._notify(
            "agent_stop",
            {"round_index": round_index, "reason": reason},
            progress_callback,
        )
        return response

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
            return self._stringify_tool_result(
                {
                    "error": f"tool handler not found: {tool_call.name}",
                    "tool_call_id": tool_call.id,
                }
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
        decision = self._notify_before(context, progress_callback)
        if decision is not None and decision.blocked:
            reason = decision.reason or "blocked by PreToolUse hook"
            result = decision.result
            if result is None:
                result = {"error": reason, "blocked": True}
            self._notify(
                "tool_blocked",
                {
                    "call_id": context.call_id,
                    "name": context.name,
                    "category": context.category,
                    "round_index": context.round_index,
                    "reason": reason,
                },
                progress_callback,
            )
            return self._stringify_tool_result(result)

        if (
            self.background_dispatcher is not None
            and self.background_dispatcher.should_run(tool_call, handler)
        ):
            try:
                result = self.background_dispatcher.dispatch(
                    tool_call,
                    handler,
                    observer=progress_callback,
                )
            except Exception as exc:
                logger.exception(
                    "background tool dispatch failed name=%s",
                    tool_call.name,
                )
                self._notify_error(context, exc, progress_callback)
                result = {"error": str(exc)}
            else:
                result = self._postprocess(context, result, progress_callback)
            return self._stringify_tool_result(result)

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
            result = self._postprocess(context, result, progress_callback)

        logger.debug(
            "tool execution finished name=%s",
            tool_call.name,
        )
        return self._stringify_tool_result(result)

    def _notify_before(
        self,
        context: ToolCallContext,
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> ToolHookDecision | None:
        for hook in self.hooks:
            callback = getattr(hook, "pre_tool_use", None)
            if not callable(callback):
                continue
            try:
                decision = self._normalize_decision(callback(context))
            except Exception:
                logger.exception("PreToolUse hook failed name=%s", context.name)
                return ToolHookDecision.block(
                    "PreToolUse hook failed; tool execution denied"
                )
            if decision is not None and decision.blocked:
                return decision

        for hook in self.hooks:
            try:
                callback = getattr(hook, "before_tool_call", None)
                if callable(callback):
                    callback(context)
            except Exception:
                logger.exception("tool before hook failed name=%s", context.name)
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
        return None

    @staticmethod
    def _normalize_decision(value: Any) -> ToolHookDecision | None:
        if value is None:
            return None
        if isinstance(value, ToolHookDecision):
            return value
        if isinstance(value, bool):
            return (
                ToolHookDecision.block("blocked by PreToolUse hook")
                if value
                else None
            )
        if isinstance(value, dict) and value.get("blocked"):
            return ToolHookDecision.block(
                str(value.get("reason") or "blocked by PreToolUse hook"),
                result=value.get("result"),
            )
        return None

    def _postprocess(
        self,
        context: ToolCallContext,
        result: Any,
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> Any:
        current = result
        for hook in self.hooks:
            callback = getattr(hook, "post_tool_use", None)
            if not callable(callback):
                continue
            try:
                value = callback(context, current)
                if isinstance(value, ToolPostprocessResult):
                    current = value.result
            except Exception:
                logger.exception("PostToolUse hook failed name=%s", context.name)
        output_chars = len(self._stringify_tool_result(current))
        warning_threshold = (
            self.compaction_pipeline.max_tool_result_chars
            if self.compaction_pipeline is not None
            else 30000
        )
        if output_chars > warning_threshold:
            logger.warning(
                "large tool output name=%s chars=%d threshold=%d",
                context.name,
                output_chars,
                warning_threshold,
            )
            self._notify(
                "tool_output_large",
                {
                    "call_id": context.call_id,
                    "name": context.name,
                    "output_chars": output_chars,
                    "threshold_chars": warning_threshold,
                },
                progress_callback,
            )
        self._notify_after(context, current, progress_callback)
        return current

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
                callback = getattr(hook, "after_tool_call", None)
                if callable(callback):
                    callback(context, result)
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
                callback = getattr(hook, "on_tool_error", None)
                if callable(callback):
                    callback(context, error)
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
