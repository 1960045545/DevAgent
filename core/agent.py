from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

from openai import OpenAI

from core.invoke_options import InvokeOptions
from core.message import Message
from core.response import ModelResponse
from core.tool_call import ToolCall
from core.tool_registry import ToolRegistry
from core.tool_space import ToolSpec
from core.user_profile import UserProfile
from error.request_error import AgentRequestError


class Agent:
    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        model_id: str | None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        default_headers: dict[str, str] | None = None,
        max_tokens: int = 1000,
        client: OpenAI | None = None,
        tool_registry: ToolRegistry | None = None,
        user_profile_path: str | Path | None = None,
        user_profile_max_items: int = 20,
        enable_user_profile: bool = True,
        user_profile_model_id: str | None = None,
        history_abstract_model_id: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.model_id = model_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.default_headers = default_headers or {}
        self.history: list[Message] = []
        self.max_tokens = max_tokens
        self.rounds = 3
        self.client = client or self._create_client()
        self.tool_registry = tool_registry or ToolRegistry()
        self.enable_user_profile = enable_user_profile
        self.user_profile_model_id = (
            user_profile_model_id or self.model_id
        )
        self.history_abstract_model_id = (
            history_abstract_model_id or self.model_id
        )

        default_profile_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "user_profile.json"
        )
        profile_path = (
            Path(user_profile_path)
            if user_profile_path
            else default_profile_path
        )
        self.user_profile = (
            UserProfile.load(
                profile_path,
                max_items=user_profile_max_items,
            )
            if enable_user_profile
            else UserProfile(
                max_items=user_profile_max_items,
            )
        )

    def register_tool(
        self,
        spec: ToolSpec,
        handler: Callable[..., Any],
    ) -> None:
        self.tool_registry.register(
            spec,
            handler,
        )

    async def ainvoke(
        self,
        message: Message,
        options: InvokeOptions | None = None,
        model_id: str | None = None,
    ) -> ModelResponse:
        if not message:
            raise ValueError("message can not be empty")

        options = options or InvokeOptions()
        kwargs = self._build_chat_completion_kwargs(
            message,
            options,
            stream=False,
            model_id=model_id,
        )

        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            **kwargs,
        )

        return self._parse_chat_response(
            self._response_to_dict(response)
        )

    def chat(
        self,
        user_message: str,
        options: InvokeOptions | None = None,
    ) -> ModelResponse:
        self._compose_history(rounds=self.rounds)
        prompt = self._build_chat_prompt(user_message)
        options = self._merge_tool_options(
            options or InvokeOptions()
        )

        if options.tools:
            response = self._chat_with_tools(
                prompt=prompt,
                user_message=user_message,
                options=options,
            )
            self._append_chat_history(
                user_message=user_message,
                assistant_message=response.text,
            )
            self._update_user_profile(
                user_message=user_message,
                assistant_message=response.text,
            )
            return response

        response = asyncio.run(
            self.ainvoke(
                Message(role="user", content=prompt),
                options=options,
            )
        )

        self._append_chat_history(
            user_message=user_message,
            assistant_message=response.text,
        )
        self._update_user_profile(
            user_message=user_message,
            assistant_message=response.text,
        )

        return response

    def stream_chat(
        self,
        user_message: str,
        options: InvokeOptions | None = None,
    ) -> Iterator[str]:
        self._compose_history(rounds=self.rounds)
        prompt = self._build_chat_prompt(user_message)
        options = self._merge_tool_options(
            options or InvokeOptions()
        )

        if options.tools:
            response = self._chat_with_tools(
                prompt=prompt,
                user_message=user_message,
                options=options,
            )
            self._append_chat_history(
                user_message=user_message,
                assistant_message=response.text,
            )
            self._update_user_profile(
                user_message=user_message,
                assistant_message=response.text,
            )
            yield response.text
            return

        kwargs = self._build_chat_completion_kwargs(
            Message(role="user", content=prompt),
            options,
            stream=True,
        )

        chunks: list[str] = []
        stream = self.client.chat.completions.create(**kwargs)

        for event in stream:
            chunk = self._stream_event_content(event)

            if not chunk:
                continue

            chunks.append(chunk)
            yield chunk

        self._append_chat_history(
            user_message=user_message,
            assistant_message="".join(chunks),
        )
        self._update_user_profile(
            user_message=user_message,
            assistant_message="".join(chunks),
        )

    def _merge_tool_options(
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

    @staticmethod
    def _tool_spec_name(tool: Any) -> str | None:
        if hasattr(tool, "name"):
            return tool.name

        if isinstance(tool, dict):
            function = tool.get("function") or {}
            return function.get("name") or tool.get("name")

        return None

    def _chat_with_tools(
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
            kwargs = self._build_chat_completion_kwargs_for_messages(
                messages,
                options,
                stream=False,
            )
            raw_response = self.client.chat.completions.create(
                **kwargs,
            )
            data = self._response_to_dict(raw_response)
            response = self._parse_chat_response(data)

            if not response.tool_calls:
                return response

            messages.append(
                self._assistant_message_from_response(data)
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

    def _create_client(self) -> OpenAI:
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        if self.base_url:
            kwargs["base_url"] = self.base_url

        return OpenAI(**kwargs)

    def _build_chat_completion_kwargs(
        self,
        message: Message,
        options: InvokeOptions,
        *,
        stream: bool,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model_id or self.model_id,
            "messages": [
                self._serialize_message(message),
            ],
            "stream": stream,
            "timeout": options.timeout or self.timeout,
        }

        if options.tools:
            kwargs["tools"] = [
                self._serialize_tool(tool)
                for tool in options.tools
            ]

        if options.tool_choice is not None:
            kwargs["tool_choice"] = options.tool_choice

        if options.response_format is not None:
            kwargs["response_format"] = options.response_format

        if options.temperature is not None:
            kwargs["temperature"] = options.temperature

        if options.top_p is not None:
            kwargs["top_p"] = options.top_p

        if options.max_output_tokens is not None:
            kwargs["max_completion_tokens"] = (
                options.max_output_tokens
            )

        if options.reasoning_effort is not None:
            kwargs["reasoning_effort"] = (
                options.reasoning_effort
            )

        if options.extra_body:
            kwargs["extra_body"] = options.extra_body

        extra_headers = dict(self.default_headers)

        if options.extra_headers:
            extra_headers.update(options.extra_headers)

        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        return kwargs

    def _build_chat_completion_kwargs_for_messages(
        self,
        messages: list[dict[str, Any]],
        options: InvokeOptions,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "stream": stream,
            "timeout": options.timeout or self.timeout,
        }

        if options.tools:
            kwargs["tools"] = [
                self._serialize_tool(tool)
                for tool in options.tools
            ]

        if options.tool_choice is not None:
            kwargs["tool_choice"] = options.tool_choice

        if options.response_format is not None:
            kwargs["response_format"] = options.response_format

        if options.temperature is not None:
            kwargs["temperature"] = options.temperature

        if options.top_p is not None:
            kwargs["top_p"] = options.top_p

        if options.max_output_tokens is not None:
            kwargs["max_completion_tokens"] = (
                options.max_output_tokens
            )

        if options.reasoning_effort is not None:
            kwargs["reasoning_effort"] = (
                options.reasoning_effort
            )

        if options.extra_body:
            kwargs["extra_body"] = options.extra_body

        extra_headers = dict(self.default_headers)

        if options.extra_headers:
            extra_headers.update(options.extra_headers)

        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        return kwargs

    @staticmethod
    def _assistant_message_from_response(
        data: dict[str, Any],
    ) -> dict[str, Any]:
        choices = data.get("choices") or []

        if not choices:
            raise AgentRequestError(
                "Model response has no choices",
                retryable=False,
            )

        raw_message = choices[0].get("message") or {}
        message: dict[str, Any] = {
            "role": "assistant",
            "content": raw_message.get("content"),
        }

        if raw_message.get("tool_calls"):
            message["tool_calls"] = raw_message["tool_calls"]

        if raw_message.get("function_call"):
            message["function_call"] = raw_message["function_call"]

        return message

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

    @staticmethod
    def _response_to_dict(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump()

        if hasattr(response, "dict"):
            return response.dict()

        if isinstance(response, dict):
            return response

        raise AgentRequestError(
            f"Unsupported model response type: {type(response)}",
            retryable=False,
        )

    @staticmethod
    def _stream_event_content(event: Any) -> str:
        data = Agent._response_to_dict(event)
        choices = data.get("choices") or []

        if not choices:
            return ""

        choice = choices[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")

        if content is None:
            content = choice.get("text")

        return content or ""

    @staticmethod
    def _serialize_message(
        message: Message | dict[str, Any],
    ) -> dict[str, Any]:
        if hasattr(message, "to_dict"):
            return message.to_dict()

        if isinstance(message, dict):
            return message

        raise TypeError(
            f"Unsupported message type: {type(message)}"
        )

    @staticmethod
    def _serialize_tool(tool: Any) -> dict[str, Any]:
        if hasattr(tool, "to_openai_chat_tool"):
            return tool.to_openai_chat_tool()

        if isinstance(tool, dict):
            return tool

        raise TypeError(
            f"Unsupported tool type: {type(tool)}"
        )

    @staticmethod
    def _parse_chat_response(
        data: dict[str, Any],
    ) -> ModelResponse:
        choices = data.get("choices") or []

        if not choices:
            raise AgentRequestError(
                "Model response has no choices",
                retryable=False,
            )

        choice = choices[0]
        message = choice.get("message") or {}
        tool_calls = []

        for raw_tool_call in message.get("tool_calls") or []:
            function = raw_tool_call.get("function", {})
            raw_arguments = function.get("arguments", "{}")

            try:
                arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else raw_arguments
                )
            except json.JSONDecodeError as exc:
                raise AgentRequestError(
                    "Tool arguments are not valid JSON",
                    retryable=False,
                ) from exc

            tool_calls.append(
                ToolCall(
                    id=raw_tool_call.get("id"),
                    name=function.get("name", ""),
                    arguments=arguments,
                    raw=raw_tool_call,
                )
            )

        return ModelResponse(
            id=data.get("id"),
            model=data.get("model"),
            text=message.get("content") or "",
            tool_calls=tool_calls,
            usage=data.get("usage"),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    def _append_chat_history(
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

    def _compose_history(self, rounds: int) -> None:
        total_tokens = sum(
            self._estimate_tokens(msg.content)
            for msg in self.history
        )
        keep_messages = rounds * 2

        if (
            total_tokens <= 0.8 * self.max_tokens
            or len(self.history) <= keep_messages
        ):
            return

        old_messages = self.history[:-keep_messages]
        recent_messages = self.history[-keep_messages:]

        if not old_messages:
            return

        prompt = self._build_prompt(
            "compress_history_prompt.md",
            old_messages,
        )
        response = asyncio.run(
            self.ainvoke(
                Message(role="user", content=prompt),
                model_id=self.history_abstract_model_id,
            )
        )
        summary = Message(
            role="system",
            content=f"以下是之前对话的摘要：\n{response.text}",
        )
        self.history = [summary] + recent_messages

    @staticmethod
    def _estimate_tokens(text: str | None) -> int:
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

    def _load_prompt(self, file_name: str) -> str:
        prompt_path = (
            Path(__file__).resolve().parent.parent
            / "prompts"
            / file_name
        )
        return prompt_path.read_text(encoding="utf-8")

    def _build_chat_prompt(self, user_message: str) -> str:
        template = self._load_prompt("chat_prompt.md")
        keep_messages = self.rounds * 2
        history_messages = self.history[:-keep_messages]
        recent_messages = self.history[-keep_messages:]

        history_text = self._format_messages(history_messages)
        recent_text = self._format_messages(recent_messages)

        return (
            template
            .replace("{history}", history_text)
            .replace("{input}", user_message)
            .replace("{recent_chat_record}", recent_text)
            .replace(
                "{user_profile}",
                self.user_profile.format(),
            )
        )

    def _update_user_profile(
        self,
        *,
        user_message: str,
        assistant_message: str,
    ) -> None:
        if not self.enable_user_profile:
            return

        template = self._load_prompt(
            "user_profile_update_prompt.md"
        )
        prompt = (
            template
            .replace(
                "{user_profile}",
                self.user_profile.format(),
            )
            .replace("{user_message}", user_message)
            .replace("{assistant_message}", assistant_message)
        )

        try:
            response = asyncio.run(
                self.ainvoke(
                    Message(role="user", content=prompt),
                    options=InvokeOptions(
                        temperature=0,
                        max_output_tokens=400,
                    ),
                    model_id=self.user_profile_model_id,
                )
            )
            update = self._parse_json_object(response.text)
            additions = self._as_string_list(
                update.get("add")
                or update.get("items")
                or update.get("profile")
            )
            removals = self._as_string_list(
                update.get("remove")
            )

            self.user_profile.merge(
                items=additions,
                remove=removals,
            )
            self.user_profile.save()
        except Exception:
            # 画像抽取是辅助流程，失败时不能影响正常聊天。
            return

    @staticmethod
    def _as_string_list(value: Any) -> list[str]:
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
    def _parse_json_object(text: str) -> dict[str, Any]:
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

    def _build_prompt(
        self,
        file_name: str,
        messages: list[Message],
    ) -> str:
        template = self._load_prompt(file_name)
        return template.replace(
            "{history}",
            self._format_messages(messages),
        )

    @staticmethod
    def _format_messages(messages: list[Message]) -> str:
        lines = [
            f"{msg.role}: {msg.content}"
            for msg in messages
            if msg.content
        ]
        return "\n".join(lines)
