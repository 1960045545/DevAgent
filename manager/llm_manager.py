from __future__ import annotations

import asyncio
import json
from typing import Any, Iterator

from openai import OpenAI

from core.invoke_options import InvokeOptions
from core.message import Message
from core.response import ModelResponse
from core.tool_call import ToolCall
from error.request_error import AgentRequestError


class LLMManager:
    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        model_id: str | None,
        timeout: float = 60.0,
        max_retries: int = 3,
        default_headers: dict[str, str] | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.model_id = model_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_headers = default_headers or {}
        self.client = client or self._create_client()

    async def ainvoke(
        self,
        message: Message,
        options: InvokeOptions | None = None,
        model_id: str | None = None,
    ) -> ModelResponse:
        if not message:
            raise ValueError("message can not be empty")

        options = options or InvokeOptions()
        kwargs = self.build_chat_completion_kwargs(
            message,
            options,
            stream=False,
            model_id=model_id,
        )
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            **kwargs,
        )

        return self.parse_chat_response(
            self.response_to_dict(response)
        )

    def invoke(
        self,
        message: Message,
        options: InvokeOptions | None = None,
        model_id: str | None = None,
    ) -> ModelResponse:
        return asyncio.run(
            self.ainvoke(
                message,
                options=options,
                model_id=model_id,
            )
        )

    def stream(
        self,
        message: Message,
        options: InvokeOptions | None = None,
        model_id: str | None = None,
    ) -> Iterator[str]:
        options = options or InvokeOptions()
        kwargs = self.build_chat_completion_kwargs(
            message,
            options,
            stream=True,
            model_id=model_id,
        )
        stream = self.client.chat.completions.create(**kwargs)

        for event in stream:
            chunk = self.stream_event_content(event)

            if chunk:
                yield chunk

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        options: InvokeOptions | None = None,
        *,
        stream: bool,
        model_id: str | None = None,
    ) -> Any:
        kwargs = self.build_chat_completion_kwargs_for_messages(
            messages,
            options or InvokeOptions(),
            stream=stream,
            model_id=model_id,
        )
        return self.client.chat.completions.create(**kwargs)

    def _create_client(self) -> OpenAI:
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        if self.base_url:
            kwargs["base_url"] = self.base_url

        return OpenAI(**kwargs)

    def build_chat_completion_kwargs(
        self,
        message: Message | dict[str, Any],
        options: InvokeOptions,
        *,
        stream: bool,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        return self.build_chat_completion_kwargs_for_messages(
            [self.serialize_message(message)],
            options,
            stream=stream,
            model_id=model_id,
        )

    def build_chat_completion_kwargs_for_messages(
        self,
        messages: list[dict[str, Any]],
        options: InvokeOptions,
        *,
        stream: bool,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model_id or self.model_id,
            "messages": messages,
            "stream": stream,
            "timeout": options.timeout or self.timeout,
        }

        if options.tools:
            kwargs["tools"] = [
                self.serialize_tool(tool)
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
    def assistant_message_from_response(
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

    @staticmethod
    def response_to_dict(response: Any) -> dict[str, Any]:
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
    def stream_event_content(event: Any) -> str:
        data = LLMManager.response_to_dict(event)
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
    def serialize_message(
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
    def serialize_tool(tool: Any) -> dict[str, Any]:
        if hasattr(tool, "to_openai_chat_tool"):
            return tool.to_openai_chat_tool()

        if isinstance(tool, dict):
            return tool

        raise TypeError(
            f"Unsupported tool type: {type(tool)}"
        )

    @staticmethod
    def parse_chat_response(
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
