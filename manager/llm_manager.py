from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Iterator

from openai import OpenAI

from core.invoke_options import InvokeOptions
from core.message import Message
from core.response import ModelResponse
from core.tool_call import ToolCall
from error.request_error import AgentRequestError
from manager.model_provider_manager import (
    ModelProviderConfig,
    ModelProviderHealth,
    ModelPurpose,
)


class LLMManager:
    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        model_id: str | None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_interval: float = 5.0,
        cooldown_seconds: float = 30.0,
        default_headers: dict[str, str] | None = None,
        client: OpenAI | None = None,
        providers: list[ModelProviderConfig] | None = None,
    ) -> None:
        self.providers = providers or [
            ModelProviderConfig(
                name="default",
                base_url=base_url,
                api_key=api_key,
                chat_model_id=model_id,
                history_model_id=model_id,
                profile_model_id=model_id,
            )
        ]
        primary_provider = self.providers[0]
        self.base_url = (
            primary_provider.base_url.rstrip("/")
            if primary_provider.base_url
            else None
        )
        self.api_key = primary_provider.api_key
        self.model_id = primary_provider.chat_model_id
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.retry_interval = retry_interval
        self.cooldown_seconds = cooldown_seconds
        self.default_headers = default_headers or {}
        self.clients = self._create_clients(client)
        self.client = (
            client
            or self.clients[primary_provider.name]
        )

    async def ainvoke(
        self,
        message: Message,
        options: InvokeOptions | None = None,
        model_id: str | None = None,
        purpose: ModelPurpose = "chat",
    ) -> ModelResponse:
        if not message:
            raise ValueError("message can not be empty")

        options = options or InvokeOptions()
        response = await asyncio.to_thread(
            self._create_completion_with_fallback,
            [self.serialize_message(message)],
            options,
            model_id,
            purpose,
        )

        return self.parse_chat_response(
            self.response_to_dict(response)
        )

    def invoke(
        self,
        message: Message,
        options: InvokeOptions | None = None,
        model_id: str | None = None,
        purpose: ModelPurpose = "chat",
    ) -> ModelResponse:
        return asyncio.run(
            self.ainvoke(
                message,
                options=options,
                model_id=model_id,
                purpose=purpose,
            )
        )

    def stream(
        self,
        message: Message,
        options: InvokeOptions | None = None,
        model_id: str | None = None,
        purpose: ModelPurpose = "chat",
    ) -> Iterator[str]:
        options = options or InvokeOptions()
        yield from self._stream_with_fallback(
            [self.serialize_message(message)],
            options,
            model_id,
            purpose,
        )

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        options: InvokeOptions | None = None,
        *,
        stream: bool,
        model_id: str | None = None,
        purpose: ModelPurpose = "chat",
    ) -> Any:
        if stream:
            return self._stream_with_fallback(
                messages,
                options or InvokeOptions(),
                model_id,
                purpose,
            )

        return self._create_completion_with_fallback(
            messages,
            options or InvokeOptions(),
            model_id,
            purpose,
        )

    def _create_clients(
        self,
        client: OpenAI | None,
    ) -> dict[str, OpenAI]:
        if client is not None:
            return {
                provider.name: client
                for provider in self.providers
            }

        return {
            provider.name: self._create_client(provider)
            for provider in self.providers
        }

    def _create_client(
        self,
        provider: ModelProviderConfig,
    ) -> OpenAI:
        kwargs: dict[str, Any] = {
            "api_key": provider.api_key or "ollama",
            "timeout": self.timeout,
            "max_retries": 0,
        }

        if provider.base_url:
            kwargs["base_url"] = provider.base_url.rstrip("/")

        return OpenAI(**kwargs)

    def _create_completion_with_fallback(
        self,
        messages: list[dict[str, Any]],
        options: InvokeOptions,
        model_id: str | None,
        purpose: ModelPurpose,
    ) -> Any:
        last_error: BaseException | None = None

        for provider in self.providers:
            if not ModelProviderHealth.can_try(provider.name):
                continue

            attempts = (
                1
                if ModelProviderHealth.needs_probe(provider.name)
                else self.max_retries
            )

            try:
                response = self._create_completion_for_provider(
                    provider,
                    messages,
                    options,
                    stream=False,
                    model_id=model_id,
                    purpose=purpose,
                    attempts=attempts,
                )
                ModelProviderHealth.record_success(provider.name)
                return response
            except Exception as exc:
                if not self._is_retryable_exception(exc):
                    raise

                last_error = exc
                ModelProviderHealth.mark_unavailable(
                    provider.name,
                    error=exc,
                    cooldown_seconds=self.cooldown_seconds,
                )

        raise AgentRequestError(
            "All model providers are unavailable",
            retryable=True,
        ) from last_error

    def _create_completion_for_provider(
        self,
        provider: ModelProviderConfig,
        messages: list[dict[str, Any]],
        options: InvokeOptions,
        *,
        stream: bool,
        model_id: str | None,
        purpose: ModelPurpose,
        attempts: int,
    ) -> Any:
        last_error: BaseException | None = None

        for attempt in range(1, attempts + 1):
            try:
                kwargs = self.build_chat_completion_kwargs_for_messages(
                    messages,
                    options,
                    stream=stream,
                    model_id=provider.model_for(
                        purpose,
                        override_model_id=model_id,
                    ),
                )
                return self.clients[
                    provider.name
                ].chat.completions.create(**kwargs)
            except Exception as exc:
                if not self._is_retryable_exception(exc):
                    raise

                last_error = exc
                ModelProviderHealth.record_failure(
                    provider.name,
                    attempt=attempt,
                    error=exc,
                )

                if attempt < attempts:
                    time.sleep(self.retry_interval)

        if last_error is None:
            raise AgentRequestError(
                f"Provider has no available model: {provider.name}",
                retryable=True,
            )

        raise last_error

    def _stream_with_fallback(
        self,
        messages: list[dict[str, Any]],
        options: InvokeOptions,
        model_id: str | None,
        purpose: ModelPurpose,
    ) -> Iterator[str]:
        last_error: BaseException | None = None

        for provider in self.providers:
            if not ModelProviderHealth.can_try(provider.name):
                continue

            attempts = (
                1
                if ModelProviderHealth.needs_probe(provider.name)
                else self.max_retries
            )

            try:
                first_chunk, stream = self._open_stream_for_provider(
                    provider,
                    messages,
                    options,
                    model_id=model_id,
                    purpose=purpose,
                    attempts=attempts,
                )
            except Exception as exc:
                if not self._is_retryable_exception(exc):
                    raise

                last_error = exc
                ModelProviderHealth.mark_unavailable(
                    provider.name,
                    error=exc,
                    cooldown_seconds=self.cooldown_seconds,
                )
                continue

            ModelProviderHealth.record_success(provider.name)
            yield first_chunk

            for event in stream:
                chunk = self.stream_event_content(event)

                if chunk:
                    yield chunk

            return

        raise AgentRequestError(
            "All model providers are unavailable",
            retryable=True,
        ) from last_error

    def _open_stream_for_provider(
        self,
        provider: ModelProviderConfig,
        messages: list[dict[str, Any]],
        options: InvokeOptions,
        *,
        model_id: str | None,
        purpose: ModelPurpose,
        attempts: int,
    ) -> tuple[str, Any]:
        stream = self._create_completion_for_provider(
            provider,
            messages,
            options,
            stream=True,
            model_id=model_id,
            purpose=purpose,
            attempts=attempts,
        )

        for event in stream:
            chunk = self.stream_event_content(event)

            if chunk:
                return chunk, stream

        raise AgentRequestError(
            f"Provider stream returned no content: {provider.name}",
            retryable=True,
        )

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
    def _is_retryable_exception(exc: BaseException) -> bool:
        retryable = getattr(exc, "retryable", None)

        if retryable is not None:
            return bool(retryable)

        status_code = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)

        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)

        if status_code is None:
            return True

        return (
            status_code == 408
            or status_code == 429
            or status_code >= 500
        )

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
