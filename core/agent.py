from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI

from core.invoke_options import InvokeOptions
from core.message import Message
from core.response import ModelResponse
from core.tool_call import ToolCall
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

    async def ainvoke(
        self,
        message: Message,
        options: InvokeOptions | None = None,
    ) -> ModelResponse:
        if not message:
            raise ValueError("message can not be empty")

        options = options or InvokeOptions()
        kwargs = self._build_chat_completion_kwargs(
            message,
            options,
            stream=False,
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

        return response

    def stream_chat(
        self,
        user_message: str,
        options: InvokeOptions | None = None,
    ) -> Iterator[str]:
        self._compose_history(rounds=self.rounds)
        prompt = self._build_chat_prompt(user_message)
        options = options or InvokeOptions()

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
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
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

        for raw_tool_call in message.get("tool_calls", []):
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
        )

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
