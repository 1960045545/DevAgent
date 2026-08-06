from __future__ import annotations

import asyncio
from datetime import datetime
import json
import urllib.error
import urllib.request
import uuid
from types import CoroutineType
from typing import Any

from core.response import ModelResponse
from core.tool_call import ToolCall
from core.message import Message
from core.invoke_options import InvokeOptions
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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.default_headers = default_headers or {}
        self.history = []

    async def ainvoke(
        self,
        messages: list["Message"],
        options: InvokeOptions | None = None,
    ) -> ModelResponse:
        if not messages:
            raise ValueError("messages 不能为空")

        options = options or InvokeOptions()

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                self._serialize_message(message)
                for message in messages
            ],
        }

        if options.tools:
            payload["tools"] = [
                self._serialize_tool(tool)
                for tool in options.tools
            ]

        if options.tool_choice is not None:
            payload["tool_choice"] = options.tool_choice

        if options.response_format is not None:
            payload["response_format"] = options.response_format

        if options.temperature is not None:
            payload["temperature"] = options.temperature

        if options.top_p is not None:
            payload["top_p"] = options.top_p

        if options.max_output_tokens is not None:
            # 部分服务商使用 max_tokens，需要按服务商调整
            payload["max_completion_tokens"] = (
                options.max_output_tokens
            )

        if options.reasoning_effort is not None:
            payload["reasoning_effort"] = (
                options.reasoning_effort
            )

        if options.extra_body:
            payload.update(options.extra_body)

        request_id = str(uuid.uuid4())

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Client-Request-Id": request_id,
        }

        headers.update(self.default_headers)

        if options.extra_headers:
            headers.update(options.extra_headers)

        raw_response = await self._post_json(
            url=f"{self.base_url}/chat/completions",
            payload=payload,
            headers=headers,
            timeout=options.timeout or self.timeout,
        )

        return self._parse_chat_response(raw_response)

    def chat(self, user_message: str) -> ModelResponse:
        msg = Message(
            role="user",
            content=user_message,
            timestamp=datetime.now().isoformat(),
        )
        response = asyncio.run(self.ainvoke([msg]))
        self.history.append({
            "role": "user",
            "message": msg,
        })
        self.history.append({
            "role": "assistant",
            "message": response,
        })
        return response


    async def _post_json(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.to_thread(
                    self._post_json_sync,
                    url,
                    payload,
                    headers,
                    timeout,
                )

            except AgentRequestError as exc:
                is_last_attempt = attempt >= self.max_retries

                if not exc.retryable or is_last_attempt:
                    raise

                delay = self.retry_backoff * (2 ** attempt)
                await asyncio.sleep(delay)

        raise RuntimeError("请求失败")

    @staticmethod
    def _post_json_sync(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                response_body = response.read()

        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            retryable = (
                exc.code == 408
                or exc.code == 429
                or exc.code >= 500
            )

            raise AgentRequestError(
                f"HTTP {exc.code}: {error_body}",
                retryable=retryable,
            ) from exc

        except urllib.error.URLError as exc:
            raise AgentRequestError(
                f"网络请求失败: {exc}",
                retryable=True,
            ) from exc

        try:
            return json.loads(
                response_body.decode("utf-8")
            )

        except json.JSONDecodeError as exc:
            raise AgentRequestError(
                "模型返回的内容不是合法 JSON",
                retryable=False,
            ) from exc

    @staticmethod
    def _serialize_message(
        message: "Message",
    ) -> dict[str, Any]:
        if hasattr(message, "to_dict"):
            return message.to_dict()

        if isinstance(message, dict):
            return message

        raise TypeError(
            f"不支持的消息类型: {type(message)}"
        )

    @staticmethod
    def _serialize_tool(tool: Any) -> dict[str, Any]:
        if hasattr(tool, "to_openai_chat_tool"):
            return tool.to_openai_chat_tool()

        if isinstance(tool, dict):
            return tool

        raise TypeError(
            f"不支持的工具类型: {type(tool)}"
        )

    @staticmethod
    def _parse_chat_response(
        data: dict[str, Any],
    ) -> "ModelResponse":
        choices = data.get("choices") or []

        if not choices:
            raise AgentRequestError(
                "模型响应中没有 choices",
                retryable=False,
            )

        choice = choices[0]
        message = choice.get("message") or {}

        tool_calls = []

        for raw_tool_call in message.get(
            "tool_calls",
            [],
        ):
            function = raw_tool_call.get(
                "function",
                {},
            )

            raw_arguments = function.get(
                "arguments",
                "{}",
            )

            try:
                arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else raw_arguments
                )

            except json.JSONDecodeError as exc:
                raise AgentRequestError(
                    "工具参数不是合法 JSON",
                    retryable=False,
                ) from exc

            tool_calls.append(
                ToolCall(
                    id=raw_tool_call.get("id"),
                    name=function.get("name", ""),
                    arguments=arguments,
                )
            )

        return ModelResponse(
            id=data.get("id"),
            model=data.get("model"),
            text=message.get("content") or "",
            tool_calls=tool_calls,
            usage=data.get("usage"),
            finish_reason=choice.get(
                "finish_reason"
            ),
            raw=data,
        )