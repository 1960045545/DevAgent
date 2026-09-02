from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterator

from openai import OpenAI

from core.invoke_options import InvokeOptions
from core.message import Message
from core.response import ModelResponse
from core.tool_call import ToolCall
from error.request_error import (
    AgentRequestError,
    PromptTooLongError,
    ProviderOverloadedError,
    RateLimitError,
)
from manager.model_provider_manager import (
    ModelProviderConfig,
    ModelProviderHealth,
    ModelPurpose,
)


logger = logging.getLogger(__name__)


@dataclass
class _RecoveryState:
    output_limit_escalated: bool = False
    continuation_count: int = 0
    context_compaction_attempted: bool = False
    reactive_compaction_attempted: bool = False


_CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly; do not apologize or recap. "
    "Pick up exactly where the previous answer stopped."
)


class LLMManager:
    INITIAL_MAX_OUTPUT_TOKENS = 8_000
    ESCALATED_MAX_OUTPUT_TOKENS = 64_000
    MAX_OUTPUT_CONTINUATIONS = 3
    MAX_RATE_LIMIT_RETRIES = 10
    MAX_OVERLOAD_RETRIES = 3
    BASE_RETRY_DELAY = 0.5
    MAX_RETRY_DELAY = 32.0
    REACTIVE_SYSTEM_MAX_CHARS = 12_000
    REACTIVE_MESSAGE_MAX_CHARS = 8_000

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
        context_compactor: Callable[[], None] | None = None,
        context_transcript_writer: Callable[..., Any] | None = None,
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
        self.context_compactor = context_compactor
        self.context_transcript_writer = context_transcript_writer
        self._context_compaction_active = False
        self.last_success_provider_name: str | None = None
        self.last_success_model_id: str | None = None
        self.clients = self._create_clients(client)
        self.client = (
            client
            or self.clients[primary_provider.name]
        )
        logger.info(
            "initialized llm manager providers=%s primary=%s",
            [provider.name for provider in self.providers],
            primary_provider.name,
        )

    def set_context_recovery_callbacks(
        self,
        *,
        context_compactor: Callable[[], None] | None = None,
        context_transcript_writer: Callable[..., Any] | None = None,
    ) -> None:
        """Attach the agent's existing memory compression and transcript hooks."""
        self.context_compactor = context_compactor
        self.context_transcript_writer = context_transcript_writer

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
        logger.debug(
            "invoke single-message purpose=%s model_override=%s",
            purpose,
            model_id,
        )
        return asyncio.run(
            self.ainvoke(
                message,
                options=options,
                model_id=model_id,
                purpose=purpose,
            )
        )

    def invoke_messages(
        self,
        messages: list[dict[str, Any]],
        options: InvokeOptions | None = None,
        model_id: str | None = None,
        purpose: ModelPurpose = "chat",
    ) -> ModelResponse:
        logger.debug(
            "invoke messages purpose=%s model_override=%s count=%d",
            purpose,
            model_id,
            len(messages),
        )
        options = options or InvokeOptions()
        response = self._create_completion_with_fallback(
            messages,
            options,
            model_id,
            purpose,
        )
        return self.parse_chat_response(
            self.response_to_dict(response)
        )

    def stream(
        self,
        message: Message,
        options: InvokeOptions | None = None,
        model_id: str | None = None,
        purpose: ModelPurpose = "chat",
    ) -> Iterator[str]:
        logger.debug(
            "stream single-message purpose=%s model_override=%s",
            purpose,
            model_id,
        )
        options = options or InvokeOptions()
        yield from self._stream_with_fallback(
            [self.serialize_message(message)],
            options,
            model_id,
            purpose,
        )

    def stream_messages(
        self,
        messages: list[dict[str, Any]],
        options: InvokeOptions | None = None,
        model_id: str | None = None,
        purpose: ModelPurpose = "chat",
    ) -> Iterator[str]:
        logger.debug(
            "stream messages purpose=%s model_override=%s count=%d",
            purpose,
            model_id,
            len(messages),
        )
        yield from self._stream_with_fallback(
            messages,
            options or InvokeOptions(),
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
        self.last_success_provider_name = None
        self.last_success_model_id = None
        last_error: BaseException | None = None

        for provider in self.providers:
            if not ModelProviderHealth.can_try(provider.name):
                logger.info(
                    "skip provider cooldown name=%s purpose=%s",
                    provider.name,
                    purpose,
                )
                continue

            attempts = self.max_retries
            model_name = provider.model_for(purpose, override_model_id=model_id)
            logger.info(
                "try provider name=%s purpose=%s stream=%s attempts=%d model=%s",
                provider.name,
                purpose,
                False,
                attempts,
                model_name,
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
                self.last_success_provider_name = provider.name
                self.last_success_model_id = provider.model_for(
                    purpose,
                    override_model_id=model_id,
                )
                logger.info(
                    "provider success name=%s purpose=%s",
                    provider.name,
                    purpose,
                )
                return response
            except ProviderOverloadedError as exc:
                last_error = exc
                ModelProviderHealth.mark_unavailable(
                    provider.name,
                    error=exc,
                    cooldown_seconds=self.cooldown_seconds,
                )
                logger.warning(
                    "provider unavailable name=%s purpose=%s cooldown=%ss error=%s",
                    provider.name,
                    purpose,
                    self.cooldown_seconds,
                    exc,
                )
                # A provider may only be replaced after the dedicated 529
                # retry budget is exhausted.  Do not use another provider for
                # prompt errors, rate limits, or unrelated failures.
                continue
            except Exception as exc:
                logger.exception(
                    "non-retryable error provider=%s purpose=%s",
                    provider.name,
                    purpose,
                )
                raise

        if last_error is not None:
            raise last_error
        raise AgentRequestError(
            "No model provider is available",
            retryable=False,
        )

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
        del attempts
        state = _RecoveryState()
        accumulated_chunks: list[str] = []
        current_options = self._options_with_initial_output_limit(options)

        while True:
            try:
                response = self._request_with_transient_retries(
                    provider,
                    messages,
                    current_options,
                    stream=stream,
                    model_id=model_id,
                    purpose=purpose,
                )
            except Exception as exc:
                if self.is_prompt_too_long_error(exc):
                    if self._recover_prompt_too_long(
                        messages,
                        state,
                        current_options,
                    ):
                        continue
                    raise PromptTooLongError(
                        "prompt remains too long after context compression",
                        retryable=False,
                        status_code=self.extract_status_code(exc),
                        error_code="prompt_too_long",
                    ) from exc

                if self.is_output_limit_error(exc):
                    if not state.output_limit_escalated:
                        state.output_limit_escalated = True
                        current_options = replace(
                            current_options,
                            max_output_tokens=self.ESCALATED_MAX_OUTPUT_TOKENS,
                        )
                        continue
                    raise AgentRequestError(
                        "model stopped mid-answer after the 64K output limit",
                        retryable=False,
                    ) from exc
                raise

            data = self.response_to_dict(response)
            if not self.response_stopped_by_length(data):
                if accumulated_chunks:
                    return self._with_accumulated_text(
                        data,
                        "".join(accumulated_chunks),
                    )
                return response

            if not state.output_limit_escalated:
                state.output_limit_escalated = True
                current_options = replace(
                    current_options,
                    max_output_tokens=self.ESCALATED_MAX_OUTPUT_TOKENS,
                )
                # The 8K response is deliberately discarded.  Retrying with
                # the original messages prevents duplicated or partial output.
                continue

            partial_text = self.response_text(data)
            accumulated_chunks.append(partial_text)
            if state.continuation_count >= self.MAX_OUTPUT_CONTINUATIONS:
                raise AgentRequestError(
                    "model output remained truncated after continuation recovery",
                    retryable=False,
                )

            self._append_continuation(messages, partial_text)
            state.continuation_count += 1

    def _request_with_transient_retries(
        self,
        provider: ModelProviderConfig,
        messages: list[dict[str, Any]],
        options: InvokeOptions,
        *,
        stream: bool,
        model_id: str | None,
        purpose: ModelPurpose,
    ) -> Any:
        last_error: BaseException | None = None
        rate_limit_attempts = 0
        overload_attempts = 0
        total_attempts = 0
        max_total_attempts = (
            self.MAX_RATE_LIMIT_RETRIES
            + self.MAX_OVERLOAD_RETRIES
        )
        while total_attempts < max_total_attempts:
            total_attempts += 1
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
                if self.is_prompt_too_long_error(exc) or self.is_output_limit_error(exc):
                    raise

                if self.is_rate_limit_error(exc):
                    last_error = exc
                    rate_limit_attempts += 1
                    overload_attempts = 0
                    ModelProviderHealth.record_failure(
                        provider.name,
                        attempt=rate_limit_attempts,
                        error=exc,
                    )
                    if rate_limit_attempts >= self.MAX_RATE_LIMIT_RETRIES:
                        raise RateLimitError(
                            "rate limit recovery exhausted after 10 retries",
                            retryable=True,
                            status_code=429,
                            retry_after=self.extract_retry_after(exc),
                        ) from exc
                    time.sleep(
                        self.retry_delay(
                            rate_limit_attempts - 1,
                            retry_after=self.extract_retry_after(exc),
                        )
                    )
                    continue

                if self.is_overloaded_error(exc):
                    last_error = exc
                    overload_attempts += 1
                    rate_limit_attempts = 0
                    ModelProviderHealth.record_failure(
                        provider.name,
                        attempt=overload_attempts,
                        error=exc,
                    )
                    if overload_attempts >= self.MAX_OVERLOAD_RETRIES:
                        raise ProviderOverloadedError(
                            "provider remained overloaded after 3 retries",
                            retryable=True,
                            status_code=529,
                            error_code="overloaded",
                        ) from exc
                    time.sleep(
                        self.retry_delay(
                            overload_attempts - 1,
                            retry_after=self.extract_retry_after(exc),
                        )
                    )
                    continue

                # 408, arbitrary 5xx, connection errors and API validation
                # failures are intentionally not folded into this policy.
                raise

        if last_error is not None:
            raise last_error
        raise AgentRequestError(
            f"Provider has no available model: {provider.name}",
            retryable=False,
        )

    def _stream_with_fallback(
        self,
        messages: list[dict[str, Any]],
        options: InvokeOptions,
        model_id: str | None,
        purpose: ModelPurpose,
    ) -> Iterator[str]:
        self.last_success_provider_name = None
        self.last_success_model_id = None
        last_error: BaseException | None = None

        for provider in self.providers:
            if not ModelProviderHealth.can_try(provider.name):
                logger.info(
                    "skip provider cooldown name=%s purpose=%s",
                    provider.name,
                    purpose,
                )
                continue

            attempts = self.max_retries
            model_name = provider.model_for(purpose, override_model_id=model_id)
            logger.info(
                "try provider name=%s purpose=%s stream=%s attempts=%d model=%s",
                provider.name,
                purpose,
                True,
                attempts,
                model_name,
            )

            try:
                chunks = self._stream_provider_with_recovery(
                    provider,
                    messages,
                    options,
                    model_id=model_id,
                    purpose=purpose,
                    attempts=attempts,
                )
                for chunk in chunks:
                    yield chunk
                return
            except ProviderOverloadedError as exc:
                last_error = exc
                ModelProviderHealth.mark_unavailable(
                    provider.name,
                    error=exc,
                    cooldown_seconds=self.cooldown_seconds,
                )
                logger.warning(
                    "provider stream unavailable name=%s purpose=%s cooldown=%ss error=%s",
                    provider.name,
                    purpose,
                    self.cooldown_seconds,
                    exc,
                )
                continue
            except Exception as exc:
                logger.exception(
                    "non-retryable stream failure name=%s purpose=%s",
                    provider.name,
                    purpose,
                )
                raise

        if last_error is not None:
            raise last_error
        raise AgentRequestError(
            "No model provider is available",
            retryable=False,
        )

    def _stream_provider_with_recovery(
        self,
        provider: ModelProviderConfig,
        messages: list[dict[str, Any]],
        options: InvokeOptions,
        *,
        model_id: str | None,
        purpose: ModelPurpose,
        attempts: int,
    ) -> Iterator[str]:
        del attempts
        state = _RecoveryState()
        accumulated_chunks: list[str] = []
        current_options = self._options_with_initial_output_limit(options)

        while True:
            try:
                stream = self._request_with_transient_retries(
                    provider,
                    messages,
                    current_options,
                    stream=True,
                    model_id=model_id,
                    purpose=purpose,
                )
                chunks, finish_reason = self._collect_stream(stream)
            except Exception as exc:
                if self.is_prompt_too_long_error(exc):
                    if self._recover_prompt_too_long(
                        messages,
                        state,
                        current_options,
                    ):
                        continue
                    raise PromptTooLongError(
                        "prompt remains too long after context compression",
                        retryable=False,
                        status_code=self.extract_status_code(exc),
                        error_code="prompt_too_long",
                    ) from exc
                if self.is_output_limit_error(exc):
                    if not state.output_limit_escalated:
                        state.output_limit_escalated = True
                        current_options = replace(
                            current_options,
                            max_output_tokens=self.ESCALATED_MAX_OUTPUT_TOKENS,
                        )
                        continue
                raise

            if not self.is_length_finish_reason(finish_reason):
                final_chunks = [*accumulated_chunks, *chunks]
                ModelProviderHealth.record_success(provider.name)
                self.last_success_provider_name = provider.name
                self.last_success_model_id = provider.model_for(
                    purpose,
                    override_model_id=model_id,
                )
                for chunk in final_chunks:
                    if chunk:
                        yield chunk
                return

            if not state.output_limit_escalated:
                state.output_limit_escalated = True
                current_options = replace(
                    current_options,
                    max_output_tokens=self.ESCALATED_MAX_OUTPUT_TOKENS,
                )
                continue

            accumulated_chunks.extend(chunks)
            if state.continuation_count >= self.MAX_OUTPUT_CONTINUATIONS:
                raise AgentRequestError(
                    "model output remained truncated after continuation recovery",
                    retryable=False,
                )
            self._append_continuation(messages, "".join(chunks))
            state.continuation_count += 1

    @staticmethod
    def _collect_stream(stream: Any) -> tuple[list[str], str | None]:
        chunks: list[str] = []
        finish_reason: str | None = None
        for event in stream:
            data = LLMManager.response_to_dict(event)
            choices = data.get("choices") or []
            if choices:
                finish_reason = choices[0].get("finish_reason") or finish_reason
            chunk = LLMManager.stream_event_content(event)
            if chunk:
                chunks.append(chunk)
        return chunks, finish_reason

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

    @classmethod
    def _is_retryable_exception(cls, exc: BaseException) -> bool:
        """Compatibility helper: only the two transient policies are retryable."""
        return cls.is_rate_limit_error(exc) or cls.is_overloaded_error(exc)

    @classmethod
    def is_rate_limit_error(cls, exc: BaseException) -> bool:
        return cls.extract_status_code(exc) == 429

    @classmethod
    def is_overloaded_error(cls, exc: BaseException) -> bool:
        return cls.extract_status_code(exc) == 529

    @classmethod
    def is_prompt_too_long_error(cls, exc: BaseException) -> bool:
        code = cls.extract_error_code(exc)
        text = str(exc).lower()
        return code in {
            "prompt_too_long",
            "context_length_exceeded",
            "context_window_exceeded",
            "input_too_long",
        } or any(
            marker in text
            for marker in (
                "prompt_too_long",
                "prompt too long",
                "context length exceeded",
                "maximum context length",
                "input is too long",
                "too many tokens",
            )
        )

    @classmethod
    def is_output_limit_error(cls, exc: BaseException) -> bool:
        code = cls.extract_error_code(exc)
        text = str(exc).lower()
        return code in {"max_tokens", "max_completion_tokens", "length"} or any(
            marker in text
            for marker in (
                "model stopped mid-answer",
                "model stopped mid answer",
                "output token limit",
                "maximum output tokens reached",
                "max completion tokens reached",
                "finish_reason: length",
            )
        )

    @staticmethod
    def extract_status_code(exc: BaseException) -> int | None:
        status_code = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        if isinstance(status_code, str) and status_code.isdigit():
            return int(status_code)
        return None

    @classmethod
    def extract_error_code(cls, exc: BaseException) -> str | None:
        for candidate in (
            getattr(exc, "code", None),
            getattr(exc, "error_code", None),
        ):
            if isinstance(candidate, str):
                return candidate.lower()

        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error") or body
            if isinstance(error, dict):
                code = error.get("code") or error.get("type")
                if isinstance(code, str):
                    return code.lower()

        return None

    @classmethod
    def extract_retry_after(cls, exc: BaseException) -> float | None:
        direct_value = getattr(exc, "retry_after", None)
        if isinstance(direct_value, (int, float)):
            return max(0.0, float(direct_value))

        headers = getattr(exc, "headers", None)
        response = getattr(exc, "response", None)
        if headers is None and response is not None:
            headers = getattr(response, "headers", None)
        if headers is None:
            return None

        value = None
        if hasattr(headers, "get"):
            value = headers.get("retry-after") or headers.get("Retry-After")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                target = parsedate_to_datetime(str(value))
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                return max(
                    0.0,
                    (target - datetime.now(timezone.utc)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                return None

    @classmethod
    def retry_delay(
        cls,
        attempt: int,
        *,
        retry_after: float | None = None,
    ) -> float:
        if retry_after is not None:
            return retry_after
        base = min(
            cls.BASE_RETRY_DELAY * (2 ** max(0, attempt)),
            cls.MAX_RETRY_DELAY,
        )
        return base + random.uniform(0.0, base * 0.25)

    def _options_with_initial_output_limit(
        self,
        options: InvokeOptions,
    ) -> InvokeOptions:
        if options.max_output_tokens is not None:
            return options
        return replace(
            options,
            max_output_tokens=self.INITIAL_MAX_OUTPUT_TOKENS,
        )

    def _recover_prompt_too_long(
        self,
        messages: list[dict[str, Any]],
        state: _RecoveryState,
        options: InvokeOptions,
    ) -> bool:
        if not state.context_compaction_attempted:
            state.context_compaction_attempted = True
            self._run_context_compactor()
            if options.context_recovery_callback is not None:
                try:
                    options.context_recovery_callback(messages)
                except Exception:
                    logger.exception("context prompt rebuild failed")
            if (
                self.context_compactor is not None
                or options.context_recovery_callback is not None
            ):
                return True
        if not state.reactive_compaction_attempted:
            state.reactive_compaction_attempted = True
            self._reactive_compact_messages(messages)
            return True
        return False

    def _run_context_compactor(self) -> None:
        if self.context_compactor is None or self._context_compaction_active:
            return
        self._context_compaction_active = True
        try:
            self.context_compactor()
        except Exception:
            logger.exception("context compaction recovery failed")
        finally:
            self._context_compaction_active = False

    def _reactive_compact_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        if self.context_transcript_writer is not None:
            try:
                self.context_transcript_writer(
                    messages,
                    reason="reactive_prompt_compact",
                )
            except Exception:
                logger.exception("failed to save reactive context transcript")

        system_message = next(
            (message for message in messages if message.get("role") == "system"),
            None,
        )
        user_messages = [
            message
            for message in messages
            if message.get("role") == "user"
        ]
        latest_user = next(
            (
                message
                for message in reversed(user_messages)
                if message.get("content") != _CONTINUATION_PROMPT
            ),
            user_messages[-1] if user_messages else None,
        )
        compacted: list[dict[str, Any]] = []
        if system_message is not None:
            compacted.append(
                self._compact_message_content(
                    system_message,
                    self.REACTIVE_SYSTEM_MAX_CHARS,
                )
            )
        if latest_user is not None:
            compacted.append(
                self._compact_message_content(
                    latest_user,
                    self.REACTIVE_MESSAGE_MAX_CHARS,
                )
            )
        if not compacted and messages:
            compacted.append(
                self._compact_message_content(
                    messages[-1],
                    self.REACTIVE_MESSAGE_MAX_CHARS,
                )
            )
        messages[:] = compacted

    @staticmethod
    def _compact_message_content(
        message: dict[str, Any],
        max_chars: int,
    ) -> dict[str, Any]:
        compacted = {
            key: value
            for key, value in message.items()
            if key in {"role", "content", "name"}
        }
        content = compacted.get("content")
        if isinstance(content, str) and len(content) > max_chars:
            head_chars = max_chars * 2 // 3
            tail_chars = max_chars - head_chars
            compacted["content"] = (
                content[:head_chars]
                + "\n...[reactively compacted]...\n"
                + content[-tail_chars:]
            )
        return compacted

    @staticmethod
    def response_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str((choices[0].get("message") or {}).get("content") or "")

    @classmethod
    def response_stopped_by_length(cls, data: dict[str, Any]) -> bool:
        choices = data.get("choices") or []
        if not choices:
            return False
        return cls.is_length_finish_reason(choices[0].get("finish_reason"))

    @staticmethod
    def is_length_finish_reason(reason: Any) -> bool:
        if not isinstance(reason, str):
            return False
        return reason.lower() in {
            "length",
            "max_tokens",
            "max_completion_tokens",
            "model_stopped_mid_answer",
        }

    @staticmethod
    def _append_continuation(
        messages: list[dict[str, Any]],
        partial_text: str,
    ) -> None:
        messages.append(
            {
                "role": "assistant",
                "content": partial_text,
            }
        )
        messages.append(
            {
                "role": "user",
                "content": _CONTINUATION_PROMPT,
            }
        )

    @classmethod
    def _with_accumulated_text(
        cls,
        data: dict[str, Any],
        accumulated_text: str,
    ) -> dict[str, Any]:
        choices = data.get("choices") or []
        if choices:
            message = choices[0].setdefault("message", {})
            message["content"] = accumulated_text + (message.get("content") or "")
        return data

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
