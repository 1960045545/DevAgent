from __future__ import annotations

import logging
import time
from datetime import datetime
from threading import RLock
from typing import Any, Callable, Iterator

from core.agent import Agent

from .event import ChatEvent, EventType
from .policy import RuntimePolicy
from .result import ChatResult
from .task_state import TaskState, TaskStatus


logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        agent: Agent,
        policy: RuntimePolicy | None = None,
        event_callback: Callable[[ChatEvent], None] | None = None,
    ) -> None:
        self.agent = agent
        self.policy = policy or RuntimePolicy()
        self.event_callback = event_callback
        self.state: TaskState | None = None
        self._state_lock = RLock()

    def chat(self, user_message: str) -> ChatResult:
        if self.policy.stream:
            return self.stream_chat_result(user_message)

        return self.response_chat(user_message)

    def response_chat(self, user_message: str) -> ChatResult:
        state = self._start_state(user_message)

        try:
            response = self.agent.chat(
                user_message,
                progress_callback=self._on_agent_progress,
            )
            assistant_text = response.text or ""
            state.assistant_output = assistant_text
            state.model_id = (
                response.model
                or self.agent.llm_manager.last_success_model_id
                or self.agent.model_id
            )
            state.provider_name = (
                self.agent.llm_manager.last_success_provider_name
            )
            state.status = TaskStatus.COMPLETED
            state.end_time = datetime.now()

            self._emit_event(
                state,
                self._build_event(
                    state,
                    EventType.COMPLETED,
                    data={"text": assistant_text},
                )
            )

            return ChatResult(
                task_id=state.task_id,
                text=assistant_text,
                success=True,
                events=state.events if self.policy.collect_events else [],
                model=state.model_id,
                provider_name=state.provider_name,
                token_usage=self._extract_token_usage(response.usage),
            )
        except Exception as exc:
            state.status = TaskStatus.FAILED
            state.error_msg = str(exc)
            state.end_time = datetime.now()

            self._emit_event(
                state,
                self._build_event(
                    state,
                    EventType.FAILED,
                    data={"text": state.assistant_output},
                    error=str(exc),
                )
            )

            logger.exception("response chat failed")

            if self.policy.raise_on_error:
                raise

            return ChatResult(
                task_id=state.task_id,
                text="模型服务暂时不可用，请稍后再试。",
                success=False,
                events=state.events if self.policy.collect_events else [],
                model=state.model_id,
                provider_name=state.provider_name,
                token_usage=None,
                error=str(exc),
            )

    def stream_chat(self, user_message: str) -> Iterator[str]:
        state = self._start_state(user_message)
        buffer: list[str] = []
        buffer_length = 0
        last_flush = time.monotonic()

        try:
            for chunk in self.agent.stream_chat(
                user_message,
                progress_callback=self._on_agent_progress,
            ):
                if not chunk:
                    continue

                if state.model_id is None:
                    state.model_id = (
                        self.agent.llm_manager.last_success_model_id
                        or self.agent.model_id
                    )
                if state.provider_name is None:
                    state.provider_name = (
                        self.agent.llm_manager.last_success_provider_name
                    )

                state.assistant_output += chunk
                self._emit_event(
                    state,
                    self._build_event(
                        state,
                        EventType.DELTA,
                        data={"text": chunk},
                    )
                )

                buffer.append(chunk)
                buffer_length += len(chunk)

                if self._should_flush_batch(
                    buffer_length=buffer_length,
                    last_flush=last_flush,
                    latest_chunk=chunk,
                ):
                    batch = "".join(buffer)
                    yield batch
                    buffer.clear()
                    buffer_length = 0
                    last_flush = time.monotonic()

            if buffer:
                yield "".join(buffer)

            state.status = TaskStatus.COMPLETED
            state.end_time = datetime.now()

            self._emit_event(
                state,
                self._build_event(
                    state,
                    EventType.COMPLETED,
                    data={"text": state.assistant_output},
                )
            )
        except Exception as exc:
            if buffer:
                yield "".join(buffer)

            state.status = TaskStatus.FAILED
            state.error_msg = str(exc)
            state.end_time = datetime.now()

            self._emit_event(
                state,
                self._build_event(
                    state,
                    EventType.FAILED,
                    data={"text": state.assistant_output},
                    error=str(exc),
                )
            )

            logger.exception("stream chat failed")

            if self.policy.raise_on_error:
                raise
        finally:
            if state.end_time is None:
                state.end_time = datetime.now()
            self.state = state

    def stream_chat_result(self, user_message: str) -> ChatResult:
        for _ in self.stream_chat(user_message):
            pass

        state = self.state or TaskState(user_input=user_message)
        success = (
            state.status == TaskStatus.COMPLETED
            and state.error_msg is None
        )

        return ChatResult(
            task_id=state.task_id,
            text=state.assistant_output,
            success=success,
            events=state.events if self.policy.collect_events else [],
            model=state.model_id,
            provider_name=state.provider_name,
            token_usage=None,
            error=state.error_msg,
        )

    def _start_state(self, user_message: str) -> TaskState:
        state = TaskState(user_input=user_message)
        state.status = TaskStatus.RUNNING
        state.start_time = datetime.now()
        self.state = state
        self._emit_event(
            state,
            self._build_event(state, EventType.STARTED),
        )
        return state

    def _on_agent_progress(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        state = self.state
        if state is None:
            return

        event_mapping = {
            "tool_started": EventType.TOOL_STARTED,
            "tool_finished": EventType.TOOL_FINISHED,
            "tool_failed": EventType.TOOL_FAILED,
            "todo_created": EventType.TODO_CREATED,
            "todo_updated": EventType.TODO_UPDATED,
            "subtask_started": EventType.SUBTASK_STARTED,
            "subtask_finished": EventType.SUBTASK_FINISHED,
            "subtask_failed": EventType.SUBTASK_FAILED,
            "subtask_deferred": EventType.SUBTASK_DEFERRED,
        }
        if event_type.startswith("todo_"):
            with self._state_lock:
                state.todos = [
                    dict(item)
                    for item in data.get("todos", [])
                    if isinstance(item, dict)
                ]

        mapped_type = event_mapping.get(event_type)
        if mapped_type is None:
            return
        self._emit_event(
            state,
            self._build_event(state, mapped_type, data=data),
        )

    def _emit_event(self, state: TaskState, event: ChatEvent) -> None:
        with self._state_lock:
            event.seq = len(state.events)
            if self.policy.collect_events:
                state.events.append(event)
        if self.event_callback is not None:
            self.event_callback(event)

    def _build_event(
        self,
        state: TaskState,
        event_type: EventType,
        *,
        data: dict[str, object] | None = None,
        error: str | None = None,
    ) -> ChatEvent:
        return ChatEvent(
            task_id=state.task_id,
            event_type=event_type,
            data=data or {},
            error=error,
        )

    def _should_flush_batch(
        self,
        *,
        buffer_length: int,
        last_flush: float,
        latest_chunk: str,
    ) -> bool:
        if buffer_length >= self.policy.stream_batch_chars:
            return True

        if time.monotonic() - last_flush >= self.policy.stream_batch_seconds:
            return True

        return latest_chunk.endswith(
            ("。", "！", "？", "!", "?", ".", "\n", "；", ";")
        )

    @staticmethod
    def _extract_token_usage(usage: object | None) -> int | None:
        if usage is None:
            return None

        token_usage = getattr(usage, "total_tokens", None)
        if isinstance(token_usage, int):
            return token_usage

        if isinstance(usage, dict):
            value = usage.get("total_tokens")
            if isinstance(value, int):
                return value

        return None
