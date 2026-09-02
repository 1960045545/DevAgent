from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from core.message import Message

if TYPE_CHECKING:
    from manager.llm_manager import LLMManager
    from manager.prompt_manager import PromptManager


logger = logging.getLogger(__name__)
_SUMMARY_METADATA_KEY = "context_summary"


class MemoryManager:
    """Stores chat context and applies bounded, traceable compression."""

    def __init__(
        self,
        *,
        max_tokens: int = 1000,
        rounds: int = 3,
        prompt_manager: PromptManager | None = None,
        llm_manager: LLMManager | None = None,
        history_abstract_model_id: str | None = None,
        conversation_head_rounds: int = 3,
        conversation_tail_rounds: int = 47,
        operation_keep_count: int = 3,
        transcripts_dir: str | Path | None = None,
        static_context: str = "",
    ) -> None:
        self.history: list[Message] = []
        self.operation_history: list[dict[str, Any]] = []
        self.max_tokens = max_tokens
        # Kept for compatibility with the existing Agent.rounds API.
        self.rounds = rounds
        self.conversation_head_rounds = max(0, conversation_head_rounds)
        self.conversation_tail_rounds = max(0, conversation_tail_rounds)
        self.operation_keep_count = max(0, operation_keep_count)
        self.prompt_manager = prompt_manager
        self.llm_manager = llm_manager
        self.history_abstract_model_id = history_abstract_model_id
        self.transcripts_dir = (
            Path(transcripts_dir).expanduser().resolve()
            if transcripts_dir is not None
            else (Path.cwd() / ".transcripts").resolve()
        )
        self.static_context = static_context
        self.last_transcript_path: Path | None = None
        self._lock = RLock()

    def append_chat(
        self,
        *,
        user_message: str,
        assistant_message: str,
    ) -> None:
        logger.debug("append chat history")
        with self._lock:
            self.history.append(
                Message(role="user", content=user_message)
            )
            self.history.append(
                Message(role="assistant", content=assistant_message)
            )

    def append_tool_result(
        self,
        tool_call: Any,
        result: Any,
        round_index: int = 0,
    ) -> None:
        """Record a host tool result without retaining it in the prompt forever."""
        record = {
            "kind": "tool",
            "call_id": getattr(tool_call, "id", None),
            "name": getattr(tool_call, "name", "unknown"),
            "arguments": self._json_safe(
                getattr(tool_call, "arguments", {})
            ),
            "result": self._json_safe(result),
            "round_index": round_index,
        }
        with self._lock:
            self.operation_history.append(record)

    def append_skill_result(
        self,
        skill_name: str,
        result: Any,
        *,
        arguments: Any = None,
        call_id: str | None = None,
        round_index: int = 0,
    ) -> None:
        """Record a skill execution using the same three-result policy."""
        record = {
            "kind": "skill",
            "call_id": call_id,
            "name": skill_name,
            "arguments": self._json_safe(arguments),
            "result": self._json_safe(result),
            "round_index": round_index,
        }
        with self._lock:
            self.operation_history.append(record)

    def compress_if_needed(self) -> None:
        with self._lock:
            full_history = list(self.history)
            full_operations = [
                dict(operation)
                for operation in self.operation_history
            ]
            retained_history = self._conversation_window(full_history)
            retained_operations = (
                full_operations[-self.operation_keep_count:]
                if self.operation_keep_count
                else []
            )
            retained_context = self._format_context(
                retained_history,
                retained_operations,
            )
            changed_by_window = (
                len(retained_history) != len(full_history)
                or len(retained_operations) != len(full_operations)
            )
            retained_tokens = self.estimate_tokens(retained_context)
            static_tokens = self.estimate_tokens(self.static_context)
            context_tokens = retained_tokens + static_tokens
            threshold = self._compression_threshold()

        if not full_history and not full_operations:
            return

        if not changed_by_window and context_tokens <= threshold:
            logger.debug(
                "skip context compression context_tokens=%d threshold=%d",
                context_tokens,
                threshold,
            )
            return

        transcript_path = self._write_transcript(
            full_history,
            full_operations,
            reason=(
                "retention_window"
                if context_tokens <= threshold
                else "context_over_threshold"
            ),
        )
        if transcript_path is None:
            logger.error(
                "context compression aborted because transcript could not be saved"
            )
            return

        if context_tokens <= threshold:
            with self._lock:
                self.history = retained_history
                self.operation_history = retained_operations
            logger.info(
                "context window applied history_messages=%d operations=%d transcript=%s",
                len(retained_history),
                len(retained_operations),
                transcript_path,
            )
            return

        if self.prompt_manager is None or self.llm_manager is None:
            logger.warning(
                "context exceeds threshold but summarizer dependencies are missing; "
                "retaining bounded window history_messages=%d operations=%d",
                len(retained_history),
                len(retained_operations),
            )
            with self._lock:
                self.history = retained_history
                self.operation_history = retained_operations
            return

        logger.info(
            "context compression start context_tokens=%d threshold=%d "
            "history_messages=%d operations=%d transcript=%s",
            context_tokens,
            threshold,
            len(retained_history),
            len(retained_operations),
            transcript_path,
        )
        prompt = self.prompt_manager.build_compress_history_prompt(
            history=retained_context,
        )

        try:
            response = self.llm_manager.invoke(
                Message(role="user", content=prompt),
                model_id=self.history_abstract_model_id,
                purpose="history",
            )
            summary_text = (response.text or "").strip()
        except Exception:
            logger.exception("context compression failed")
            with self._lock:
                self.history = retained_history
                self.operation_history = retained_operations
            return

        if not summary_text:
            logger.warning("context compression returned an empty summary")
            with self._lock:
                self.history = retained_history
                self.operation_history = retained_operations
            return

        summary = Message(
            role="system",
            content=(
                "以下是之前对话、技能和工具调用的摘要；"
                "原始完整上下文已保存到 transcript："
                f"{transcript_path}\n"
                f"{summary_text}"
            ),
            metadata={
                _SUMMARY_METADATA_KEY: True,
                "transcript_path": str(transcript_path),
            },
        )
        summary_history = [summary]
        summary_operations = retained_operations
        summary_tokens = self.estimate_tokens(
            self._format_context(summary_history, summary_operations)
        ) + static_tokens
        if summary_tokens > threshold:
            # Recent operation output can itself be very large. The summary
            # already includes it, so remove the duplicated raw results.
            summary_operations = []

        with self._lock:
            self.history = summary_history
            self.operation_history = summary_operations
        logger.info(
            "context compression finished summary_chars=%d transcript=%s",
            len(summary_text),
            transcript_path,
        )

    def get_prompt_texts(self) -> tuple[str, str]:
        with self._lock:
            messages = list(self.history)

        pinned_messages, conversation_messages = (
            self._split_pinned_messages(messages)
        )
        head_messages = self.conversation_head_rounds * 2
        pinned_text = self.format_messages(pinned_messages)
        if len(conversation_messages) <= head_messages:
            return pinned_text, self.format_messages(conversation_messages)

        history_text = self.format_messages(
            conversation_messages[:head_messages]
        )
        if pinned_text:
            history_text = "\n".join(
                part
                for part in (pinned_text, history_text)
                if part
            )

        return (
            history_text,
            self.format_messages(conversation_messages[head_messages:]),
        )

    def get_operation_prompt_text(self) -> str:
        with self._lock:
            operations = list(
                self.operation_history[-self.operation_keep_count:]
                if self.operation_keep_count
                else []
            )
        return self.format_operations(operations)

    def _conversation_window(self, messages: list[Message]) -> list[Message]:
        pinned_messages, conversation_messages = (
            self._split_pinned_messages(messages)
        )
        head_count = self.conversation_head_rounds * 2
        tail_count = self.conversation_tail_rounds * 2
        if len(conversation_messages) <= head_count + tail_count:
            return [*pinned_messages, *conversation_messages]
        if head_count == 0:
            window = conversation_messages[-tail_count:] if tail_count else []
            return [*pinned_messages, *window]
        if tail_count == 0:
            return [*pinned_messages, *conversation_messages[:head_count]]
        return [
            *pinned_messages,
            *conversation_messages[:head_count],
            *conversation_messages[-tail_count:],
        ]

    @staticmethod
    def _split_pinned_messages(
        messages: list[Message],
    ) -> tuple[list[Message], list[Message]]:
        pinned: list[Message] = []
        conversation: list[Message] = []
        for message in messages:
            metadata = getattr(message, "metadata", None) or {}
            if metadata.get(_SUMMARY_METADATA_KEY) is True:
                pinned.append(message)
            else:
                conversation.append(message)
        return pinned, conversation

    def _compression_threshold(self) -> int:
        return max(1, int(max(1, self.max_tokens) * 0.8))

    def _write_transcript(
        self,
        messages: list[Message],
        operations: list[dict[str, Any]],
        *,
        reason: str,
    ) -> Path | None:
        payload = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "messages": [
                self._message_to_dict(message)
                for message in messages
            ],
            "operations": operations,
            "static_context": self.static_context,
            "parent_transcript": (
                str(self.last_transcript_path)
                if self.last_transcript_path is not None
                else None
            ),
        }
        try:
            self.transcripts_dir.mkdir(parents=True, exist_ok=True)
            path = self.transcripts_dir / (
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
                f"-{uuid4().hex}.json"
            )
            path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError):
            logger.exception(
                "failed to persist context transcript path=%s",
                self.transcripts_dir,
            )
            return None

        self.last_transcript_path = path
        return path

    def write_runtime_transcript(
        self,
        messages: list[dict[str, Any]],
        *,
        reason: str,
    ) -> Path | None:
        """Persist an in-flight request before reactive prompt compaction."""
        with self._lock:
            history = list(self.history)
            operations = [dict(operation) for operation in self.operation_history]
        payload = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "messages": [self._message_to_dict(message) for message in history],
            "operations": operations,
            "request_messages": self._json_safe(messages),
            "static_context": self.static_context,
            "parent_transcript": (
                str(self.last_transcript_path)
                if self.last_transcript_path is not None
                else None
            ),
        }
        try:
            self.transcripts_dir.mkdir(parents=True, exist_ok=True)
            path = self.transcripts_dir / (
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
                f"-{uuid4().hex}.json"
            )
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError):
            logger.exception(
                "failed to persist runtime transcript path=%s",
                self.transcripts_dir,
            )
            return None
        self.last_transcript_path = path
        return path

    def _format_context(
        self,
        messages: list[Message],
        operations: list[dict[str, Any]],
    ) -> str:
        message_text = self.format_messages(messages)
        operation_text = self.format_operations(operations)
        sections = []
        if message_text:
            sections.append("【对话消息】\n" + message_text)
        if operation_text:
            sections.append("【技能和工具调用结果】\n" + operation_text)
        return "\n\n".join(sections)

    @staticmethod
    def format_operations(operations: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for index, operation in enumerate(operations, start=1):
            kind = str(operation.get("kind", "tool"))
            name = str(operation.get("name", "unknown"))
            result = operation.get("result", "")
            lines.append(
                f"{index}. [{kind}] {name}\n结果：{result}"
            )
        return "\n".join(lines)

    @staticmethod
    def _message_to_dict(message: Message) -> dict[str, Any]:
        return {
            "role": message.role,
            "content": message.content,
            "timestamp": str(getattr(message, "timestamp", "")),
            "metadata": getattr(message, "metadata", None),
        }

    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            return json.loads(
                json.dumps(value, ensure_ascii=False, default=str)
            )
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def estimate_tokens(text: str | None) -> int:
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

    @staticmethod
    def format_messages(messages: list[Message]) -> str:
        lines = [
            f"{msg.role}: {msg.content}"
            for msg in messages
            if msg.content
        ]
        return "\n".join(lines)
