from __future__ import annotations

import copy
import json
import os
from typing import Any, Callable


TranscriptWriter = Callable[..., Any]
CompactionObserver = Callable[[str, dict[str, Any]], None]


class MessageCompactionPipeline:
    """Bound in-flight tool-loop messages without breaking tool pairing."""

    STAGES = (
        "tool_result_budget",
        "snip_compact",
        "micro_compact",
        "compact_history",
    )

    def __init__(
        self,
        *,
        max_context_chars: int | None = None,
        tool_result_budget_chars: int | None = None,
        max_tool_result_chars: int | None = None,
        old_tool_result_chars: int | None = None,
        keep_recent_tool_results: int = 3,
        transcript_writer: TranscriptWriter | None = None,
    ) -> None:
        self.max_context_chars = max_context_chars or _positive_int(
            "AGENT_CONTEXT_MAX_CHARS", 240000
        )
        self.tool_result_budget_chars = tool_result_budget_chars or _positive_int(
            "AGENT_TOOL_RESULT_BUDGET_CHARS", 100000
        )
        self.max_tool_result_chars = max_tool_result_chars or _positive_int(
            "AGENT_TOOL_RESULT_MAX_CHARS", 30000
        )
        self.old_tool_result_chars = old_tool_result_chars or _positive_int(
            "AGENT_OLD_TOOL_RESULT_MAX_CHARS", 2000
        )
        self.keep_recent_tool_results = max(0, keep_recent_tool_results)
        self.transcript_writer = transcript_writer
        self.last_stage_order: list[str] = []

    def compact(
        self,
        messages: list[dict[str, Any]],
        *,
        observer: CompactionObserver | None = None,
    ) -> bool:
        before_chars = self.message_chars(messages)
        candidate = copy.deepcopy(messages)
        self.last_stage_order = []

        for stage_name in self.STAGES:
            self.last_stage_order.append(stage_name)
            getattr(self, f"_{stage_name}")(candidate)

        if candidate == messages:
            return False

        if self.transcript_writer is not None:
            self.transcript_writer(
                copy.deepcopy(messages),
                reason="pre_llm_compaction_pipeline",
            )
        messages[:] = candidate
        if observer is not None:
            observer(
                "context_compacted",
                {
                    "stages": list(self.last_stage_order),
                    "before_chars": before_chars,
                    "after_chars": self.message_chars(messages),
                },
            )
        return True

    def _tool_result_budget(self, messages: list[dict[str, Any]]) -> None:
        tool_messages = [
            message for message in messages if message.get("role") == "tool"
        ]
        total = sum(self._content_chars(message) for message in tool_messages)
        overflow = total - self.tool_result_budget_chars
        if overflow <= 0:
            return

        for message in tool_messages:
            if overflow <= 0:
                break
            content = self._content_text(message)
            if not content:
                continue
            target = max(160, len(content) - overflow)
            replacement = self._snip(
                content,
                target,
                marker="tool result budget compacted",
            )
            overflow -= len(content) - len(replacement)
            message["content"] = replacement

    def _snip_compact(self, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            if message.get("role") != "tool":
                continue
            content = self._content_text(message)
            if len(content) > self.max_tool_result_chars:
                message["content"] = self._snip(
                    content,
                    self.max_tool_result_chars,
                    marker="large tool result snipped",
                )

    def _micro_compact(self, messages: list[dict[str, Any]]) -> None:
        if self.message_chars(messages) <= self.max_context_chars:
            return
        indexes = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "tool"
        ]
        old_indexes = (
            indexes[:-self.keep_recent_tool_results]
            if self.keep_recent_tool_results
            else indexes
        )
        for index in old_indexes:
            content = self._content_text(messages[index])
            if len(content) > self.old_tool_result_chars:
                messages[index]["content"] = self._snip(
                    content,
                    self.old_tool_result_chars,
                    marker="old tool result compacted",
                )

    def _compact_history(self, messages: list[dict[str, Any]]) -> None:
        removed: list[str] = []
        while self.message_chars(messages) > self.max_context_chars:
            exchange = self._oldest_complete_tool_exchange(messages)
            if exchange is None:
                break
            start, end, names = exchange
            removed.extend(names)
            del messages[start:end]

        if not removed:
            return
        summary = {
            "removed_tool_exchanges": len(removed),
            "tools": removed,
            "note": "Full messages were saved in .transcripts before compaction.",
        }
        insert_at = 1 if messages and messages[0].get("role") == "system" else 0
        messages.insert(
            insert_at,
            {
                "role": "system",
                "content": "<compacted_history>\n"
                + json.dumps(summary, ensure_ascii=False)
                + "\n</compacted_history>",
            },
        )

    @classmethod
    def _oldest_complete_tool_exchange(
        cls,
        messages: list[dict[str, Any]],
    ) -> tuple[int, int, list[str]] | None:
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls") or []
            call_ids = {
                str(call.get("id"))
                for call in tool_calls
                if isinstance(call, dict) and call.get("id")
            }
            if not call_ids:
                continue
            result_ids: set[str] = set()
            end = index + 1
            while end < len(messages) and messages[end].get("role") == "tool":
                tool_call_id = messages[end].get("tool_call_id")
                if tool_call_id:
                    result_ids.add(str(tool_call_id))
                end += 1
            if call_ids != result_ids:
                continue
            names = [
                str((call.get("function") or {}).get("name") or "unknown")
                for call in tool_calls
                if isinstance(call, dict)
            ]
            return index, end, names
        return None

    @staticmethod
    def _content_text(message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return json.dumps(content, ensure_ascii=False, default=str)

    @classmethod
    def _content_chars(cls, message: dict[str, Any]) -> int:
        return len(cls._content_text(message))

    @classmethod
    def message_chars(cls, messages: list[dict[str, Any]]) -> int:
        return sum(
            len(json.dumps(message, ensure_ascii=False, default=str))
            for message in messages
        )

    @staticmethod
    def _snip(content: str, limit: int, *, marker: str) -> str:
        if len(content) <= limit:
            return content
        label = f"\n...[{marker}; original_chars={len(content)}]...\n"
        if limit <= len(label) + 2:
            return label[:limit]
        remaining = limit - len(label)
        head = remaining * 2 // 3
        tail = remaining - head
        return content[:head] + label + content[-tail:]


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed
