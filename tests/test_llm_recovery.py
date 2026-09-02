from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from core.invoke_options import InvokeOptions
from error.request_error import (
    AgentRequestError,
    PromptTooLongError,
)
from manager.llm_manager import LLMManager
from manager.model_provider_manager import ModelProviderConfig


class FakeAPIError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int,
        *,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.headers = headers or {}


class FakeCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(outcomes)


def response(
    text: str,
    *,
    finish_reason: str = "stop",
) -> dict:
    return {
        "id": "response",
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }


class LLMRecoveryTests(unittest.TestCase):
    def make_manager(
        self,
        outcomes: list[object],
        *,
        providers: list[ModelProviderConfig] | None = None,
        **kwargs,
    ) -> tuple[LLMManager, FakeClient]:
        client = FakeClient(outcomes)
        manager = LLMManager(
            base_url="http://example.test",
            api_key="key",
            model_id="test-model",
            client=client,
            providers=providers,
            max_retries=1,
            cooldown_seconds=0,
            **kwargs,
        )
        return manager, client

    def test_length_finish_reason_escalates_from_8k_to_64k(self) -> None:
        manager, client = self.make_manager(
            [
                response("discard this", finish_reason="length"),
                response("complete"),
            ]
        )

        result = manager.invoke_messages(
            [{"role": "user", "content": "question"}],
            InvokeOptions(),
        )

        self.assertEqual(result.text, "complete")
        self.assertEqual(
            [call["max_completion_tokens"] for call in client.chat.completions.calls],
            [8000, 64000],
        )
        self.assertEqual(
            client.chat.completions.calls[0]["messages"],
            [{"role": "user", "content": "question"}],
        )

    def test_length_response_uses_bounded_continuation(self) -> None:
        manager, client = self.make_manager(
            [
                response("discard", finish_reason="length"),
                response("part one", finish_reason="length"),
                response("part two", finish_reason="length"),
                response("done"),
            ]
        )

        result = manager.invoke_messages(
            [{"role": "user", "content": "question"}],
            InvokeOptions(),
        )

        self.assertEqual(result.text, "part onepart twodone")
        self.assertEqual(len(client.chat.completions.calls), 4)
        self.assertEqual(
            client.chat.completions.calls[-1]["messages"][-1]["content"],
            "Output token limit hit. Resume directly; do not apologize or recap. "
            "Pick up exactly where the previous answer stopped.",
        )

    def test_prompt_too_long_compacts_then_fails_after_second_attempt(self) -> None:
        manager, _ = self.make_manager(
            [
                FakeAPIError(
                    "prompt_too_long",
                    400,
                    code="prompt_too_long",
                ),
                FakeAPIError(
                    "prompt_too_long",
                    400,
                    code="prompt_too_long",
                ),
                FakeAPIError(
                    "prompt_too_long",
                    400,
                    code="prompt_too_long",
                ),
            ],
            context_compactor=lambda: None,
        )
        transcript_calls: list[str] = []
        manager.context_transcript_writer = (
            lambda _messages, *, reason: transcript_calls.append(reason)
        )

        with self.assertRaises(PromptTooLongError):
            manager.invoke_messages(
                [{"role": "user", "content": "x" * 10000}],
                InvokeOptions(),
            )

        self.assertEqual(transcript_calls, ["reactive_prompt_compact"])

    def test_rate_limit_honors_retry_after_and_does_not_switch_provider(self) -> None:
        manager, client = self.make_manager(
            [
                FakeAPIError(
                    "rate limited",
                    429,
                    headers={"Retry-After": "0"},
                ),
                response("ok"),
            ]
        )

        with patch("manager.llm_manager.time.sleep") as sleep:
            result = manager.invoke_messages(
                [{"role": "user", "content": "question"}],
                InvokeOptions(),
            )

        self.assertEqual(result.text, "ok")
        sleep.assert_called_once_with(0.0)
        self.assertEqual(len(client.chat.completions.calls), 2)

    def test_three_529_responses_allow_backup_provider(self) -> None:
        primary = ModelProviderConfig(
            name="primary",
            base_url="http://primary.test",
            api_key="key",
            chat_model_id="primary-model",
        )
        backup = ModelProviderConfig(
            name="backup",
            base_url="http://backup.test",
            api_key="key",
            chat_model_id="backup-model",
        )
        primary_client = FakeClient(
            [
                FakeAPIError("overloaded", 529),
                FakeAPIError("overloaded", 529),
                FakeAPIError("overloaded", 529),
            ]
        )
        backup_client = FakeClient([response("backup ok")])
        manager = LLMManager(
            base_url=None,
            api_key=None,
            model_id=None,
            client=None,
            providers=[primary, backup],
            max_retries=1,
            cooldown_seconds=0,
        )
        manager.clients = {
            "primary": primary_client,
            "backup": backup_client,
        }

        with patch("manager.llm_manager.time.sleep"):
            result = manager.invoke_messages(
                [{"role": "user", "content": "question"}],
                InvokeOptions(),
            )

        self.assertEqual(result.text, "backup ok")
        self.assertEqual(len(primary_client.chat.completions.calls), 3)
        self.assertEqual(len(backup_client.chat.completions.calls), 1)
        self.assertEqual(manager.last_success_provider_name, "backup")

    def test_other_errors_are_not_retried_or_fallbacked(self) -> None:
        manager, client = self.make_manager(
            [FakeAPIError("bad request", 400)]
        )

        with self.assertRaises(FakeAPIError):
            manager.invoke_messages(
                [{"role": "user", "content": "question"}],
                InvokeOptions(),
            )

        self.assertEqual(len(client.chat.completions.calls), 1)


if __name__ == "__main__":
    unittest.main()
