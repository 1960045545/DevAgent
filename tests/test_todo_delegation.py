from __future__ import annotations

import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from core.delegation import SubtaskExecutor
from core.invoke_options import InvokeOptions
from core.response import ModelResponse
from core.tool_call import ToolCall
from core.tool_registry import ToolRegistry
from core.tool_space import ToolSpec
from core.todo import TodoList, TodoToolset
from manager.tool_manager import ToolManager
from runtime.chat_service import ChatService
from runtime.event import EventType
from runtime.task_state import TaskStatus


class FakeChildAgent:
    def __init__(self, response_text: str, messages: list[list[dict]]):
        self.response_text = response_text
        self.messages = messages
        self.llm_manager = SimpleNamespace(
            invoke_messages=self.invoke_messages,
        )
        self.tool_manager = SimpleNamespace()

    def invoke_messages(self, messages, **_kwargs):
        self.messages.append(messages)
        return ModelResponse(text=self.response_text)


class FakeToolLLM:
    def __init__(self, responses: list[ModelResponse]):
        self.responses = responses
        self.calls: list[list[dict]] = []

    def create_chat_completion(self, messages, _options, *, stream, purpose):
        self.calls.append(messages[:])
        return self.responses.pop(0).raw or {}

    @staticmethod
    def response_to_dict(response):
        return response

    @staticmethod
    def parse_chat_response(data):
        return ModelResponse(**data)

    @staticmethod
    def assistant_message_from_response(data):
        return {
            "role": "assistant",
            "content": data.get("text", ""),
            "tool_calls": data.get("tool_calls", []),
        }


class TodoDelegationTests(unittest.TestCase):
    def test_child_receives_only_explicit_subtask_messages(self) -> None:
        todo_list = TodoList()
        todo_list.create(["Inspect the file"])
        host_registry = ToolRegistry()
        host = SimpleNamespace(
            base_url="http://example.test",
            api_key="key",
            model_id="model",
            timeout=10,
            max_retries=1,
            default_headers={},
            client=object(),
            max_tokens=1000,
            prompt_manager=SimpleNamespace(prompt_dir="prompts"),
            llm_manager=SimpleNamespace(providers=[]),
            tool_registry=host_registry,
            tool_manager=SimpleNamespace(hooks=[]),
        )
        sent_messages: list[list[dict]] = []

        def factory(**_kwargs):
            return FakeChildAgent(
                json.dumps(
                    {
                        "task_id": "wrong-id",
                        "title": "wrong-title",
                        "status": "completed",
                        "summary": "file inspected",
                    }
                ),
                sent_messages,
            )

        host_history = "主 agent secret history"
        result = SubtaskExecutor(
            host,
            todo_list,
            child_agent_factory=factory,
        ).delegate(
            "todo-1",
            "Inspect only README.md and report its first heading.",
        )

        self.assertEqual(result["task_id"], "todo-1")
        self.assertEqual(result["title"], "Inspect the file")
        self.assertEqual(todo_list.items[0].status, "completed")
        self.assertEqual(len(sent_messages), 1)
        self.assertEqual(
            sent_messages[0][-1]["content"],
            "Inspect only README.md and report its first heading.",
        )
        self.assertNotIn(host_history, json.dumps(sent_messages))
        self.assertEqual(len(sent_messages[0]), 2)

    def test_summary_updates_todo_and_emits_progress(self) -> None:
        todo_list = TodoList()
        events: list[tuple[str, dict]] = []
        todo_list.create(["Build artifact"])
        host = self._host_with_child_response(
            json.dumps(
                {
                    "task_id": "todo-1",
                    "title": "Build artifact",
                    "status": "completed",
                    "summary": "built",
                    "artifacts": ["dist/app.zip"],
                }
            )
        )
        result = SubtaskExecutor(
            host,
            todo_list,
            progress_callback=lambda event_type, data: events.append(
                (event_type, data)
            ),
            child_agent_factory=host.child_factory,
        ).delegate("todo-1", "Build the artifact.")

        self.assertEqual(result["summary"], "built")
        self.assertEqual(result["artifacts"], ["dist/app.zip"])
        self.assertEqual(todo_list.items[0].status, "completed")
        self.assertEqual(
            [event[0] for event in events],
            ["subtask_started", "todo_updated", "subtask_finished"],
        )

    def test_child_failure_is_not_reported_as_completed(self) -> None:
        todo_list = TodoList()
        todo_list.create(["Run checks"])
        host = self._host_with_child_response("not json")
        result = SubtaskExecutor(
            host,
            todo_list,
            child_agent_factory=host.child_factory,
        ).delegate("todo-1", "Run the checks.")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(todo_list.items[0].status, "failed")
        self.assertNotEqual(todo_list.items[0].status, "completed")

    def test_dependencies_are_waited_and_can_run_in_parallel(self) -> None:
        todo_list = TodoList()
        todo_list.create(
            [
                {"title": "First", "dependencies": []},
                {"title": "Second", "dependencies": ["todo-1"]},
                {"title": "Third", "dependencies": []},
            ]
        )
        self.assertFalse(todo_list.is_ready("todo-2"))
        self.assertTrue(todo_list.is_ready("todo-1"))
        self.assertTrue(todo_list.is_ready("todo-3"))

    def test_dependency_cycles_are_rejected(self) -> None:
        todo_list = TodoList()

        with self.assertRaisesRegex(ValueError, "dependency cycle"):
            todo_list.create(
                [
                    {"title": "First", "dependencies": ["todo-2"]},
                    {"title": "Second", "dependencies": ["todo-1"]},
                ]
            )

    def test_waiting_subtask_is_deferred_without_blocking_todo(self) -> None:
        todo_list = TodoList()
        todo_list.create(
            [
                "First",
                {"title": "Second", "dependencies": ["todo-1"]},
            ]
        )
        events: list[str] = []
        host = self._host_with_child_response(
            json.dumps(
                {
                    "status": "completed",
                    "summary": "must not run",
                }
            )
        )

        result = SubtaskExecutor(
            host,
            todo_list,
            progress_callback=lambda event_type, _data: events.append(event_type),
            child_agent_factory=host.child_factory,
        ).delegate("todo-2", "Wait for the first task.")

        self.assertEqual(result["status"], "deferred")
        self.assertEqual(todo_list.items[1].status, "pending")
        self.assertIn("subtask_deferred", events)

    def test_child_inherits_registered_non_todo_tools_by_default(self) -> None:
        todo_list = TodoList()
        todo_list.create(["Inspect files"])
        registry = ToolRegistry()
        inspect_spec = ToolSpec(
            name="inspect_file",
            description="inspect a file",
            parameters={"type": "object", "properties": {}},
        )
        registry.register(inspect_spec, lambda: "ok")
        host = self._host_with_child_response(
            json.dumps({"status": "completed", "summary": "inspected"})
        )
        host.tool_registry = registry
        child_kwargs: list[dict] = []

        def factory(**kwargs):
            child_kwargs.append(kwargs)
            child = FakeChildAgent(
                json.dumps({"status": "completed", "summary": "inspected"}),
                [],
            )
            child.tool_manager.chat_with_tools = (
                lambda **_options: ModelResponse(
                    text=json.dumps(
                        {"status": "completed", "summary": "inspected"}
                    )
                )
            )
            return child

        result = SubtaskExecutor(
            host,
            todo_list,
            child_agent_factory=factory,
        ).delegate("todo-1", "Inspect the configured file.")

        child_registry = child_kwargs[0]["tool_registry"]
        self.assertEqual(
            [spec.name for spec in child_registry.list_specs()],
            ["inspect_file"],
        )
        self.assertEqual(result["status"], "completed")

    def test_concurrent_claim_allows_only_one_executor(self) -> None:
        todo_list = TodoList()
        todo_list.create(["One item"])
        barrier = threading.Barrier(2)

        def claim() -> str:
            barrier.wait()
            try:
                todo_list.claim("todo-1")
            except ValueError:
                return "rejected"
            return "claimed"

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _index: claim(), (1, 2)))

        self.assertEqual(sorted(results), ["claimed", "rejected"])
        self.assertEqual(todo_list.items[0].status, "in_progress")

    def test_first_round_plan_executes_before_other_tools(self) -> None:
        calls: list[str] = []
        recorded_results: list[tuple[str, str, int, str]] = []
        spec = ToolSpec(
            name="dangerous_tool",
            description="test",
            parameters={"type": "object", "properties": {}},
        )
        responses = [
            self._response(
                tool_calls=[
                    ToolCall(
                        id="plan-call",
                        name="todo_create",
                        arguments={"items": ["one"]},
                    ),
                    ToolCall(
                        id="other-call",
                        name="dangerous_tool",
                        arguments={},
                    ),
                ]
            ),
            self._response(tool_calls=[
                ToolCall(
                    id="retry-call",
                    name="dangerous_tool",
                    arguments={},
                )
            ]),
            self._response(tool_calls=[
                ToolCall(
                    id="update-call",
                    name="todo_update",
                    arguments={
                        "item_id": "todo-1",
                        "status": "completed",
                    },
                )
            ]),
            self._response(text="done"),
        ]
        llm = FakeToolLLM(responses)
        manager = ToolManager(llm_manager=llm)
        todo_toolset = TodoToolset()
        options = manager.merge_options(
            InvokeOptions(
                tools=[spec],
                tool_handlers={
                    "dangerous_tool": lambda: calls.append("other")
                },
            ),
            extra_specs=todo_toolset.specs,
            extra_handlers=todo_toolset.handlers,
        )
        original_create = todo_toolset.todo_list.create
        todo_toolset.todo_list.create = lambda items: (
            calls.append("plan"), original_create(items)
        )[1]
        # The handler map must point at the wrapped function used for ordering.
        options.tool_handlers["todo_create"] = todo_toolset.todo_list.create

        result = manager.chat_with_tools(
            prompt="system",
            user_message="complex request",
            options=options,
            todo_list=todo_toolset.todo_list,
            tool_result_callback=(
                lambda tool_call, tool_result, round_index: recorded_results.append(
                    (tool_call.name, tool_result, round_index, tool_call.id)
                )
            ),
        )

        self.assertEqual(result.text, "done")
        self.assertEqual(calls, ["plan", "other"])
        first_call_messages = llm.calls[1]
        self.assertEqual(
            [message["tool_call_id"] for message in first_call_messages if message["role"] == "tool"],
            ["plan-call", "other-call"],
        )
        self.assertEqual(
            [name for name, _result, _round, _call_id in recorded_results],
            ["todo_create", "dangerous_tool", "todo_update"],
        )
        self.assertEqual(
            [call_id for _name, _result, _round, call_id in recorded_results],
            ["plan-call", "retry-call", "update-call"],
        )

    def test_parallel_delegate_calls_overlap_and_emit_tool_events(self) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()

        def delegate(item_id: str, instructions: str) -> dict:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return {"task_id": item_id, "status": "completed", "summary": instructions}

        todo_list = TodoList()
        todo_list.create(["one", "two"])
        spec = ToolSpec(
            name="todo_delegate",
            description="test",
            parameters={"type": "object", "properties": {}},
        )
        calls = [
            ToolCall(id="a", name="todo_delegate", arguments={"item_id": "todo-1", "instructions": "a"}),
            ToolCall(id="b", name="todo_delegate", arguments={"item_id": "todo-2", "instructions": "b"}),
        ]
        manager = ToolManager(llm_manager=Mock())
        result = manager._parallel_delegated_calls(
            calls,
            todo_list,
            handlers={"todo_delegate": delegate},
            spec_by_name={"todo_delegate": spec},
            round_index=0,
            progress_callback=None,
        )

        self.assertEqual(set(result), {"a", "b"})
        self.assertEqual(max_active, 2)

    def test_chat_service_maps_subtask_progress_event(self) -> None:
        service = ChatService(Mock())
        state = service._start_state("request")
        service._on_agent_progress(
            "subtask_finished",
            {"task_id": "todo-1", "status": "completed"},
        )

        self.assertEqual(state.status, TaskStatus.RUNNING)
        self.assertEqual(state.events[-1].event_type, EventType.SUBTASK_FINISHED)

    def test_chat_service_maps_deferred_subtask_event(self) -> None:
        service = ChatService(Mock())
        state = service._start_state("request")
        service._on_agent_progress(
            "subtask_deferred",
            {"task_id": "todo-1", "status": "deferred"},
        )

        self.assertEqual(state.events[-1].event_type, EventType.SUBTASK_DEFERRED)

    @staticmethod
    def _response(text: str = "", tool_calls: list[ToolCall] | None = None):
        raw_tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        call.arguments,
                        ensure_ascii=False,
                    ),
                },
            }
            for call in tool_calls or []
        ]
        return ModelResponse(
            text=text,
            tool_calls=tool_calls or [],
            raw={
                "id": "response",
                "model": "test-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": text,
                            "tool_calls": raw_tool_calls,
                        },
                        "finish_reason": "tool_calls" if raw_tool_calls else "stop",
                    }
                ],
            },
        )

    @staticmethod
    def _host_with_child_response(response_text: str):
        messages: list[list[dict]] = []

        def factory(**_kwargs):
            return FakeChildAgent(response_text, messages)

        return SimpleNamespace(
            child_factory=factory,
            base_url="http://example.test",
            api_key="key",
            model_id="model",
            timeout=10,
            max_retries=1,
            default_headers={},
            client=object(),
            max_tokens=1000,
            prompt_manager=SimpleNamespace(prompt_dir="prompts"),
            llm_manager=SimpleNamespace(providers=[]),
            tool_registry=ToolRegistry(),
            tool_manager=SimpleNamespace(hooks=[]),
        )


if __name__ == "__main__":
    unittest.main()
