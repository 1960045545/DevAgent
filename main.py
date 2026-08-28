from __future__ import annotations

import os

from config.settings import load_project_env
from core import agent
from core.tool_space import ToolSpec
from core.workspace_tools import register_workspace_tools
from manager.log_manager import configure_logging
from manager.model_provider_manager import ModelProviderConfig
from runtime.chat_service import ChatService
from runtime.event import ChatEvent, EventType
from runtime.policy import RuntimePolicy


load_project_env()
configure_logging()


def add(a: int, b: int) -> int:
    return a + b


add_tool = ToolSpec(
    name="add",
    description="calculate the sum of two integers",
    parameters={
        "type": "object",
        "properties": {
            "a": {
                "type": "integer",
                "description": "first number",
            },
            "b": {
                "type": "integer",
                "description": "second number",
            },
        },
        "required": ["a", "b"],
    },
    category="calculation",
)


def build_model_providers() -> list[ModelProviderConfig]:
    providers: list[ModelProviderConfig] = []

    if os.getenv("SILICON_BASE_URL") and os.getenv("SILICON_LLM_MODEL_ID"):
        providers.append(
            ModelProviderConfig(
                name="silicon",
                base_url=os.getenv("SILICON_BASE_URL"),
                api_key=os.getenv("SILICON_API_KEY"),
                chat_model_id=os.getenv("SILICON_LLM_MODEL_ID"),
                history_model_id=os.getenv("SILICON_HISTORY_ABSTRACT_MODEL_ID"),
                profile_model_id=os.getenv("SILICON_USER_PROFILE_MODEL_ID"),
            )
        )

    if os.getenv("BAI_LIAN_BASE_URL") and os.getenv("BAI_LIAN_LLM_MODEL_ID"):
        providers.append(
            ModelProviderConfig(
                name="bailian",
                base_url=os.getenv("BAI_LIAN_BASE_URL"),
                api_key=os.getenv("BAI_LIAN_API_KEY"),
                chat_model_id=os.getenv("BAI_LIAN_LLM_MODEL_ID"),
                history_model_id=os.getenv("BAI_LIAN_HISTORY_ABSTRACT_MODEL_ID"),
                profile_model_id=os.getenv("BAI_LIAN_USER_PROFILE_MODEL_ID"),
            )
        )

    if os.getenv("OLLAMA_BASE_URL") and os.getenv("OLLAMA_MODEL_ID"):
        ollama_model_id = os.getenv("OLLAMA_MODEL_ID")
        providers.append(
            ModelProviderConfig(
                name="ollama",
                base_url=os.getenv("OLLAMA_BASE_URL"),
                api_key=os.getenv("OLLAMA_API_KEY"),
                chat_model_id=ollama_model_id,
                history_model_id=ollama_model_id,
                profile_model_id=ollama_model_id,
            )
        )

    return providers


def print_stream_result(chat_service: ChatService, user_msg: str) -> None:
    print("assistant: ", end="", flush=True)
    for part in chat_service.stream_chat(user_msg):
        print(part, end="", flush=True)
    print()

    state = chat_service.state
    if state is not None:
        print(
            f"[task] id={state.task_id} "
            f"status={state.status.value} "
            f"provider={state.provider_name} "
            f"model={state.model_id} "
            f"events={len(state.events)}"
        )


def print_progress_event(event: ChatEvent) -> None:
    if event.event_type == EventType.TODO_CREATED:
        print("\n[todo] plan created")
        _print_todo_snapshot(event.data)
    elif event.event_type == EventType.TODO_UPDATED:
        print("\n[todo] progress updated")
        _print_todo_snapshot(event.data)
    elif event.event_type == EventType.TOOL_STARTED:
        print(
            f"\n[tool] {event.data.get('category', 'general')}/"
            f"{event.data.get('name', 'unknown')} started",
        )
    elif event.event_type == EventType.TOOL_FINISHED:
        print(
            f"\n[tool] {event.data.get('category', 'general')}/"
            f"{event.data.get('name', 'unknown')} finished",
        )
    elif event.event_type == EventType.TOOL_FAILED:
        print(
            f"\n[tool] {event.data.get('category', 'general')}/"
            f"{event.data.get('name', 'unknown')} failed: "
            f"{event.data.get('error', event.error or 'unknown error')}",
        )


def _print_todo_snapshot(data: dict[str, object]) -> None:
    for todo in data.get("todos", []):
        if not isinstance(todo, dict):
            continue
        status = str(todo.get("status", "pending"))
        marker = "[x]" if status == "completed" else "[!]" if status == "blocked" else "[>]" if status == "in_progress" else "[ ]"
        note = f" ({todo['note']})" if todo.get("note") else ""
        print(
            f"[todo] {marker} {todo.get('item_id', '?')}: "
            f"{todo.get('title', '')}{note}",
        )


def print_normal_result(chat_service: ChatService, user_msg: str) -> None:
    result = chat_service.response_chat(user_msg)
    print(result.text)
    print(
        f"[task] id={result.task_id} "
        f"success={result.success} "
        f"provider={result.provider_name} "
        f"model={result.model} "
        f"events={len(result.events)}"
    )


if __name__ == "__main__":
    model_providers = build_model_providers()

    chat_agent = agent.Agent(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        model_id=os.getenv("LLM_MODEL_ID"),
        providers=model_providers or None,
        max_retries=3,
        retry_interval=5,
        cooldown_seconds=30,
        max_tokens=3000,
        timeout=120,
    )

    chat_agent.register_tool(add_tool, add)
    register_workspace_tools(chat_agent.tool_registry)

    chat_service = ChatService(
        chat_agent,
        RuntimePolicy(
            stream=True,
            collect_events=True,
            stream_batch_chars=12,
            stream_batch_seconds=0.1,
        ),
        event_callback=print_progress_event,
    )

    print("Runtime chat test started.")
    print("Type end to exit.")
    print("Type /normal <message> to test normal response mode.")

    while True:
        user_msg = input("You: ").strip()

        if user_msg == "end":
            print("Bye.")
            break

        if user_msg.startswith("/normal "):
            print_normal_result(chat_service, user_msg.removeprefix("/normal ").strip())
            continue

        print_stream_result(chat_service, user_msg)
