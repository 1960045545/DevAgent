from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event, RLock, Thread
from typing import TYPE_CHECKING, Any, Callable, Iterable

from core.invoke_options import InvokeOptions
from core.response import ModelResponse
from core.subagent_communication import (
    SubagentCommunication,
    SubagentCommunicationToolset,
)
from core.tool_registry import ToolRegistry
from core.todo import TodoList
from core.worktree import WorktreeManager

if TYPE_CHECKING:
    from core.agent import Agent


logger = logging.getLogger(__name__)
_CHILD_EXCLUDED_TOOLS = {
    "todo_create",
    "todo_claim",
    "todo_complete",
    "todo_block",
    "todo_update",
    "todo_list",
    "todo_delegate",
    "todo_run_background",
    "background_list",
    "background_get",
    "background_notifications",
    "worktree_create",
    "worktree_bind",
    "worktree_list",
    "worktree_keep",
    "worktree_remove",
}


@dataclass(frozen=True, slots=True)
class SubtaskSummary:
    task_id: str
    title: str
    status: str
    summary: str
    artifacts: list[str] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task_id": self.task_id,
            "task": self.title,
            "title": self.title,
            "status": self.status,
            "summary": self.summary,
        }
        if self.artifacts:
            result["artifacts"] = list(self.artifacts)
        if self.error:
            result["error"] = self.error
        return result


class AutonomousTaskWorker:
    """Poll one agent inbox and the shared Todo board while it is idle."""

    def __init__(
        self,
        executor: "SubtaskExecutor",
        agent_name: str,
        *,
        poll_interval: float,
        idle_timeout: float,
    ) -> None:
        self.executor = executor
        self.agent_name = agent_name
        self.poll_interval = max(0.01, poll_interval)
        self.idle_timeout = max(self.poll_interval, idle_timeout)
        self._stop_event = Event()
        self._thread = Thread(
            target=self._run,
            name=f"agent-idle-{agent_name}",
            daemon=True,
        )

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        communication = self.executor.communication
        if communication is None:
            return
        try:
            communication.register_task(self.agent_name)
            idle_started = time.monotonic()
            while not self._stop_event.wait(self.poll_interval):
                if self._consume_inbox(communication):
                    idle_started = time.monotonic()
                    continue

                task = self.executor.todo_list.claim_next(self.agent_name)
                if task is not None:
                    idle_started = time.monotonic()
                    self.executor._notify(
                        "autonomous_task_claimed",
                        {
                            "agent_name": self.agent_name,
                            "task_id": task["task_id"],
                            "owner": task.get("owner"),
                            "task": task.get("task", ""),
                            "source": "task_board",
                        },
                    )
                    self.executor.delegate(
                        task["task_id"],
                        str(task.get("task", "")),
                        owner=self.agent_name,
                        preclaimed=True,
                        start_autonomous=False,
                    )
                    continue

                if time.monotonic() - idle_started >= self.idle_timeout:
                    self.executor._notify(
                        "autonomous_shutdown",
                        {
                            "agent_name": self.agent_name,
                            "reason": "idle timeout",
                            "idle_timeout": self.idle_timeout,
                        },
                    )
                    return
        except Exception:
            logger.exception(
                "autonomous worker failed agent_name=%s",
                self.agent_name,
            )
            self.executor._notify(
                "autonomous_failed",
                {
                    "agent_name": self.agent_name,
                    "error": "autonomous worker stopped unexpectedly",
                },
            )

    def _consume_inbox(self, communication: SubagentCommunication) -> bool:
        peeked = communication.peek_inbox(self.agent_name)
        messages = peeked.get("messages", [])
        selected_message_id = self._select_inbox_message(messages)
        if selected_message_id is None:
            return False
        inbox = communication.consume_inbox(
            self.agent_name,
            message_ids=[selected_message_id],
        )
        messages = inbox.get("messages", [])
        if not messages:
            return False

        for routed in messages:
            message = routed.get("message", {})
            message_type = str(message.get("type", ""))
            if message_type == "shutdown_request":
                request_id = str(
                    message.get(
                        "protocol_request_id",
                        message.get("request_id", ""),
                    )
                )
                try:
                    communication.respond_protocol(
                        request_id,
                        responder=self.agent_name,
                        payload={"approve": True},
                        content="shutdown acknowledged",
                    )
                    communication.consume_inbox("main")
                except (TypeError, ValueError, PermissionError):
                    logger.exception(
                        "failed to acknowledge shutdown agent_name=%s",
                        self.agent_name,
                    )
                self.executor._notify(
                    "autonomous_shutdown",
                    {
                        "agent_name": self.agent_name,
                        "reason": "shutdown request",
                        "request_id": request_id,
                    },
                )
                self.stop()
                return True

            if message_type == "task_request":
                payload = message.get("payload")
                if not isinstance(payload, dict):
                    continue
                task_id = str(payload.get("task_id", "")).strip()
                instructions = str(
                    payload.get("instructions", payload.get("title", ""))
                ).strip()
                if not task_id or not instructions:
                    continue
                try:
                    task = self.executor.todo_list.get(task_id)
                    if (
                        task.status != "pending"
                        or task.owner
                        or not self.executor.todo_list.can_start(task_id)
                    ):
                        continue
                    self.executor.todo_list.claim(
                        task_id,
                        note="claimed from agent inbox",
                        owner=self.agent_name,
                    )
                except ValueError:
                    continue
                self.executor._notify(
                    "autonomous_task_claimed",
                    {
                        "agent_name": self.agent_name,
                        "task_id": task_id,
                        "owner": self.agent_name,
                        "source": "inbox",
                        "request_id": message.get("protocol_request_id"),
                    },
                )
                self.executor.delegate(
                    task_id,
                    instructions,
                    tool_names=payload.get("tool_names"),
                    owner=self.agent_name,
                    preclaimed=True,
                    protocol_request_id=(
                        str(message.get("protocol_request_id", "")).strip()
                        or None
                    ),
                    protocol_responder=self.agent_name,
                    start_autonomous=False,
                )
                return True

            self.executor._notify(
                "autonomous_inbox_message",
                {
                    "agent_name": self.agent_name,
                    "message_id": message.get("message_id"),
                    "message_type": message_type,
                },
            )
        return True

    def _select_inbox_message(
        self,
        messages: list[dict[str, Any]],
    ) -> str | None:
        """Select one runnable inbox message without consuming blocked work."""
        for message in messages:
            message_type = str(message.get("type", ""))
            if message_type != "task_request":
                return str(message.get("message_id", "")) or None

            payload = message.get("payload")
            if not isinstance(payload, dict):
                return str(message.get("message_id", "")) or None
            task_id = str(payload.get("task_id", "")).strip()
            instructions = str(
                payload.get("instructions", payload.get("title", ""))
            ).strip()
            if not task_id or not instructions:
                return str(message.get("message_id", "")) or None
            try:
                task = self.executor.todo_list.get(task_id)
            except ValueError:
                return str(message.get("message_id", "")) or None
            if (
                task.status == "pending"
                and not task.owner
                and self.executor.todo_list.can_start(task_id)
            ):
                return str(message.get("message_id", "")) or None
        return None


class SubtaskExecutor:
    """Runs isolated child agents and returns summaries, never their history."""

    def __init__(
        self,
        host_agent: Agent,
        todo_list: TodoList,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        child_agent_factory: Callable[..., Agent] | None = None,
        communication: SubagentCommunication | None = None,
        autonomous_enabled: bool = True,
        autonomous_poll_interval: float = 5.0,
        autonomous_idle_timeout: float = 60.0,
        worktree_manager: WorktreeManager | None = None,
    ) -> None:
        self.host_agent = host_agent
        self.todo_list = todo_list
        self.progress_callback = progress_callback
        self.child_agent_factory = child_agent_factory
        self.communication = communication
        self.autonomous_enabled = autonomous_enabled
        self.autonomous_poll_interval = max(0.01, autonomous_poll_interval)
        self.autonomous_idle_timeout = max(
            self.autonomous_poll_interval,
            autonomous_idle_timeout,
        )
        self.worktree_manager = worktree_manager
        self._autonomous_workers: dict[str, AutonomousTaskWorker] = {}
        self._autonomous_lock = RLock()

    def delegate(
        self,
        item_id: str | None = None,
        instructions: str = "",
        tool_names: list[str] | None = None,
        *,
        task_id: str | None = None,
        owner: str | None = None,
        preclaimed: bool = False,
        protocol_request_id: str | None = None,
        protocol_responder: str | None = None,
        start_autonomous: bool = True,
    ) -> dict[str, Any]:
        item_id = (task_id or item_id or "").strip()
        item = self.todo_list.get(item_id)
        owner = (owner or item_id).strip()
        protocol_responder = (protocol_responder or item_id).strip()
        if not instructions.strip():
            return self._finish_without_child(
                item,
                status="failed",
                summary="子任务说明为空。",
                error="subtask instructions cannot be empty",
            )

        try:
            if preclaimed:
                current = self.todo_list.get(item_id)
                if current.status != "in_process" or current.owner != owner:
                    raise ValueError(
                        f"task {item_id} is not claimed by {owner}"
                    )
            else:
                self.todo_list.claim(
                    item_id,
                    note="delegated to an isolated sub-agent",
                    owner=owner,
                )
        except ValueError as exc:
            current_status = self.todo_list.get(item_id).status
            if current_status in {"pending", "in_process", "in_progress"}:
                result = SubtaskSummary(
                    task_id=item_id,
                    title=item.title,
                    status="deferred",
                    summary=(
                        "子任务暂未执行，可能正在等待依赖或已被其它执行器占用。"
                    ),
                    error=str(exc),
                ).to_dict()
                self._notify("subtask_deferred", result)
                return result
            return SubtaskSummary(
                task_id=item_id,
                title=item.title,
                status=current_status,
                summary="子任务已经处于终态，未重复执行。",
                error=str(exc),
            ).to_dict()
        worktree_record: dict[str, Any] | None = None
        worktree_path: Path | None = None
        worktree_data: dict[str, Any] = {}
        if self.worktree_manager is not None:
            try:
                worktree_name = item.worktree or self.worktree_manager.task_worktree_name(
                    item_id
                )
                worktree_record = self.worktree_manager.create_worktree(
                    worktree_name,
                    task_id=item_id,
                )
                if item.worktree is None:
                    self.todo_list.bind_worktree(item_id, worktree_name)
                self.worktree_manager.bind_task_to_worktree(
                    item_id,
                    worktree_name,
                )
                worktree_path = self.worktree_manager.resolve_worktree(worktree_name)
                worktree_data = self._worktree_data(worktree_record)
            except Exception as exc:
                logger.exception("worktree setup failed item_id=%s", item_id)
                error = str(exc)
                self.todo_list.update(item_id, "failed", note=error)
                self._notify_task_update_if_unobserved(item_id, action="failed")
                result = SubtaskSummary(
                    task_id=item_id,
                    title=item.title,
                    status="failed",
                    summary="子任务工作区创建失败。",
                    error=error,
                ).to_dict()
                self._notify("subtask_failed", result)
                return result

        self._notify(
            "subtask_started",
            {
                "task_id": item_id,
                "task": item.task,
                "title": item.title,
                "summary": item.summary,
                "status": item.status,
                "dependencies": list(item.dependencies),
                **worktree_data,
                **(
                    {
                        "request_id": self.communication.request_id,
                        "communication_root": self.communication.relative_root,
                    }
                    if self.communication is not None
                    else {}
                ),
            },
        )

        try:
            if self.communication is not None:
                self.communication.write_task_input(
                    item_id,
                    task=item.task,
                    instructions=instructions,
                    summary=item.summary,
                    dependencies=list(item.dependencies),
                    tool_names=tool_names,
                )
                if protocol_request_id is None:
                    protocol_request = self.communication.create_protocol_request(
                        protocol_type="task",
                        sender="main",
                        target=item_id,
                        payload={
                            "task_id": item_id,
                            "title": item.task,
                            "instructions": instructions,
                            "dependencies": list(item.dependencies),
                            "tool_names": list(tool_names or []),
                        },
                        content=instructions,
                    )
                    protocol_request_id = str(
                        protocol_request["protocol_request_id"]
                    )
                    # Route a locally-created task request through the child
                    # inbox before the isolated child starts.
                    self.communication.consume_inbox(item_id)
            response = self._run_child(
                item_id=item_id,
                title=item.title,
                instructions=instructions,
                tool_names=tool_names,
                worktree_path=worktree_path,
                worktree_record=worktree_record,
            )
            summary = self._parse_summary(item_id, item.title, response)
            result_payload = {**summary.to_dict(), **worktree_data}
            if self.communication is not None:
                self.communication.write_result(item_id, result_payload)
                if protocol_request_id is not None:
                    self.communication.respond_protocol(
                        protocol_request_id,
                        responder=protocol_responder,
                        payload=result_payload,
                        content=json.dumps(result_payload, ensure_ascii=False),
                    )
                    self.communication.consume_inbox("main")
            final_status = summary.status
            if final_status == "completed":
                self.todo_list.complete(
                    item_id,
                    summary=summary.summary,
                )
            elif final_status == "blocked":
                self.todo_list.block(
                    item_id,
                    reason=summary.error or summary.summary,
                )
            else:
                self.todo_list.update(
                    item_id,
                    "failed",
                    note=summary.error or summary.summary,
                )
            self._notify_task_update_if_unobserved(
                item_id,
                action=(
                    "complete"
                    if final_status == "completed"
                    else "block"
                    if final_status == "blocked"
                    else "failed"
                ),
            )

            event_type = (
                "subtask_finished"
                if final_status == "completed"
                else "subtask_failed"
            )
            self._notify(event_type, result_payload)
            self._notify_worktree_cleanup_pending(item_id, worktree_data)
            if start_autonomous:
                self.start_autonomous_worker(item_id)
            return result_payload
        except Exception as exc:
            logger.exception("subtask failed item_id=%s", item_id)
            error = str(exc)
            try:
                self.todo_list.update(item_id, "failed", note=error)
                self._notify_task_update_if_unobserved(item_id, action="failed")
            except Exception:
                logger.exception("failed to update subtask status item_id=%s", item_id)
            summary = SubtaskSummary(
                task_id=item_id,
                title=item.title,
                status="failed",
                summary="子任务执行失败。",
                error=error,
            )
            result_payload = {**summary.to_dict(), **worktree_data}
            if self.communication is not None:
                try:
                    self.communication.write_result(item_id, result_payload)
                    if protocol_request_id is not None:
                        self.communication.respond_protocol(
                            protocol_request_id,
                            responder=protocol_responder,
                            payload=result_payload,
                            content=json.dumps(
                                result_payload,
                                ensure_ascii=False,
                            ),
                        )
                        self.communication.consume_inbox("main")
                except Exception:
                    logger.exception(
                        "failed to persist failed subtask result item_id=%s",
                        item_id,
                    )
            self._notify("subtask_failed", result_payload)
            self._notify_worktree_cleanup_pending(item_id, worktree_data)
            if start_autonomous:
                self.start_autonomous_worker(item_id)
            return result_payload

    def delegate_many(
        self,
        requests: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run ready independent requests concurrently; dependent work waits."""
        requests = list(requests)
        if not requests:
            return []

        results: list[dict[str, Any]] = []
        remaining = requests[:]
        while remaining:
            ready: list[dict[str, Any]] = []
            waiting: list[dict[str, Any]] = []
            for request in remaining:
                item_id = str(
                    request.get("task_id", request.get("item_id", ""))
                )
                try:
                    if self.todo_list.is_ready(item_id):
                        ready.append(request)
                    else:
                        waiting.append(request)
                except Exception as exc:
                    ready.append({**request, "_validation_error": str(exc)})

            if not ready:
                for request in waiting:
                    results.append(
                        self.delegate(
                            str(
                                request.get(
                                    "task_id",
                                    request.get("item_id", ""),
                                )
                            ),
                            str(request.get("instructions", "")),
                            request.get("tool_names"),
                        )
                    )
                break

            with ThreadPoolExecutor(max_workers=len(ready)) as executor:
                futures = [
                    executor.submit(
                        self._delegate_request,
                        request,
                    )
                    for request in ready
                ]
                results.extend(future.result() for future in futures)
            remaining = waiting

        return results

    def start_autonomous_worker(
        self,
        agent_name: str,
        *,
        poll_interval: float | None = None,
        idle_timeout: float | None = None,
    ) -> dict[str, Any]:
        """Keep one completed child alive so it can claim later work."""
        if not self.autonomous_enabled:
            return {
                "started": False,
                "agent_name": agent_name,
                "reason": "autonomous workers are disabled",
            }
        if self.communication is None:
            return {
                "started": False,
                "agent_name": agent_name,
                "reason": "file communication is required",
            }
        agent_name = agent_name.strip()
        if not agent_name:
            raise ValueError("autonomous agent name cannot be empty")
        with self._autonomous_lock:
            worker = self._autonomous_workers.get(agent_name)
            if worker is not None and worker.is_alive:
                return {
                    "started": False,
                    "agent_name": agent_name,
                    "reason": "worker already running",
                }
            worker = AutonomousTaskWorker(
                self,
                agent_name,
                poll_interval=(
                    self.autonomous_poll_interval
                    if poll_interval is None
                    else poll_interval
                ),
                idle_timeout=(
                    self.autonomous_idle_timeout
                    if idle_timeout is None
                    else idle_timeout
                ),
            )
            self._autonomous_workers[agent_name] = worker
            worker.start()
            return {
                "started": True,
                "agent_name": agent_name,
                "poll_interval": worker.poll_interval,
                "idle_timeout": worker.idle_timeout,
            }

    def stop_autonomous_workers(self, *, wait: bool = True) -> None:
        with self._autonomous_lock:
            workers = list(self._autonomous_workers.values())
        for worker in workers:
            worker.stop()
        if wait:
            for worker in workers:
                worker.join(timeout=worker.idle_timeout + worker.poll_interval)

    def _delegate_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("_validation_error"):
            item_id = str(
                request.get("task_id", request.get("item_id", ""))
            )
            item = self.todo_list.get(item_id)
            error = str(request["_validation_error"])
            self.todo_list.update(item_id, "failed", note=error)
            self._notify_task_update_if_unobserved(item_id, action="failed")
            summary = SubtaskSummary(
                task_id=item_id,
                title=item.title,
                status="failed",
                summary="子任务参数无效。",
                error=error,
            )
            self._notify("subtask_failed", summary.to_dict())
            return summary.to_dict()
        return self.delegate(
            str(request.get("task_id", request.get("item_id", ""))),
            str(request.get("instructions", "")),
            request.get("tool_names"),
        )

    def _run_child(
        self,
        *,
        item_id: str,
        title: str,
        instructions: str,
        tool_names: list[str] | None,
        worktree_path: Path | None = None,
        worktree_record: dict[str, Any] | None = None,
    ) -> ModelResponse:
        registry = ToolRegistry()
        communication_toolset = None
        if self.communication is not None:
            communication_toolset = SubagentCommunicationToolset(
                self.communication,
                item_id,
            )
        registered_specs = self.host_agent.tool_registry.list_specs()
        communication_specs = (
            communication_toolset.specs
            if communication_toolset is not None
            else []
        )
        available_specs = [*registered_specs, *communication_specs]
        if tool_names is None:
            allowed_names = {
                spec.name
                for spec in available_specs
                if spec.name not in _CHILD_EXCLUDED_TOOLS
            }
        else:
            # Communication is part of the child runtime contract even when
            # the caller narrows the task's business tools explicitly.
            allowed_names = set(tool_names) | {
                spec.name for spec in communication_specs
            }
            excluded_names = allowed_names & _CHILD_EXCLUDED_TOOLS
            if excluded_names:
                raise ValueError(
                    "sub-agents cannot use Todo tools: "
                    f"{sorted(excluded_names)}"
                )
        specs_by_name = {
            spec.name: spec
            for spec in available_specs
        }
        specs = [
            spec
            for name, spec in specs_by_name.items()
            if name in allowed_names
        ]
        unknown_names = allowed_names - set(specs_by_name)
        if unknown_names:
            raise ValueError(f"unknown sub-agent tools: {sorted(unknown_names)}")
        for spec in specs:
            handler = (
                communication_toolset.handlers.get(spec.name)
                if communication_toolset is not None
                and spec.name in communication_toolset.handlers
                else self.host_agent.tool_registry.get_handler(spec.name)
            )
            if worktree_path is not None:
                handler = self._workspace_handler_for_child(
                    spec.name,
                    handler,
                    worktree_path,
                )
            if handler is not None:
                registry.register(spec, handler)

        if self.child_agent_factory is None:
            from core.agent import Agent

            child_agent_factory = Agent
        else:
            child_agent_factory = self.child_agent_factory

        child_kwargs: dict[str, Any] = {
            "base_url": self.host_agent.base_url,
            "api_key": self.host_agent.api_key,
            "model_id": self.host_agent.model_id,
            "timeout": self.host_agent.timeout,
            "max_retries": self.host_agent.max_retries,
            "default_headers": self.host_agent.default_headers,
            "max_tokens": self.host_agent.max_tokens,
            "client": self.host_agent.client,
            "providers": self.host_agent.llm_manager.providers,
            "tool_registry": registry,
            "enable_user_profile": False,
            "enable_background": False,
            "prompt_dir": self.host_agent.prompt_manager.prompt_dir,
            "tool_hooks": list(self.host_agent.tool_manager.hooks),
        }
        startup_dir = getattr(self.host_agent, "startup_dir", None)
        skills_dir = getattr(self.host_agent, "skills_dir", None)
        workspace_root = worktree_path or getattr(
            self.host_agent,
            "workspace_root",
            None,
        )
        if startup_dir is not None:
            child_kwargs["startup_dir"] = startup_dir
        if skills_dir is not None:
            child_kwargs["skills_dir"] = skills_dir
        if workspace_root is not None:
            child_kwargs["workspace_root"] = workspace_root

        child = child_agent_factory(
            **child_kwargs,
        )
        child_skill_toolset = getattr(child, "skill_toolset", None)
        child_specs = list(specs)
        child_handlers = registry.handlers()
        if child_skill_toolset is not None:
            child_specs.extend(child_skill_toolset.specs)
            child_handlers.update(child_skill_toolset.handlers)
        child_skills_prompt = getattr(
            child,
            "skills_prompt",
            getattr(self.host_agent, "skills_prompt", "（未发现本地技能）"),
        )
        base_prompt_builder = getattr(child, "_build_chat_system_prompt", None)
        if callable(base_prompt_builder):
            child_prompt = base_prompt_builder(enabled_tools=child_specs)
        else:
            child_prompt = (
                "你是一个独立子任务执行 agent。你没有主 agent 的历史消息，"
                "也不知道主任务的其它内容。只处理下面给出的子任务。需要工具时"
                "必须调用提供的工具。"
            )
            child_prompt += "\n\n## 本地技能\n" + child_skills_prompt

        if worktree_path is not None:
            branch = str((worktree_record or {}).get("branch", ""))
            child_prompt += (
                "\n\n## 当前任务 Git 工作区\n"
                f"工作区：{worktree_path}\n"
                f"Git 分支：{branch or 'unknown'}\n"
                "所有文件、Shell 和 Python 操作必须限制在此工作区内，"
                "不得访问父工作区或其他 worktree。"
            )

        if self.communication is not None:
            child_prompt += (
                "\n\n"
                + self.communication.prompt_context(item_id)
                + "\npredecessor_task_ids: "
                + ", ".join(
                    self.todo_list.get(item_id).dependencies
                )
                + "\n"
                "Before completing, read predecessor task results when this "
                "task has dependencies. Write substantial outputs with "
                "subagent_write_artifact and report its returned path in the "
                "JSON summary. Use subagent_request_protocol for any "
                "structured request and subagent_respond_protocol for its "
                "response; never invent a request_id."
            )

        child_prompt += (
            "\n\n# 子任务输出协议\n"
            "你只处理下面给出的子任务。完成后只输出 JSON 对象，不要输出 Markdown："
            '{"task_id":"%s","task":"%s","title":"%s",'
            '"status":"completed|blocked|failed",'
            '"summary":"简短处理摘要","artifacts":["可选产物"],"error":"可选错误"}'
            % (item_id, title, title)
        )
        options = InvokeOptions(
            tools=child_specs or None,
            tool_handlers=child_handlers or None,
            max_tool_rounds=5,
            temperature=0,
            response_format={"type": "json_object"},
        )
        if child_specs:
            return child.tool_manager.chat_with_tools(
                prompt=child_prompt,
                user_message=instructions,
                options=options,
                progress_callback=self._child_progress(item_id),
            )
        return child.llm_manager.invoke_messages(
            [
                {"role": "system", "content": child_prompt},
                {"role": "user", "content": instructions},
            ],
            options=options,
            purpose="chat",
        )

    @staticmethod
    def _workspace_handler_for_child(
        name: str,
        handler: Callable[..., Any] | None,
        worktree_path: Path,
    ) -> Callable[..., Any] | None:
        if not name.startswith("workspace_") or handler is None:
            return handler
        owner = getattr(handler, "__self__", None)
        clone = getattr(owner, "for_workspace", None)
        if not callable(clone):
            return handler
        child_toolset = clone(worktree_path)
        return {
            "workspace_list_files": child_toolset.list_files,
            "workspace_read_file": child_toolset.read_file,
            "workspace_write_file": child_toolset.write_file,
            "workspace_replace_text": child_toolset.replace_text,
            "workspace_run_shell": child_toolset.run_shell,
            "workspace_run_python": child_toolset.run_python,
        }.get(name, handler)

    @staticmethod
    def _worktree_data(
        record: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not record:
            return {}
        return {
            "worktree": record.get("name"),
            "worktree_path": record.get("path"),
            "worktree_branch": record.get("branch"),
        }

    def _notify_worktree_cleanup_pending(
        self,
        task_id: str,
        worktree_data: dict[str, Any],
    ) -> None:
        if self.worktree_manager is None or not worktree_data.get("worktree"):
            return
        try:
            inspection = self.worktree_manager.inspect_worktree(
                str(worktree_data["worktree"])
            )
        except Exception as exc:
            self._notify(
                "worktree_failed",
                {
                    "task_id": task_id,
                    **worktree_data,
                    "error": str(exc),
                },
            )
            return
        self._notify(
            "worktree_cleanup_pending",
            {
                "task_id": task_id,
                **worktree_data,
                "dirty": inspection.get("dirty", False),
                "changed_files": inspection.get("changed_files", 0),
                "commits": inspection.get("commits", 0),
                "actions": ["keep_worktree", "remove_worktree"],
                "cleanup_required": True,
            },
        )

    def _child_progress(
        self,
        item_id: str,
    ) -> Callable[[str, dict[str, Any]], None]:
        def callback(event_type: str, data: dict[str, Any]) -> None:
            self._notify(
                event_type,
                {"subtask_id": item_id, **data},
            )

        return callback

    def _parse_summary(
        self,
        item_id: str,
        title: str,
        response: ModelResponse,
    ) -> SubtaskSummary:
        try:
            value = self._parse_json_object(response.text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("sub-agent did not return valid JSON summary") from exc
        if not isinstance(value, dict):
            raise ValueError("sub-agent summary must be a JSON object")

        status = value.get("status")
        if status not in {"completed", "blocked", "failed"}:
            raise ValueError("sub-agent summary has an invalid status")
        summary = str(value.get("summary", "")).strip()
        if not summary:
            raise ValueError("sub-agent summary is empty")
        artifacts = value.get("artifacts")
        normalized_artifacts = (
            [str(artifact) for artifact in artifacts if artifact]
            if isinstance(artifacts, list)
            else None
        )
        error = str(value.get("error", "")).strip() or None
        return SubtaskSummary(
            task_id=item_id,
            title=title,
            status=status,
            summary=summary,
            artifacts=normalized_artifacts,
            error=error,
        )

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        value = json.loads(cleaned)
        if not isinstance(value, dict):
            raise json.JSONDecodeError(
                "summary must be an object",
                cleaned,
                0,
            )
        return value

    def _notify(self, event_type: str, data: dict[str, Any]) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event_type, data)

    def _notify_task_update_if_unobserved(
        self,
        task_id: str,
        *,
        action: str,
    ) -> None:
        if self.todo_list.observer is not None:
            return
        task = self.todo_list.get(task_id)
        self._notify(
            "todo_updated",
            {
                **self.todo_list.snapshot(),
                "action": action,
                "updated_task_id": task_id,
                "updated_item_id": task_id,
                "updated_task": task.to_dict(),
            },
        )

    def _finish_without_child(
        self,
        item: Any,
        *,
        status: str,
        summary: str,
        error: str,
    ) -> dict[str, Any]:
        self.todo_list.update(item.item_id, status, note=error)
        self._notify_task_update_if_unobserved(item.item_id, action=status)
        result = SubtaskSummary(
            task_id=item.item_id,
            title=item.title,
            status=status,
            summary=summary,
            error=error,
        ).to_dict()
        self._notify("subtask_failed", result)
        return result
