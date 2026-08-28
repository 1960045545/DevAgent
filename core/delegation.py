from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable

from core.invoke_options import InvokeOptions
from core.response import ModelResponse
from core.tool_registry import ToolRegistry
from core.todo import TodoList

if TYPE_CHECKING:
    from core.agent import Agent


logger = logging.getLogger(__name__)
_CHILD_EXCLUDED_TOOLS = {
    "todo_create",
    "todo_update",
    "todo_list",
    "todo_delegate",
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
            "title": self.title,
            "status": self.status,
            "summary": self.summary,
        }
        if self.artifacts:
            result["artifacts"] = list(self.artifacts)
        if self.error:
            result["error"] = self.error
        return result


class SubtaskExecutor:
    """Runs isolated child agents and returns summaries, never their history."""

    def __init__(
        self,
        host_agent: Agent,
        todo_list: TodoList,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        child_agent_factory: Callable[..., Agent] | None = None,
    ) -> None:
        self.host_agent = host_agent
        self.todo_list = todo_list
        self.progress_callback = progress_callback
        self.child_agent_factory = child_agent_factory

    def delegate(
        self,
        item_id: str,
        instructions: str,
        tool_names: list[str] | None = None,
    ) -> dict[str, Any]:
        item = self.todo_list.get(item_id)
        if not instructions.strip():
            return self._finish_without_child(
                item,
                status="failed",
                summary="子任务说明为空。",
                error="subtask instructions cannot be empty",
            )

        try:
            self.todo_list.claim(
                item_id,
                note="delegated to an isolated sub-agent",
            )
        except ValueError as exc:
            current_status = self.todo_list.get(item_id).status
            if current_status in {"pending", "in_progress"}:
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
        self._notify(
            "subtask_started",
            {
                "task_id": item_id,
                "title": item.title,
            },
        )

        try:
            response = self._run_child(
                item_id=item_id,
                title=item.title,
                instructions=instructions,
                tool_names=tool_names,
            )
            summary = self._parse_summary(item_id, item.title, response)
            final_status = summary.status
            if final_status == "completed":
                self.todo_list.update(
                    item_id,
                    "completed",
                    note=summary.summary,
                )
            elif final_status == "blocked":
                self.todo_list.update(
                    item_id,
                    "blocked",
                    note=summary.error or summary.summary,
                )
            else:
                self.todo_list.update(
                    item_id,
                    "failed",
                    note=summary.error or summary.summary,
                )

            event_type = (
                "subtask_finished"
                if final_status == "completed"
                else "subtask_failed"
            )
            self._notify(event_type, summary.to_dict())
            return summary.to_dict()
        except Exception as exc:
            logger.exception("subtask failed item_id=%s", item_id)
            error = str(exc)
            try:
                self.todo_list.update(item_id, "failed", note=error)
            except Exception:
                logger.exception("failed to update subtask status item_id=%s", item_id)
            summary = SubtaskSummary(
                task_id=item_id,
                title=item.title,
                status="failed",
                summary="子任务执行失败。",
                error=error,
            )
            self._notify("subtask_failed", summary.to_dict())
            return summary.to_dict()

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
                item_id = str(request.get("item_id", ""))
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
                            str(request.get("item_id", "")),
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

    def _delegate_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("_validation_error"):
            item_id = str(request.get("item_id", ""))
            item = self.todo_list.get(item_id)
            error = str(request["_validation_error"])
            self.todo_list.update(item_id, "failed", note=error)
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
            str(request.get("item_id", "")),
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
    ) -> ModelResponse:
        registry = ToolRegistry()
        registered_specs = self.host_agent.tool_registry.list_specs()
        if tool_names is None:
            allowed_names = {
                spec.name
                for spec in registered_specs
                if spec.name not in _CHILD_EXCLUDED_TOOLS
            }
        else:
            allowed_names = set(tool_names)
            excluded_names = allowed_names & _CHILD_EXCLUDED_TOOLS
            if excluded_names:
                raise ValueError(
                    "sub-agents cannot use Todo tools: "
                    f"{sorted(excluded_names)}"
                )
        specs = [spec for spec in registered_specs if spec.name in allowed_names]
        unknown_names = allowed_names - {spec.name for spec in specs}
        if unknown_names:
            raise ValueError(f"unknown sub-agent tools: {sorted(unknown_names)}")
        for spec in specs:
            handler = self.host_agent.tool_registry.get_handler(spec.name)
            if handler is not None:
                registry.register(spec, handler)

        from core.agent import Agent

        child_agent_factory = self.child_agent_factory or Agent

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
            "prompt_dir": self.host_agent.prompt_manager.prompt_dir,
            "tool_hooks": list(self.host_agent.tool_manager.hooks),
        }
        startup_dir = getattr(self.host_agent, "startup_dir", None)
        skills_dir = getattr(self.host_agent, "skills_dir", None)
        if startup_dir is not None:
            child_kwargs["startup_dir"] = startup_dir
        if skills_dir is not None:
            child_kwargs["skills_dir"] = skills_dir

        child = child_agent_factory(
            **child_kwargs,
        )
        child_skills_prompt = getattr(
            child,
            "skills_prompt",
            getattr(self.host_agent, "skills_prompt", "（未发现本地技能）"),
        )
        child_prompt = (
            "你是一个独立子任务执行 agent。你没有主 agent 的历史消息，"
            "也不知道主任务的其它内容。只处理下面给出的子任务。需要工具时"
            "必须调用提供的工具。完成后只输出 JSON 对象，不要输出 Markdown："
            '{"task_id":"%s","title":"%s","status":"completed|blocked|failed",'
            '"summary":"简短处理摘要","artifacts":["可选产物"],"error":"可选错误"}'
            % (item_id, title)
            + "\n\n## 本地技能\n"
            + child_skills_prompt
        )
        options = InvokeOptions(
            tools=specs or None,
            tool_handlers=registry.handlers() or None,
            max_tool_rounds=5,
            temperature=0,
            response_format={"type": "json_object"},
        )
        if specs:
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

    def _finish_without_child(
        self,
        item: Any,
        *,
        status: str,
        summary: str,
        error: str,
    ) -> dict[str, Any]:
        self.todo_list.update(item.item_id, status, note=error)
        result = SubtaskSummary(
            task_id=item.item_id,
            title=item.title,
            status=status,
            summary=summary,
            error=error,
        ).to_dict()
        self._notify("subtask_failed", result)
        return result
