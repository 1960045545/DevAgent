from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

from openai import OpenAI

from core.invoke_options import InvokeOptions
from core.delegation import SubtaskExecutor
from core.message import Message
from core.response import ModelResponse
from core.skills import Skill, SkillLoader, SkillToolset
from core.tool_registry import ToolRegistry
from core.tool_hooks import ToolHook
from core.tool_space import ToolSpec
from core.todo import TodoToolset, is_complex_task
from manager.llm_manager import LLMManager
from manager.memory_manager import MemoryManager
from manager.model_provider_manager import ModelProviderConfig
from manager.prompt_manager import PromptManager
from manager.tool_manager import ToolManager
from manager.user_profile_manager import UserProfileManager


logger = logging.getLogger(__name__)


class Agent:
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
        retry_backoff: float = 1.5,
        default_headers: dict[str, str] | None = None,
        max_tokens: int = 1000,
        client: OpenAI | None = None,
        providers: list[ModelProviderConfig] | None = None,
        tool_registry: ToolRegistry | None = None,
        user_profile_path: str | Path | None = None,
        user_profile_max_items: int = 20,
        enable_user_profile: bool = True,
        user_profile_model_id: str | None = None,
        history_abstract_model_id: str | None = None,
        prompt_dir: str | Path | None = None,
        tool_hooks: list[ToolHook] | None = None,
        startup_dir: str | Path | None = None,
        skills_dir: str | Path | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.retry_backoff = retry_backoff
        self.startup_dir = (
            Path(startup_dir).expanduser().resolve()
            if startup_dir is not None
            else Path.cwd().resolve()
        )
        self.skills_dir = (
            Path(skills_dir).expanduser().resolve()
            if skills_dir is not None
            else self.startup_dir / "skills"
        )
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve()
            if workspace_root is not None
            else Path(
                os.getenv("AGENT_WORKSPACE_ROOT", str(self.startup_dir))
            ).expanduser().resolve()
        )
        self.skill_loader = SkillLoader(self.skills_dir)
        self.skills: tuple[Skill, ...] = self.skill_loader.load()
        self.skill_toolset = SkillToolset(self.skill_loader)
        self.skills_prompt = SkillLoader.format_for_prompt(self.skills)
        self.llm_manager = LLMManager(
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            timeout=timeout,
            max_retries=max_retries,
            retry_interval=retry_interval,
            cooldown_seconds=cooldown_seconds,
            default_headers=default_headers,
            client=client,
            providers=providers,
        )
        self.prompt_manager = PromptManager(prompt_dir)
        self.memory_manager = MemoryManager(
            max_tokens=max_tokens,
            rounds=3,
            prompt_manager=self.prompt_manager,
            llm_manager=self.llm_manager,
            history_abstract_model_id=history_abstract_model_id,
            transcripts_dir=self.startup_dir / ".transcripts",
            static_context=self.skills_prompt,
        )
        self.llm_manager.set_context_recovery_callbacks(
            context_compactor=self.memory_manager.compress_if_needed,
            context_transcript_writer=self.memory_manager.write_runtime_transcript,
        )
        self.profile_manager = UserProfileManager(
            prompt_manager=self.prompt_manager,
            llm_manager=self.llm_manager,
            model_id=user_profile_model_id,
            profile_path=user_profile_path,
            max_items=user_profile_max_items,
            enabled=enable_user_profile,
        )
        self.tool_manager = ToolManager(
            llm_manager=self.llm_manager,
            tool_registry=tool_registry,
            hooks=tool_hooks,
        )

        self.base_url = self.llm_manager.base_url
        self.api_key = self.llm_manager.api_key
        self.model_id = self.llm_manager.model_id
        self.timeout = self.llm_manager.timeout
        self.max_retries = self.llm_manager.max_retries
        self.default_headers = self.llm_manager.default_headers
        self.client = self.llm_manager.client
        self.tool_registry = self.tool_manager.tool_registry
        logger.info(
            "agent initialized model=%s providers=%s skills=%s",
            self.model_id,
            [provider.name for provider in (providers or [])] or ["default"],
            [skill.name for skill in self.skills],
        )

    @property
    def history(self) -> list[Message]:
        return self.memory_manager.history

    @history.setter
    def history(self, value: list[Message]) -> None:
        self.memory_manager.history = value

    @property
    def max_tokens(self) -> int:
        return self.memory_manager.max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self.memory_manager.max_tokens = value

    @property
    def rounds(self) -> int:
        return self.memory_manager.rounds

    @rounds.setter
    def rounds(self, value: int) -> None:
        self.memory_manager.rounds = value

    @property
    def enable_user_profile(self) -> bool:
        return self.profile_manager.enabled

    @enable_user_profile.setter
    def enable_user_profile(self, value: bool) -> None:
        self.profile_manager.enabled = value

    @property
    def user_profile(self):
        return self.profile_manager.profile

    def register_tool(
        self,
        spec: ToolSpec,
        handler: Callable[..., Any],
    ) -> None:
        self.tool_manager.register_tool(
            spec,
            handler,
        )

    def register_tool_hook(self, hook: ToolHook) -> None:
        self.tool_manager.register_hook(hook)

    def unregister_tool_hook(self, hook: ToolHook) -> None:
        self.tool_manager.unregister_hook(hook)

    def record_skill_result(
        self,
        skill_name: str,
        result: Any,
        *,
        arguments: Any = None,
    ) -> None:
        """Record a skill executor result for bounded prompt retention."""
        self.memory_manager.append_skill_result(
            skill_name,
            result,
            arguments=arguments,
        )

    async def ainvoke(
        self,
        message: Message,
        options: InvokeOptions | None = None,
        model_id: str | None = None,
    ) -> ModelResponse:
        return await self.llm_manager.ainvoke(
            message,
            options=options,
            model_id=model_id,
        )

    def chat(
        self,
        user_message: str,
        options: InvokeOptions | None = None,
        *,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> ModelResponse:
        logger.info("chat start")
        self.memory_manager.compress_if_needed()
        request_options = options or InvokeOptions()
        todo_toolset = self._build_todo_toolset(
            user_message,
            progress_callback,
        )
        request_options = self._merge_runtime_tools(
            request_options,
            todo_toolset,
        )
        system_prompt = self._build_chat_system_prompt(
            enabled_tools=request_options.tools,
            todo_toolset=todo_toolset,
        )
        request_options = replace(
            request_options,
            context_recovery_callback=lambda messages: self._refresh_system_message(
                messages,
                lambda: self._build_chat_system_prompt(
                    enabled_tools=request_options.tools,
                    todo_toolset=todo_toolset,
                ),
            ),
        )

        if request_options.tools:
            logger.info("chat route=tool")
            response = self.tool_manager.chat_with_tools(
                prompt=system_prompt,
                user_message=user_message,
                options=request_options,
                todo_list=(
                    todo_toolset.todo_list
                    if todo_toolset is not None
                    else None
                ),
                progress_callback=progress_callback,
                tool_result_callback=self._record_tool_result,
                prompt_builder=lambda: self._build_chat_system_prompt(
                    enabled_tools=request_options.tools,
                    todo_toolset=todo_toolset,
                ),
            )
        else:
            logger.info("chat route=normal")
            response = self.llm_manager.invoke_messages(
                [
                    Message(
                        role="system",
                        content=system_prompt,
                    ).to_dict(),
                    Message(role="user", content=user_message).to_dict(),
                ],
                options=request_options,
                purpose="chat",
            )

        self._finish_chat_turn(
            user_message=user_message,
            assistant_message=response.text,
        )
        return response

    def stream_chat(
        self,
        user_message: str,
        options: InvokeOptions | None = None,
        *,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> Iterator[str]:
        logger.info("stream_chat start")
        self.memory_manager.compress_if_needed()
        request_options = options or InvokeOptions()
        todo_toolset = self._build_todo_toolset(
            user_message,
            progress_callback,
        )
        request_options = self._merge_runtime_tools(
            request_options,
            todo_toolset,
        )
        system_prompt = self._build_chat_system_prompt(
            enabled_tools=request_options.tools,
            todo_toolset=todo_toolset,
        )
        request_options = replace(
            request_options,
            context_recovery_callback=lambda messages: self._refresh_system_message(
                messages,
                lambda: self._build_chat_system_prompt(
                    enabled_tools=request_options.tools,
                    todo_toolset=todo_toolset,
                ),
            ),
        )

        if request_options.tools:
            logger.info("stream_chat route=tool")
            response = self.tool_manager.chat_with_tools(
                prompt=system_prompt,
                user_message=user_message,
                options=request_options,
                todo_list=(
                    todo_toolset.todo_list
                    if todo_toolset is not None
                    else None
                ),
                progress_callback=progress_callback,
                tool_result_callback=self._record_tool_result,
                prompt_builder=lambda: self._build_chat_system_prompt(
                    enabled_tools=request_options.tools,
                    todo_toolset=todo_toolset,
                ),
            )
            self._finish_chat_turn(
                user_message=user_message,
                assistant_message=response.text,
            )
            yield response.text
            return

        chunks: list[str] = []

        logger.info("stream_chat route=normal")
        for chunk in self.llm_manager.stream_messages(
            [
                Message(
                    role="system",
                    content=system_prompt,
                ).to_dict(),
                Message(role="user", content=user_message).to_dict(),
            ],
            options=request_options,
            purpose="chat",
        ):
            chunks.append(chunk)
            yield chunk

        self._finish_chat_turn(
            user_message=user_message,
            assistant_message="".join(chunks),
        )

    def _build_chat_system_prompt(
        self,
        *,
        enabled_tools: list[ToolSpec] | None = None,
        todo_toolset: TodoToolset | None = None,
    ) -> str:
        history_text, recent_text = (
            self.memory_manager.get_prompt_texts()
        )
        return self.prompt_manager.build_chat_system_prompt(
            history=history_text,
            recent_chat_record=recent_text,
            user_profile=self.profile_manager.format(),
            enabled_tools=enabled_tools,
            workspace=self.workspace_root,
            skills_catalog=self.skills_prompt if self.skills else "",
            operation_history=self.memory_manager.get_operation_prompt_text(),
            todo_enabled=todo_toolset is not None,
            delegation_enabled=(
                todo_toolset is not None
                and todo_toolset.delegate_handler is not None
            ),
            todo_state=(
                todo_toolset.todo_list.snapshot()
                if todo_toolset is not None
                else None
            ),
        )

    def _merge_runtime_tools(
        self,
        options: InvokeOptions,
        todo_toolset: TodoToolset | None,
    ) -> InvokeOptions:
        extra_specs = list(self.skill_toolset.specs)
        extra_handlers = dict(self.skill_toolset.handlers)
        if todo_toolset is not None:
            extra_specs.extend(todo_toolset.specs)
            extra_handlers.update(todo_toolset.handlers)
        return self.tool_manager.merge_options(
            options,
            extra_specs=extra_specs or None,
            extra_handlers=extra_handlers or None,
        )

    @staticmethod
    def _refresh_system_message(
        messages: list[dict[str, Any]],
        prompt_builder: Callable[[], str],
    ) -> None:
        for message in messages:
            if message.get("role") == "system":
                message["content"] = prompt_builder()
                return

    def _record_tool_result(
        self,
        tool_call: Any,
        result: str,
        round_index: int,
    ) -> None:
        if getattr(tool_call, "name", "") == "load_skill":
            self.memory_manager.append_skill_result(
                "load_skill",
                result,
                arguments=getattr(tool_call, "arguments", None),
                call_id=getattr(tool_call, "id", None),
                round_index=round_index,
            )
            return
        self.memory_manager.append_tool_result(
            tool_call,
            result,
            round_index,
        )

    def _build_todo_toolset(
        self,
        user_message: str,
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> TodoToolset | None:
        if not is_complex_task(user_message):
            return None

        todo_toolset = TodoToolset(observer=progress_callback)
        executor = SubtaskExecutor(
            self,
            todo_toolset.todo_list,
            progress_callback=progress_callback,
        )
        todo_toolset.delegate_handler = executor.delegate
        return todo_toolset

    def _build_chat_prompt(self, user_message: str) -> str:
        todo_toolset = self._build_todo_toolset(user_message, None)
        request_options = self._merge_runtime_tools(
            InvokeOptions(),
            todo_toolset,
        )
        system_prompt = self._build_chat_system_prompt(
            enabled_tools=request_options.tools,
            todo_toolset=todo_toolset,
        )
        return f"{system_prompt}\n\n用户最新问题：\n{user_message}"

    def _finish_chat_turn(
        self,
        *,
        user_message: str,
        assistant_message: str,
    ) -> None:
        logger.debug("finish chat turn")
        self.memory_manager.append_chat(
            user_message=user_message,
            assistant_message=assistant_message,
        )
        self.profile_manager.update(
            user_message=user_message,
            assistant_message=assistant_message,
        )
