from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from manager.llm_manager import LLMManager
    from manager.log_manager import configure_logging
    from manager.memory_manager import MemoryManager
    from manager.model_provider_manager import ModelProviderConfig
    from manager.prompt_manager import PromptManager
    from manager.tool_manager import ToolManager
    from manager.user_profile_manager import UserProfileManager

__all__ = [
    "configure_logging",
    "LLMManager",
    "MemoryManager",
    "ModelProviderConfig",
    "PromptManager",
    "ToolManager",
    "UserProfileManager",
]


_EXPORT_MODULES = {
    "configure_logging": "manager.log_manager",
    "LLMManager": "manager.llm_manager",
    "MemoryManager": "manager.memory_manager",
    "ModelProviderConfig": "manager.model_provider_manager",
    "PromptManager": "manager.prompt_manager",
    "ToolManager": "manager.tool_manager",
    "UserProfileManager": "manager.user_profile_manager",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'manager' has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
