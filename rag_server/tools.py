from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.tool_registry import ToolRegistry
from core.tool_space import ToolSpec
from rag_server.service import RagService


KNOWLEDGE_SEARCH_SPEC = ToolSpec(
    name="knowledge_search",
    description=(
        "Search the knowledge base and return evidence passages. "
        "Use the returned passages as context; do not invent missing facts."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's knowledge-base question.",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 3,
            },
            "tenant_id": {
                "type": ["string", "null"],
                "description": "Tenant scope used for access filtering.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    strict=False,
    category="retrieval",
)


def make_knowledge_search_handler(
    service: RagService,
) -> Callable[..., dict[str, Any]]:
    def knowledge_search(
        query: str,
        top_k: int = 3,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        filters = None if tenant_id is None else {"tenant_id": tenant_id}
        return service.search(
            query=query,
            top_k=top_k,
            filters=filters,
        ).to_dict()

    return knowledge_search


def register_rag_tools(
    registry: ToolRegistry,
    service: RagService,
) -> None:
    registry.register(
        KNOWLEDGE_SEARCH_SPEC,
        make_knowledge_search_handler(service),
    )
