"""RAG application layer.

The package intentionally keeps the orchestration layer independent from
MySQL, Elasticsearch, Milvus, and the embedding provider. Concrete adapters
can be added later without changing the public service API.
"""

from rag_server.bootstrap import (
    build_in_memory_service,
    build_rag_service,
)
from rag_server.config import RagSettings
from rag_server.schemas import (
    ChunkRecord,
    DocumentRecord,
    IngestionResult,
    SearchHit,
    SearchResponse,
)
from rag_server.prompt import build_context, build_grounded_prompt
from rag_server.service import RagService

__all__ = [
    "ChunkRecord",
    "build_context",
    "build_grounded_prompt",
    "build_in_memory_service",
    "build_rag_service",
    "DocumentRecord",
    "IngestionResult",
    "RagService",
    "RagSettings",
    "SearchHit",
    "SearchResponse",
]
