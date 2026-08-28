from __future__ import annotations

from rag_server.adapters import (
    HashEmbeddingProvider,
    InMemoryDocumentRepository,
    InMemoryKeywordIndex,
    InMemoryVectorIndex,
)
from rag_server.adapters.elasticsearch_index import (
    ElasticsearchKeywordIndex,
)
from rag_server.adapters.embedding import QwenEmbeddingProvider
from rag_server.adapters.milvus_index import MilvusVectorIndex
from rag_server.adapters.mysql_repository import MySQLDocumentRepository
from rag_server.adapters.qwen_reranker import Qwen3Reranker
from rag_server.config import RagSettings
from rag_server.service import RagService


def build_in_memory_service(
    settings: RagSettings | None = None,
) -> RagService:
    """Build a dependency-free service for smoke tests and examples."""
    resolved_settings = settings or RagSettings(
        embedding_dimension=64,
    )
    embedder = HashEmbeddingProvider(
        dimension=resolved_settings.embedding_dimension or 64,
    )
    return RagService(
        settings=resolved_settings,
        embedder=embedder,
        document_repository=InMemoryDocumentRepository(),
        keyword_index=InMemoryKeywordIndex(),
        vector_index=InMemoryVectorIndex(),
    )


def build_rag_service(
    settings: RagSettings | None = None,
) -> RagService:
    """Build the production MySQL + Elasticsearch + Milvus RAG service."""
    resolved_settings = settings or RagSettings.from_env()
    if not resolved_settings.embedding_model:
        raise ValueError("RAG_EMBEDDING_MODEL must be configured")

    embedder = QwenEmbeddingProvider(
        model_name=resolved_settings.embedding_model,
        device=resolved_settings.embedding_device,
        batch_size=resolved_settings.embedding_batch_size,
        max_length=resolved_settings.embedding_max_length,
        dimension=resolved_settings.embedding_dimension,
    )
    dimension = (
        resolved_settings.embedding_dimension
        or embedder.dimension
    )
    return RagService(
        settings=resolved_settings,
        embedder=embedder,
        document_repository=MySQLDocumentRepository(resolved_settings),
        keyword_index=ElasticsearchKeywordIndex(resolved_settings),
        vector_index=MilvusVectorIndex(
            resolved_settings,
            dimension=dimension,
        ),
        reranker=Qwen3Reranker(resolved_settings),
    )
