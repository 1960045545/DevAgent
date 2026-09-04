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
from rag_server.chunker import ChunkedText, MarkdownChunker, TextChunker
from rag_server.evaluation import (
    CaseEvaluation,
    EvaluationCase,
    EvaluationReport,
    evaluate_retrieval,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_server.loaders import DocumentLoader, LocalDocumentLoader
from rag_server.query import DefaultQueryRouter, QueryPlanner
from rag_server.schemas import (
    ChunkRecord,
    BatchIngestionResult,
    DocumentRecord,
    IngestionResult,
    QueryPlan,
    RetrievalTrace,
    SearchHit,
    SearchResponse,
)
from rag_server.prompt import (
    build_context,
    build_grounded_messages,
    build_grounded_prompt,
)
from rag_server.service import RagService

__all__ = [
    "ChunkRecord",
    "BatchIngestionResult",
    "ChunkedText",
    "CaseEvaluation",
    "build_context",
    "build_grounded_prompt",
    "build_grounded_messages",
    "DefaultQueryRouter",
    "DocumentLoader",
    "EvaluationCase",
    "EvaluationReport",
    "evaluate_retrieval",
    "LocalDocumentLoader",
    "MarkdownChunker",
    "ndcg_at_k",
    "precision_at_k",
    "QueryPlan",
    "QueryPlanner",
    "recall_at_k",
    "RetrievalTrace",
    "reciprocal_rank",
    "build_in_memory_service",
    "build_rag_service",
    "DocumentRecord",
    "IngestionResult",
    "RagService",
    "RagSettings",
    "SearchHit",
    "SearchResponse",
    "TextChunker",
]
