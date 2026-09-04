from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Metadata = dict[str, Any]


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    title: str
    content: str
    source_uri: str | None = None
    tenant_id: str | None = None
    version: int = 1
    permission_ids: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)
    content_hash: str = ""
    source_type: str = "text"


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    title: str
    version_id: int | None = None
    source_uri: str | None = None
    tenant_id: str | None = None
    version: int = 1
    permission_ids: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)
    content_hash: str = ""
    token_count: int | None = None

    def to_metadata(self) -> Metadata:
        return {
            **self.metadata,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "version_id": self.version_id,
            "source_uri": self.source_uri,
            "tenant_id": self.tenant_id,
            "version": self.version,
            "permission_ids": list(self.permission_ids),
            "token_count": self.token_count,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    content: str
    title: str = ""
    version_id: int | None = None
    source_uri: str | None = None
    tenant_id: str | None = None
    score: float = 0.0
    rank: int | None = None
    sources: tuple[str, ...] = ()
    rerank_score: float | None = None
    keyword_score: float | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    metadata: Metadata = field(default_factory=dict)

    def to_dict(self) -> Metadata:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "title": self.title,
            "version_id": self.version_id,
            "source_uri": self.source_uri,
            "tenant_id": self.tenant_id,
            "score": self.score,
            "rank": self.rank,
            "sources": list(self.sources),
            "rerank_score": self.rerank_score,
            "keyword_score": self.keyword_score,
            "vector_score": self.vector_score,
            "rrf_score": self.rrf_score,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """The normalized retrieval plan used for one user query."""

    original_query: str
    queries: tuple[str, ...]
    route: str = "hybrid"
    rewrite_applied: bool = False
    reason: str = ""

    def to_dict(self) -> Metadata:
        return {
            "original_query": self.original_query,
            "queries": list(self.queries),
            "route": self.route,
            "rewrite_applied": self.rewrite_applied,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """Small, serializable diagnostics record for retrieval tuning."""

    route: str
    queries: tuple[str, ...]
    keyword_candidates: int
    vector_candidates: int
    fused_candidates: int
    reranked_candidates: int
    filtered_candidates: int

    def to_dict(self) -> Metadata:
        return {
            "route": self.route,
            "queries": list(self.queries),
            "keyword_candidates": self.keyword_candidates,
            "vector_candidates": self.vector_candidates,
            "fused_candidates": self.fused_candidates,
            "reranked_candidates": self.reranked_candidates,
            "filtered_candidates": self.filtered_candidates,
        }


@dataclass(slots=True)
class SearchResponse:
    query: str
    hits: list[SearchHit]
    total_candidates: int
    keyword_candidates: int
    vector_candidates: int
    query_plan: QueryPlan | None = None
    trace: RetrievalTrace | None = None

    @property
    def has_evidence(self) -> bool:
        return bool(self.hits)

    def to_dict(self) -> Metadata:
        return {
            "query": self.query,
            "has_evidence": self.has_evidence,
            "total_candidates": self.total_candidates,
            "keyword_candidates": self.keyword_candidates,
            "vector_candidates": self.vector_candidates,
            "query_plan": (
                self.query_plan.to_dict()
                if self.query_plan is not None
                else None
            ),
            "trace": (
                self.trace.to_dict()
                if self.trace is not None
                else None
            ),
            "results": [hit.to_dict() for hit in self.hits],
        }


@dataclass(slots=True)
class IngestionResult:
    doc_id: str
    chunk_count: int
    indexed: bool
    error: str | None = None

    def to_dict(self) -> Metadata:
        return {
            "doc_id": self.doc_id,
            "chunk_count": self.chunk_count,
            "indexed": self.indexed,
            "error": self.error,
        }


@dataclass(slots=True)
class BatchIngestionResult:
    """Summary for a batch run; individual document results remain inspectable."""

    results: list[IngestionResult]

    @property
    def succeeded(self) -> int:
        return sum(1 for result in self.results if result.indexed)

    @property
    def failed(self) -> int:
        return len(self.results) - self.succeeded

    def to_dict(self) -> Metadata:
        return {
            "total": len(self.results),
            "succeeded": self.succeeded,
            "failed": self.failed,
            "results": [result.to_dict() for result in self.results],
        }
