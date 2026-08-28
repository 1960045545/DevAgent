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
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SearchResponse:
    query: str
    hits: list[SearchHit]
    total_candidates: int
    keyword_candidates: int
    vector_candidates: int

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
