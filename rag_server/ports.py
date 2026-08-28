from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from rag_server.schemas import (
    ChunkRecord,
    DocumentRecord,
    SearchHit,
)


SearchFilters = Mapping[str, object]


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class DocumentRepository(Protocol):
    def save_document(self, document: DocumentRecord) -> int | None:
        ...

    def save_chunks(
        self,
        chunks: Sequence[ChunkRecord],
    ) -> list[ChunkRecord]:
        ...

    def get_chunks(
        self,
        chunk_ids: Sequence[str],
    ) -> list[ChunkRecord]:
        ...

class KeywordIndex(Protocol):
    def delete_document(self, doc_id: str) -> None:
        ...

    def index_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        ...

    def search(
        self,
        query: str,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        ...


class VectorIndex(Protocol):
    def delete_document(self, doc_id: str) -> None:
        ...

    def index_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        ...

    def search(
        self,
        vector: Sequence[float],
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        ...


class Reranker(Protocol):
    def rank(
        self,
        query: str,
        candidates: Sequence[SearchHit],
        limit: int,
    ) -> list[SearchHit]:
        ...
