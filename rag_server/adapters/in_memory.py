from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence

from rag_server.ports import SearchFilters
from rag_server.schemas import (
    ChunkRecord,
    DocumentRecord,
    SearchHit,
)


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"\w+", text, flags=re.UNICODE)
        if token.strip()
    ]


def _matches(
    metadata: Mapping[str, object],
    filters: SearchFilters | None,
) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, (list, tuple, set, frozenset)):
            if isinstance(actual, (list, tuple, set, frozenset)):
                if not set(actual).intersection(expected):
                    return False
            elif actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class HashEmbeddingProvider:
    """Deterministic local embedding used only for development and tests."""

    def __init__(
        self,
        dimension: int = 64,
        model_name: str = "hash-dev-embedding",
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        self.dimension = dimension
        self.model_name = model_name

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _tokens(text) or [text.lower()]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class InMemoryDocumentRepository:
    """Small repository for local smoke tests."""

    def __init__(self) -> None:
        self.documents: dict[str, DocumentRecord] = {}
        self.chunks: dict[str, ChunkRecord] = {}

    def save_document(self, document: DocumentRecord) -> int:
        self.documents[document.doc_id] = document
        return document.version

    def save_chunks(
        self,
        chunks: Sequence[ChunkRecord],
    ) -> list[ChunkRecord]:
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
        return list(chunks)

    def get_chunks(
        self,
        chunk_ids: Sequence[str],
    ) -> list[ChunkRecord]:
        return [
            self.chunks[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self.chunks
        ]


class InMemoryKeywordIndex:
    """Naive keyword index that mirrors the KeywordIndex protocol."""

    def __init__(self) -> None:
        self._chunks: dict[str, ChunkRecord] = {}

    def index_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def delete_document(self, doc_id: str) -> None:
        self._chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._chunks.items()
            if chunk.doc_id != doc_id
        }

    def search(
        self,
        query: str,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        query_tokens = set(_tokens(query))
        scored: list[SearchHit] = []

        for chunk in self._chunks.values():
            if not _matches(chunk.to_metadata(), filters):
                continue
            content_tokens = _tokens(
                f"{chunk.title} {chunk.content}",
            )
            overlap = sum(
                1
                for token in content_tokens
                if token in query_tokens
            )
            if overlap == 0:
                continue
            scored.append(self._to_hit(chunk, float(overlap)))

        scored.sort(key=lambda item: item.score, reverse=True)
        for rank, hit in enumerate(scored[:limit], start=1):
            hit.rank = rank
        return scored[:limit]

    @staticmethod
    def _to_hit(chunk: ChunkRecord, score: float) -> SearchHit:
        return SearchHit(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            content=chunk.content,
            title=chunk.title,
            version_id=chunk.version_id,
            source_uri=chunk.source_uri,
            tenant_id=chunk.tenant_id,
            score=score,
            sources=("keyword",),
            metadata=chunk.to_metadata(),
        )


class InMemoryVectorIndex:
    """Brute-force cosine index that mirrors the VectorIndex protocol."""

    def __init__(self) -> None:
        self._records: dict[
            str,
            tuple[ChunkRecord, list[float]],
        ] = {}

    def index_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk and vector counts must match")
        for chunk, vector in zip(chunks, vectors):
            self._records[chunk.chunk_id] = (
                chunk,
                list(vector),
            )

    def delete_document(self, doc_id: str) -> None:
        self._records = {
            chunk_id: record
            for chunk_id, record in self._records.items()
            if record[0].doc_id != doc_id
        }

    def search(
        self,
        vector: Sequence[float],
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        scored: list[SearchHit] = []
        for chunk, stored_vector in self._records.values():
            if not _matches(chunk.to_metadata(), filters):
                continue
            score = self._cosine(vector, stored_vector)
            scored.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    title=chunk.title,
                    version_id=chunk.version_id,
                    source_uri=chunk.source_uri,
                    tenant_id=chunk.tenant_id,
                    score=score,
                    sources=("vector",),
                    metadata=chunk.to_metadata(),
                ),
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        results = scored[:limit]
        for rank, hit in enumerate(results, start=1):
            hit.rank = rank
        return results

    @staticmethod
    def _cosine(
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        if len(left) != len(right):
            raise ValueError("vector dimensions must match")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        dot = sum(
            left_value * right_value
            for left_value, right_value in zip(left, right)
        )
        return dot / (left_norm * right_norm)
