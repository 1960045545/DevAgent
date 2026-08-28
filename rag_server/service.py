from __future__ import annotations

import hashlib
from dataclasses import replace

from rag_server.chunker import TextChunker
from rag_server.config import RagSettings
from rag_server.fusion import rrf_fuse
from rag_server.ports import (
    DocumentRepository,
    EmbeddingProvider,
    KeywordIndex,
    Reranker,
    SearchFilters,
    VectorIndex,
)
from rag_server.reranker import NoOpReranker
from rag_server.schemas import (
    ChunkRecord,
    DocumentRecord,
    IngestionResult,
    SearchHit,
    SearchResponse,
)


class RagService:
    """Application-level RAG orchestration.

    The service owns the workflow, while storage and model integrations are
    injected through small protocols.
    """

    def __init__(
        self,
        *,
        settings: RagSettings,
        embedder: EmbeddingProvider,
        document_repository: DocumentRepository,
        keyword_index: KeywordIndex,
        vector_index: VectorIndex,
        chunker: TextChunker | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.document_repository = document_repository
        self.keyword_index = keyword_index
        self.vector_index = vector_index
        self.chunker = chunker or TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.reranker = reranker or NoOpReranker()

        if (
            settings.embedding_dimension
            and settings.embedding_dimension != embedder.dimension
        ):
            raise ValueError(
                "configured embedding dimension does not match the provider",
            )

    def ingest(self, document: DocumentRecord) -> IngestionResult:
        if not document.doc_id:
            raise ValueError("doc_id must not be empty")
        if not document.content.strip():
            raise ValueError("document content must not be empty")

        content_hash = self._hash(document.content)
        stored_document = replace(
            document,
            content_hash=content_hash,
        )
        text_chunks = self.chunker.split(document.content)
        chunks = [
            ChunkRecord(
                chunk_id=f"{document.doc_id}:{document.version}:{index}",
                doc_id=document.doc_id,
                content=content,
                chunk_index=index,
                title=document.title,
                source_uri=document.source_uri,
                tenant_id=document.tenant_id,
                version=document.version,
                permission_ids=document.permission_ids,
                metadata=dict(document.metadata),
                content_hash=self._hash(content),
            )
            for index, content in enumerate(text_chunks)
        ]
        vectors = self.embedder.embed_documents(
            [chunk.content for chunk in chunks],
        )
        if len(vectors) != len(chunks):
            raise ValueError("embedding count does not match chunk count")
        for vector in vectors:
            if len(vector) != self.embedder.dimension:
                raise ValueError(
                    "embedding dimension does not match the provider",
                )

        version_id = self.document_repository.save_document(stored_document)
        if version_id is not None:
            chunks = [
                replace(chunk, version_id=version_id)
                for chunk in chunks
            ]
        chunks = self.document_repository.save_chunks(chunks)
        self.keyword_index.delete_document(document.doc_id)
        self.vector_index.delete_document(document.doc_id)
        self.keyword_index.index_chunks(chunks)
        self.vector_index.index_chunks(chunks, vectors)

        return IngestionResult(
            doc_id=document.doc_id,
            chunk_count=len(chunks),
            indexed=True,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
    ) -> SearchResponse:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        final_top_k = (
            self.settings.final_top_k
            if top_k is None
            else top_k
        )
        if final_top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        keyword_hits = self.keyword_index.search(
            query=query,
            limit=self.settings.keyword_top_k,
            filters=filters,
        )
        vector_hits = self.vector_index.search(
            vector=self.embedder.embed_query(query),
            limit=self.settings.vector_top_k,
            filters=filters,
        )
        keyword_hits = self._hydrate_hits(keyword_hits)
        vector_hits = self._hydrate_hits(vector_hits)
        fused = rrf_fuse(
            keyword_hits,
            vector_hits,
            rrf_k=self.settings.rrf_k,
            limit=self.settings.rerank_top_k,
        )
        reranked = self.reranker.rank(
            query=query,
            candidates=fused,
            limit=self.settings.rerank_top_k,
        )

        if self.settings.rerank_threshold is not None:
            reranked = [
                hit
                for hit in reranked
                if (
                    hit.rerank_score is not None
                    and hit.rerank_score >= self.settings.rerank_threshold
                )
            ]

        return SearchResponse(
            query=query,
            hits=reranked[:final_top_k],
            total_candidates=len(fused),
            keyword_candidates=len(keyword_hits),
            vector_candidates=len(vector_hits),
        )

    def _hydrate_hits(
        self,
        hits: list[SearchHit],
    ) -> list[SearchHit]:
        if not hits:
            return hits

        chunk_ids = [hit.chunk_id for hit in hits]
        chunks = self.document_repository.get_chunks(chunk_ids)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        hydrated: list[SearchHit] = []

        for hit in hits:
            chunk = by_id.get(hit.chunk_id)
            if chunk is None:
                hydrated.append(hit)
                continue
            hydrated.append(
                replace(
                    hit,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    title=chunk.title,
                    version_id=chunk.version_id,
                    source_uri=chunk.source_uri,
                    tenant_id=chunk.tenant_id,
                    metadata={
                        **hit.metadata,
                        **chunk.to_metadata(),
                    },
                ),
            )
        return hydrated

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
