from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import replace

from rag_server.chunker import ChunkedText, TextChunker
from rag_server.config import RagSettings
from rag_server.fusion import rrf_fuse
from rag_server.query import QueryPlanner
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
    BatchIngestionResult,
    DocumentRecord,
    IngestionResult,
    RetrievalTrace,
    SearchHit,
    QueryPlan,
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
        query_planner: QueryPlanner | None = None,
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
        self.query_planner = query_planner or QueryPlanner()

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
        chunked: list[ChunkedText] = self._split_document(document.content)
        chunks = [
            ChunkRecord(
                chunk_id=f"{document.doc_id}:{document.version}:{index}",
                doc_id=document.doc_id,
                content=item.content,
                chunk_index=index,
                title=document.title,
                source_uri=document.source_uri,
                tenant_id=document.tenant_id,
                version=document.version,
                permission_ids=document.permission_ids,
                metadata={
                    **document.metadata,
                    **item.metadata,
                },
                content_hash=self._hash(item.content),
            )
            for index, item in enumerate(chunked)
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

    def ingest_many(
        self,
        documents: Iterable[DocumentRecord],
        *,
        continue_on_error: bool = False,
    ) -> BatchIngestionResult:
        """Ingest documents one by one for worker-friendly retry behavior."""
        results: list[IngestionResult] = []
        for document in documents:
            try:
                results.append(self.ingest(document))
            except Exception as exc:
                result = IngestionResult(
                    doc_id=document.doc_id,
                    chunk_count=0,
                    indexed=False,
                    error=str(exc),
                )
                results.append(result)
                if not continue_on_error:
                    raise
        return BatchIngestionResult(results)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
        route: str | None = None,
    ) -> SearchResponse:
        final_top_k = (
            self.settings.final_top_k
            if top_k is None
            else top_k
        )
        if final_top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        plan = self.query_planner.build(query, route=route)
        backend_filters, post_filters = self._split_filters(filters)
        keyword_limit = self.settings.keyword_top_k
        vector_limit = self.settings.vector_top_k
        if post_filters:
            # ACLs are checked after MySQL hydration because Milvus schemas
            # may not contain the full permission list.
            keyword_limit = max(keyword_limit, final_top_k * 4)
            vector_limit = max(vector_limit, final_top_k * 4)

        keyword_hits = (
            self._search_keywords(plan, keyword_limit, backend_filters)
            if plan.route in {"hybrid", "keyword"}
            else []
        )
        vector_hits = (
            self._search_vectors(plan, vector_limit, backend_filters)
            if plan.route in {"hybrid", "vector"}
            else []
        )
        keyword_hits = self._filter_hits(
            self._hydrate_hits(keyword_hits),
            post_filters,
        )
        vector_hits = self._filter_hits(
            self._hydrate_hits(vector_hits),
            post_filters,
        )
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
        reranked_count = len(reranked)

        if self.settings.rerank_threshold is not None:
            reranked = [
                hit
                for hit in reranked
                if (
                    hit.rerank_score is not None
                    and hit.rerank_score >= self.settings.rerank_threshold
                )
            ]

        final_hits = reranked[:final_top_k]
        for rank, hit in enumerate(final_hits, start=1):
            hit.rank = rank

        trace = RetrievalTrace(
            route=plan.route,
            queries=plan.queries,
            keyword_candidates=len(keyword_hits),
            vector_candidates=len(vector_hits),
            fused_candidates=len(fused),
            reranked_candidates=reranked_count,
            filtered_candidates=len(reranked),
        )
        return SearchResponse(
            query=plan.original_query,
            hits=final_hits,
            total_candidates=len(fused),
            keyword_candidates=len(keyword_hits),
            vector_candidates=len(vector_hits),
            query_plan=plan,
            trace=trace,
        )

    def _search_keywords(
        self,
        plan: QueryPlan,
        limit: int,
        filters: SearchFilters | None,
    ) -> list[SearchHit]:
        return self._merge_ranked_hits(
            [
                self.keyword_index.search(
                    query=query,
                    limit=limit,
                    filters=filters,
                )
                for query in plan.queries
            ],
        )

    def _search_vectors(
        self,
        plan: QueryPlan,
        limit: int,
        filters: SearchFilters | None,
    ) -> list[SearchHit]:
        return self._merge_ranked_hits(
            [
                self.vector_index.search(
                    vector=self.embedder.embed_query(query),
                    limit=limit,
                    filters=filters,
                )
                for query in plan.queries
            ],
        )

    @staticmethod
    def _merge_ranked_hits(
        ranked_lists: Sequence[list[SearchHit]],
    ) -> list[SearchHit]:
        best: dict[str, SearchHit] = {}
        for hits in ranked_lists:
            for hit in hits:
                current = best.get(hit.chunk_id)
                if current is None or hit.score > current.score:
                    best[hit.chunk_id] = hit
        result = sorted(
            best.values(),
            key=lambda item: item.score,
            reverse=True,
        )
        for rank, hit in enumerate(result, start=1):
            hit.rank = rank
        return result

    @staticmethod
    def _split_filters(
        filters: SearchFilters | None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Keep backend filters portable and apply ACL filters after hydration."""
        provided = dict(filters or {})
        backend_keys = {
            "tenant_id",
            "doc_id",
            "version_id",
            "is_active",
        }
        post_filter_keys = {"permission_ids"}
        unknown = set(provided) - backend_keys - post_filter_keys
        if unknown:
            raise ValueError(
                "unsupported search filter(s): " + ", ".join(sorted(unknown)),
            )
        return (
            {
                key: value
                for key, value in provided.items()
                if key in backend_keys
            },
            {
                key: value
                for key, value in provided.items()
                if key in post_filter_keys
            },
        )

    @staticmethod
    def _filter_hits(
        hits: list[SearchHit],
        filters: SearchFilters,
    ) -> list[SearchHit]:
        if "permission_ids" not in filters:
            return hits
        expected = RagService._as_string_set(filters.get("permission_ids"))
        result: list[SearchHit] = []
        for hit in hits:
            if "permission_ids" not in hit.metadata:
                continue
            actual = RagService._as_string_set(
                hit.metadata.get("permission_ids"),
            )
            # An empty permission list denotes a public chunk.
            if not actual or actual.intersection(expected):
                result.append(hit)
        return result

    @staticmethod
    def _as_string_set(value: object) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set, frozenset)):
            return {str(item) for item in value}
        return {str(value)}

    def _split_document(self, content: str) -> list[ChunkedText]:
        splitter = getattr(self.chunker, "split_with_metadata", None)
        if callable(splitter):
            return list(splitter(content))
        return [ChunkedText(item) for item in self.chunker.split(content)]

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
