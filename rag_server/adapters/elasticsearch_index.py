from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rag_server.config import RagSettings
from rag_server.ports import SearchFilters
from rag_server.schemas import ChunkRecord, SearchHit


class ElasticsearchKeywordIndex:
    """Elasticsearch BM25 adapter for one-document-per-chunk indexing."""

    def __init__(
        self,
        settings: RagSettings,
        *,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or self._build_client()

    def ensure_index(self) -> None:
        if self.client.indices.exists(index=self.settings.elasticsearch_index):
            return
        self.client.indices.create(
            index=self.settings.elasticsearch_index,
            mappings=self.mapping(),
        )

    def index_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        if not chunks:
            return
        self.ensure_index()

        try:
            from elasticsearch import helpers
        except ImportError as exc:
            raise RuntimeError(
                "elasticsearch is required for Elasticsearch RAG adapter",
            ) from exc

        actions = [
            {
                "_op_type": "index",
                "_index": self.settings.elasticsearch_index,
                "_id": chunk.chunk_id,
                "_source": self._source(chunk),
            }
            for chunk in chunks
        ]
        # Make a just-ingested document visible to the next search request.
        helpers.bulk(self.client, actions, refresh="wait_for")

    def delete_document(self, doc_id: str) -> None:
        self.ensure_index()
        self.client.delete_by_query(
            index=self.settings.elasticsearch_index,
            query={"term": {"doc_id": doc_id}},
            conflicts="proceed",
            refresh="wait_for",
        )

    def search(
        self,
        query: str,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        self.ensure_index()
        response = self.client.search(
            index=self.settings.elasticsearch_index,
            size=limit,
            query={
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "content^2",
                                    "title^3",
                                    "heading_path^2",
                                ],
                                "type": "best_fields",
                            },
                        },
                    ],
                    "filter": self._filters(filters),
                },
            },
        )
        hits: list[SearchHit] = []
        for rank, item in enumerate(
            response.get("hits", {}).get("hits", []),
            start=1,
        ):
            source = item.get("_source", {})
            hits.append(
                SearchHit(
                    chunk_id=str(item.get("_id")),
                    doc_id=str(source.get("doc_id", "")),
                    content=str(source.get("content", "")),
                    title=str(source.get("title", "")),
                    version_id=source.get("version_id"),
                    source_uri=source.get("source_uri"),
                    tenant_id=source.get("tenant_id"),
                    score=float(item.get("_score") or 0.0),
                    rank=rank,
                    sources=("keyword",),
                    metadata=source,
                ),
            )
        return hits

    @staticmethod
    def mapping() -> dict[str, Any]:
        return {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "version_id": {"type": "long"},
                "title": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256,
                        },
                    },
                },
                "content": {"type": "text"},
                "heading_path": {"type": "text"},
                "chunk_index": {"type": "integer"},
                "source_uri": {
                    "type": "keyword",
                    "ignore_above": 1024,
                },
                "tenant_id": {"type": "keyword"},
                "permission_ids": {"type": "keyword"},
                "is_active": {"type": "boolean"},
                "metadata": {"type": "flattened"},
                "updated_at": {"type": "date"},
            },
        }

    @staticmethod
    def _source(chunk: ChunkRecord) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "version_id": chunk.version_id,
            "title": chunk.title,
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            "heading_path": chunk.metadata.get("heading_path", ""),
            "source_uri": chunk.source_uri,
            "tenant_id": chunk.tenant_id,
            "permission_ids": list(chunk.permission_ids),
            "metadata": dict(chunk.metadata),
            "is_active": True,
        }

    @staticmethod
    def _filters(
        filters: SearchFilters | None,
    ) -> list[dict[str, Any]]:
        clauses: list[dict[str, Any]] = []
        provided = filters or {}
        if "is_active" not in provided:
            clauses.append({"term": {"is_active": True}})
        for key, value in provided.items():
            if isinstance(value, (list, tuple, set, frozenset)):
                clauses.append({"terms": {key: list(value)}})
            else:
                clauses.append({"term": {key: value}})
        return clauses

    def _build_client(self) -> Any:
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise RuntimeError(
                "elasticsearch is required for Elasticsearch RAG adapter",
            ) from exc

        kwargs: dict[str, Any] = {
            "hosts": [self.settings.elasticsearch_url],
        }
        if (
            self.settings.elasticsearch_username
            and self.settings.elasticsearch_password
        ):
            kwargs["basic_auth"] = (
                self.settings.elasticsearch_username,
                self.settings.elasticsearch_password,
            )
        return Elasticsearch(**kwargs)
