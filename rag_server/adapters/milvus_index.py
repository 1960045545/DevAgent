from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag_server.config import RagSettings
from rag_server.ports import SearchFilters
from rag_server.schemas import ChunkRecord, SearchHit


class MilvusVectorIndex:
    """Milvus vector index with scalar fields for query-time filtering."""

    def __init__(
        self,
        settings: RagSettings,
        *,
        dimension: int,
        client: Any | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("Milvus vector dimension must be greater than zero")
        self.settings = settings
        self.dimension = dimension
        self.client = client or self._build_client()

    def ensure_collection(self) -> None:
        if self.client.has_collection(
            collection_name=self.settings.milvus_collection,
        ):
            return

        try:
            from pymilvus import DataType
        except ImportError as exc:
            raise RuntimeError(
                "pymilvus is required for Milvus RAG adapter",
            ) from exc

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dimension,
        )
        schema.add_field(
            field_name="doc_id",
            datatype=DataType.VARCHAR,
            max_length=128,
        )
        schema.add_field(
            field_name="tenant_id",
            datatype=DataType.VARCHAR,
            max_length=128,
        )
        schema.add_field(
            field_name="version_id",
            datatype=DataType.INT64,
        )
        schema.add_field(
            field_name="is_active",
            datatype=DataType.BOOL,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=self.settings.milvus_collection,
            schema=schema,
            index_params=index_params,
        )

    def index_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk and vector counts must match")
        if not chunks:
            return
        self.ensure_collection()

        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    "vector dimension does not match Milvus collection",
                )

        data = [
            {
                "chunk_id": chunk.chunk_id,
                "vector": list(vector),
                "doc_id": chunk.doc_id,
                "tenant_id": chunk.tenant_id or "",
                "version_id": chunk.version_id or 0,
                "is_active": True,
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(
            collection_name=self.settings.milvus_collection,
            data=data,
        )

    def delete_document(self, doc_id: str) -> None:
        self.ensure_collection()
        escaped = doc_id.replace("\\", "\\\\").replace('"', '\\"')
        self.client.delete(
            collection_name=self.settings.milvus_collection,
            filter=f'doc_id == "{escaped}"',
        )

    def search(
        self,
        vector: Sequence[float],
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        self.ensure_collection()
        if len(vector) != self.dimension:
            raise ValueError(
                "query vector dimension does not match Milvus collection",
            )

        kwargs: dict[str, Any] = {
            "collection_name": self.settings.milvus_collection,
            "data": [list(vector)],
            "limit": limit,
            "output_fields": [
                "doc_id",
                "tenant_id",
                "version_id",
                "is_active",
            ],
        }
        expression = self._filter_expression(filters)
        if expression:
            kwargs["filter"] = expression

        response = self.client.search(**kwargs)
        raw_hits = response[0] if response else []
        hits: list[SearchHit] = []
        for rank, item in enumerate(raw_hits, start=1):
            entity = item.get("entity") or {}
            chunk_id = item.get("id") or item.get("pk")
            hits.append(
                SearchHit(
                    chunk_id=str(chunk_id),
                    doc_id=str(entity.get("doc_id", "")),
                    content="",
                    version_id=entity.get("version_id"),
                    tenant_id=entity.get("tenant_id"),
                    score=float(
                        item.get("distance")
                        or item.get("score")
                        or 0.0
                    ),
                    rank=rank,
                    sources=("vector",),
                    metadata=entity,
                ),
            )
        return hits

    @staticmethod
    def _filter_expression(
        filters: SearchFilters | None,
    ) -> str:
        expressions: list[str] = []
        provided = filters or {}
        if "is_active" not in provided:
            expressions.append("is_active == true")
        for key, value in provided.items():
            if key not in {
                "tenant_id",
                "doc_id",
                "version_id",
                "is_active",
            }:
                raise ValueError(
                    f"Milvus filter field is not supported: {key}",
                )
            if isinstance(value, bool):
                literal = "true" if value else "false"
            elif isinstance(value, (int, float)):
                literal = str(value)
            else:
                escaped = str(value).replace("\\", "\\\\").replace(
                    '"',
                    '\\"',
                )
                literal = f'"{escaped}"'
            expressions.append(f"{key} == {literal}")
        return " and ".join(expressions)

    def _build_client(self) -> Any:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError(
                "pymilvus is required for Milvus RAG adapter",
            ) from exc

        kwargs: dict[str, Any] = {"uri": self.settings.milvus_uri}
        if self.settings.milvus_token:
            kwargs["token"] = self.settings.milvus_token
        return MilvusClient(**kwargs)
