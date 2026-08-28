from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Callable

from rag_server.config import RagSettings
from rag_server.schemas import (
    ChunkRecord,
    DocumentRecord,
)


class MySQLDocumentRepository:
    """MySQL repository for the four-table RAG schema.

    Expected tables:
    `rag_document`, `rag_document_version`, `rag_chunk`, and
    `rag_index_task`. The index-task table is managed by a later ingestion
    worker and is intentionally not required by this repository.
    """

    def __init__(
        self,
        settings: RagSettings,
        *,
        connection_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self._connection_factory = connection_factory

    def save_document(self, document: DocumentRecord) -> int:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO rag_document (
                    doc_id, tenant_id, title, source_uri, source_type,
                    current_version, status, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, 'ACTIVE', %s
                )
                ON DUPLICATE KEY UPDATE
                    tenant_id = VALUES(tenant_id),
                    title = VALUES(title),
                    source_uri = VALUES(source_uri),
                    source_type = VALUES(source_type),
                    current_version = GREATEST(
                        current_version,
                        VALUES(current_version)
                    ),
                    status = 'ACTIVE',
                    metadata = VALUES(metadata)
                """,
                (
                    document.doc_id,
                    document.tenant_id or "",
                    document.title,
                    document.source_uri,
                    document.source_type,
                    document.version,
                    self._json(document.metadata),
                ),
            )
            cursor.execute(
                """
                INSERT INTO rag_document_version (
                    doc_id, version_no, content, content_hash,
                    embedding_model, embedding_dimension,
                    chunk_size, chunk_overlap, status
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, 'ACTIVE'
                )
                ON DUPLICATE KEY UPDATE
                    content = VALUES(content),
                    content_hash = VALUES(content_hash),
                    embedding_model = VALUES(embedding_model),
                    embedding_dimension = VALUES(embedding_dimension),
                    chunk_size = VALUES(chunk_size),
                    chunk_overlap = VALUES(chunk_overlap),
                    status = 'ACTIVE'
                """,
                (
                    document.doc_id,
                    document.version,
                    document.content,
                    document.content_hash,
                    self.settings.embedding_model,
                    self.settings.embedding_dimension or None,
                    self.settings.chunk_size,
                    self.settings.chunk_overlap,
                ),
            )
            cursor.execute(
                """
                SELECT version_id
                FROM rag_document_version
                WHERE doc_id = %s AND version_no = %s
                """,
                (document.doc_id, document.version),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("failed to resolve document version_id")
            version_id = int(row[0])
            connection.commit()
            return version_id
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def save_chunks(
        self,
        chunks: Sequence[ChunkRecord],
    ) -> list[ChunkRecord]:
        if not chunks:
            return []

        connection = self._connect()
        try:
            cursor = connection.cursor()
            persisted: list[ChunkRecord] = []
            resolved_version_ids: dict[tuple[str, int], int] = {}
            for chunk in chunks:
                version_id = chunk.version_id
                if version_id is None:
                    key = (chunk.doc_id, chunk.version)
                    version_id = resolved_version_ids.get(key)
                    if version_id is None:
                        cursor.execute(
                            """
                            SELECT version_id
                            FROM rag_document_version
                            WHERE doc_id = %s AND version_no = %s
                            """,
                            key,
                        )
                        row = cursor.fetchone()
                        if row is None:
                            raise RuntimeError(
                                f"document version not found: "
                                f"{chunk.doc_id}:{chunk.version}",
                            )
                        version_id = int(row[0])
                resolved_version_ids[(chunk.doc_id, chunk.version)] = int(
                    version_id,
                )

                if version_id not in resolved_version_ids.values():
                    resolved_version_ids[
                        (chunk.doc_id, chunk.version)
                    ] = version_id

            for version_id in set(resolved_version_ids.values()):
                cursor.execute(
                    "DELETE FROM rag_chunk WHERE version_id = %s",
                    (version_id,),
                )

            for chunk in chunks:
                version_id = (
                    chunk.version_id
                    or resolved_version_ids[(chunk.doc_id, chunk.version)]
                )
                cursor.execute(
                    """
                    INSERT INTO rag_chunk (
                        chunk_id, version_id, doc_id, chunk_index,
                        content, content_hash, title, heading_path,
                        token_count, metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON DUPLICATE KEY UPDATE
                        version_id = VALUES(version_id),
                        doc_id = VALUES(doc_id),
                        chunk_index = VALUES(chunk_index),
                        content = VALUES(content),
                        content_hash = VALUES(content_hash),
                        title = VALUES(title),
                        heading_path = VALUES(heading_path),
                        token_count = VALUES(token_count),
                        metadata = VALUES(metadata)
                    """,
                    (
                        chunk.chunk_id,
                        version_id,
                        chunk.doc_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.content_hash,
                        chunk.title,
                        chunk.metadata.get("heading_path"),
                        chunk.token_count,
                        self._json(
                            {
                                "permission_ids": list(chunk.permission_ids),
                                **chunk.metadata,
                            },
                        ),
                    ),
                )
                persisted.append(
                    ChunkRecord(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        content=chunk.content,
                        chunk_index=chunk.chunk_index,
                        title=chunk.title,
                        version_id=version_id,
                        source_uri=chunk.source_uri,
                        tenant_id=chunk.tenant_id,
                        version=chunk.version,
                        permission_ids=chunk.permission_ids,
                        metadata=dict(chunk.metadata),
                        content_hash=chunk.content_hash,
                        token_count=chunk.token_count,
                    ),
                )
            connection.commit()
            return persisted
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_chunks(
        self,
        chunk_ids: Sequence[str],
    ) -> list[ChunkRecord]:
        if not chunk_ids:
            return []

        placeholders = ", ".join(["%s"] * len(chunk_ids))
        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                f"""
                SELECT
                    c.chunk_id,
                    c.doc_id,
                    c.chunk_index,
                    c.content,
                    c.content_hash,
                    c.title,
                    c.heading_path,
                    c.token_count,
                    c.metadata,
                    c.version_id,
                    v.version_no,
                    d.source_uri,
                    d.tenant_id
                FROM rag_chunk AS c
                INNER JOIN rag_document_version AS v
                    ON v.version_id = c.version_id
                INNER JOIN rag_document AS d
                    ON d.doc_id = c.doc_id
                WHERE c.chunk_id IN ({placeholders})
                """,
                tuple(chunk_ids),
            )
            rows = cursor.fetchall()
            return [self._row_to_chunk(row) for row in rows]
        finally:
            connection.close()

    def _connect(self) -> Any:
        if self._connection_factory is not None:
            return self._connection_factory()

        try:
            import mysql.connector
        except ImportError as exc:
            raise RuntimeError(
                "mysql-connector-python is required for MySQL RAG adapter",
            ) from exc

        return mysql.connector.connect(
            host=self.settings.mysql_host,
            port=self.settings.mysql_port,
            user=self.settings.mysql_user,
            password=self.settings.mysql_password,
            database=self.settings.mysql_database,
        )

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def _row_to_chunk(row: dict[str, Any]) -> ChunkRecord:
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if not isinstance(metadata, dict):
            metadata = {}

        permission_ids = metadata.get("permission_ids") or ()
        if not isinstance(permission_ids, (list, tuple)):
            permission_ids = ()

        return ChunkRecord(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            content=row["content"],
            chunk_index=int(row["chunk_index"]),
            title=row.get("title") or "",
            version_id=int(row["version_id"]),
            source_uri=row.get("source_uri"),
            tenant_id=row.get("tenant_id"),
            version=int(row["version_no"]),
            permission_ids=tuple(str(item) for item in permission_ids),
            metadata=metadata,
            content_hash=row.get("content_hash") or "",
            token_count=row.get("token_count"),
        )
