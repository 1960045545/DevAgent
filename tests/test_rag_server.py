from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from config.settings import _load_env_fallback
from rag_server.adapters import (
    HashEmbeddingProvider,
    InMemoryDocumentRepository,
    InMemoryKeywordIndex,
    InMemoryVectorIndex,
)
from rag_server.chunker import TextChunker
from rag_server.bootstrap import build_in_memory_service
from rag_server.config import RagSettings
from rag_server.fusion import rrf_fuse
from rag_server.prompt import build_grounded_prompt
from rag_server.schemas import DocumentRecord, SearchHit
from rag_server.service import RagService


class RagServerTests(unittest.TestCase):
    def test_chunker_respects_size_and_overlap(self) -> None:
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.split("one two three four five six seven eight nine")

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 20 for chunk in chunks))

    def test_settings_can_load_defaults_and_environment(self) -> None:
        settings = RagSettings.from_env(
            {
                "RAG_MYSQL_PORT": "3307",
                "RAG_CHUNK_SIZE": "400",
                "RAG_RERANK_THRESHOLD": "0.8",
            },
        )

        self.assertEqual(settings.mysql_port, 3307)
        self.assertEqual(settings.chunk_size, 400)
        self.assertEqual(settings.rerank_threshold, 0.8)

    def test_settings_empty_mapping_does_not_read_process_environment(self) -> None:
        settings = RagSettings.from_env({})

        self.assertEqual(settings.mysql_port, 3306)

    def test_project_env_fallback_preserves_existing_values(self) -> None:
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "RAG_TEST_EXISTING=file\nRAG_TEST_NEW='new value'\n",
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {"RAG_TEST_EXISTING": "process"},
                clear=True,
            ):
                _load_env_fallback(env_file, override=False)
                self.assertEqual(
                    os.environ["RAG_TEST_EXISTING"],
                    "process",
                )
                self.assertEqual(
                    os.environ["RAG_TEST_NEW"],
                    "new value",
                )

    def test_long_unit_does_not_create_overlap_only_chunk(self) -> None:
        chunker = TextChunker(chunk_size=10, chunk_overlap=3)
        chunks = chunker.split("abcdefghijklmnop")

        self.assertEqual(chunks, ["abcdefghij", "hijklmnop"])

    def test_rrf_fuses_duplicate_hits(self) -> None:
        keyword_hit = SearchHit(
            chunk_id="same",
            doc_id="doc",
            content="content",
            score=100,
            sources=("keyword",),
        )
        vector_hit = SearchHit(
            chunk_id="same",
            doc_id="doc",
            content="content",
            score=0.9,
            sources=("vector",),
        )

        fused = rrf_fuse(
            [keyword_hit],
            [vector_hit],
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].sources, ("keyword", "vector"))

    def test_in_memory_ingest_and_search(self) -> None:
        settings = RagSettings(
            embedding_dimension=32,
            chunk_size=200,
            chunk_overlap=20,
            final_top_k=3,
        )
        service = RagService(
            settings=settings,
            embedder=HashEmbeddingProvider(dimension=32),
            document_repository=InMemoryDocumentRepository(),
            keyword_index=InMemoryKeywordIndex(),
            vector_index=InMemoryVectorIndex(),
        )

        result = service.ingest(
            DocumentRecord(
                doc_id="doc-001",
                title="Storage guide",
                content=(
                    "Milvus stores vector embeddings for semantic search. "
                    "Elasticsearch stores keyword indexes for lexical search."
                ),
                source_uri="guide.md",
                tenant_id="tenant-a",
            ),
        )

        response = service.search(
            "Where are vector embeddings stored?",
            filters={"tenant_id": "tenant-a"},
        )

        self.assertTrue(result.indexed)
        self.assertEqual(result.chunk_count, 1)
        self.assertTrue(response.has_evidence)
        self.assertEqual(response.hits[0].doc_id, "doc-001")

    def test_reingest_replaces_old_chunks(self) -> None:
        service = build_in_memory_service(
            RagSettings(
                embedding_dimension=32,
                chunk_size=10,
                chunk_overlap=2,
            ),
        )
        service.ingest(
            DocumentRecord(
                doc_id="doc-replace",
                title="Old",
                content="alpha beta gamma delta",
            ),
        )
        service.ingest(
            DocumentRecord(
                doc_id="doc-replace",
                title="New",
                content="replacement",
            ),
        )

        response = service.search("alpha")

        self.assertFalse(
            any(
                hit.doc_id == "doc-replace" and "alpha" in hit.content
                for hit in response.hits
            ),
        )

    def test_chunk_metadata_cannot_override_identity_fields(self) -> None:
        from rag_server.schemas import ChunkRecord

        chunk = ChunkRecord(
            chunk_id="canonical",
            doc_id="doc",
            content="content",
            chunk_index=0,
            title="title",
            metadata={"doc_id": "attacker-value", "category": "guide"},
        )

        metadata = chunk.to_metadata()

        self.assertEqual(metadata["doc_id"], "doc")
        self.assertEqual(metadata["category"], "guide")

    def test_search_rejects_non_positive_top_k(self) -> None:
        service = build_in_memory_service()

        with self.assertRaises(ValueError):
            service.search("anything", top_k=0)

    def test_grounded_prompt_contains_evidence_marker(self) -> None:
        service = build_in_memory_service()
        service.ingest(
            DocumentRecord(
                doc_id="doc-002",
                title="Prompt guide",
                content="The answer is in this evidence.",
            ),
        )
        response = service.search("What is in this evidence?")
        prompt = build_grounded_prompt(
            response.query,
            response.hits,
        )

        self.assertIn("[证据 1]", prompt)
        self.assertIn(response.hits[0].content, prompt)


if __name__ == "__main__":
    unittest.main()
