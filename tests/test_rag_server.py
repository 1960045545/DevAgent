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
from rag_server.chunker import MarkdownChunker
from rag_server.evaluation import (
    EvaluationCase,
    evaluate_retrieval,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_server.bootstrap import build_in_memory_service
from rag_server.config import RagSettings
from rag_server.fusion import rrf_fuse
from rag_server.loaders import LocalDocumentLoader
from rag_server.prompt import build_grounded_messages, build_grounded_prompt
from rag_server.query import QueryPlanner
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

    def test_settings_rejects_rerank_threshold_outside_probability_range(self) -> None:
        with self.assertRaises(ValueError):
            RagSettings.from_env({"RAG_RERANK_THRESHOLD": "1.1"})

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
        self.assertEqual(response.hits[0].rank, 1)

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

    def test_markdown_chunker_preserves_heading_path(self) -> None:
        chunker = MarkdownChunker(chunk_size=80, chunk_overlap=10)
        chunks = chunker.split_with_metadata(
            "# Guide\n\nIntroduction.\n\n## Install\n\nRun the installer."
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["heading_path"], "Guide")
        self.assertEqual(
            chunks[1].metadata["heading_path"],
            "Guide > Install",
        )

    def test_local_loader_converts_files_and_skips_runtime_directories(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.md").write_text(
                "# Guide\n\nUse Milvus.",
                encoding="utf-8",
            )
            (root / ".git").mkdir()
            (root / ".git" / "ignored.md").write_text("secret", encoding="utf-8")
            loader = LocalDocumentLoader(root)

            documents = list(loader.iter_documents())
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].source_uri, "guide.md")
            self.assertEqual(documents[0].source_type, "md")
            self.assertEqual(documents[0].metadata["extension"], ".md")

            with self.assertRaises(ValueError):
                loader.load("../outside.md")

    def test_query_planner_rewrites_deduplicates_and_routes(self) -> None:
        planner = QueryPlanner(
            rewriter=lambda query: [query, "semantic " + query, "third"],
            router=lambda _query: "keyword",
            max_queries=2,
        )

        plan = planner.build("  original question  ")

        self.assertEqual(plan.route, "keyword")
        self.assertEqual(plan.queries, ("original question", "semantic original question"))
        self.assertTrue(plan.rewrite_applied)

    def test_search_route_and_permission_filter_are_reported(self) -> None:
        service = build_in_memory_service(
            RagSettings(
                embedding_dimension=32,
                keyword_top_k=10,
                vector_top_k=10,
                rerank_top_k=10,
            ),
        )
        service.ingest(
            DocumentRecord(
                doc_id="public",
                title="Public",
                content="Milvus vector store",
            ),
        )
        service.ingest(
            DocumentRecord(
                doc_id="restricted",
                title="Restricted",
                content="Milvus vector store private",
                permission_ids=("admin",),
            ),
        )

        response = service.search(
            "Milvus vector",
            route="keyword",
            filters={"permission_ids": []},
        )

        self.assertEqual(response.query_plan.route, "keyword")
        self.assertEqual(response.trace.route, "keyword")
        self.assertEqual(response.trace.vector_candidates, 0)
        self.assertTrue(all(hit.doc_id == "public" for hit in response.hits))

    def test_grounded_messages_split_instructions_from_evidence(self) -> None:
        hit = SearchHit(
            chunk_id="chunk-1",
            doc_id="doc-1",
            title="Guide",
            content="Milvus stores vectors.",
        )

        messages = build_grounded_messages("Where are vectors?", [hit])

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("[证据 1]", messages[1]["content"])

    def test_grounded_messages_explain_when_evidence_is_missing(self) -> None:
        messages = build_grounded_messages("Where are vectors?", [])

        self.assertIn("没有检索到可用的知识库证据", messages[1]["content"])

    def test_retrieval_evaluation_reports_standard_metrics(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"b", "d"}
        self.assertEqual(recall_at_k(retrieved, relevant, 2), 0.5)
        self.assertEqual(precision_at_k(retrieved, relevant, 2), 0.5)
        self.assertEqual(reciprocal_rank(retrieved, relevant, k=3), 0.5)
        self.assertGreater(ndcg_at_k(retrieved, relevant, 3), 0.0)

        report = evaluate_retrieval(
            lambda _query, top_k, filters=None: {
                "results": [{"chunk_id": "b"}, {"chunk_id": "a"}],
            },
            [EvaluationCase("question", frozenset({"b"}), case_id="case-1")],
            ks=(1, 2),
        )
        self.assertEqual(report.aggregate["recall@1"], 1.0)
        self.assertEqual(report.to_dict()["case_count"], 1)

    def test_batch_ingest_can_continue_after_one_document_fails(self) -> None:
        service = build_in_memory_service(RagSettings(embedding_dimension=16))
        report = service.ingest_many(
            [
                DocumentRecord("ok", "OK", "usable content"),
                DocumentRecord("bad", "Bad", ""),
            ],
            continue_on_error=True,
        )

        self.assertEqual(report.succeeded, 1)
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.results[1].error, "document content must not be empty")


if __name__ == "__main__":
    unittest.main()
