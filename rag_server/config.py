from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from config.settings import load_project_env


def _get(
    env: Mapping[str, str],
    name: str,
    default: str,
) -> str:
    value = env.get(name)
    return default if value is None or not value.strip() else value.strip()


def _get_int(
    env: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    return int(_get(env, name, str(default)))


def _get_float(
    env: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    return float(_get(env, name, str(default)))


@dataclass(frozen=True, slots=True)
class RagSettings:
    """Runtime settings for the RAG application layer.

    The defaults are safe for local development. Credentials and model
    settings are intentionally read from environment variables.
    """

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_database: str = "rag"
    mysql_user: str = "rag_user"
    mysql_password: str = ""

    elasticsearch_url: str = "http://127.0.0.1:9200"
    elasticsearch_username: str = ""
    elasticsearch_password: str = ""
    elasticsearch_index: str = "rag_chunks_v1"

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_collection: str = "rag_chunks_v1"

    embedding_model: str = ""
    embedding_dimension: int = 0
    embedding_device: str = "auto"
    embedding_batch_size: int = 8
    embedding_max_length: int = 8192

    chunk_size: int = 800
    chunk_overlap: int = 120
    chunk_strategy: str = "text"

    keyword_top_k: int = 30
    vector_top_k: int = 30
    rerank_top_k: int = 10
    final_top_k: int = 3
    rrf_k: int = 60
    rerank_threshold: float | None = None
    reranker_model: str = "Qwen/Qwen3-Reranker-8B"
    reranker_device: str = "auto"
    reranker_batch_size: int = 2
    reranker_max_length: int = 8192
    reranker_instruction: str = (
        "Given a web search query, retrieve relevant passages "
        "that answer the query"
    )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "RagSettings":
        if env is None:
            load_project_env()
        values = os.environ if env is None else env
        defaults = cls()
        threshold = values.get("RAG_RERANK_THRESHOLD")

        settings = cls(
            mysql_host=_get(values, "RAG_MYSQL_HOST", defaults.mysql_host),
            mysql_port=_get_int(values, "RAG_MYSQL_PORT", defaults.mysql_port),
            mysql_database=_get(
                values,
                "RAG_MYSQL_DATABASE",
                defaults.mysql_database,
            ),
            mysql_user=_get(values, "RAG_MYSQL_USER", defaults.mysql_user),
            mysql_password=values.get("RAG_MYSQL_PASSWORD", ""),
            elasticsearch_url=_get(
                values,
                "RAG_ELASTICSEARCH_URL",
                defaults.elasticsearch_url,
            ),
            elasticsearch_username=values.get(
                "RAG_ELASTICSEARCH_USERNAME",
                "",
            ),
            elasticsearch_password=values.get(
                "RAG_ELASTICSEARCH_PASSWORD",
                "",
            ),
            elasticsearch_index=_get(
                values,
                "RAG_ELASTICSEARCH_INDEX",
                defaults.elasticsearch_index,
            ),
            milvus_uri=_get(values, "RAG_MILVUS_URI", defaults.milvus_uri),
            milvus_token=values.get("RAG_MILVUS_TOKEN", ""),
            milvus_collection=_get(
                values,
                "RAG_MILVUS_COLLECTION",
                defaults.milvus_collection,
            ),
            embedding_model=values.get("RAG_EMBEDDING_MODEL", ""),
            embedding_dimension=_get_int(
                values,
                "RAG_EMBEDDING_DIMENSION",
                defaults.embedding_dimension,
            ),
            embedding_device=_get(
                values,
                "RAG_EMBEDDING_DEVICE",
                defaults.embedding_device,
            ),
            embedding_batch_size=_get_int(
                values,
                "RAG_EMBEDDING_BATCH_SIZE",
                defaults.embedding_batch_size,
            ),
            embedding_max_length=_get_int(
                values,
                "RAG_EMBEDDING_MAX_LENGTH",
                defaults.embedding_max_length,
            ),
            chunk_size=_get_int(
                values,
                "RAG_CHUNK_SIZE",
                defaults.chunk_size,
            ),
            chunk_overlap=_get_int(
                values,
                "RAG_CHUNK_OVERLAP",
                defaults.chunk_overlap,
            ),
            chunk_strategy=_get(
                values,
                "RAG_CHUNK_STRATEGY",
                defaults.chunk_strategy,
            ).lower(),
            keyword_top_k=_get_int(
                values,
                "RAG_KEYWORD_TOP_K",
                defaults.keyword_top_k,
            ),
            vector_top_k=_get_int(
                values,
                "RAG_VECTOR_TOP_K",
                defaults.vector_top_k,
            ),
            rerank_top_k=_get_int(
                values,
                "RAG_RERANK_TOP_K",
                defaults.rerank_top_k,
            ),
            final_top_k=_get_int(
                values,
                "RAG_FINAL_TOP_K",
                defaults.final_top_k,
            ),
            rrf_k=_get_int(values, "RAG_RRF_K", defaults.rrf_k),
            rerank_threshold=(
                None
                if threshold is None or not threshold.strip()
                else float(threshold)
            ),
            reranker_model=_get(
                values,
                "RAG_RERANKER_MODEL",
                defaults.reranker_model,
            ),
            reranker_device=_get(
                values,
                "RAG_RERANKER_DEVICE",
                defaults.reranker_device,
            ),
            reranker_batch_size=_get_int(
                values,
                "RAG_RERANKER_BATCH_SIZE",
                defaults.reranker_batch_size,
            ),
            reranker_max_length=_get_int(
                values,
                "RAG_RERANKER_MAX_LENGTH",
                defaults.reranker_max_length,
            ),
            reranker_instruction=_get(
                values,
                "RAG_RERANKER_INSTRUCTION",
                defaults.reranker_instruction,
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.chunk_strategy not in {"text", "markdown"}:
            raise ValueError(
                "chunk_strategy must be one of: text, markdown",
            )
        if self.embedding_dimension < 0:
            raise ValueError("embedding_dimension must not be negative")
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be greater than zero")
        if self.embedding_max_length <= 0:
            raise ValueError("embedding_max_length must be greater than zero")
        if self.keyword_top_k <= 0:
            raise ValueError("keyword_top_k must be greater than zero")
        if self.vector_top_k <= 0:
            raise ValueError("vector_top_k must be greater than zero")
        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be greater than zero")
        if self.final_top_k <= 0:
            raise ValueError("final_top_k must be greater than zero")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        if self.rerank_threshold is not None and not 0 <= self.rerank_threshold <= 1:
            raise ValueError("rerank_threshold must be between zero and one")
        if self.reranker_batch_size <= 0:
            raise ValueError("reranker_batch_size must be greater than zero")
        if self.reranker_max_length <= 0:
            raise ValueError("reranker_max_length must be greater than zero")
