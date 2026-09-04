# RAG Server

This package contains the RAG application layer for AgentDemo.

## Current boundaries

- `schemas.py`: document, chunk, search, and ingestion data contracts.
- `ports.py`: interfaces for embeddings, MySQL metadata, Elasticsearch,
  Milvus, and reranking.
- `chunker.py`: dependency-free first-pass text chunker.
- `loaders.py`: safe local text/Markdown/HTML/JSON/CSV to document adapters.
- `query.py`: optional query rewrite, multi-query expansion, and route selection.
- `evaluation.py`: offline Recall, Precision, MRR, and nDCG evaluation helpers.
- `fusion.py`: reciprocal rank fusion for keyword and vector results.
- `service.py`: ingestion and retrieval orchestration.
- `adapters/in_memory.py`: deterministic local implementation for smoke tests.
- `adapters/mysql_repository.py`: MySQL document and chunk persistence.
- `adapters/elasticsearch_index.py`: Elasticsearch BM25 index.
- `adapters/milvus_index.py`: Milvus vector index.
- `adapters/embedding.py`: Qwen3 embedding model adapter.
- `adapters/qwen_reranker.py`: Qwen3 reranker adapter.
- `tools.py`: `knowledge_search` tool definition and registration helper.
- `config.py`: environment-backed runtime settings.

## Retrieval workflow

The default path is:

```text
DocumentLoader -> Chunker -> Embedding -> MySQL metadata
                                      -> Elasticsearch BM25
                                      -> Milvus cosine search
                                      -> RRF -> Qwen reranker -> threshold -> Top-K
```

`RAG_CHUNK_STRATEGY=markdown` enables heading-aware chunks. Each search uses
one original query by default. Inject `QueryPlanner(rewriter=..., router=...)`
when query rewriting, multi-query retrieval, or channel routing is needed.
Supported routes are `hybrid`, `keyword`, and `vector`.

Search responses include a serializable `query_plan` and `trace` with channel
candidate counts and fusion/reranking counts. These fields are intended for
observability and retrieval tuning, not for model-generated facts.

`permission_ids` is applied after metadata hydration. Empty document
permissions are public; a restricted chunk is returned only when the caller's
permission set intersects the chunk permissions. Tenant/document/version
filters are pushed to both search backends.

The application reads one runtime configuration file:
`D:\python_program\agentDemo\.env`. The root `.env.example` is a template,
not a second runtime configuration.

## Minimal smoke test

```python
from rag_server.adapters import HashEmbeddingProvider
from rag_server.bootstrap import build_in_memory_service
from rag_server.schemas import DocumentRecord

service = build_in_memory_service()
service.ingest(
    DocumentRecord(
        doc_id="doc-001",
        title="RAG notes",
        content="Milvus stores vectors. Elasticsearch stores keyword indexes.",
        source_uri="notes.md",
    ),
)

print(service.search("Where are vectors stored?").to_dict())
```

## Local loading and evaluation

```python
from rag_server.evaluation import EvaluationCase, evaluate_retrieval
from rag_server.loaders import LocalDocumentLoader

loader = LocalDocumentLoader("./knowledge")
batch = service.ingest_many(loader.iter_documents(), continue_on_error=True)

report = evaluate_retrieval(
    service.search,
    [EvaluationCase("Where are vectors stored?", frozenset({"doc-001:1:0"}))],
)
print(report.to_dict())
```

The loader only reads caller-provided files. It does not include tutorial
content or automatically index a directory.

## Production service

Install the optional RAG dependencies:

```powershell
python -m pip install -r requirements-rag.txt
```

Build the service from the root `.env`:

```python
from rag_server.bootstrap import build_rag_service

service = build_rag_service()
```

The reranker is loaded on the first search instead of package import.

## Register with AgentDemo tools

```python
from core.tool_registry import ToolRegistry
from rag_server.bootstrap import build_rag_service
from rag_server.tools import register_rag_tools

registry = ToolRegistry()
register_rag_tools(registry, build_rag_service())
```

The same registry can be passed to the existing MCP server factory or to
`Agent(..., tool_registry=registry)`. The tool is intentionally retrieval-only:
the caller receives evidence and remains responsible for the final answer.

## Expected storage names

- MySQL: `rag_document`, `rag_document_version`, `rag_chunk`,
  `rag_index_task`.
- Elasticsearch: the index named by `RAG_ELASTICSEARCH_INDEX`.
- Milvus: the collection named by `RAG_MILVUS_COLLECTION`, containing
  `chunk_id`, `vector`, `doc_id`, `tenant_id`, `version_id`, and
  `is_active`.

The MCP and Agent integrations call the same `knowledge_search` handler so
retrieval behavior is not duplicated between transports.

## Workspace tools

The AgentDemo entry point also registers the workspace tools from
`core/workspace_tools.py`:

- `workspace_list_files`
- `workspace_read_file`
- `workspace_write_file`
- `workspace_replace_text`
- `workspace_run_shell`
- `workspace_run_python`

They are registered in the same `ToolRegistry` as RAG tools. Pass that same
registry to `build_mcp_server()` to expose them through MCP as well.

Workspace tools default to the project root and reject paths outside it.
Sensitive files such as `.env`, private keys, and `.git` are protected.
Shell commands are restricted to one command, a timeout, an output limit,
and the configured command allowlist.
