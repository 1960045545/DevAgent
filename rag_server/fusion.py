from __future__ import annotations

from collections.abc import Sequence

from rag_server.schemas import SearchHit


def rrf_fuse(
    keyword_hits: Sequence[SearchHit],
    vector_hits: Sequence[SearchHit],
    *,
    rrf_k: int = 60,
    limit: int | None = None,
) -> list[SearchHit]:
    """Fuse ranked lists without comparing their raw score scales."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")

    merged: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    raw_scores: dict[str, dict[str, float]] = {}

    for source, hits in (
        ("keyword", keyword_hits),
        ("vector", vector_hits),
    ):
        for rank, hit in enumerate(hits, start=1):
            if hit.chunk_id not in merged:
                merged[hit.chunk_id] = hit
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )
            sources.setdefault(hit.chunk_id, set()).add(source)
            raw_scores.setdefault(hit.chunk_id, {})[source] = hit.score

    fused: list[SearchHit] = []
    for hit in merged.values():
        fused.append(
            SearchHit(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                content=hit.content,
                title=hit.title,
                version_id=hit.version_id,
                source_uri=hit.source_uri,
                tenant_id=hit.tenant_id,
                score=scores[hit.chunk_id],
                sources=tuple(sorted(sources[hit.chunk_id])),
                keyword_score=raw_scores[hit.chunk_id].get("keyword"),
                vector_score=raw_scores[hit.chunk_id].get("vector"),
                rrf_score=scores[hit.chunk_id],
                metadata=dict(hit.metadata),
            )
        )

    fused.sort(key=lambda item: item.score, reverse=True)
    for rank, hit in enumerate(fused, start=1):
        hit.rank = rank
    return fused if limit is None else fused[:limit]
