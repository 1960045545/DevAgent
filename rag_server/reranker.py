from __future__ import annotations

from collections.abc import Sequence

from rag_server.schemas import SearchHit


class NoOpReranker:
    """Default reranker for the skeleton.

    It preserves the fused order. A cross-encoder or remote reranker can
    replace this class later through the Reranker protocol.
    """

    def rank(
        self,
        query: str,
        candidates: Sequence[SearchHit],
        limit: int,
    ) -> list[SearchHit]:
        ranked = list(candidates[:limit])
        for hit in ranked:
            hit.rerank_score = hit.score
        return ranked
