from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag_server.schemas import QueryPlan


VALID_ROUTES = frozenset({"hybrid", "keyword", "vector"})


@dataclass(frozen=True, slots=True)
class DefaultQueryRouter:
    """Conservative router: use both retrieval channels by default."""

    def route(self, query: str) -> str:
        del query
        return "hybrid"


class QueryPlanner:
    """Build query variants and select a retrieval route.

    Rewriting is deliberately optional. A caller can inject a local rule
    based rewriter or an LLM-backed implementation without coupling the RAG
    service to a specific model provider.
    """

    def __init__(
        self,
        *,
        rewriter: Any | None = None,
        router: Any | None = None,
        max_queries: int = 3,
    ) -> None:
        if max_queries <= 0:
            raise ValueError("max_queries must be greater than zero")
        self.rewriter = rewriter
        self.router = router or DefaultQueryRouter()
        self.max_queries = max_queries

    def build(
        self,
        query: str,
        *,
        route: str | None = None,
    ) -> QueryPlan:
        original = " ".join(query.split())
        if not original:
            raise ValueError("query must not be empty")

        selected_route = (
            str(route).strip().lower()
            if route is not None
            else self._call_router(original)
        )
        if selected_route not in VALID_ROUTES:
            raise ValueError(
                f"unsupported retrieval route: {selected_route}; "
                f"expected one of {sorted(VALID_ROUTES)}",
            )

        variants = [original]
        if self.rewriter is not None:
            rewritten = self._call_rewriter(original)
            for candidate in rewritten:
                normalized = " ".join(str(candidate).split())
                if normalized and normalized not in variants:
                    variants.append(normalized)
                if len(variants) >= self.max_queries:
                    break

        return QueryPlan(
            original_query=original,
            queries=tuple(variants[: self.max_queries]),
            route=selected_route,
            rewrite_applied=len(variants) > 1,
            reason=(
                "injected query router/rewriter"
                if self.rewriter
                else "default hybrid route"
            ),
        )

    def _call_router(self, query: str) -> str:
        route = (
            self.router(query)
            if callable(self.router)
            else self.router.route(query)
        )
        return str(route).strip().lower()

    def _call_rewriter(self, query: str) -> Sequence[str]:
        rewritten = (
            self.rewriter(query)
            if callable(self.rewriter)
            else self.rewriter.rewrite(query)
        )
        if isinstance(rewritten, str):
            return [rewritten]
        if not isinstance(rewritten, Sequence):
            raise ValueError("query rewriter must return a string or sequence")
        return rewritten
