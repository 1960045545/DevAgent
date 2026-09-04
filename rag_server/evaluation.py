from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_server.ports import SearchFilters


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """A query and the chunk ids judged relevant by a human or benchmark."""

    query: str
    relevant_chunk_ids: frozenset[str]
    case_id: str = ""
    filters: SearchFilters | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("evaluation query must not be empty")
        if not self.relevant_chunk_ids:
            raise ValueError("evaluation case needs at least one relevant chunk")


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    query: str
    retrieved_chunk_ids: tuple[str, ...]
    metrics: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    cases: tuple[CaseEvaluation, ...]
    aggregate: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_count": len(self.cases),
            "aggregate": dict(self.aggregate),
            "cases": [case.to_dict() for case in self.cases],
        }


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    _validate_k(k)
    expected = set(relevant)
    return (
        len(set(retrieved[:k]).intersection(expected)) / len(expected)
        if expected
        else 0.0
    )


def precision_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    _validate_k(k)
    expected = set(relevant)
    return len(set(retrieved[:k]).intersection(expected)) / k


def reciprocal_rank(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    *,
    k: int | None = None,
) -> float:
    expected = set(relevant)
    candidates = retrieved if k is None else retrieved[:k]
    for index, chunk_id in enumerate(candidates, start=1):
        if chunk_id in expected:
            return 1.0 / index
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    _validate_k(k)
    expected = set(relevant)
    if not expected:
        return 0.0
    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, chunk_id in enumerate(retrieved[:k])
        if chunk_id in expected
    )
    ideal_length = min(k, len(expected))
    idcg = sum(
        1.0 / math.log2(index + 2)
        for index in range(ideal_length)
    )
    return dcg / idcg if idcg else 0.0


def evaluate_retrieval(
    search: Callable[..., Any],
    cases: Iterable[EvaluationCase],
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> EvaluationReport:
    """Run an offline retrieval evaluation without persisting benchmark data."""
    normalized_ks = tuple(dict.fromkeys(ks))
    if not normalized_ks:
        raise ValueError("ks must not be empty")
    for k in normalized_ks:
        _validate_k(k)

    case_results: list[CaseEvaluation] = []
    for index, case in enumerate(cases, start=1):
        response = search(
            case.query,
            top_k=max(normalized_ks),
            filters=case.filters,
        )
        hits = getattr(response, "hits", None)
        if hits is None and isinstance(response, dict):
            hits = response.get("results", [])
        hits = hits or []
        retrieved = tuple(
            str(
                hit.chunk_id
                if hasattr(hit, "chunk_id")
                else hit.get("chunk_id", "")
            )
            for hit in hits
        )
        metrics: dict[str, float] = {}
        for k in normalized_ks:
            metrics[f"recall@{k}"] = recall_at_k(
                retrieved,
                case.relevant_chunk_ids,
                k,
            )
            metrics[f"precision@{k}"] = precision_at_k(
                retrieved,
                case.relevant_chunk_ids,
                k,
            )
            metrics[f"ndcg@{k}"] = ndcg_at_k(
                retrieved,
                case.relevant_chunk_ids,
                k,
            )
            metrics[f"mrr@{k}"] = reciprocal_rank(
                retrieved,
                case.relevant_chunk_ids,
                k=k,
            )
        case_results.append(
            CaseEvaluation(
                case_id=case.case_id or str(index),
                query=case.query,
                retrieved_chunk_ids=retrieved,
                metrics=metrics,
            ),
        )

    aggregate: dict[str, float] = {}
    if case_results:
        keys = case_results[0].metrics.keys()
        aggregate = {
            key: sum(result.metrics[key] for result in case_results)
            / len(case_results)
            for key in keys
        }
    return EvaluationReport(tuple(case_results), aggregate)


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("evaluation k must be greater than zero")
