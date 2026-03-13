"""Retrieval evaluation metrics with bootstrap confidence intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ── Per-query metric functions ────────────────────────────────────────────────


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in the top-k retrieved."""
    if not relevant:
        return 0.0
    hits = len(relevant & set(retrieved[:k]))
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved that are relevant."""
    if k == 0:
        return 0.0
    hits = len(relevant & set(retrieved[:k]))
    return hits / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant result, or 0 if none found."""
    for i, rid in enumerate(retrieved):
        if rid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Average precision across all relevant items."""
    if not relevant:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, rid in enumerate(retrieved):
        if rid in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, rid in enumerate(retrieved[:k])
        if rid in relevant
    )
    ideal_k = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
    return dcg / idcg if idcg > 0 else 0.0


# ── Aggregate results ─────────────────────────────────────────────────────────


@dataclass
class QueryMetrics:
    """Metrics for a single query."""

    query_id: str
    category: str
    language: str
    difficulty: str
    recall: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ap: float = 0.0


@dataclass
class ConfidenceInterval:
    """Mean with 95% bootstrap confidence interval."""

    mean: float
    lower: float
    upper: float

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple queries."""

    n_queries: int
    recall: dict[int, ConfidenceInterval] = field(default_factory=dict)
    precision: dict[int, ConfidenceInterval] = field(default_factory=dict)
    ndcg: dict[int, ConfidenceInterval] = field(default_factory=dict)
    mrr: ConfidenceInterval | None = None
    map: ConfidenceInterval | None = None


def compute_query_metrics(
    query_id: str,
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int],
    category: str = "",
    language: str = "",
    difficulty: str = "",
) -> QueryMetrics:
    """Compute all metrics for a single query."""
    qm = QueryMetrics(
        query_id=query_id,
        category=category,
        language=language,
        difficulty=difficulty,
    )
    for k in k_values:
        qm.recall[k] = recall_at_k(retrieved_ids, relevant_ids, k)
        qm.precision[k] = precision_at_k(retrieved_ids, relevant_ids, k)
        qm.ndcg[k] = ndcg_at_k(retrieved_ids, relevant_ids, k)
    qm.mrr = reciprocal_rank(retrieved_ids, relevant_ids)
    qm.ap = average_precision(retrieved_ids, relevant_ids)
    return qm


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    """Compute mean with bootstrap confidence interval."""
    if not values:
        return ConfidenceInterval(0.0, 0.0, 0.0)
    arr = np.array(values)
    rng = np.random.RandomState(seed)
    means = [
        float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    return ConfidenceInterval(
        mean=float(np.mean(arr)),
        lower=float(np.percentile(means, alpha * 100)),
        upper=float(np.percentile(means, (1 - alpha) * 100)),
    )


def aggregate_metrics(
    query_metrics: list[QueryMetrics],
    k_values: list[int],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> AggregateMetrics:
    """Aggregate per-query metrics into means with confidence intervals."""
    n = len(query_metrics)
    if n == 0:
        return AggregateMetrics(n_queries=0)

    agg = AggregateMetrics(n_queries=n)
    for k in k_values:
        agg.recall[k] = bootstrap_ci([qm.recall[k] for qm in query_metrics], n_bootstrap, seed=seed)
        agg.precision[k] = bootstrap_ci(
            [qm.precision[k] for qm in query_metrics], n_bootstrap, seed=seed
        )
        agg.ndcg[k] = bootstrap_ci([qm.ndcg[k] for qm in query_metrics], n_bootstrap, seed=seed)
    agg.mrr = bootstrap_ci([qm.mrr for qm in query_metrics], n_bootstrap, seed=seed)
    agg.map = bootstrap_ci([qm.ap for qm in query_metrics], n_bootstrap, seed=seed)
    return agg
