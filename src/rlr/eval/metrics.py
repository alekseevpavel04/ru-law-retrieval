"""Article-level retrieval metrics. One implementation for every evaluation.

``ranked``: article ids ordered by score (best first, no duplicates).
``qrels``:  {article_id: relevance} with relevance > 0 for relevant articles.
"""

import math
from collections.abc import Sequence

import numpy as np

METRICS = ("ndcg@10", "recall@5", "recall@10", "mrr@10")


def dcg(gains: Sequence[float]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], qrels: dict[str, float], k: int = 10) -> float:
    gains = [qrels.get(d, 0.0) for d in ranked[:k]]
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], qrels: dict[str, float], k: int) -> float:
    relevant = {d for d, g in qrels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: Sequence[str], qrels: dict[str, float], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if qrels.get(d, 0.0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def query_metrics(ranked: Sequence[str], qrels: dict[str, float]) -> dict[str, float]:
    return {
        "ndcg@10": ndcg_at_k(ranked, qrels, 10),
        "recall@5": recall_at_k(ranked, qrels, 5),
        "recall@10": recall_at_k(ranked, qrels, 10),
        "mrr@10": mrr_at_k(ranked, qrels, 10),
    }


def mean_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {m: float("nan") for m in METRICS}
    return {m: float(np.mean([q[m] for q in per_query])) for m in METRICS}
