import math

import pytest

from rlr.eval.metrics import mean_metrics, mrr_at_k, ndcg_at_k, query_metrics, recall_at_k


def test_single_relevant_at_rank_1():
    m = query_metrics(["a", "b", "c"], {"a": 1})
    assert m == {"ndcg@10": 1.0, "recall@5": 1.0, "recall@10": 1.0, "mrr@10": 1.0}


def test_single_relevant_at_rank_3():
    m = query_metrics(["x", "y", "a"], {"a": 1})
    assert m["ndcg@10"] == pytest.approx(1 / math.log2(4))  # 0.5
    assert m["mrr@10"] == pytest.approx(1 / 3)
    assert m["recall@5"] == 1.0


def test_relevant_at_rank_6_counts_for_recall10_not_recall5():
    ranked = ["x1", "x2", "x3", "x4", "x5", "a"]
    assert recall_at_k(ranked, {"a": 1}, 5) == 0.0
    assert recall_at_k(ranked, {"a": 1}, 10) == 1.0
    assert mrr_at_k(ranked, {"a": 1}) == pytest.approx(1 / 6)


def test_not_found_in_top10():
    ranked = [f"x{i}" for i in range(10)] + ["a"]
    assert query_metrics(ranked, {"a": 1}) == {"ndcg@10": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr@10": 0.0}


def test_multiple_relevant_ndcg_by_hand():
    # relevant a, b; ranked b at 1, a at 3
    ranked = ["b", "x", "a"]
    dcg = 1 + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(ranked, {"a": 1, "b": 1}) == pytest.approx(dcg / idcg)
    assert recall_at_k(ranked, {"a": 1, "b": 1, "c": 1}, 5) == pytest.approx(2 / 3)


def test_zero_relevance_is_ignored():
    assert ndcg_at_k(["a"], {"a": 0, "b": 1}) == 0.0
    assert recall_at_k(["a"], {"a": 0}, 5) == 0.0


def test_mean_metrics():
    m = mean_metrics([query_metrics(["a"], {"a": 1}), query_metrics(["x"], {"a": 1})])
    assert m["ndcg@10"] == 0.5
    assert math.isnan(mean_metrics([])["mrr@10"])
