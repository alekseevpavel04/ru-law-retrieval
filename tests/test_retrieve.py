import numpy as np
import torch

from rlr.eval.retrieve import Corpus, aggregate_max, bm25_tokenize, dense_search, rrf, topk_articles


def test_aggregate_max_over_chunks():
    # 2 articles: units 0,1 -> article 0, unit 2 -> article 1
    scores = torch.tensor([[0.1, 0.9, 0.5], [0.3, 0.2, 0.4]])
    agg = aggregate_max(scores, np.array([0, 0, 1]), 2)
    assert torch.allclose(agg, torch.tensor([[0.9, 0.5], [0.3, 0.4]]))


def test_article_without_chunks_gets_minus_inf():
    agg = aggregate_max(torch.tensor([[0.5]]), np.array([1]), 2)
    assert agg[0, 0] == float("-inf")


def test_topk_articles_order():
    res = topk_articles(torch.tensor([[0.1, 0.7, 0.3]]), ["a", "b", "c"], k=2)
    assert [d for d, _ in res[0]] == ["b", "c"]


def test_dense_search_chunk_view_returns_unique_articles():
    corpus = Corpus("chunk", ["a#0", "a#1", "b#0"], ["", "", ""], np.array([0, 0, 1]), ["a", "b"])
    units = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    q = np.array([[0.6, 0.8]], dtype=np.float32)
    (ranking,) = dense_search(q, units, corpus, k=5, device="cpu")
    assert [d for d, _ in ranking] == ["a", "b"]
    assert ranking[0][1] == np.float32(0.96).item() or abs(ranking[0][1] - 0.96) < 1e-6


def test_rrf():
    run1 = [[("a", 9.0), ("b", 8.0)]]
    run2 = [[("b", 0.9), ("c", 0.8)]]
    (fused,) = rrf([run1, run2], k=60)
    assert fused[0][0] == "b"
    assert {d for d, _ in fused} == {"a", "b", "c"}


class _FakeStemmer:
    def stemWords(self, words):
        return [w[:5] for w in words]


def test_bm25_tokenize_lower_yo_and_stem():
    assert bm25_tokenize(["Ёлки-Палки, ЗАЁМ 2024"], None) == [["елки", "палки", "заем", "2024"]]
    assert bm25_tokenize(["увольнение"], _FakeStemmer()) == [["уволь"]]
