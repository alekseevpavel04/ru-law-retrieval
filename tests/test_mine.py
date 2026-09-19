import numpy as np

from rlr.train.mine_negatives import mine


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_mine_skips_gold_article_and_too_close_candidates():
    # chunks: 0,1 -> article 0 (gold); 2 -> article 1 (almost identical to query); 3 -> article 2
    chunk_emb = _unit([[1, 0, 0], [0.9, 0.1, 0], [1, 0.01, 0], [0.7, 0.7, 0]])
    chunk_article = np.array([0, 0, 1, 2])
    q = _unit([[1, 0, 0]])
    pos_score = np.array([float(chunk_emb[1] @ q[0])])  # positive is chunk 1
    neg, stats = mine(q, pos_score, np.array([0]), chunk_emb, chunk_article, top_k=4, margin=0.95)
    assert neg.tolist() == [3]  # chunk 2 is >= 0.95 * pos -> likely false negative
    assert stats["candidates_same_article"] == 2
    assert stats["candidates_too_close"] == 1


def test_mine_returns_minus_one_when_nothing_left():
    chunk_emb = _unit([[1, 0], [0.99, 0.01]])
    neg, stats = mine(_unit([[1, 0]]), np.array([1.0]), np.array([0]), chunk_emb, np.array([0, 0]), top_k=2)
    assert neg.tolist() == [-1]
    assert stats["queries_without_negative"] == 1
