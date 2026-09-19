import numpy as np

from rlr.data.build import near_dup_against, near_dup_within


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_near_dup_within_keeps_first():
    emb = _unit([[1, 0], [1, 0.01], [0, 1], [0.01, 1]])
    assert near_dup_within(emb, 0.99).tolist() == [False, True, False, True]


def test_near_dup_against_reference():
    emb = _unit([[1, 0], [0, 1]])
    ref = _unit([[1, 0.01]])
    assert near_dup_against(emb, ref, 0.99).tolist() == [True, False]
