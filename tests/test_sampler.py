import torch

from rlr.train.sampler import ArticleNoDuplicatesBatchSampler


def test_no_shared_article_in_batch_and_all_rows_used():
    # 12 rows over 4 articles, some rows also touch a negative article
    keys = [frozenset({f"a{i % 4}"}) for i in range(12)]
    keys[0] = frozenset({"a0", "a1"})
    sampler = ArticleNoDuplicatesBatchSampler(list(range(12)), 3, False, keys, generator=torch.Generator(), seed=1)
    batches = list(sampler)
    seen = [i for b in batches for i in b]
    assert sorted(seen) == list(range(12))
    for b in batches:
        used: list[str] = [k for i in b for k in keys[i]]
        assert len(used) == len(set(used)), b


def test_deterministic_per_epoch():
    keys = [frozenset({f"a{i}"}) for i in range(20)]
    s1 = ArticleNoDuplicatesBatchSampler(list(range(20)), 4, False, keys, generator=torch.Generator(), seed=3)
    s2 = ArticleNoDuplicatesBatchSampler(list(range(20)), 4, False, keys, generator=torch.Generator(), seed=3)
    assert list(s1) == list(s2)
    s1.set_epoch(1)
    assert list(s1) != list(s2)
