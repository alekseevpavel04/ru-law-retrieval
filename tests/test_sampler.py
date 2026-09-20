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


def test_batch_count_matches_len_under_heavy_conflicts():
    # 8 of 12 rows share one article, so the greedy pass cannot fill a batch from disjoint rows.
    # The trainer plans the schedule from __len__, so __iter__ must yield exactly that many batches
    # and lose no row (before the fix it yielded 8 short batches instead of 3 full ones).
    keys = [frozenset({"a0"})] * 8 + [frozenset({f"a{i}"}) for i in range(1, 5)]
    sampler = ArticleNoDuplicatesBatchSampler(list(range(12)), 4, False, keys, generator=torch.Generator(), seed=7)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert sorted(i for b in batches for i in b) == list(range(12))
    assert all(len(b) == 4 for b in batches)


def test_drop_last_keeps_full_batches():
    keys = [frozenset({f"a{i % 3}"}) for i in range(10)]
    sampler = ArticleNoDuplicatesBatchSampler(list(range(10)), 3, True, keys, generator=torch.Generator(), seed=2)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 3
    assert all(len(b) == 3 for b in batches)


def test_deterministic_per_epoch():
    keys = [frozenset({f"a{i}"}) for i in range(20)]
    s1 = ArticleNoDuplicatesBatchSampler(list(range(20)), 4, False, keys, generator=torch.Generator(), seed=3)
    s2 = ArticleNoDuplicatesBatchSampler(list(range(20)), 4, False, keys, generator=torch.Generator(), seed=3)
    assert list(s1) == list(s2)
    s1.set_epoch(1)
    assert list(s1) != list(s2)
