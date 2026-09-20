"""Article-level splits: the guarantees the whole benchmark rests on, checked without any data.

``tests/test_leaks.py`` checks the built dataset and therefore needs ``data/``; these tests check
the function that creates the guarantee, so they run on a fresh clone.
"""

from collections import Counter

import pytest

from rlr.data.splits import make_splits, stats


def articles(n_per_code: int = 60) -> list[dict]:
    rows = []
    for code, group in (("tk", "in_domain"), ("gk", "in_domain"), ("sk", "ood")):
        for i in range(1, n_per_code + 1):
            rows.append(
                {
                    "doc_id": f"{code}-{i}",
                    "code": code,
                    "group": group,
                    "title": f"Статья {i}",
                    "text": "текст статьи " * 30,
                }
            )
    return rows


@pytest.fixture
def splits() -> dict:
    return make_splits(articles(), seed=42, n_test_seen=20, n_test_unseen=8, n_dev_seen=15, n_dev_unseen=5)


def test_dev_and_test_articles_are_disjoint(splits):
    by_split = {s: {it["doc_id"] for it in splits["eval_items"] if it["split"] == s} for s in ("dev", "test")}
    assert by_split["dev"] & by_split["test"] == set()


def test_held_out_articles_are_never_training_articles(splits):
    assert set(splits["held_out"]) & set(splits["train_articles"]) == set()


def test_unseen_articles_slice_comes_only_from_held_out(splits):
    held = set(splits["held_out"])
    for it in splits["eval_items"]:
        if it["slice"] == "unseen_articles":
            assert it["doc_id"] in held
        elif it["slice"] == "seen":
            assert it["doc_id"] not in held


def test_unseen_codes_slice_is_exactly_the_out_of_domain_codes(splits):
    ood = {it["doc_id"] for it in splits["eval_items"] if it["slice"] == "unseen_codes"}
    assert ood and all(d.startswith("sk-") for d in ood)
    assert not any(d.startswith("sk-") for d in splits["train_articles"])


def test_held_out_fraction_is_stratified_per_code(splits):
    per_code = Counter(d.split("-")[0] for d in splits["held_out"])
    assert per_code == {"tk": 6, "gk": 6}  # 10% of 60 in each in-domain code, no OOD code


def test_question_types_are_balanced_within_each_slice(splits):
    groups = {(it["split"], it["slice"]) for it in splits["eval_items"]}
    assert groups
    for split, slice_name in groups:
        types = Counter(
            it["qtype"] for it in splits["eval_items"] if (it["split"], it["slice"]) == (split, slice_name)
        )
        assert max(types.values()) - min(types.values()) <= 1, (split, slice_name, types)


def small(rows: list[dict], seed: int = 42) -> dict:
    return make_splits(rows, seed=seed, n_test_seen=20, n_test_unseen=8, n_dev_seen=15, n_dev_unseen=5)


def test_same_seed_gives_the_same_splits():
    a, b = small(articles()), small(articles())
    assert a == b
    assert small(articles(), seed=1)["held_out"] != a["held_out"]


def test_short_articles_are_not_used_for_evaluation():
    rows = articles()
    rows[0]["text"] = "коротко"  # below min_chars_eval
    s = small(rows)
    assert rows[0]["doc_id"] not in {it["doc_id"] for it in s["eval_items"]}


def test_stats_reports_the_same_counts(splits):
    s = stats(splits)
    assert s["held_out_articles"] == len(splits["held_out"])
    assert sum(s["eval_articles"].values()) == len(splits["eval_items"])
