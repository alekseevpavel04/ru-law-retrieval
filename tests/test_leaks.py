"""Leakage tests on the built dataset (skipped if the dataset has not been built yet)."""

import json

import pytest

from rlr.data.parse import CORPUS, read_jsonl
from rlr.data.splits import SPLITS
from rlr.env import DATA
from rlr.gen.filters import normalize

DATASET = DATA / "dataset"
pytestmark = pytest.mark.skipif(not (DATASET / "test.jsonl").exists(), reason="dataset not built")


@pytest.fixture(scope="module")
def data():
    splits = json.loads((SPLITS / "splits.json").read_text(encoding="utf-8"))
    articles = {a["doc_id"]: a for a in read_jsonl(CORPUS / "articles.jsonl")}
    train = read_jsonl(DATASET / "train_llm.jsonl") + read_jsonl(DATASET / "train_titles.jsonl")
    dev = read_jsonl(DATASET / "dev.jsonl")
    test = read_jsonl(DATASET / "test.jsonl")
    golden = read_jsonl(DATASET / "golden.jsonl") if (DATASET / "golden.jsonl").exists() else []
    return splits, articles, train, dev, test, golden


def test_no_train_question_on_held_out_or_ood(data):
    splits, articles, train, *_ = data
    held = set(splits["held_out"])
    for q in train:
        assert q["doc_id"] not in held, q["qid"]
        assert articles[q["doc_id"]]["group"] == "in_domain", q["qid"]


def test_positive_and_negative_chunks_respect_articles(data):
    _, _, train, *_ = data
    for q in train:
        assert q["pos_chunk_id"].split("#")[0] == q["doc_id"]


def test_eval_texts_not_in_train_or_dev(data):
    _, _, train, dev, test, _ = data
    train_texts = {normalize(q["text"]) for q in train}
    dev_texts = {normalize(q["text"]) for q in dev}
    for q in test:
        assert normalize(q["text"]) not in train_texts, q["qid"]
        assert normalize(q["text"]) not in dev_texts, q["qid"]
    for q in dev:
        assert normalize(q["text"]) not in train_texts, q["qid"]


def test_dev_and_test_articles_disjoint(data):
    _, _, _, dev, test, _ = data
    assert not {q["doc_id"] for q in dev} & {q["doc_id"] for q in test}


def test_slices_are_consistent(data):
    splits, articles, _, dev, test, _ = data
    held = set(splits["held_out"])
    for q in dev + test:
        a = articles[q["doc_id"]]
        expected = "unseen_codes" if a["group"] == "ood" else ("unseen_articles" if q["doc_id"] in held else "seen")
        assert q["slice"] == expected, q["qid"]


def test_qrels_exist_in_corpus(data):
    _, articles, _, dev, test, golden = data
    for q in dev + test + golden:
        assert q["qrels"], q["qid"]
        for d in q["qrels"]:
            assert d in articles, (q["qid"], d)


def test_golden_is_subset_of_test(data):
    *_, test, golden = data
    test_ids = {q["qid"] for q in test}
    assert all(q["qid"] in test_ids for q in golden)


TK_HARD = DATASET / "tk_hard.jsonl"


@pytest.mark.skipif(not TK_HARD.exists(), reason="tk_hard not built")
def test_tk_hard_is_disjoint_from_dev_and_train(data):
    _, articles, train, dev, _, _ = data
    tk_hard = read_jsonl(TK_HARD)
    dev_articles = {q["doc_id"] for q in dev}
    train_texts = {normalize(q["text"]) for q in train}
    for extra in ("train_llm_v12.jsonl",):
        if (DATASET / extra).exists():
            train_texts |= {normalize(q["text"]) for q in read_jsonl(DATASET / extra)}
    for q in tk_hard:
        assert q["doc_id"] not in dev_articles, q["qid"]
        assert normalize(q["text"]) not in train_texts, q["qid"]
        assert articles[q["doc_id"]]["code"] == "tk"


@pytest.mark.skipif(not (DATASET / "train_llm_v12.jsonl").exists(), reason="v2 train not built")
def test_v2_train_respects_held_out_articles(data):
    splits, articles, *_ = data
    held = set(splits["held_out"])
    for q in read_jsonl(DATASET / "train_llm_v12.jsonl"):
        assert q["doc_id"] not in held and articles[q["doc_id"]]["group"] == "in_domain", q["qid"]
