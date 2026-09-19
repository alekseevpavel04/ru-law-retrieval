import json

import numpy as np

import rlr.train.train as train_mod
from rlr.train.teacher import best_chunk_per_article
from rlr.train.train import load_teacher_rows, to_distill_dataset


def test_best_chunk_per_article():
    scores = np.array([[0.1, 0.9, 0.5, 0.2]], dtype=np.float32)
    best, idx = best_chunk_per_article(scores, np.array([0, 0, 1, 1]), 2)
    assert best.tolist() == [[np.float32(0.9), np.float32(0.5)]]
    assert idx.tolist() == [[1, 2]]


def _row(i: int, rank: int) -> dict:
    return {
        "qid": f"q{i}",
        "text": f"вопрос {i}",
        "qtype": "legal",
        "code": "tk",
        "doc_id": f"tk-{i}",
        "prompt_version": "v1" if i % 2 else "v2",
        "teacher_gold_rank": rank,
        "pos_text": "A",
        "pos_b_text": "B",
        "neg_text": "nA",
        "neg_b_text": "nB",
        "neg_chunk_id": "gk-1#0",
        "cand_texts": ["cA1", "cA2"],
        "cand_b_texts": ["cB1", "cB2"],
        "labels": [0.9, 0.5, 0.4],
        "labels_b": [0.8, 0.5, 0.3],
    }


def test_teacher_rows_filter_versions_and_format_aug(tmp_path, monkeypatch):
    rows = [_row(i, rank=1 if i < 8 else 99) for i in range(10)]
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(train_mod, "DATASET", tmp_path)
    out, info = load_teacher_rows({"train_file": "t", "teacher_filter_rank": 50, "seed": 0})
    assert info["dropped_by_teacher_rank"] == 2 and len(out) == 8
    out, _ = load_teacher_rows({"train_file": "t", "prompt_versions": ["v1"], "seed": 0})
    assert {r["prompt_version"] for r in out} == {"v1"}
    out, info = load_teacher_rows({"train_file": "t", "format_aug": 1.0, "seed": 0})
    assert info["format_aug_rows"] == 10
    assert all(r["pos_text"] == "B" and r["neg_text"] == "nB" and r["labels"] == [0.8, 0.5, 0.3] for r in out)


def test_distill_dataset_columns_and_scaled_labels():
    ds = to_distill_dataset([_row(1, 1)], teacher_temperature=0.05)
    assert ds.column_names == ["query", "positive", "cand_0", "cand_1", "label"]
    assert np.allclose(ds[0]["label"], [18.0, 10.0, 8.0])
