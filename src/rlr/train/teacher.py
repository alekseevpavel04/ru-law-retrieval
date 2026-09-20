"""Teacher annotations for training (v2): the best baseline model (FRIDA) labels the training questions.

For every training question (train-only data, no dev/test involved):
- ``pos_text`` / ``pos_b_text``: the teacher's best chunk of the gold article in the main view (600/90 with the
  code name in the header) and in the tk-rf-rag view (500/75, "Статья N. Title" header);
- ``neg_text`` / ``neg_b_text``: a hard negative mined with the teacher (other articles only, candidates scoring
  >= ``margin`` x positive are skipped as likely false negatives, fallback as in ``mine_negatives``);
- ``cand_texts`` + ``labels``: the positive and ``n_cands`` best chunks of other articles with teacher cosine
  scores, for listwise distillation (DistillKLDivLoss);
- ``teacher_gold_rank``: rank of the gold article in the teacher's article ranking (MaxP over chunks), used to
  drop noisy synthetic questions the teacher cannot connect to their article.

Usage: python -m rlr teacher --train train_llm_v12 [--teacher FRIDA]
"""

import argparse
import json
from collections import Counter

import numpy as np
import torch

from rlr.config import load_yaml
from rlr.data.parse import read_jsonl, write_jsonl
from rlr.env import DATA, RESULTS
from rlr.eval.retrieve import DenseEncoder, aggregate_max, load_corpus

DATASET = DATA / "dataset"
TOP_K = 50
MARGIN = 0.95


def best_chunk_per_article(
    scores: np.ndarray, chunk_article: np.ndarray, n_articles: int
) -> tuple[np.ndarray, np.ndarray]:
    """For each query and article: max score and the index of the best chunk (queries x articles)."""
    best_score = np.full((scores.shape[0], n_articles), -np.inf, dtype=np.float32)
    best_idx = np.full((scores.shape[0], n_articles), -1, dtype=np.int64)
    for j, a in enumerate(chunk_article):
        better = scores[:, j] > best_score[:, a]
        best_score[better, a] = scores[better, j]
        best_idx[better, a] = j
    return best_score, best_idx


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr teacher")
    parser.add_argument("--train", default="train_llm_v12", help="train file name in data/dataset (without .jsonl)")
    parser.add_argument("--teacher", default="FRIDA", help="model name from configs/baselines.yaml")
    parser.add_argument("--n-cands", type=int, default=7)
    args = parser.parse_args(argv)

    cfg = next(m for m in load_yaml("baselines.yaml")["models"] if m["name"] == args.teacher)
    enc = DenseEncoder(
        cfg["path"], cfg["query_prompt"], cfg["doc_prompt"], name=cfg["name"], dtype=cfg.get("dtype", "float16")
    )
    main_view, b_view = load_corpus("chunk"), load_corpus("chunk_tkfmt")
    emb_main = enc.encode_corpus(main_view, cache_key=cfg["name"])
    emb_b = enc.encode_corpus(b_view, cache_key=cfg["name"])
    art_index = {d: i for i, d in enumerate(main_view.article_ids)}
    n_art = len(main_view.article_ids)

    rows = read_jsonl(DATASET / f"{args.train}.jsonl")
    q_emb = enc.encode_queries([r["text"] for r in rows])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    um, ub = torch.as_tensor(emb_main, device=dev), torch.as_tensor(emb_b, device=dev)
    stats: Counter = Counter()
    out = []
    for i in range(0, len(rows), 256):
        batch = rows[i : i + 256]
        q = torch.as_tensor(q_emb[i : i + 256], device=dev)
        s_main = (q @ um.T).float()
        s_b = (q @ ub.T).float()
        art_scores = aggregate_max(s_main, main_view.unit_article, n_art)
        ranks = (
            art_scores > art_scores.gather(1, torch.tensor([[art_index[r["doc_id"]]] for r in batch], device=dev))
        ).sum(1) + 1
        s_main_np, s_b_np = s_main.cpu().numpy(), s_b.cpu().numpy()
        best_m, idx_m = best_chunk_per_article(s_main_np, main_view.unit_article, n_art)
        best_b, idx_b = best_chunk_per_article(s_b_np, b_view.unit_article, n_art)
        top_scores, top_idx = torch.topk(s_main, TOP_K, dim=1)
        top_scores, top_idx = top_scores.cpu().numpy(), top_idx.cpu().numpy()
        for r_i, r in enumerate(batch):
            g = art_index[r["doc_id"]]
            pos_score = float(best_m[r_i, g])
            # distillation candidates: best chunk of each of the top other articles
            order = np.argsort(-best_m[r_i])
            cands = [a for a in order if a != g][: args.n_cands]
            # hard negative: best non-gold chunk under the margin rule; fallback = lowest non-gold chunk
            # in the top-k. An article long enough to fill the whole top-k (koap-19.5 has 72 chunks)
            # leaves no non-gold chunk there at all, so the last resort is the best chunk of the best
            # other article - never an out-of-range index.
            neg_j, fallback = -1, -1
            for s, j in zip(top_scores[r_i], top_idx[r_i], strict=True):
                if main_view.unit_article[j] == g:
                    continue
                fallback = j
                if s < MARGIN * pos_score and neg_j < 0:
                    neg_j = j
            if neg_j >= 0:
                stats["neg_margin_rule"] += 1
            elif fallback >= 0:
                stats["neg_fallback"] += 1
                neg_j = fallback
            else:
                stats["neg_best_other_article"] += 1
                neg_j = int(idx_m[r_i, cands[0]])
            neg_art = main_view.unit_article[neg_j]
            out.append(
                {
                    **{k: r[k] for k in ("qid", "text", "qtype", "code", "doc_id") if k in r},
                    "prompt_version": r.get("prompt_version", "v1"),
                    "teacher_gold_rank": int(ranks[r_i]),
                    "pos_chunk_id": main_view.unit_ids[idx_m[r_i, g]],
                    "pos_text": main_view.texts[idx_m[r_i, g]],
                    "pos_b_text": b_view.texts[idx_b[r_i, g]],
                    "neg_chunk_id": main_view.unit_ids[neg_j],
                    "neg_text": main_view.texts[neg_j],
                    "neg_b_text": b_view.texts[idx_b[r_i, neg_art]],
                    "cand_texts": [main_view.texts[idx_m[r_i, a]] for a in cands],
                    "cand_b_texts": [b_view.texts[idx_b[r_i, a]] for a in cands],
                    "labels": [pos_score] + [float(best_m[r_i, a]) for a in cands],
                    "labels_b": [float(best_b[r_i, g])] + [float(best_b[r_i, a]) for a in cands],
                }
            )
    write_jsonl(DATASET / f"{args.train}_teacher.jsonl", out)
    ranks_all = np.array([r["teacher_gold_rank"] for r in out])
    report = {
        "teacher": cfg["path"],
        "train": args.train,
        "rows": len(out),
        **dict(stats),
        "teacher_gold_rank": {f"<= {k}": float((ranks_all <= k).mean()) for k in (1, 5, 10, 20, 50, 100)},
        "labels_mean": float(np.mean([r["labels"][0] for r in out])),
        "cands_mean": float(np.mean([np.mean(r["labels"][1:]) for r in out])),
    }
    (RESULTS / f"teacher_{args.train}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
