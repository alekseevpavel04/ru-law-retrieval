"""Hard negative mining that knows about articles.

The stock ``mine_hard_negatives`` does not know that other chunks of the gold
article are not negatives. Algorithm (per training query, with the base model):

1. score all chunks, take top-50;
2. drop every chunk of the gold article;
3. keep candidates with score < ``margin`` x score(query, positive) (likely false negatives removed);
4. the negative is the best remaining candidate.

For title pairs the positive is a chunk body without the header line, so the
negative is also taken without its header (same text format on both sides).

Usage: python -m rlr mine --base intfloat/multilingual-e5-small --name e5-small
"""

import argparse
from collections import Counter

import numpy as np
import torch

from rlr.data.parse import read_jsonl, write_jsonl
from rlr.env import DATA
from rlr.eval.retrieve import DenseEncoder, load_corpus

DATASET = DATA / "dataset"
TOP_K = 50
MARGIN = 0.95


def mine(
    q_emb: np.ndarray,
    pos_scores: np.ndarray,
    gold_article: np.ndarray,
    chunk_emb: np.ndarray,
    chunk_article: np.ndarray,
    top_k: int = TOP_K,
    margin: float = MARGIN,
) -> tuple[np.ndarray, Counter]:
    """Return the index of the negative chunk per query (-1 if none) and drop reasons."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    chunks = torch.as_tensor(chunk_emb, device=dev)
    art = torch.as_tensor(chunk_article, device=dev)
    out = np.full(len(q_emb), -1, dtype=np.int64)
    stats: Counter = Counter()
    for i in range(0, len(q_emb), 512):
        q = torch.as_tensor(q_emb[i : i + 512], device=dev)
        scores, idx = (q @ chunks.T).topk(top_k, dim=1)
        gold = torch.as_tensor(gold_article[i : i + 512], device=dev).unsqueeze(1)
        same_article = art[idx] == gold
        pos = torch.as_tensor(pos_scores[i : i + 512], device=dev).unsqueeze(1)
        too_close = scores >= margin * pos
        ok = ~same_article & ~too_close
        stats["candidates_same_article"] += int(same_article.sum())
        stats["candidates_too_close"] += int((too_close & ~same_article).sum())
        first_ok = torch.where(ok.any(dim=1), ok.float().argmax(dim=1), torch.full_like(ok[:, 0], -1, dtype=torch.long))
        for r, j in enumerate(first_ok.cpu().numpy()):
            out[i + r] = int(idx[r, j]) if j >= 0 else -1
    stats["queries_without_negative"] = int((out < 0).sum())
    return out, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr mine")
    parser.add_argument("--base", default="intfloat/multilingual-e5-small")
    parser.add_argument("--name", default="e5-small")
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args(argv)

    enc = DenseEncoder(args.base, "query: ", "passage: ", max_seq_length=args.max_seq_length, name=args.name)
    corpus = load_corpus("chunk")
    chunks = read_jsonl(DATA / "corpus" / "chunks.jsonl")
    chunk_emb = enc.encode_corpus(corpus, cache_key=args.name)
    art_index = {d: i for i, d in enumerate(corpus.article_ids)}

    for source in ("llm", "titles"):
        rows = read_jsonl(DATASET / f"train_{source}.jsonl")
        q_emb = enc.encode_queries([r["text"] for r in rows])
        if source == "titles":
            # score against body-only texts, same format as the positive
            bodies = [c["body"] for c in chunks]
            cand_emb = enc.encode(bodies, "passage: ", batch_size=64)
        else:
            cand_emb = chunk_emb
        pos_emb = enc.encode([r["pos_text"] for r in rows], "passage: ", batch_size=64)
        pos_scores = (q_emb * pos_emb).sum(axis=1)
        gold = np.array([art_index[r["doc_id"]] for r in rows])
        neg, stats = mine(q_emb, pos_scores, gold, cand_emb, corpus.unit_article)
        for r, j in zip(rows, neg, strict=True):
            if j < 0:
                r["neg_chunk_id"], r["neg_text"] = None, None
                continue
            c = chunks[j]
            r["neg_chunk_id"] = c["chunk_id"]
            r["neg_text"] = c["body"] if source == "titles" else c["text"]
        write_jsonl(DATASET / f"train_{source}_hn_{args.name}.jsonl", rows)
        print(source, len(rows), dict(stats))


if __name__ == "__main__":
    main()
