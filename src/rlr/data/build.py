"""Build the final dataset from raw generations.

dev/test: generation -> filters (no n-gram copy filter, see DECISIONS.md) -> LLM judge -> exact dedup.
train:    generation -> filters -> exact dedup -> near-dup dedup inside train ->
          near-dups of dev/test questions are removed from train (never from test) ->
          positive chunk = best chunk of the own article by base e5-small.
titles:   free pairs "article title -> first chunk body without the header line".

Every drop is counted per filter and saved to ``results/dataset_stats.json``.

Usage: python -m rlr build-dataset
"""

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import torch

from rlr.config import codes_config
from rlr.data.parse import CORPUS, read_jsonl, write_jsonl
from rlr.data.splits import SPLITS
from rlr.env import DATA, RESULTS
from rlr.gen.filters import exact_duplicates, normalize, run_filters
from rlr.gen.generate import article_parts
from rlr.gen.prompts import QUESTION_TYPES

GEN = DATA / "gen"
DATASET = DATA / "dataset"
NEAR_DUP = 0.95  # cosine of base e5-small query embeddings


def load_encoder():
    from rlr.eval.retrieve import DenseEncoder

    return DenseEncoder("intfloat/multilingual-e5-small", "query: ", "passage: ", name="e5-small")


def near_dup_within(emb: np.ndarray, threshold: float) -> np.ndarray:
    """True for items that have a near-duplicate anywhere earlier in the list.

    Simple deterministic rule (keeps the first occurrence); in a chain A~B~C both
    B and C are dropped even if C is not close to A, which is fine for dedup.
    """
    x = torch.as_tensor(emb, device="cuda" if torch.cuda.is_available() else "cpu")
    flags = []
    for i in range(0, len(emb), 1024):
        sims = x[i : i + 1024] @ x.T
        rows = torch.arange(sims.shape[0], device=x.device) + i
        later_or_self = torch.arange(len(emb), device=x.device).unsqueeze(0) >= rows.unsqueeze(1)
        sims[later_or_self] = -1
        flags.append((sims >= threshold).any(dim=1).cpu().numpy())
    return np.concatenate(flags) if flags else np.zeros(0, dtype=bool)


def near_dup_against(emb: np.ndarray, ref: np.ndarray, threshold: float) -> np.ndarray:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x, r = torch.as_tensor(emb, device=dev), torch.as_tensor(ref, device=dev)
    out = []
    for i in range(0, len(emb), 2048):
        out.append(((x[i : i + 2048] @ r.T) >= threshold).any(dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=bool)


def build_eval(articles: dict, splits: dict, counts: Counter) -> list[dict]:
    judged = {r["key"]: r["answerable"] for r in read_jsonl(GEN / "eval_judged.jsonl")}
    rows = read_jsonl(GEN / "eval_raw.jsonl")
    out = []
    for r in rows:
        counts["eval_generated"] += 1
        q = (r.get("output") or {}).get("question", "")
        if not q:
            counts["eval_drop_invalid_json"] += 1
            continue
        text = article_parts(articles[r["doc_id"]]["text"])[r["part"]]
        # the n-gram copy filter is train-only: for dev/test it removed ~18% of "legal" questions
        # (legal terms are long fixed phrases); lexical overlap is measured instead (analysis lexical)
        reason = run_filters(q, r["qtype"], text, copy_check=False)
        if reason:
            counts[f"eval_drop_{reason}"] += 1
            continue
        if judged.get(r["key"]) != "yes":
            counts["eval_drop_judge_" + str(judged.get(r["key"], "missing"))] += 1
            continue
        out.append(
            {
                "qid": f"{r['split']}-{r['doc_id']}",
                "text": q.strip(),
                "split": r["split"],
                "slice": r["slice"],
                "qtype": r["qtype"],
                "code": articles[r["doc_id"]]["code"],
                "doc_id": r["doc_id"],
                "qrels": {r["doc_id"]: 1},
                "generator": r["model"],
            }
        )
    dup = exact_duplicates(q["text"] for q in out)
    counts["eval_drop_exact_dup"] += sum(dup)
    return [q for q, d in zip(out, dup, strict=True) if not d]


def build_train_raw(
    articles: dict, splits: dict, counts: Counter, raw_files: tuple[str, ...] = ("train_raw.jsonl",)
) -> list[dict]:
    held = set(splits["held_out"])
    out = []
    raw_rows = [r for f in raw_files for r in read_jsonl(GEN / f)]
    for r in raw_rows:
        a = articles[r["doc_id"]]
        assert r["doc_id"] not in held and a["group"] == "in_domain"
        text = article_parts(a["text"])[r["part"]]
        for qtype in QUESTION_TYPES:
            counts["train_generated"] += 1
            q = ((r.get("output") or {}).get(qtype) or "").strip()
            if r.get("output") is None:
                counts["train_drop_invalid_json"] += 1
                continue
            reason = run_filters(q, qtype, text)
            if reason:
                counts[f"train_drop_{reason}"] += 1
                continue
            version = r.get("prompt_version", "v1")
            out.append(
                {
                    "qid": f"tr-{r['doc_id']}-{r['part']}-{qtype}" + ("" if version == "v1" else f"-{version}"),
                    "prompt_version": version,
                    "text": q,
                    "split": "train",
                    "qtype": qtype,
                    "code": a["code"],
                    "doc_id": r["doc_id"],
                    "part": r["part"],
                    "generator": r["model"],
                }
            )
    dup = exact_duplicates(q["text"] for q in out)
    counts["train_drop_exact_dup"] += sum(dup)
    return [q for q, d in zip(out, dup, strict=True) if not d]


def build_tkhard(articles: dict, train_emb_sets: list[np.ndarray], enc, counts: Counter) -> list[dict]:
    """Harder extra TK test set (everyday + search). Test-only: items too close to ANY train question
    (v1 or v2) are dropped from the test side, so no model saw a near-copy of a test question."""
    judged = {r["key"]: r["answerable"] for r in read_jsonl(GEN / "tkhard_judged.jsonl")}
    out = []
    for r in read_jsonl(GEN / "tkhard_raw.jsonl"):
        counts["tkhard_generated"] += 1
        q = ((r.get("output") or {}).get("question") or "").strip()
        if not q:
            counts["tkhard_drop_invalid_json"] += 1
            continue
        text = article_parts(articles[r["doc_id"]]["text"])[r["part"]]
        reason = run_filters(q, r["qtype"], text, copy_check=False)
        if reason:
            counts[f"tkhard_drop_{reason}"] += 1
            continue
        if judged.get(r["key"]) != "yes":
            counts["tkhard_drop_judge_" + str(judged.get(r["key"], "missing"))] += 1
            continue
        out.append(
            {
                "qid": f"tkhard-{r['doc_id']}-{r['qtype']}",
                "text": q,
                "split": "tk_hard",
                "slice": r["slice"],
                "qtype": r["qtype"],
                "code": "tk",
                "doc_id": r["doc_id"],
                "qrels": {r["doc_id"]: 1},
                "generator": r["model"],
            }
        )
    dup = exact_duplicates(q["text"] for q in out)
    counts["tkhard_drop_exact_dup"] += sum(dup)
    out = [q for q, d in zip(out, dup, strict=True) if not d]
    emb = enc.encode_queries([q["text"] for q in out])
    near = np.zeros(len(out), dtype=bool)
    for ref in train_emb_sets:
        near |= near_dup_against(emb, ref, NEAR_DUP)
    counts["tkhard_drop_near_dup_of_train"] += int(near.sum())
    return [q for q, d in zip(out, near, strict=True) if not d]


def build_train_extra(raw_files: tuple[str, ...], out_name: str) -> None:
    """Train set from several raw generation files (v1 + v2) without touching dev/test/titles.

    Same filters and dedup as the main build; near-duplicates of dev, test and tk_hard questions are removed
    from train. Positive chunk = best chunk of the own article by base e5-small (as in v1)."""
    counts: Counter = Counter()
    articles = {a["doc_id"]: a for a in read_jsonl(CORPUS / "articles.jsonl")}
    splits = json.loads((SPLITS / "splits.json").read_text(encoding="utf-8"))
    chunks = read_jsonl(CORPUS / "chunks.jsonl")
    chunks_by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        chunks_by_doc[c["doc_id"]].append(c)
    train_q = build_train_raw(articles, splits, counts, raw_files)
    eval_q = read_jsonl(DATASET / "dev.jsonl") + read_jsonl(DATASET / "test.jsonl")
    if (DATASET / "tk_hard.jsonl").exists():
        eval_q += read_jsonl(DATASET / "tk_hard.jsonl")
    enc = load_encoder()
    train_emb = enc.encode_queries([q["text"] for q in train_q])
    eval_emb = enc.encode_queries([q["text"] for q in eval_q])
    within = near_dup_within(train_emb, NEAR_DUP)
    against = near_dup_against(train_emb, eval_emb, NEAR_DUP) & ~within
    counts["train_drop_near_dup"] += int(within.sum())
    counts["train_drop_near_dup_of_eval"] += int(against.sum())
    keep = ~(within | against)
    train_q = [q for q, k in zip(train_q, keep, strict=True) if k]
    train_emb = train_emb[keep]
    chunk_emb = enc.encode_corpus(_chunk_corpus(chunks), cache_key="e5-small")
    chunk_index = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    for q, e in zip(train_q, train_emb, strict=True):
        own = chunks_by_doc[q["doc_id"]]
        best = int(np.argmax(chunk_emb[[chunk_index[c["chunk_id"]] for c in own]] @ e))
        q["pos_chunk_id"], q["pos_text"] = own[best]["chunk_id"], own[best]["text"]
    write_jsonl(DATASET / f"{out_name}.jsonl", train_q)
    stats = {
        "raw_files": list(raw_files),
        "filters": dict(sorted(counts.items())),
        "train": len(train_q),
        "by_prompt_version": dict(Counter(q["prompt_version"] for q in train_q)),
        "by_type": dict(Counter(q["qtype"] for q in train_q)),
        "by_code": dict(Counter(q["code"] for q in train_q)),
    }
    (RESULTS / f"dataset_stats_{out_name}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def build_tkhard_main() -> None:
    counts: Counter = Counter()
    articles = {a["doc_id"]: a for a in read_jsonl(CORPUS / "articles.jsonl")}
    enc = load_encoder()
    refs = []
    for name in ("train_llm", "train_titles", "train_llm_v12"):
        if (DATASET / f"{name}.jsonl").exists():
            refs.append(enc.encode_queries([q["text"] for q in read_jsonl(DATASET / f"{name}.jsonl")]))
    if (GEN / "train_raw_v2.jsonl").exists():  # raw v2 questions too (before its own dedup)
        v2 = [
            ((r.get("output") or {}).get(t) or "").strip()
            for r in read_jsonl(GEN / "train_raw_v2.jsonl")
            for t in QUESTION_TYPES
        ]
        refs.append(enc.encode_queries([t for t in v2 if t]))
    rows = build_tkhard(articles, refs, enc, counts)
    write_jsonl(DATASET / "tk_hard.jsonl", rows)
    stats = {
        "filters": dict(sorted(counts.items())),
        "tk_hard": len(rows),
        "by_slice_type": {
            f"{s}/{t}": n for (s, t), n in sorted(Counter((q["slice"], q["qtype"]) for q in rows).items())
        },
    }
    (RESULTS / "dataset_stats_tk_hard.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr build-dataset")
    parser.add_argument(
        "--train-raw", nargs="*", help="build only a train set from these raw files (one per generation pass)"
    )
    parser.add_argument("--train-out", default="train_llm_v12")
    parser.add_argument("--tk-hard", action="store_true", help="build only the tk_hard test set")
    args = parser.parse_args(argv)
    if args.tk_hard:
        build_tkhard_main()
        return
    if args.train_raw:
        build_train_extra(tuple(args.train_raw), args.train_out)
        return
    counts: Counter = Counter()
    articles = {a["doc_id"]: a for a in read_jsonl(CORPUS / "articles.jsonl")}
    splits = json.loads((SPLITS / "splits.json").read_text(encoding="utf-8"))
    chunks = read_jsonl(CORPUS / "chunks.jsonl")
    chunks_by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        chunks_by_doc[c["doc_id"]].append(c)

    eval_q = build_eval(articles, splits, counts)
    train_q = build_train_raw(articles, splits, counts)

    enc = load_encoder()
    train_emb = enc.encode_queries([q["text"] for q in train_q])
    eval_emb = enc.encode_queries([q["text"] for q in eval_q])

    within = near_dup_within(train_emb, NEAR_DUP)
    counts["train_drop_near_dup"] += int(within.sum())
    against = near_dup_against(train_emb, eval_emb, NEAR_DUP) & ~within
    counts["train_drop_near_dup_of_eval"] += int(against.sum())
    keep = ~(within | against)
    train_q = [q for q, k in zip(train_q, keep, strict=True) if k]
    train_emb = train_emb[keep]

    # positive chunk: the best chunk of the own article by the base model
    chunk_emb = enc.encode_corpus(_chunk_corpus(chunks), cache_key="e5-small")
    chunk_index = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    for q, e in zip(train_q, train_emb, strict=True):
        own = chunks_by_doc[q["doc_id"]]
        idx = [chunk_index[c["chunk_id"]] for c in own]
        best = int(np.argmax(chunk_emb[idx] @ e))
        q["pos_chunk_id"] = own[best]["chunk_id"]
        q["pos_text"] = own[best]["text"]

    # free pairs: title -> first chunk body (without the header line)
    display = codes_config()["display"]
    titles = []
    for doc_id in splits["train_articles"]:
        a = articles[doc_id]
        first = chunks_by_doc[doc_id][0]
        if not first["body"].strip() or first["body"] == first["header"]:
            counts["titles_drop_no_body"] += 1
            continue
        titles.append(
            {
                "qid": f"title-{doc_id}",
                "text": a["title"],
                "split": "train",
                "qtype": "title",
                "code": a["code"],
                "doc_id": doc_id,
                "pos_chunk_id": first["chunk_id"],
                "pos_text": first["body"],
                "code_display": display[a["code"]],
            }
        )

    # the same rule for title pairs: a title that (nearly) equals a dev/test query is removed from train
    title_emb = enc.encode_queries([t["text"] for t in titles])
    eval_norm = {normalize(q["text"]) for q in eval_q}
    title_dup = near_dup_against(title_emb, eval_emb, NEAR_DUP) | np.array(
        [normalize(t["text"]) in eval_norm for t in titles], dtype=bool
    )
    counts["titles_drop_near_dup_of_eval"] += int(title_dup.sum())
    titles = [t for t, d in zip(titles, title_dup, strict=True) if not d]

    DATASET.mkdir(parents=True, exist_ok=True)
    write_jsonl(DATASET / "train_llm.jsonl", train_q)
    write_jsonl(DATASET / "train_titles.jsonl", titles)
    write_jsonl(DATASET / "dev.jsonl", [q for q in eval_q if q["split"] == "dev"])
    write_jsonl(DATASET / "test.jsonl", [q for q in eval_q if q["split"] == "test"])

    stats = {
        "filters": dict(sorted(counts.items())),
        "near_dup_threshold": NEAR_DUP,
        "train_llm": len(train_q),
        "train_titles": len(titles),
        "train_by_type": dict(Counter(q["qtype"] for q in train_q)),
        "train_by_code": dict(Counter(q["code"] for q in train_q)),
        "train_articles_covered": len({q["doc_id"] for q in train_q}),
        "eval": {f"{s}/{sl}": n for (s, sl), n in sorted(Counter((q["split"], q["slice"]) for q in eval_q).items())},
        "eval_by_type": {
            f"{s}/{t}": n for (s, t), n in sorted(Counter((q["split"], q["qtype"]) for q in eval_q).items())
        },
        "test_by_code": dict(Counter(q["code"] for q in eval_q if q["split"] == "test")),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def _chunk_corpus(chunks: list[dict]):
    from rlr.eval.retrieve import load_corpus

    corpus = load_corpus("chunk")
    assert corpus.unit_ids == [c["chunk_id"] for c in chunks]
    return corpus


if __name__ == "__main__":
    main()
