"""Golden subset of the test set: selection, candidate pools, final qrels.

select: 180 test questions stratified by slice x question type; each gets a pool of
        5 candidate articles (BM25 + best baseline, gold excluded) for multi-relevance.
build:  annotations -> ``data/dataset/golden.jsonl`` with multi-relevance qrels,
        plus agreement stats with the automatic (synthetic) qrels.

Usage:
  python -m rlr golden select --dense e5-large --n 180
  python -m rlr golden build
"""

import argparse
import json
import random
from collections import Counter, defaultdict

from rlr.config import codes_config
from rlr.data.chunk import doc_header
from rlr.data.parse import CORPUS, read_jsonl, write_jsonl
from rlr.env import DATA, RESULTS
from rlr.eval.retrieve import load_run

GOLDEN = DATA / "golden"
DATASET = DATA / "dataset"
RUNS = DATA / "runs"
N_CANDIDATES = 5


def stratified_pick(test: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for q in test:
        groups[(q["slice"], q["qtype"])].append(q)
    per_group = n // len(groups)
    picked = []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda q: q["qid"])
        picked += rng.sample(items, min(per_group, len(items)))
    return sorted(picked, key=lambda q: q["qid"])


def candidate_pool(qid: str, gold: str, runs: list[dict], k: int = N_CANDIDATES) -> list[tuple[str, str]]:
    """Interleave BM25 and dense rankings (gold excluded) until k unique candidates."""
    out: list[tuple[str, str]] = []
    seen = {gold}
    depth = 0
    while len(out) < k and depth < 100:
        for name, run in runs:
            ranking = run.get(qid, [])
            if depth < len(ranking):
                d = ranking[depth][0]
                if d not in seen:
                    seen.add(d)
                    out.append((d, name))
                    if len(out) == k:
                        break
        depth += 1
    return out


def select(dense: str, n: int, seed: int) -> None:
    display = codes_config()["display"]
    articles = {a["doc_id"]: a for a in read_jsonl(CORPUS / "articles.jsonl")}
    test = read_jsonl(DATASET / "test.jsonl")
    picked = stratified_pick(test, n, seed)
    runs = [(name, load_run(RUNS / name / "test_chunk.jsonl")) for name in ("BM25", dense)]

    def art(doc_id: str) -> dict:
        a = articles[doc_id]
        return {"doc_id": doc_id, "title": doc_header(display[a["code"]], a["number"], a["title"]), "text": a["text"]}

    items = []
    for q in picked:
        pool = candidate_pool(q["qid"], q["doc_id"], runs)
        items.append(
            {
                "qid": q["qid"],
                "text": q["text"],
                "slice": q["slice"],
                "qtype": q["qtype"],
                "code": q["code"],
                "gold": art(q["doc_id"]),
                "candidates": [{**art(d), "source": src} for d, src in pool],
            }
        )
    GOLDEN.mkdir(parents=True, exist_ok=True)
    write_jsonl(GOLDEN / "items.jsonl", items)
    meta = {"n": len(items), "seed": seed, "pool": ["BM25", dense], "candidates": N_CANDIDATES}
    meta["by_slice_type"] = {
        f"{s}/{t}": c for (s, t), c in sorted(Counter((i["slice"], i["qtype"]) for i in items).items())
    }
    (GOLDEN / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


def latest_annotations() -> dict[str, dict]:
    path = GOLDEN / "annotations.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for r in read_jsonl(path):
        out[r["qid"]] = r  # the last answer wins
    return out


def build() -> None:
    items = {i["qid"]: i for i in read_jsonl(GOLDEN / "items.jsonl")}
    ann = latest_annotations()
    test = {q["qid"]: q for q in read_jsonl(DATASET / "test.jsonl")}
    golden, stats = [], Counter()
    extra_counts = []
    for qid, item in items.items():
        a = ann.get(qid)
        if a is None:
            stats["not_annotated"] += 1
            continue
        stats[f"question_{a['question_status']}"] += 1
        if a["question_status"] == "drop":
            continue
        qrels = {d: 1 for d in a.get("extra_relevant", [])}
        if a["gold_relevant"]:
            qrels[item["gold"]["doc_id"]] = 1
        else:
            stats["gold_not_relevant"] += 1
        extra_counts.append(len(a.get("extra_relevant", [])))
        if not qrels:
            stats["dropped_no_relevant"] += 1
            continue
        q = dict(test[qid])
        q["qid"] = qid
        if a["question_status"] == "fix" and a.get("fixed_text"):
            q["text"] = a["fixed_text"]
        q["qrels"] = qrels
        q["qrels_auto"] = {item["gold"]["doc_id"]: 1}
        q["annotator"] = a.get("annotator", "")
        golden.append(q)
    write_jsonl(DATASET / "golden.jsonl", golden)
    kept = golden
    report = {
        "items": len(items),
        "annotated": len(ann),
        "golden_queries": len(golden),
        **dict(stats),
        "share_gold_relevant": round(1 - stats["gold_not_relevant"] / max(len(extra_counts), 1), 4),
        "share_with_extra_relevant": round(sum(c > 0 for c in extra_counts) / max(len(extra_counts), 1), 4),
        "mean_extra_relevant": round(sum(extra_counts) / max(len(extra_counts), 1), 4),
        "mean_relevant_per_query": round(sum(len(q["qrels"]) for q in kept) / max(len(kept), 1), 4),
        "annotators": dict(Counter(a.get("annotator", "") for a in ann.values())),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "golden_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr golden")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--dense", required=True, help="best baseline run name for the candidate pool")
    s.add_argument("--n", type=int, default=180)
    s.add_argument("--seed", type=int, default=42)
    sub.add_parser("build")
    args = parser.parse_args(argv)
    if args.cmd == "select":
        select(args.dense, args.n, args.seed)
    else:
        build()


if __name__ == "__main__":
    main()
