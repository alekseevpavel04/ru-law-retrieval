"""Evaluate retrievers on dev / test / golden / external query sets, both protocols.

Per-query metrics go to ``results/per_query/<model>.csv`` (used by bootstrap and
slice analysis), rankings (top-100 articles) to ``data/runs/<model>/<set>_<protocol>.jsonl``,
summaries to ``results/summary/<model>.json``.

Usage:
  python -m rlr baselines [--models e5-small bge-m3 ...] [--sets dev test] [--bm25]
  python -m rlr baselines --hybrid BM25 e5-large          # RRF of two saved runs
  python -m rlr evaluate --path models/xxx --name ft-e5-small --query-prompt "query: " --doc-prompt "passage: "
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from rlr.config import load_yaml
from rlr.data.parse import read_jsonl
from rlr.env import DATA, RESULTS
from rlr.eval.metrics import METRICS, mean_metrics, query_metrics
from rlr.eval.retrieve import BM25Retriever, DenseEncoder, dense_search, load_corpus, load_run, rrf, save_run

DATASET = DATA / "dataset"
RUNS = DATA / "runs"
PROTOCOLS = ("article", "chunk")
EXTERNAL = DATA / "external" / "tk_rf_rag_questions.jsonl"
TK_RF_RAG_QUESTIONS = Path(r"D:\VScode_projects\github-portfolio\tk-rf-rag\eval\questions.jsonl")


def prepare_external() -> None:
    """tk-rf-rag questions -> our format (article numbers -> ``tk-<n>``), answerable only."""
    if EXTERNAL.exists() or not TK_RF_RAG_QUESTIONS.exists():
        return
    rows = []
    for r in read_jsonl(TK_RF_RAG_QUESTIONS):
        if not r["articles"]:
            continue
        rows.append(
            {
                "qid": f"ext-{r['id']}",
                "text": r["question"],
                "split": "external",
                "slice": "tk_rf_rag",
                "qtype": "external",
                "code": "tk",
                "doc_id": f"tk-{r['articles'][0]}",
                "qrels": {f"tk-{a}": 1 for a in r["articles"]},
            }
        )
    EXTERNAL.parent.mkdir(parents=True, exist_ok=True)
    with EXTERNAL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_query_sets(names: list[str]) -> dict[str, list[dict]]:
    prepare_external()
    files = {
        "dev": DATASET / "dev.jsonl",
        "test": DATASET / "test.jsonl",
        "golden": DATASET / "golden.jsonl",
        "external": EXTERNAL,
    }
    out = {}
    for n in names:
        if files[n].exists():
            out[n] = read_jsonl(files[n])
    return out


def score_run(run: list[list[tuple[str, float]]], queries: list[dict]) -> list[dict]:
    rows = []
    for q, ranking in zip(queries, run, strict=True):
        m = query_metrics([d for d, _ in ranking], q["qrels"])
        rows.append({"qid": q["qid"], "slice": q.get("slice", ""), "qtype": q.get("qtype", ""), "code": q["code"], **m})
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {"all": {**mean_metrics(rows), "n": len(rows)}}
    for key in ("slice", "qtype", "code"):
        for value in sorted({r[key] for r in rows}):
            sub = [r for r in rows if r[key] == value]
            out[f"{key}={value}"] = {**mean_metrics(sub), "n": len(sub)}
    return out


def write_results(name: str, per_query: list[dict], meta: dict) -> None:
    (RESULTS / "per_query").mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_query)
    df.to_csv(RESULTS / "per_query" / f"{name}.csv", index=False, float_format="%.5f")
    summary = {}
    for (qset, protocol), g in df.groupby(["set", "protocol"]):
        summary[f"{qset}/{protocol}"] = summarize(g.to_dict("records"))
    (RESULTS / "summary" / f"{name}.json").write_text(
        json.dumps({"model": name, **meta, "results": summary}, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def eval_dense(
    enc: DenseEncoder, query_sets: dict[str, list[dict]], protocols=PROTOCOLS, cache_key: str | None = None
) -> tuple[list[dict], dict]:
    per_query, timing = [], {}
    for protocol in protocols:
        corpus = load_corpus(protocol)
        t0 = time.time()
        unit_emb = enc.encode_corpus(corpus, cache_key=cache_key)
        timing[f"encode_{protocol}_s"] = round(time.time() - t0, 1)
        for qset, queries in query_sets.items():
            q_emb = enc.encode_queries([q["text"] for q in queries])
            run = dense_search(q_emb, unit_emb, corpus)
            save_run(RUNS / enc.name / f"{qset}_{protocol}.jsonl", [q["qid"] for q in queries], run)
            per_query += [{"set": qset, "protocol": protocol, **r} for r in score_run(run, queries)]
    return per_query, timing


def eval_bm25(query_sets: dict[str, list[dict]], cfg: dict) -> list[dict]:
    per_query = []
    for protocol in PROTOCOLS:
        retr = BM25Retriever(load_corpus(protocol), cfg["k1"], cfg["b"], cfg["stemmer"])
        for qset, queries in query_sets.items():
            run = retr.search([q["text"] for q in queries])
            save_run(RUNS / "BM25" / f"{qset}_{protocol}.jsonl", [q["qid"] for q in queries], run)
            per_query += [{"set": qset, "protocol": protocol, **r} for r in score_run(run, queries)]
    return per_query


def eval_hybrid(names: list[str], query_sets: dict[str, list[dict]], k: int) -> tuple[str, list[dict]]:
    hybrid = "RRF(" + "+".join(names) + ")"
    per_query = []
    for protocol in PROTOCOLS:
        for qset, queries in query_sets.items():
            runs = []
            for n in names:
                saved = load_run(RUNS / n / f"{qset}_{protocol}.jsonl")
                runs.append([saved[q["qid"]] for q in queries])
            run = rrf(runs, k=k)
            save_run(RUNS / hybrid / f"{qset}_{protocol}.jsonl", [q["qid"] for q in queries], run)
            per_query += [{"set": qset, "protocol": protocol, **r} for r in score_run(run, queries)]
    return hybrid, per_query


def print_table(names: list[str], qset: str = "test") -> None:
    rows = []
    for n in names:
        p = RESULTS / "summary" / f"{n}.json"
        if not p.exists():
            continue
        res = json.loads(p.read_text(encoding="utf-8"))["results"]
        for protocol in PROTOCOLS:
            r = res.get(f"{qset}/{protocol}")
            if r:
                rows.append({"model": n, "protocol": protocol, **{m: round(r["all"][m], 4) for m in METRICS}})
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr baselines")
    parser.add_argument("--config", default="baselines.yaml")
    parser.add_argument("--models", nargs="*", help="model names from the config (default: all)")
    parser.add_argument("--sets", nargs="*", default=["dev", "test", "golden", "external"])
    parser.add_argument("--bm25", action="store_true", help="also evaluate BM25")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--hybrid", nargs="*", help="RRF of saved runs, e.g. BM25 e5-large")
    # ad-hoc model (fine-tuned checkpoints)
    parser.add_argument("--path")
    parser.add_argument("--name")
    parser.add_argument("--query-prompt", default="query: ")
    parser.add_argument("--doc-prompt", default="passage: ")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    query_sets = load_query_sets(args.sets)
    print({k: len(v) for k, v in query_sets.items()}, flush=True)

    if args.hybrid:
        name, per_query = eval_hybrid(args.hybrid, query_sets, cfg["rrf_k"])
        write_results(name, per_query, {"type": "hybrid", "components": args.hybrid, "rrf_k": cfg["rrf_k"]})
        print_table([name])
        return

    if args.bm25:
        write_results("BM25", eval_bm25(query_sets, cfg["bm25"]), {"type": "bm25", **cfg["bm25"]})
        print_table(["BM25"])

    models = []
    if args.path:
        models = [{"name": args.name, "path": args.path, "query_prompt": args.query_prompt,
                   "doc_prompt": args.doc_prompt}]  # fmt: skip
    elif not args.no_dense:
        models = [m for m in cfg["models"] if not args.models or m["name"] in args.models]

    for m in models:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        enc = DenseEncoder(
            m["path"],
            m["query_prompt"],
            m["doc_prompt"],
            max_seq_length=cfg["max_seq_length"],
            dtype=cfg["dtype"],
            padding_side=m.get("padding_side"),
            name=m["name"],
        )
        n_params = sum(p.numel() for p in enc.model.parameters())
        per_query, timing = eval_dense(enc, query_sets)
        meta = {
            "type": "dense",
            "path": m["path"],
            "query_prompt": m["query_prompt"],
            "doc_prompt": m["doc_prompt"],
            "max_seq_length": enc.model.max_seq_length,
            "dtype": cfg["dtype"],
            "params": n_params,
            "dim": enc.model.get_sentence_embedding_dimension(),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
            "total_s": round(time.time() - t0, 1),
            **timing,
        }
        write_results(m["name"], per_query, meta)
        print(f"== {m['name']} ({meta['total_s']} s, peak {meta['peak_mem_gb']} GB)", flush=True)
        print_table([m["name"]], "dev")
        del enc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
