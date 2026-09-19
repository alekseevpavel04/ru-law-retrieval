"""Analysis: lexical difficulty, result tables and figures from ``results/``.

Every figure is saved together with the exact table it was drawn from (CSV), so
all numbers in README can be traced to a file produced by a script.

Usage:
  python -m rlr analysis lexical
  python -m rlr analysis tables --models ... --finetuned ...
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from rlr.data.parse import CORPUS, read_jsonl
from rlr.env import DATA, RESULTS
from rlr.eval.retrieve import TOKEN_RE

DATASET = DATA / "dataset"
FIG = RESULTS / "figures"


def stem_set(text: str, stemmer) -> set[str]:
    return set(stemmer.stemWords([w.lower().replace("ё", "е") for w in TOKEN_RE.findall(text)]))


def lexical_overlap() -> pd.DataFrame:
    """Share of (stemmed) question words that also occur in the gold article."""
    import Stemmer

    stemmer = Stemmer.Stemmer("russian")
    articles = {a["doc_id"]: a for a in read_jsonl(CORPUS / "articles.jsonl")}
    art_stems: dict[str, set[str]] = {}
    rows = []
    sources = {
        "train_llm": DATASET / "train_llm.jsonl",
        "train_titles": DATASET / "train_titles.jsonl",
        "dev": DATASET / "dev.jsonl",
        "test": DATASET / "test.jsonl",
    }
    for split, path in sources.items():
        for q in read_jsonl(path):
            d = q["doc_id"]
            if d not in art_stems:
                art_stems[d] = stem_set(articles[d]["title"] + " " + articles[d]["text"], stemmer)
            qs = stem_set(q["text"], stemmer)
            if not qs:
                continue
            rows.append(
                {
                    "split": split,
                    "generator": q.get("generator", "none"),
                    "qtype": q["qtype"],
                    "slice": q.get("slice", "train"),
                    "overlap": len(qs & art_stems[d]) / len(qs),
                    "n_words": len(qs),
                }
            )
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["split", "generator", "qtype"])
        .agg(
            n=("overlap", "size"),
            overlap_mean=("overlap", "mean"),
            overlap_median=("overlap", "median"),
            words_mean=("n_words", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(RESULTS / "lexical_overlap.csv", index=False, float_format="%.4f")
    return agg


def load_summary(name: str) -> dict:
    return json.loads((RESULTS / "summary" / f"{name}.json").read_text(encoding="utf-8"))


def main_table(models: list[str], qset: str = "test", protocols=("chunk", "article")) -> pd.DataFrame:
    rows = []
    for m in models:
        s = load_summary(m)
        for protocol in protocols:
            res = s["results"].get(f"{qset}/{protocol}")
            if not res:
                continue
            row = {"model": m, "protocol": protocol, "params_m": round((s.get("params") or 0) / 1e6, 1)}
            for key, label in (
                ("all", "all"),
                ("slice=seen", "seen"),
                ("slice=unseen_articles", "unseen_articles"),
                ("slice=unseen_codes", "unseen_codes"),
            ):
                if key in res:
                    row[f"ndcg@10_{label}"] = res[key]["ndcg@10"]
                    row[f"recall@10_{label}"] = res[key]["recall@10"]
            rows.append(row)
    return pd.DataFrame(rows)


def breakdown(models: list[str], key: str, qset: str = "test", protocol: str = "chunk") -> pd.DataFrame:
    rows = defaultdict(dict)
    for m in models:
        res = load_summary(m)["results"].get(f"{qset}/{protocol}", {})
        for k, v in res.items():
            if k.startswith(f"{key}="):
                rows[m][k.split("=", 1)[1]] = v["ndcg@10"]
    return pd.DataFrame(rows).T


def train_runs(prefix: str = "") -> pd.DataFrame:
    """One row per training run: dev (base / best), time, memory. Plus overlay figure per experiment."""
    from rlr.plots import plot_runs_overlay

    summaries = []
    for p in sorted((RESULTS / "train").glob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        if s["name"].startswith(prefix):
            if s["name"].endswith(("_s43", "_s44")):  # seed repeats of the chosen configs
                s["config"]["experiment"] = "E3"
            summaries.append(s)
    rows = []
    for s in summaries:
        best, base = s["dev_best"], s["dev_base"]
        rows.append(
            {
                "run": s["name"],
                "experiment": s["config"].get("experiment", ""),
                "base_model": s["base_model"].split("/")[-1],
                "data": "+".join(k for k in ("llm", "titles") if s["config"]["data"].get(k)),
                "fraction": s["config"]["data"].get("fraction", 1.0),
                "hard_negatives": s["config"].get("hard_negatives", False),
                "seed": s["seed"],
                "train_rows": s["train_rows"],
                "steps": s["total_steps"],
                "best_step": best["step"],
                "best_epoch": round(best["step"] / s["steps_per_epoch"], 2),
                "dev_ndcg@10_base": base["ndcg@10"],
                "dev_ndcg@10_best": best["ndcg@10"],
                "dev_ndcg@10_best_seen": best.get("ndcg@10_seen"),
                "dev_ndcg@10_best_unseen_articles": best.get("ndcg@10_unseen_articles"),
                "dev_recall@10_best": best["recall@10"],
                "train_min": round(s["train_seconds"] / 60, 2),
                "sec_per_step": s["seconds_per_step"],
                "peak_mem_gb": s["peak_mem_gb"],
                "git_commit": s["git_commit"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "train_runs.csv", index=False, float_format="%.4f")
    for exp in sorted({s["config"].get("experiment", "") for s in summaries}):
        group = [s for s in summaries if s["config"].get("experiment", "") == exp]
        if group:
            plot_runs_overlay(group, FIG / "train" / f"overlay_{exp}.png", f"{exp}: dev nDCG@10 and train loss")
    return df


def mean_ci(values: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def per_query(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / "per_query" / f"{name}.csv")


def report(models: list[str], finetuned: list[str], focus: list[str]) -> None:
    """Main tables and figures. ``focus`` = short list of models for slice / type charts."""
    from rlr.plots import plot_grouped, plot_model_bars, plot_scatter

    FIG.mkdir(parents=True, exist_ok=True)
    highlight = set(finetuned)
    rows = []
    for m in models + finetuned:
        s = load_summary(m)
        pq = per_query(m)
        for qset in ("test", "golden", "external", "dev", "tk_hard"):
            for protocol in ("chunk", "article", "chunk_tkfmt"):
                sub = pq[(pq["set"] == qset) & (pq["protocol"] == protocol)]
                if sub.empty:
                    continue
                lo, hi = mean_ci(sub["ndcg@10"].to_numpy())
                row = {
                    "model": m,
                    "set": qset,
                    "protocol": protocol,
                    "n": len(sub),
                    "ci_low": lo,
                    "ci_high": hi,
                    "params_m": round((s.get("params") or 0) / 1e6, 1),
                    "finetuned": m in highlight,
                }
                for metric in ("ndcg@10", "recall@5", "recall@10", "mrr@10"):
                    row[metric] = sub[metric].mean()
                for sl in ("seen", "unseen_articles", "unseen_codes"):
                    ss = sub[sub["slice"] == sl]
                    if not ss.empty:
                        row[f"ndcg@10_{sl}"] = ss["ndcg@10"].mean()
                        row[f"recall@10_{sl}"] = ss["recall@10"].mean()
                rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "main_table.csv", index=False, float_format="%.4f")

    for qset, protocol in (
        ("test", "chunk"),
        ("test", "article"),
        ("golden", "chunk"),
        ("tk_hard", "chunk"),
        ("tk_hard", "chunk_tkfmt"),
        ("test", "chunk_tkfmt"),
    ):
        sub = table[(table["set"] == qset) & (table["protocol"] == protocol)]
        if sub.empty:
            continue
        sub[["model", "ndcg@10", "ci_low", "ci_high"]].to_csv(FIG / f"models_{qset}_{protocol}.csv", index=False)
        plot_model_bars(
            sub, FIG / f"models_{qset}_{protocol}.png", f"{qset} set, {protocol} protocol: nDCG@10", highlight
        )

    for key, fname, title in (
        ("slice", "slices", "Test nDCG@10 by slice (chunk)"),
        ("qtype", "qtypes", "Test nDCG@10 by question type (chunk)"),
        ("code", "codes", "Test nDCG@10 by code (chunk)"),
    ):
        df = breakdown(focus, key)
        order = {"slice": ["seen", "unseen_articles", "unseen_codes"], "qtype": ["everyday", "search", "legal"]}
        if key in order:
            df = df[order[key]]
        df.to_csv(FIG / f"{fname}.csv", float_format="%.4f")
        plot_grouped(df, FIG / f"{fname}.png", title)

    speed_path = RESULTS / "speed.json"
    if speed_path.exists():
        sp = pd.DataFrame(json.loads(speed_path.read_text(encoding="utf-8")))
        test_chunk = table[(table["set"] == "test") & (table["protocol"] == "chunk")].set_index("model")["ndcg@10"]
        for device in sp["device"].unique():
            d = sp[sp["device"] == device].copy()
            d["ndcg@10"] = d["model"].map(test_chunk)
            d = d.dropna(subset=["ndcg@10"])
            d.to_csv(FIG / f"quality_vs_latency_{device}.csv", index=False)
            hw = "CPU: Ryzen 5 5600X, 6 threads, fp32" if device == "cpu" else "GPU: RTX 3070, fp16"
            plot_scatter(
                d,
                "latency_ms_p50",
                "ndcg@10",
                FIG / f"quality_vs_latency_{device}.png",
                f"Quality vs latency of one query ({hw})",
                "latency p50, ms (encode + search), log scale",
                highlight,
                arrows=[("e5-small", "ft-e5-small")],
            )
        d = sp[sp["device"] == sp["device"].iloc[0]].copy()
        d["ndcg@10"] = d["model"].map(test_chunk)
        d = d.dropna(subset=["ndcg@10"])
        plot_scatter(
            d,
            "params_m",
            "ndcg@10",
            FIG / "quality_vs_size.png",
            "Quality vs model size",
            "parameters, M, log scale",
            highlight,
            arrows=[("e5-small", "ft-e5-small")],
        )
    print(
        table[(table["set"] == "test") & (table["protocol"] == "chunk")]
        .sort_values("ndcg@10", ascending=False)[
            ["model", "ndcg@10", "ci_low", "ci_high", "ndcg@10_seen", "ndcg@10_unseen_articles", "ndcg@10_unseen_codes"]
        ]
        .to_string(index=False)
    )


def hero(base: str, finetuned: str, refs: list[str], v1: str | None = None) -> None:
    """Base -> (v1) -> fine-tuned on the test set by slice and question type, plus the harder TK set."""
    from rlr.plots import plot_before_after

    labels = [
        ("test", "all", "test: all questions"),
        ("test", "slice=seen", "test: seen articles"),
        ("test", "slice=unseen_articles", "test: unseen articles"),
        ("test", "slice=unseen_codes", "test: unseen codes (SK, ZoZPP)"),
        ("test", "qtype=everyday", "test: everyday questions"),
        ("test", "qtype=search", "test: search queries"),
        ("test", "qtype=legal", "test: lawyer questions"),
        ("tk_hard", "all", "TK-hard: all"),
        ("tk_hard", "qtype=everyday", "TK-hard: everyday"),
        ("tk_hard", "qtype=search", "TK-hard: search"),
    ]
    models = [base, finetuned, *refs] + ([v1] if v1 else [])
    res = {m: load_summary(m)["results"] for m in models}
    rows = []
    for qset, key, label in labels:
        if f"{qset}/chunk" not in res[finetuned] or key not in res[base].get(f"{qset}/chunk", {}):
            continue
        n = res[base][f"{qset}/chunk"][key]["n"]
        row = {
            "group": f"{label} (n={n})",
            "base": res[base][f"{qset}/chunk"][key]["ndcg@10"],
            "finetuned": res[finetuned][f"{qset}/chunk"][key]["ndcg@10"],
        }
        row.update({r: res[r][f"{qset}/chunk"][key]["ndcg@10"] for r in refs})
        if v1:
            row["v1"] = res[v1][f"{qset}/chunk"][key]["ndcg@10"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(FIG / "before_after.csv", index=False, float_format="%.4f")
    plot_before_after(df, FIG / "before_after.png", "e5-small (118M): base vs fine-tuned", "fine-tuned v2")
    print(df.round(3).to_string(index=False))


def learning_curve(full_run: str, fraction_runs: list[str], refs: list[str]) -> None:
    """Dev and test nDCG@10 of the chosen checkpoints vs number of LLM questions (final evaluation code)."""
    from rlr.plots import plot_learning_curve

    eval_name = {full_run: "ft-e5-small", **{r: f"abl-{r}" for r in fraction_runs}}
    rows = [
        {
            "run": "e5-small (no fine-tuning)",
            "train_questions": 0,
            "dev_ndcg@10": load_summary("e5-small")["results"]["dev/chunk"]["all"]["ndcg@10"],
            "test_ndcg@10": load_summary("e5-small")["results"]["test/chunk"]["all"]["ndcg@10"],
        }
    ]
    for name in [*fraction_runs, full_run]:
        s = json.loads((RESULTS / "train" / f"{name}.json").read_text(encoding="utf-8"))
        res = load_summary(eval_name[name])["results"]
        rows.append(
            {
                "run": name,
                "train_questions": s["data"].get("llm", 0),
                "dev_ndcg@10": res["dev/chunk"]["all"]["ndcg@10"],
                "test_ndcg@10": res["test/chunk"]["all"]["ndcg@10"],
            }
        )
    df = pd.DataFrame(rows).sort_values("train_questions")
    df.to_csv(FIG / "learning_curve.csv", index=False, float_format="%.4f")
    ref_vals = {r: load_summary(r)["results"]["dev/chunk"]["all"]["ndcg@10"] for r in refs if r != "e5-small"}
    plot_learning_curve(df, FIG / "learning_curve.png", ref_vals)
    print(df.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr analysis")
    parser.add_argument("what", choices=["lexical", "train", "report", "learning", "hero"])
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--finetuned", nargs="*", default=[])
    parser.add_argument("--focus", nargs="*", default=[])
    parser.add_argument("--full-run")
    parser.add_argument("--fraction-runs", nargs="*", default=[])
    parser.add_argument("--refs", nargs="*", default=[])
    args = parser.parse_args(argv)
    if args.what == "report":
        report(args.models, args.finetuned, args.focus)
    elif args.what == "hero":
        if (RESULTS / "summary" / "ft2-e5-small.json").exists():
            hero("e5-small", "ft2-e5-small", ["e5-large", "FRIDA"], v1="ft-e5-small")
        else:
            hero("e5-small", "ft-e5-small", ["e5-large", "FRIDA"])
    elif args.what == "learning":
        learning_curve(args.full_run, args.fraction_runs, args.refs)
    if args.what == "lexical":
        print(lexical_overlap().to_string(index=False))
    elif args.what == "train":
        print(train_runs().to_string(index=False))


if __name__ == "__main__":
    main()
