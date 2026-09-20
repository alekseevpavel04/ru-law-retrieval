"""Paired bootstrap over queries for the difference of a metric between two systems.

Usage:
  python -m rlr bootstrap --a e5-small-ru-law --b e5-small --set test --protocol chunk [--by slice]
  python -m rlr bootstrap --pairs-file configs/bootstrap_pairs.yaml
"""

import argparse
import json

import numpy as np
import pandas as pd

from rlr.config import load_yaml
from rlr.env import RESULTS

N_RESAMPLES = 10_000


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = N_RESAMPLES, seed: int = 0, alpha: float = 0.05) -> dict:
    """CI of mean(a - b) by resampling queries with replacement (same resample for both systems)."""
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("paired arrays of equal non-zero length expected")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    # two-sided p-value: share of resamples on the other side of zero, with the usual +1 correction
    # so that it is never reported as exactly 0 (the floor is 2 / (n + 1))
    p = 2 * (min((means <= 0).sum(), (means >= 0).sum()) + 1) / (n + 1)
    return {
        "n_queries": len(diff),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "delta": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "n_resamples": n,
    }


def compare(a: str, b: str, qset: str, protocol: str, metric: str = "ndcg@10", by: str | None = None) -> list[dict]:
    da = pd.read_csv(RESULTS / "per_query" / f"{a}.csv")
    db = pd.read_csv(RESULTS / "per_query" / f"{b}.csv")
    da = da[(da["set"] == qset) & (da["protocol"] == protocol)].set_index("qid")
    db = db[(db["set"] == qset) & (db["protocol"] == protocol)].set_index("qid")
    common = da.index.intersection(db.index)
    if len(common) != len(da) or len(common) != len(db):
        raise ValueError(f"query sets differ: {len(da)} vs {len(db)} (common {len(common)})")
    da, db = da.loc[common], db.loc[common]
    groups = [("all", common)]
    if by:
        groups += [(f"{by}={v}", da.index[da[by] == v]) for v in sorted(da[by].unique())]
    out = []
    for name, idx in groups:
        r = paired_bootstrap(da.loc[idx, metric].to_numpy(), db.loc[idx, metric].to_numpy())
        out.append({"a": a, "b": b, "set": qset, "protocol": protocol, "metric": metric, "group": name, **r})
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr bootstrap")
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--set", default="test")
    parser.add_argument("--protocol", default="chunk")
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--by", default="slice")
    parser.add_argument("--pairs-file", help="YAML with a list of {a, b, set, protocol}")
    parser.add_argument("--out", default="bootstrap.csv")
    args = parser.parse_args(argv)

    pairs = (
        load_yaml(args.pairs_file)["pairs"]
        if args.pairs_file
        else [{"a": args.a, "b": args.b, "set": args.set, "protocol": args.protocol}]
    )
    rows = []
    for p in pairs:
        by = p.get("by", args.by)
        rows += compare(
            p["a"], p["b"], p.get("set", "test"), p.get("protocol", "chunk"), p.get("metric", args.metric), by
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / args.out, index=False, float_format="%.5f")
    (RESULTS / args.out.replace(".csv", ".json")).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(df[["a", "b", "set", "protocol", "group", "n_queries", "delta", "ci_low", "ci_high", "p_value"]].to_string())


if __name__ == "__main__":
    main()
