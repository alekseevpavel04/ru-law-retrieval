"""Aggregate headline numbers for README / CV_NOTES into results/headline.json and results/ablation.csv."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")


def ndcg(name: str, qset: str = "test", protocol: str = "chunk", key: str = "all") -> float:
    s = json.loads((R / "summary" / f"{name}.json").read_text(encoding="utf-8"))["results"]
    return s[f"{qset}/{protocol}"][key]["ndcg@10"]


out: dict = {}
FAMILIES = {
    "ft-e5-small": ["ft-e5-small", "ft-e5-small-s43", "ft-e5-small-s44"],  # v1 (3 seeds)
    "ft-e5-small-v2": ["ft2-e5-small", "ft-e5-small-v2", "ft2-e5-small-s44"],  # v2 (seeds 42, 43, 44)
    "ft-e5-base": ["ft-e5-base", "ft-e5-base-s43", "ft-e5-base-s44"],
}
for fam, runs in FAMILIES.items():
    for qset in ("test", "golden", "tk_hard", "dev"):
        for protocol in ("chunk", "article", "chunk_tkfmt"):
            try:
                v = np.array([ndcg(r, qset, protocol) for r in runs])
            except (FileNotFoundError, KeyError):
                continue
            out[f"{fam}/{qset}/{protocol}"] = {
                "seeds": [42, 43, 44],
                "values": v.round(4).tolist(),
                "mean": round(float(v.mean()), 4),
                "std": round(float(v.std(ddof=1)), 4),
            }
    for sl in ("seen", "unseen_articles", "unseen_codes"):
        try:
            v = np.array([ndcg(r, "test", "chunk", f"slice={sl}") for r in runs])
        except KeyError:
            continue
        out[f"{fam}/test/chunk/{sl}"] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4)}

published = "ft-e5-small-v2"  # seed 43: best dev among the three seeds
base, v1, v2 = ndcg("e5-small"), ndcg("ft-e5-small"), ndcg(published)
large, frida = ndcg("e5-large"), ndcg("FRIDA")
out["published_model"] = {
    "name": published,
    "checkpoint": "models/v2f_distill_from_v2e_ep3_s43/best",
    "selected_by": "best dev nDCG@10 (mean of two chunk views) among seeds 42/43/44",
}
out["gap_closed_to_e5_large_v1"] = round((v1 - base) / (large - base), 4)
out["gap_closed_to_e5_large_v2"] = round((v2 - base) / (large - base), 4)
out["gap_closed_to_FRIDA_v2"] = round((v2 - base) / (frida - base), 4)
for qset in ("test", "tk_hard", "golden"):
    out[f"{qset}/published_vs_base"] = round(ndcg(published, qset) - ndcg("e5-small", qset), 4)

# ablation: E1/E2/E4 checkpoints, dev and test (chunk)
names = {
    "e1_small_titles_inb": "abl-e1_small_titles_inb",
    "e1_small_titles_hn": "abl-e1_small_titles_hn",
    "e1_small_llm_inb": "abl-e1_small_llm_inb",
    "e1_small_llm_hn": "ft-e5-small",
    "e1_small_both_inb": "abl-e1_small_both_inb",
    "e1_small_both_hn": "abl-e1_small_both_hn",
    "e2_base_llm_hn": "ft-e5-base",
    "e2_base_both_hn": "abl-e2_base_both_hn",
    "e4_small_llm_hn_f25": "abl-e4_small_llm_hn_f25",
    "e4_small_llm_hn_f50": "abl-e4_small_llm_hn_f50",
}
rows = [
    {"run": "e5-small (base)", "dev": ndcg("e5-small", "dev"), "test": ndcg("e5-small")},
    {"run": "e5-base (base)", "dev": ndcg("e5-base", "dev"), "test": ndcg("e5-base")},
]
for run, ev in names.items():
    rows.append(
        {
            "run": run,
            "dev": ndcg(ev, "dev"),
            "test": ndcg(ev),
            **{sl: ndcg(ev, "test", "chunk", f"slice={sl}") for sl in ("seen", "unseen_articles", "unseen_codes")},
        }
    )
pd.DataFrame(rows).to_csv(R / "ablation.csv", index=False, float_format="%.4f")
print(json.dumps(out, indent=1))
print(pd.DataFrame(rows).round(3).to_string(index=False))
