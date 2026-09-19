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
for fam, runs in {
    "ft-e5-small": ["ft-e5-small", "ft-e5-small-s43", "ft-e5-small-s44"],
    "ft-e5-base": ["ft-e5-base", "ft-e5-base-s43", "ft-e5-base-s44"],
}.items():
    for qset in ("test", "golden", "dev"):
        for protocol in ("chunk", "article"):
            v = np.array([ndcg(r, qset, protocol) for r in runs])
            out[f"{fam}/{qset}/{protocol}"] = {
                "seeds": [42, 43, 44],
                "values": v.round(4).tolist(),
                "mean": round(float(v.mean()), 4),
                "std": round(float(v.std(ddof=1)), 4),
            }
    for sl in ("seen", "unseen_articles", "unseen_codes"):
        v = np.array([ndcg(r, "test", "chunk", f"slice={sl}") for r in runs])
        out[f"{fam}/test/chunk/{sl}"] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4)}

base, ft, large, frida = ndcg("e5-small"), ndcg("ft-e5-small"), ndcg("e5-large"), ndcg("FRIDA")
out["gap_closed_to_e5_large_seed42"] = round((ft - base) / (large - base), 4)
out["gap_closed_to_FRIDA_seed42"] = round((ft - base) / (frida - base), 4)
m = out["ft-e5-small/test/chunk"]["mean"]
out["gap_closed_to_e5_large_seed_mean"] = round((m - base) / (large - base), 4)

sp = pd.DataFrame(json.loads((R / "speed.json").read_text(encoding="utf-8")))
cpu = sp[sp.device == "cpu"].set_index("model")
out["cpu_latency_ratio_e5_large_vs_ft_small"] = round(
    cpu.loc["e5-large", "latency_ms_p50"] / cpu.loc["ft-e5-small", "latency_ms_p50"], 2
)
out["cpu_latency_ratio_FRIDA_vs_ft_small"] = round(
    cpu.loc["FRIDA", "latency_ms_p50"] / cpu.loc["ft-e5-small", "latency_ms_p50"], 2
)
out["index_ratio_e5_large_vs_ft_small"] = round(
    cpu.loc["e5-large", "index_mb_fp32"] / cpu.loc["ft-e5-small", "index_mb_fp32"], 2
)
out["params_ratio_e5_large_vs_small"] = round(cpu.loc["e5-large", "params_m"] / cpu.loc["ft-e5-small", "params_m"], 2)

fg = json.loads((R / "forgetting.json").read_text(encoding="utf-8"))
out["rubq_ndcg@10"] = {k: v["tasks"]["RuBQRetrieval"]["ndcg_at_10"] for k, v in fg.items()}

train = pd.read_csv(R / "train_runs.csv")
e1_to_e4 = train[~train.run.str.contains("_s4")]
out["training_total_min_all_runs"] = round(float(train["train_min"].sum()), 1)
out["training_min_final_small"] = float(train.set_index("run").loc["e1_small_llm_hn", "train_min"])
out["peak_mem_gb_max"] = float(train["peak_mem_gb"].max())
(R / "headline.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

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
