"""Aggregate every headline number of the README into results/headline.json (and results/ablation.csv).

Nothing in the README is typed by hand: each claim points either here, to results/bootstrap.csv,
or to the table it was drawn from.

Usage: python scripts/headline.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
PUBLISHED = "e5-small-ru-law"  # seed 43: best dev among the three seeds of the final recipe
SEEDS = {42: "e5-small-ru-law-s42", 43: PUBLISHED, 44: "e5-small-ru-law-s44"}
FIRST_RECIPE = "abl-e1_small_llm_hn"  # E1 winner: the one-stage recipe the final one is built on


def summary(name: str) -> dict:
    return json.loads((R / "summary" / f"{name}.json").read_text(encoding="utf-8"))["results"]


def ndcg(name: str, qset: str = "test", protocol: str = "chunk", key: str = "all") -> float:
    return summary(name)[f"{qset}/{protocol}"][key]["ndcg@10"]


def seed_stats(qset: str, protocol: str = "chunk", key: str = "all") -> dict:
    v = np.array([ndcg(n, qset, protocol, key) for n in SEEDS.values()])
    return {
        "seeds": list(SEEDS),
        "values": v.round(4).tolist(),
        "mean": round(float(v.mean()), 4),
        "std": round(float(v.std(ddof=1)), 4),
    }


out: dict = {
    "published": {
        "name": PUBLISHED,
        "checkpoint": "models/v2f_distill_from_v2e_ep3_s43/best",
        "hub": "alekseevpavel04/multilingual-e5-small-ru-law",
        "base": "intfloat/multilingual-e5-small",
        "selected_by": "best dev nDCG@10 (mean over the two chunk views) among seeds 42/43/44",
    }
}

# --- quality: published checkpoint, its base, and the two reference models -------------------
out["ndcg@10"] = {
    f"{qset}/{protocol}": {m: round(ndcg(m, qset, protocol), 4) for m in ("e5-small", PUBLISHED, "e5-large", "FRIDA")}
    for qset, protocol in (("test", "chunk"), ("test", "article"), ("tk_hard", "chunk"), ("golden", "chunk"))
}
out["seed_spread"] = {
    f"{qset}/{protocol}": seed_stats(qset, protocol)
    for qset, protocol in (("test", "chunk"), ("tk_hard", "chunk"), ("golden", "chunk"), ("dev", "chunk"))
}
out["seed_spread"]["test/chunk/slices"] = {
    sl: seed_stats("test", "chunk", f"slice={sl}") for sl in ("seen", "unseen_articles", "unseen_codes")
}

base, model = ndcg("e5-small"), ndcg(PUBLISHED)
out["gap_closed"] = {
    "to_e5_large": round((model - base) / (ndcg("e5-large") - base), 4),
    "to_FRIDA": round((model - base) / (ndcg("FRIDA") - base), 4),
}

# --- dev-only selection metric (mean of the two chunk views), for the recipe table ------------
def dev_selection(name: str) -> float:
    return round((ndcg(name, "dev", "chunk") + ndcg(name, "dev", "chunk_tkfmt")) / 2, 4)


out["dev_selection_metric"] = {
    "note": "mean dev nDCG@10 over both chunk views (600/90 and the 500/75 view of the tk-rf-rag service)",
    "e5-small": dev_selection("e5-small"),
    "one_stage_recipe": dev_selection(FIRST_RECIPE),
    "published": dev_selection(PUBLISHED),
}

# --- cost: speed, index size, training time --------------------------------------------------
sp = pd.DataFrame(json.loads((R / "speed.json").read_text(encoding="utf-8")))
cpu = sp[sp["device"] == "cpu"].set_index("model")
out["cost"] = {
    "cpu_latency_ms_p50": {m: float(cpu.loc[m, "latency_ms_p50"]) for m in cpu.index},
    "index_mb_fp32": {m: float(cpu.loc[m, "index_mb_fp32"]) for m in cpu.index},
    "cpu_latency_ratio_e5_large": round(cpu.loc["e5-large", "latency_ms_p50"] / cpu.loc[PUBLISHED, "latency_ms_p50"], 2),
    "cpu_latency_ratio_FRIDA": round(cpu.loc["FRIDA", "latency_ms_p50"] / cpu.loc[PUBLISHED, "latency_ms_p50"], 2),
    "index_ratio_e5_large": round(cpu.loc["e5-large", "index_mb_fp32"] / cpu.loc[PUBLISHED, "index_mb_fp32"], 2),
    "params_ratio_e5_large": round(cpu.loc["e5-large", "params_m"] / cpu.loc[PUBLISHED, "params_m"], 2),
}

runs = pd.read_csv(R / "train_runs.csv").set_index("run")
stage1, stage2 = "v2e_ep3_s43", "v2f_distill_from_v2e_ep3_s43"
out["training"] = {
    "stage1_min": float(runs.loc[stage1, "train_min"]),
    "stage2_min": float(runs.loc[stage2, "train_min"]),
    "total_min": round(float(runs.loc[stage1, "train_min"] + runs.loc[stage2, "train_min"]), 1),
    "peak_mem_gb": float(max(runs.loc[stage1, "peak_mem_gb"], runs.loc[stage2, "peak_mem_gb"])),
    "train_rows": int(runs.loc[stage1, "train_rows"]),
    "all_runs": len(runs),
    "all_runs_gpu_hours": round(float(runs["train_min"].sum()) / 60, 2),
}

# --- general-domain retrieval after fine-tuning ----------------------------------------------
forg = json.loads((R / "forgetting.json").read_text(encoding="utf-8"))
out["rubq_ndcg@10"] = {k: v["tasks"]["RuBQRetrieval"]["ndcg_at_10"] for k, v in forg.items()}

(R / "headline.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

# --- ablation table: E1 / E2 / E4 checkpoints, dev and test (chunk) ---------------------------
ABLATIONS = [
    "abl-e1_small_titles_inb",
    "abl-e1_small_titles_hn",
    "abl-e1_small_llm_inb",
    FIRST_RECIPE,
    "abl-e1_small_both_inb",
    "abl-e1_small_both_hn",
    "abl-e2_base_llm_hn",
    "abl-e2_base_both_hn",
    "abl-e4_small_llm_hn_f25",
    "abl-e4_small_llm_hn_f50",
]
rows = [
    {"run": "e5-small (base)", "dev": ndcg("e5-small", "dev"), "test": ndcg("e5-small")},
    {"run": "e5-base (base)", "dev": ndcg("e5-base", "dev"), "test": ndcg("e5-base")},
]
rows += [
    {
        "run": name.removeprefix("abl-"),
        "dev": ndcg(name, "dev"),
        "test": ndcg(name),
        **{sl: ndcg(name, "test", "chunk", f"slice={sl}") for sl in ("seen", "unseen_articles", "unseen_codes")},
    }
    for name in ABLATIONS
]
rows.append(
    {
        "run": "published (two stages, teacher + distillation)",
        "dev": ndcg(PUBLISHED, "dev"),
        "test": ndcg(PUBLISHED),
        **{sl: ndcg(PUBLISHED, "test", "chunk", f"slice={sl}") for sl in ("seen", "unseen_articles", "unseen_codes")},
    }
)
pd.DataFrame(rows).to_csv(R / "ablation.csv", index=False, float_format="%.4f")

print(json.dumps(out, ensure_ascii=False, indent=1))
print(pd.DataFrame(rows).round(3).to_string(index=False))
