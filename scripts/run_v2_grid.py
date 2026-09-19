"""v2 training grid with automatic, dev-only selection. Every decision is written to results/v2_selection.json.

A: data + teacher        v1 questions + teacher | v1+v2 questions + teacher | + teacher noise filter (rank <= 50)
B: format augmentation   best(A) + 50% of rows with chunks in the tk-rf-rag format (500/75, no code name)
C: distillation (stage 2) from best(A/B) checkpoint, DistillKL on FRIDA scores, teacher T in {0.05, 0.02}
D: seeds 43, 44 of the final recipe (for mean +- std)

Selection metric: dev nDCG@10 averaged over two chunk formats (ours 600/90 and tk-rf-rag 500/75).
A later stage is kept only if it beats the current best on dev. Test sets are not touched here.

Usage: python scripts/run_v2_grid.py
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs" / "train"
RESULTS = ROOT / "results"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

BASE = {
    "experiment": "V2",
    "base_model": "intfloat/multilingual-e5-small",
    "train_file": "train_llm_v12_teacher",
    "negatives": "teacher",
    "loss": "mnrl",
    "batch_size": 128,
    "mini_batch_size": 32,
    "lr": 3.0e-5,
    "warmup_ratio": 0.1,
    "epochs": 2,
    "eval_every": 0.25,
    "max_seq_length": 208,
    "seed": 42,
    "freeze_embeddings": True,
    "scale": 20.0,
    "weight_decay": 0.01,
    "logging_steps": 2,
    "dev_views": ["chunk", "chunk_tkfmt"],
}


def run(cfg: dict, seed: int | None = None) -> dict:
    path = CFG_DIR / f"{cfg['name']}.yaml"
    path.write_text(
        "# v2 grid (scripts/run_v2_grid.py)\n" + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    cmd = [PY, "-m", "rlr", "train", str(path)] + (["--seed", str(seed)] if seed else [])
    print(">>", " ".join(cmd[3:]), flush=True)
    subprocess.run(cmd, check=True)
    name = cfg["name"] + (f"_s{seed}" if seed and seed != cfg["seed"] else "")
    summary = json.loads((RESULTS / "train" / f"{name}.json").read_text(encoding="utf-8"))
    best = summary["dev_best"]
    print(
        f"   {name}: dev {best['ndcg@10']:.4f} (views: "
        + ", ".join(f"{k[14:]}={v:.4f}" for k, v in best.items() if k.startswith("ndcg@10_view_"))
        + ")",
        flush=True,
    )
    return {"name": name, "dev": best["ndcg@10"], "best": best, "model_dir": summary["best_model_dir"]}


def main() -> None:
    log: dict = {"metric": "dev nDCG@10, mean over chunk views [chunk, chunk_tkfmt]", "stages": {}}
    out = RESULTS / "v2_selection.json"

    def save() -> None:
        out.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")

    # A: data + teacher
    a_cfgs = [
        {**BASE, "name": "v2a_small_teacher_v1data", "prompt_versions": ["v1"]},
        {**BASE, "name": "v2a_small_teacher_v12"},
        {**BASE, "name": "v2a_small_teacher_v12_f50", "teacher_filter_rank": 50},
    ]
    a_res = [run(c) for c in a_cfgs]
    best_i = max(range(len(a_res)), key=lambda i: a_res[i]["dev"])
    best_cfg, best = a_cfgs[best_i], a_res[best_i]
    log["stages"]["A"] = {"runs": a_res, "chosen": best["name"]}
    save()

    # B: format augmentation
    b_cfg = {**copy.deepcopy(best_cfg), "name": best_cfg["name"].replace("v2a_", "v2b_") + "_aug50", "format_aug": 0.5}
    b = run(b_cfg)
    keep_b = b["dev"] > best["dev"]
    log["stages"]["B"] = {"runs": [b], "kept": keep_b}
    if keep_b:
        best_cfg, best = b_cfg, b
    save()

    # C: distillation stage from the best checkpoint
    c_res, c_cfgs = [], []
    for t in (0.05, 0.02):
        c_cfg = {
            **copy.deepcopy(best_cfg),
            "name": f"v2c_distill_t{int(t * 100):02d}",
            "base_model": best["model_dir"],
            "loss": "distill_kl",
            "teacher_temperature": t,
            "batch_size": 32,
            "lr": 1.0e-5,
            "epochs": 1,
            "stage1": best["name"],
        }
        c_cfgs.append(c_cfg)
        c_res.append(run(c_cfg))
    ci = max(range(len(c_res)), key=lambda i: c_res[i]["dev"])
    keep_c = c_res[ci]["dev"] > best["dev"]
    log["stages"]["C"] = {"runs": c_res, "kept": keep_c, "chosen": c_res[ci]["name"] if keep_c else None}
    stage1_cfg = best_cfg
    if keep_c:
        best_cfg, best = c_cfgs[ci], c_res[ci]
    log["final"] = {
        "name": best["name"],
        "dev": best["dev"],
        "model_dir": best["model_dir"],
        "recipe": "stage1 + distillation" if keep_c else "stage1",
    }
    save()

    # D: seeds for the final recipe (both stages re-run with the new seed)
    seeds = []
    for seed in (43, 44):
        s1 = run(stage1_cfg, seed)
        if keep_c:
            c_cfg = {
                **copy.deepcopy(best_cfg),
                "name": f"{best_cfg['name']}_from_s{seed}",
                "base_model": s1["model_dir"],
                "seed": seed,
            }
            seeds.append(run(c_cfg))
        else:
            seeds.append(s1)
    log["stages"]["D"] = {"runs": seeds}
    save()
    print(json.dumps(log["final"], indent=1))


if __name__ == "__main__":
    sys.exit(main())
