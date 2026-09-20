"""v2 stage E: training-length / batch / lr variants of the current v2 winner, dev-only selection.

The v2 dev curves peaked at (or near) the last step, so the model is probably under-trained. Variants of the
stage-1 winner: 3 epochs, batch 256 (more in-batch negatives, GradCache keeps memory flat), lr 5e-5.
A variant replaces the current best only if it is better on dev; then seeds 43/44 are trained for it.
Decisions are appended to results/recipe_selection.json under "E".

Usage: python scripts/run_recipe_stage_e.py
"""

import copy
import json

import yaml
from run_recipe import CFG_DIR, RESULTS, run


def main() -> None:
    sel_path = RESULTS / "recipe_selection.json"
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    final = sel["final"]
    stage1_name = (
        sel["stages"]["A"]["chosen"] if not sel["stages"]["B"]["kept"] else sel["stages"]["B"]["runs"][0]["name"]
    )
    base_cfg = yaml.safe_load((CFG_DIR / f"{stage1_name}.yaml").read_text(encoding="utf-8"))
    variants = [
        {**copy.deepcopy(base_cfg), "name": "v2e_ep3", "epochs": 3},
        {**copy.deepcopy(base_cfg), "name": "v2e_bs256", "batch_size": 256},
        {**copy.deepcopy(base_cfg), "name": "v2e_lr5e5", "lr": 5.0e-5},
    ]
    res = [run(v) for v in variants]
    best_i = max(range(len(res)), key=lambda i: res[i]["dev"])
    improved = res[best_i]["dev"] > final["dev"]
    stage = {"runs": res, "previous_best": final, "kept": improved, "chosen": res[best_i]["name"] if improved else None}
    if improved:
        stage["seeds"] = [run(variants[best_i], seed) for seed in (43, 44)]
        sel["final"] = {
            "name": res[best_i]["name"],
            "dev": res[best_i]["dev"],
            "model_dir": res[best_i]["model_dir"],
            "recipe": f"stage1 ({res[best_i]['name']})",
        }
    sel["stages"]["E"] = stage
    sel_path.write_text(json.dumps(sel, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(sel["final"], indent=1))


if __name__ == "__main__":
    main()
