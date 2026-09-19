"""v2 stage F: distillation (T=0.02, the stage-C winner) on top of the stage-E winner and its seeds.

Kept only if it improves dev over the stage-E winner. Appended to results/v2_selection.json under "F".

Usage: cd scripts && python run_v2_stage_f.py
"""

import copy
import json

import yaml
from run_v2_grid import CFG_DIR, RESULTS, run


def main() -> None:
    sel_path = RESULTS / "v2_selection.json"
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    e = sel["stages"]["E"]
    if not e["kept"]:
        print("stage E kept nothing, stage F skipped")
        return
    c_cfg = yaml.safe_load((CFG_DIR / f"{sel['stages']['C']['chosen']}.yaml").read_text(encoding="utf-8"))
    base = next(r for r in e["runs"] if r["name"] == e["chosen"])
    starts = [(base, 42)] + [(s, seed) for s, seed in zip(e["seeds"], (43, 44), strict=True)]
    res = []
    for start, seed in starts:
        cfg = {
            **copy.deepcopy(c_cfg),
            "name": f"v2f_distill_from_{start['name']}",
            "base_model": start["model_dir"],
            "stage1": start["name"],
            "seed": seed,
        }
        res.append(run(cfg))
    improved = res[0]["dev"] > base["dev"]
    sel["stages"]["F"] = {"runs": res, "kept": improved, "compared_to": base["name"]}
    if improved:
        sel["final"] = {
            "name": res[0]["name"],
            "dev": res[0]["dev"],
            "model_dir": res[0]["model_dir"],
            "recipe": f"stage1 ({base['name']}) + distillation",
            "seeds": [r["model_dir"] for r in res[1:]],
        }
    else:
        sel["final"]["seeds"] = [s["model_dir"] for s in e["seeds"]]
    sel_path.write_text(json.dumps(sel, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(sel["final"], indent=1))


if __name__ == "__main__":
    main()
