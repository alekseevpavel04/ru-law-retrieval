"""Print nDCG@10 per model / set / protocol from results/summary (quick look)."""

import json
from pathlib import Path

for p in sorted(Path("results/summary").glob("*.json")):
    s = json.loads(p.read_text(encoding="utf-8"))["results"]
    print(p.stem.ljust(22), "  ".join(f"{k}:{v['all']['ndcg@10']:.3f}" for k, v in sorted(s.items())))
