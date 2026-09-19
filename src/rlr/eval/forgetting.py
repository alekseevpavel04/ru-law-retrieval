"""E5: catastrophic forgetting check on general Russian retrieval (MTEB RuBQRetrieval).

Evaluates base and fine-tuned models with the same e5 prompts through mteb.

Usage:
  python -m rlr forgetting --models e5-small=intfloat/multilingual-e5-small ft=models/x/best
"""

import argparse
import json
import os
import time
from pathlib import Path

from rlr.env import RESULTS


def run_task(name: str, path: str, tasks: list[str], query_prompt: str, doc_prompt: str) -> dict:
    import mteb
    import torch
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(
        path, device="cuda" if torch.cuda.is_available() else "cpu", model_kwargs={"torch_dtype": torch.float16}
    )
    st.max_seq_length = 512
    model = mteb.SentenceTransformerEncoderWrapper(st, model_prompts={"query": query_prompt, "document": doc_prompt})
    cache = mteb.ResultCache(cache_path=Path(os.environ["MTEB_CACHE"]) / name)
    t0 = time.time()
    res = mteb.evaluate(
        model,
        mteb.get_tasks(tasks=tasks),
        cache=cache,
        overwrite_strategy="always",
        encode_kwargs={"batch_size": 64},
        show_progress_bar=False,
    )
    out = {"model": name, "path": path, "seconds": round(time.time() - t0, 1), "tasks": {}}
    for tr in res.task_results:
        scores = tr.scores["test"][0] if "test" in tr.scores else next(iter(tr.scores.values()))[0]
        out["tasks"][tr.task_name] = {
            k: scores[k] for k in ("main_score", "ndcg_at_10", "recall_at_10", "mrr_at_10") if k in scores
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr forgetting")
    parser.add_argument("--models", nargs="+", required=True, help="name=path pairs")
    parser.add_argument("--tasks", nargs="+", default=["RuBQRetrieval"])
    parser.add_argument("--query-prompt", default="query: ")
    parser.add_argument("--doc-prompt", default="passage: ")
    parser.add_argument("--out", default="forgetting.json")
    args = parser.parse_args(argv)

    path = RESULTS / args.out
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for pair in args.models:
        name, model_path = pair.split("=", 1)
        results[name] = run_task(name, model_path, args.tasks, args.query_prompt, args.doc_prompt)
        print(json.dumps(results[name], indent=1), flush=True)
        path.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
