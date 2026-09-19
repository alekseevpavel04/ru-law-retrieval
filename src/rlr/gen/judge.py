"""LLM judge for dev/test questions: "can this question be answered by the article?".

The judge is the train generator (Qwen3), a different model family than the test
generator. Questions judged "no" are dropped later in ``build-dataset``.

Usage: python -m rlr judge --inp data/gen/eval_raw.jsonl --out data/gen/eval_judged.jsonl
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rlr.config import codes_config
from rlr.data.chunk import doc_header
from rlr.data.parse import CORPUS, read_jsonl
from rlr.gen import prompts
from rlr.gen.generate import article_parts, done_keys, stable_seed
from rlr.gen.llm_client import LLMClient


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr judge")
    parser.add_argument("--inp", default="data/gen/eval_raw.jsonl")
    parser.add_argument("--out", default="data/gen/eval_judged.jsonl")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args(argv)

    display = codes_config()["display"]
    articles = {a["doc_id"]: a for a in read_jsonl(CORPUS / "articles.jsonl")}
    rows = [r for r in read_jsonl(Path(args.inp)) if r.get("output") and r["output"].get("question")]
    out = Path(args.out)
    done = done_keys(out)
    todo = [r for r in rows if r["key"] not in done]
    client = LLMClient(args.base_url)
    model = client.model_id()
    print(f"judge {model}: {len(rows)} questions, {len(todo)} to do", flush=True)

    def run(row: dict) -> dict:
        a = articles[row["doc_id"]]
        header = doc_header(display[a["code"]], a["number"], a["title"])
        text = article_parts(a["text"])[row["part"]]
        user = prompts.JUDGE_USER.format(header=header, text=text, question=row["output"]["question"])
        messages = [{"role": "system", "content": prompts.JUDGE_SYSTEM}, {"role": "user", "content": user}]
        parsed, raw, _ = client.chat_json(
            messages, prompts.JUDGE_SCHEMA, temperature=0.0, seed=stable_seed(row["key"]), max_tokens=32
        )
        verdict = (parsed or {}).get("answerable", "invalid")
        return {"key": row["key"], "judge_model": model, "answerable": verdict, "raw": raw}

    started = time.time()
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(run, r) for r in todo]), 1):
            f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
