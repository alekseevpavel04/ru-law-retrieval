"""Generate synthetic questions with a local LLM (llama.cpp server).

Modes:
- ``train``: train generator, one call per article part (articles longer than
  ``--max-chars`` are split), three questions of three types per call;
- ``eval``: test generator, one question of the assigned type per dev/test article.

Output is appended to a JSONL file; items already present are skipped, so an
interrupted run resumes where it stopped.

Usage:
  python -m rlr generate --mode train --out data/gen/train_raw.jsonl [--limit 10]
  python -m rlr generate --mode eval  --out data/gen/eval_raw.jsonl  [--limit 10]
  python -m rlr generate --mode tkhard --out data/gen/tkhard_raw.jsonl   # harder TK test set
"""

import argparse
import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rlr.config import codes_config
from rlr.data.chunk import doc_header, split_text
from rlr.data.parse import CORPUS, read_jsonl
from rlr.data.splits import SPLITS
from rlr.gen import prompts
from rlr.gen.llm_client import LLMClient

MAX_CHARS = 4000


def stable_seed(key: str, base: int = 42) -> int:
    return base + int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 1_000_000


def article_parts(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    # parts of roughly equal size, split on paragraph/sentence boundaries
    n = -(-len(text) // max_chars)
    return split_text(text, -(-len(text) // n) + 200, 0)


def build_tasks(mode: str, limit: int, sample_seed: int = 0) -> list[dict]:
    display = codes_config()["display"]
    articles = {a["doc_id"]: a for a in read_jsonl(CORPUS / "articles.jsonl")}
    splits = json.loads((SPLITS / "splits.json").read_text(encoding="utf-8"))
    tasks: list[dict] = []
    if mode == "train":
        for doc_id in splits["train_articles"]:
            a = articles[doc_id]
            header = doc_header(display[a["code"]], a["number"], a["title"])
            for i, part in enumerate(article_parts(a["text"])):
                tasks.append({"key": f"{doc_id}|{i}", "doc_id": doc_id, "part": i, "header": header, "text": part})
    elif mode == "tkhard":
        # harder extra test set for the TK service: everyday + search questions for every TK article except
        # dev articles (dev is only for model selection); test-only, never used for choosing anything
        dev_articles = {it["doc_id"] for it in splits["eval_items"] if it["split"] == "dev"}
        held = set(splits["held_out"])
        for a in articles.values():
            if a["code"] != "tk" or a["doc_id"] in dev_articles or len(a["text"]) < 150:
                continue
            header = doc_header(display[a["code"]], a["number"], a["title"])
            parts = article_parts(a["text"])
            i = random.Random(stable_seed("tkhard" + a["doc_id"])).randrange(len(parts))
            slice_name = "unseen_articles" if a["doc_id"] in held else "seen"
            for qtype in ("everyday", "search"):
                tasks.append(
                    {
                        "key": f"tkhard|{a['doc_id']}|{qtype}",
                        "split": "tk_hard",
                        "slice": slice_name,
                        "doc_id": a["doc_id"],
                        "qtype": qtype,
                        "part": i,
                        "header": header,
                        "text": parts[i],
                    }
                )
    else:
        for it in splits["eval_items"]:
            a = articles[it["doc_id"]]
            header = doc_header(display[a["code"]], a["number"], a["title"])
            parts = article_parts(a["text"])
            i = random.Random(stable_seed(it["doc_id"])).randrange(len(parts))
            tasks.append({"key": it["doc_id"], **it, "part": i, "header": header, "text": parts[i]})
    if limit:
        rng = random.Random(sample_seed)
        tasks = rng.sample(tasks, min(limit, len(tasks)))
    return tasks


def messages_for(mode: str, task: dict, prompt_version: str = "v1") -> tuple[list[dict], dict]:
    if mode == "train":
        template = prompts.TRAIN_USER_V2 if prompt_version == "v2" else prompts.TRAIN_USER
        user = template.format(header=task["header"], text=task["text"])
        return [{"role": "system", "content": prompts.TRAIN_SYSTEM}, {"role": "user", "content": user}], (
            prompts.TRAIN_SCHEMA
        )
    user = prompts.TEST_USER.format(
        header=task["header"], text=task["text"], instruction=prompts.TEST_TYPE_INSTRUCTIONS[task["qtype"]]
    )
    return [{"role": "system", "content": prompts.TEST_SYSTEM}, {"role": "user", "content": user}], prompts.TEST_SCHEMA


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                keys.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue  # a partially written last line after a crash
    return keys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr generate")
    parser.add_argument("--mode", choices=["train", "eval", "tkhard"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="random sample of N tasks (probe)")
    parser.add_argument("--workers", type=int, default=3, help="parallel requests (= server slots)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v1", help="train prompt variant")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args.mode, args.limit)
    done = done_keys(out)
    todo = [t for t in tasks if t["key"] not in done]
    client = LLMClient(args.base_url)
    model = client.model_id()
    print(f"model {model}: {len(tasks)} tasks, {len(todo)} to do", flush=True)

    lock = threading.Lock()
    started = time.time()
    n_done, n_tokens = 0, 0

    def run(task: dict) -> dict:
        messages, schema = messages_for(args.mode, task, args.prompt_version)
        t0 = time.time()
        parsed, raw, usage = client.chat_json(
            messages, schema, temperature=args.temperature, seed=stable_seed(task["key"], args.seed)
        )
        row = {k: v for k, v in task.items() if k not in ("header", "text")}
        return {
            **row,
            "model": model,
            "prompt_version": args.prompt_version,
            "output": parsed,
            "raw": raw,
            "usage": usage,
            "seconds": time.time() - t0,
        }

    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, t) for t in todo]
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
                n_tokens += row["usage"].get("completion_tokens", 0)
                if n_done % 25 == 0 or n_done == len(todo):
                    el = time.time() - started
                    eta = el / n_done * (len(todo) - n_done)
                    print(
                        f"{n_done}/{len(todo)} {el / 60:.1f} min, {n_done / el:.2f} req/s, "
                        f"{n_tokens / el:.0f} out tok/s, ETA {eta / 60:.0f} min",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
