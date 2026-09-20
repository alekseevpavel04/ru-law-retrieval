"""Fix article-level splits once, by seed, and save ID lists.

- held-out articles: ``held_out_frac`` of every in-domain code (no training questions);
- test articles:  ``seen`` (train articles), ``unseen_articles`` (held-out), ``unseen_codes`` (OOD codes);
- dev articles:   in-domain, both train and held-out articles, disjoint from test articles.

Every dev/test article gets one question type, balanced within each slice.

Usage: python -m rlr splits
"""

import argparse
import json
import random
from collections import defaultdict

from rlr.data.parse import CORPUS, read_jsonl
from rlr.env import DATA, RESULTS
from rlr.gen.prompts import QUESTION_TYPES

SPLITS = DATA / "splits"


def stratified_sample(ids_by_code: dict[str, list[str]], frac: float, rng: random.Random) -> list[str]:
    out: list[str] = []
    for code in sorted(ids_by_code):
        ids = sorted(ids_by_code[code])
        k = round(len(ids) * frac)
        out += rng.sample(ids, k)
    return sorted(out)


def proportional_sample(ids_by_code: dict[str, list[str]], n: int, rng: random.Random) -> list[str]:
    total = sum(len(v) for v in ids_by_code.values())
    if total == 0:
        return []
    out: list[str] = []
    for code in sorted(ids_by_code):
        ids = sorted(ids_by_code[code])
        k = min(len(ids), round(n * len(ids) / total))
        out += rng.sample(ids, k)
    return sorted(out)


def assign_types(ids: list[str], rng: random.Random) -> dict[str, str]:
    shuffled = ids[:]
    rng.shuffle(shuffled)
    return {doc_id: QUESTION_TYPES[i % len(QUESTION_TYPES)] for i, doc_id in enumerate(shuffled)}


def make_splits(
    articles: list[dict],
    seed: int = 42,
    held_out_frac: float = 0.10,
    n_test_seen: int = 300,
    n_test_unseen: int = 220,
    n_dev_seen: int = 225,
    n_dev_unseen: int = 75,
    min_chars_eval: int = 150,
) -> dict:
    rng = random.Random(seed)
    in_domain: dict[str, list[str]] = defaultdict(list)
    ood: dict[str, list[str]] = defaultdict(list)
    length = {a["doc_id"]: len(a["text"]) for a in articles}
    for a in articles:
        (in_domain if a["group"] == "in_domain" else ood)[a["code"]].append(a["doc_id"])

    held_out = stratified_sample(in_domain, held_out_frac, rng)
    held_set = set(held_out)
    train_articles = sorted(d for ids in in_domain.values() for d in ids if d not in held_set)

    def eligible(ids: list[str]) -> list[str]:
        return [d for d in ids if length[d] >= min_chars_eval]

    seen_pool = {c: eligible([d for d in ids if d not in held_set]) for c, ids in in_domain.items()}
    unseen_pool = {c: eligible([d for d in ids if d in held_set]) for c, ids in in_domain.items()}

    test_seen = proportional_sample(seen_pool, n_test_seen, rng)
    test_unseen = proportional_sample(unseen_pool, n_test_unseen, rng)
    used = set(test_seen) | set(test_unseen)
    dev_seen = proportional_sample({c: [d for d in v if d not in used] for c, v in seen_pool.items()}, n_dev_seen, rng)
    dev_unseen = proportional_sample(
        {c: [d for d in v if d not in used] for c, v in unseen_pool.items()}, n_dev_unseen, rng
    )
    test_ood = sorted(d for ids in ood.values() for d in eligible(ids))

    slices = {
        "test": {"seen": test_seen, "unseen_articles": test_unseen, "unseen_codes": test_ood},
        "dev": {"seen": dev_seen, "unseen_articles": dev_unseen},
    }
    items = []
    for split, by_slice in slices.items():
        for slice_name, ids in by_slice.items():
            types = assign_types(ids, rng)
            items += [{"split": split, "slice": slice_name, "doc_id": d, "qtype": types[d]} for d in ids]
    return {
        "seed": seed,
        "held_out_frac": held_out_frac,
        "min_chars_eval": min_chars_eval,
        "held_out": held_out,
        "train_articles": train_articles,
        "eval_items": items,
    }


def stats(splits: dict) -> dict:
    """Article counts per split and slice (``data/`` is not committed, this summary is)."""
    counts: dict[str, int] = defaultdict(int)
    for it in splits["eval_items"]:
        counts[f"{it['split']}/{it['slice']}"] += 1
    return {
        "seed": splits["seed"],
        "held_out_frac": splits["held_out_frac"],
        "min_chars_eval": splits["min_chars_eval"],
        "held_out_articles": len(splits["held_out"]),
        "train_articles": len(splits["train_articles"]),
        "eval_articles": dict(sorted(counts.items())),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="overwrite existing splits")
    args = parser.parse_args(argv)

    SPLITS.mkdir(parents=True, exist_ok=True)
    out = SPLITS / "splits.json"
    if out.exists() and not args.force:
        print(f"{out} exists: splits are fixed once (use --force to recreate); refreshing the stats file only")
        splits = json.loads(out.read_text(encoding="utf-8"))
    else:
        splits = make_splits(read_jsonl(CORPUS / "articles.jsonl"), seed=args.seed)
        out.write_text(json.dumps(splits, ensure_ascii=False, indent=1), encoding="utf-8")
    s = stats(splits)
    (RESULTS / "splits_stats.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
