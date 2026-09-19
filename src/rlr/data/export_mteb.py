"""Export the dataset in MTEB retrieval format (corpus / queries / qrels) for Hugging Face.

Output: ``data/export/hf_dataset/`` with README.md (= docs/DATASET_CARD.md), jsonl files,
statistics and the Yandex license notice.

Usage: python -m rlr export-mteb
"""

import argparse
import json
import shutil

from rlr.data.parse import CORPUS, read_jsonl, write_jsonl
from rlr.env import DATA, RESULTS, ROOT

OUT = DATA / "export" / "hf_dataset"
DATASET = DATA / "dataset"


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(prog="rlr export-mteb").parse_args(argv)
    if OUT.exists():
        shutil.rmtree(OUT)
    docs = read_jsonl(CORPUS / "docs_article.jsonl")
    write_jsonl(
        OUT / "corpus" / "corpus.jsonl",
        [{"_id": d["doc_id"], "title": d["title"], "text": d["text"], "code": d["code"]} for d in docs],
    )
    shutil.copy(CORPUS / "sources.json", OUT / "corpus" / "sources.json")
    chunks = read_jsonl(CORPUS / "chunks.jsonl")
    write_jsonl(
        OUT / "chunks" / "chunks.jsonl",
        [{"_id": c["chunk_id"], "doc_id": c["doc_id"], "text": c["text"]} for c in chunks],
    )

    splits = {
        "train": read_jsonl(DATASET / "train_llm.jsonl") + read_jsonl(DATASET / "train_titles.jsonl"),
        "dev": read_jsonl(DATASET / "dev.jsonl"),
        "test": read_jsonl(DATASET / "test.jsonl"),
        "golden": read_jsonl(DATASET / "golden.jsonl"),
    }
    counts = {}
    for split, rows in splits.items():
        queries, qrels = [], []
        for q in rows:
            queries.append(
                {
                    "_id": q["qid"],
                    "text": q["text"],
                    "slice": q.get("slice", "train"),
                    "qtype": q["qtype"],
                    "code": q["code"],
                }
            )
            rel = q.get("qrels") or {q["doc_id"]: 1}
            qrels += [{"query-id": q["qid"], "corpus-id": d, "score": int(s)} for d, s in rel.items()]
        write_jsonl(OUT / "queries" / f"{split}.jsonl", queries)
        write_jsonl(OUT / "qrels" / f"{split}.jsonl", qrels)
        counts[split] = {"queries": len(queries), "qrels": len(qrels)}

    (OUT / "stats").mkdir(parents=True)
    for name in ("dataset_stats.json", "corpus_stats.json", "golden_stats.json", "lexical_overlap.csv"):
        if (RESULTS / name).exists():
            shutil.copy(RESULTS / name, OUT / "stats" / name)
    shutil.copy(DATA / "splits" / "splits.json", OUT / "stats" / "splits.json")
    shutil.copy(ROOT / "docs" / "DATASET_CARD.md", OUT / "README.md")
    (OUT / "stats" / "export_counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    print(json.dumps(counts, indent=1))
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"export size: {total / 2**20:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
