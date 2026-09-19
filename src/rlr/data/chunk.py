"""Build the two corpus views and corpus statistics.

- article view (``articles.jsonl`` -> ``docs_article.jsonl``): header + full text;
- chunk view (``chunks.jsonl``): paragraphs packed greedily up to ``chunk_size``
  characters inside one article, with overlap, each chunk starts with the header.

The chunker is adapted from tk-rf-rag (``app/ingest/chunker.py``): chunks never
cross article boundaries, long paragraphs are split by sentences, then by words.

Usage: python -m rlr chunk [--chunk-size 600 --overlap 90]
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass

import numpy as np

from rlr.config import codes_config
from rlr.data.parse import CORPUS, read_jsonl, write_jsonl
from rlr.env import RESULTS

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:!?])\s+")


@dataclass
class Chunk:
    chunk_id: str  # "{doc_id}#{position}"
    doc_id: str
    position: int
    header: str
    body: str  # chunk text without the header line

    @property
    def text(self) -> str:
        return f"{self.header}\n{self.body}"


def doc_header(code_display: str, number: str, title: str) -> str:
    return f"{code_display}, Статья {number}. {title}"


def _split_to_units(text: str, chunk_size: int) -> list[str]:
    """Paragraphs -> sentences -> words, so that every unit fits into chunk_size."""
    units: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for sentence in SENTENCE_SPLIT_RE.split(paragraph):
            if len(sentence) <= chunk_size:
                units.append(sentence)
                continue
            piece = ""
            for word in sentence.split():
                if piece and len(piece) + 1 + len(word) > chunk_size:
                    units.append(piece)
                    piece = word
                else:
                    piece = f"{piece} {word}" if piece else word
            if piece:
                units.append(piece)
    return units


def _joined_len(units: list[str]) -> int:
    return sum(len(u) for u in units) + max(len(units) - 1, 0)


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    pieces: list[str] = []
    current: list[str] = []
    has_new = False  # current contains something not yet emitted
    for unit in _split_to_units(text, chunk_size):
        if current and _joined_len([*current, unit]) > chunk_size:
            pieces.append("\n".join(current))
            tail: list[str] = []
            for prev in reversed(current):
                if _joined_len([prev, *tail]) > overlap:
                    break
                tail.insert(0, prev)
            current = tail if _joined_len([*tail, unit]) <= chunk_size else []
            has_new = False
        current.append(unit)
        has_new = True
    if current and has_new:
        pieces.append("\n".join(current))
    return pieces


def chunk_article(doc_id: str, header: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    pieces = split_text(text, chunk_size, overlap) or [header]
    return [Chunk(f"{doc_id}#{i}", doc_id, i, header, piece) for i, piece in enumerate(pieces)]


def _stats(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values)
    return {
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }


def token_lengths(texts: list[str], prefix: str, model_name: str) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    lengths: list[int] = []
    for i in range(0, len(texts), 512):
        enc = tok([prefix + t for t in texts[i : i + 512]], add_special_tokens=True)
        lengths += [len(ids) for ids in enc["input_ids"]]
    return lengths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr chunk")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--overlap", type=int, default=90)
    parser.add_argument("--tokenizer", default="intfloat/multilingual-e5-small")
    args = parser.parse_args(argv)

    display = codes_config()["display"]
    articles = read_jsonl(CORPUS / "articles.jsonl")
    docs, chunks = [], []
    for a in articles:
        header = doc_header(display[a["code"]], a["number"], a["title"])
        docs.append({"doc_id": a["doc_id"], "code": a["code"], "title": header, "text": a["text"]})
        chunks += chunk_article(a["doc_id"], header, a["text"], args.chunk_size, args.overlap)
    write_jsonl(CORPUS / "docs_article.jsonl", docs)
    write_jsonl(CORPUS / "chunks.jsonl", [{**asdict(c), "text": c.text} for c in chunks])
    print(f"articles {len(docs)}, chunks {len(chunks)}")

    # corpus statistics table (stage 1 report)
    art_tokens = token_lengths([f"{d['title']}\n{d['text']}" for d in docs], "passage: ", args.tokenizer)
    chunk_tokens = token_lengths([c.text for c in chunks], "passage: ", args.tokenizer)
    all_rows = read_jsonl(CORPUS / "articles_all.jsonl")
    by_code: dict[str, dict] = {}
    for code in dict.fromkeys(r["code"] for r in all_rows):
        idx = [i for i, d in enumerate(docs) if d["code"] == code]
        cidx = [i for i, c in enumerate(chunks) if c.doc_id.split("-", 1)[0] == code]
        rows = [r for r in all_rows if r["code"] == code]
        by_code[code] = {
            "articles_total": len(rows),
            "articles_active": len(idx),
            "excluded_repealed": sum(r["status"] == "repealed" for r in rows),
            "excluded_empty": sum(r["status"] == "empty" for r in rows),
            "chars": _stats([len(docs[i]["title"]) + 1 + len(docs[i]["text"]) for i in idx]),
            "tokens": _stats([art_tokens[i] for i in idx]),
            "over_512_tokens": sum(art_tokens[i] > 512 for i in idx),
            "chunks": len(cidx),
        }
    report = {
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "tokenizer": args.tokenizer,
        "articles": len(docs),
        "chunks": len(chunks),
        "article_tokens": _stats(art_tokens),
        "chunk_chars": _stats([len(c.text) for c in chunks]),
        "chunk_tokens": _stats(chunk_tokens),
        "articles_over_512_tokens": sum(t > 512 for t in art_tokens),
        "by_code": by_code,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "corpus_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
