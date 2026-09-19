"""Local annotation UI for the golden set (FastAPI + one HTML page).

Every answer is appended to ``data/golden/annotations.jsonl`` immediately, so the
work can be interrupted and resumed at any time (the last answer per question wins).

Usage: python -m rlr annotate [--port 8765] [--annotator name]
"""

import argparse
import json
import threading
import time
from pathlib import Path

from rlr.annotate.golden import GOLDEN, latest_annotations
from rlr.data.parse import read_jsonl

STATIC = Path(__file__).parent / "static"
_lock = threading.Lock()


def create_app(annotator: str):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel

    items = read_jsonl(GOLDEN / "items.jsonl")
    by_qid = {i["qid"]: i for i in items}
    app = FastAPI(title="ru-law-retrieval golden annotation")

    class Annotation(BaseModel):
        qid: str
        question_status: str  # ok | fix | drop
        fixed_text: str = ""
        gold_relevant: bool
        extra_relevant: list[str] = []
        comment: str = ""

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/items")
    def list_items():
        ann = latest_annotations()
        return [{"qid": i["qid"], "slice": i["slice"], "qtype": i["qtype"], "done": i["qid"] in ann} for i in items]

    @app.get("/api/item/{qid}")
    def get_item(qid: str):
        if qid not in by_qid:
            raise HTTPException(404)
        return {**by_qid[qid], "annotation": latest_annotations().get(qid)}

    @app.post("/api/annotate")
    def annotate(a: Annotation):
        if a.qid not in by_qid or a.question_status not in ("ok", "fix", "drop"):
            raise HTTPException(400)
        allowed = {c["doc_id"] for c in by_qid[a.qid]["candidates"]}
        if not set(a.extra_relevant) <= allowed:
            raise HTTPException(400, "unknown candidate")
        row = {**a.model_dump(), "annotator": annotator, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with _lock, (GOLDEN / "annotations.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"ok": True}

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="rlr annotate")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--annotator", default="human")
    args = parser.parse_args(argv)
    print(f"open http://127.0.0.1:{args.port}")
    uvicorn.run(create_app(args.annotator), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
