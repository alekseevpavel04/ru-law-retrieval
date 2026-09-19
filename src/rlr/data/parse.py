"""Parse raw legalacts.ru HTML into ``data/corpus/articles.jsonl``.

The parsing logic works on a flat list of text blocks (``Block``) so that it can
be unit-tested without HTML. Known bugs of the tk-rf-rag parser that are fixed
here (each has a regression test in ``tests/test_parse.py``):

- article numbers with dots and dashes: ``22.1``, ``341.1-1``, ``5.27.1``;
- repealed articles still have a body (a reference to the repealing law);
- related documents appended to the page after the signature of the code.

Usage: python -m rlr parse
"""

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from rlr.config import codes_config
from rlr.data.download import RAW, slug_of
from rlr.env import DATA

CORPUS = DATA / "corpus"

ARTICLE_RE = re.compile(r"^Статья\s+(\d+(?:[.\-]\d+)*)\s*\.?\s*(.*)$")
CHAPTER_RE = re.compile(r"^Глава\s+(\d+(?:[.\-]\d+)*)\s*\.?\s*(.*)$", re.IGNORECASE)
REPEALED_RE = re.compile(r"^\(?\s*(Утратил[аи]? силу|Исключен[аы]?)", re.IGNORECASE)
EDITION_RE = re.compile(r"\((ред\.[^)]*)\)")
SPACE_RE = re.compile(r"\s+")


@dataclass
class Block:
    kind: str  # "p" (paragraph), "center" (centered heading), "right" (signature), "stop"
    text: str


@dataclass
class Article:
    doc_id: str
    code: str
    number: str
    title: str
    chapter: str = ""
    chapter_title: str = ""
    text: str = ""
    status: str = "active"  # active | repealed | empty
    source_url: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def header(self) -> str:
        return f"Статья {self.number}. {self.title}"


def clean(text: str) -> str:
    return SPACE_RE.sub(" ", text.replace("\xa0", " ")).strip()


def article_status(title: str, body: str) -> str:
    if REPEALED_RE.match(title) or (REPEALED_RE.match(body) and len(body) < 400):
        return "repealed"
    if not body.strip():
        return "empty"
    return "active"


def parse_blocks(blocks: list[Block], code: str) -> list[Article]:
    """Split a flat block stream of one code into articles.

    Parsing stops at the first signature block ("Президент ...") or a "stop"
    block after at least one article was seen: everything after it is related
    documents that legalacts appends to the page.
    """
    articles: list[Article] = []
    chapter, chapter_title = "", ""
    current: Article | None = None
    body: list[str] = []

    def flush() -> None:
        if current is None:
            return
        current.text = "\n".join(body).strip()
        current.status = article_status(current.title, current.text)
        articles.append(current)

    for block in blocks:
        text = clean(block.text)
        if block.kind in ("right", "stop") and current is not None:
            break
        if not text:
            continue
        if m := CHAPTER_RE.match(text):
            flush()
            current, body = None, []
            chapter, chapter_title = m.group(1), m.group(2).strip()
            continue
        if m := ARTICLE_RE.match(text):
            flush()
            number, title = m.group(1), m.group(2).strip()
            current = Article(f"{code}-{number}", code, number, title, chapter, chapter_title)
            body = []
            continue
        if block.kind == "center":
            # section / subsection / paragraph (§) headings are not article text
            continue
        if current is not None:
            body.append(text)
    flush()
    return articles


def blocks_from_full_page(html: str) -> list[Block]:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("div", class_="main-center-block")
    if root is None:
        raise ValueError("no main-center-block on page")
    blocks: list[Block] = []
    for el in root.find_all(["p", "div"]):
        assert isinstance(el, Tag)
        classes = el.get("class") or []
        if el.name == "div":
            if "pb-2" in classes:  # teaser of a related document
                blocks.append(Block("stop", ""))
            continue
        kind = "center" if "pCenter" in classes else "right" if "pRight" in classes else "p"
        blocks.append(Block(kind, el.get_text(" ")))
    return blocks


def edition_of(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    if h1 is None:
        return ""
    return "; ".join(clean(m) for m in EDITION_RE.findall(h1.get_text(" "))) or clean(h1.get_text(" "))


def article_from_page(html: str, code: str, href: str) -> Article | None:
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    if h1 is None:
        return None
    m = ARTICLE_RE.match(clean(h1.get_text(" ")))
    if m is None:
        return None
    number, title = m.group(1), m.group(2).strip()
    chapter = ""
    if cm := re.search(r"/glava-([^/]+)/", href):
        chapter = cm.group(1)
    chapter_title = ""
    crumbs = soup.find("div", class_="main-center-block-crambs")
    if crumbs is not None:
        for a in crumbs.find_all("a"):
            if mm := CHAPTER_RE.match(clean(a.get_text(" "))):
                chapter, chapter_title = mm.group(1), mm.group(2)
    body_el = soup.find("div", class_="main-center-block-article-text")
    paragraphs: list[str] = []
    if body_el is not None:
        for p in body_el.find_all("p"):
            text = clean(p.get_text(" "))
            if text:
                paragraphs.append(text)
    text = "\n".join(paragraphs)
    art = Article(f"{code}-{number}", code, number, title, chapter, chapter_title, text)
    art.status = article_status(title, text)
    return art


def parse_code(entry: dict, base: str) -> tuple[list[Article], dict]:
    slug = slug_of(entry["path"])
    code_dir = RAW / slug
    meta = json.loads((code_dir / "meta.json").read_text(encoding="utf-8"))
    index_html = (code_dir / "index.html").read_text(encoding="utf-8")
    edition = edition_of(index_html)
    if entry["layout"] == "full":
        articles = parse_blocks(blocks_from_full_page(index_html), entry["code"])
        for a in articles:
            a.source_url = base + entry["path"]
    else:
        links = json.loads((code_dir / "links.json").read_text(encoding="utf-8"))
        articles = []
        for href in links:
            from rlr.data.download import article_file_name

            html = (code_dir / "articles" / article_file_name(href)).read_text(encoding="utf-8")
            art = article_from_page(html, entry["code"], href)
            if art is not None:
                art.source_url = base + href
                articles.append(art)
    for a in articles:
        a.extra = {"part": entry["name"], "edition": edition, "downloaded": meta["downloaded"]}
    info = {"slug": slug, "name": entry["name"], "edition": edition, "downloaded": meta["downloaded"]}
    return articles, info


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(prog="rlr parse").parse_args(argv)
    cfg = codes_config()
    all_articles: list[Article] = []
    sources = []
    for entry in cfg["codes"]:
        articles, info = parse_code(entry, cfg["source"])
        sources.append({**info, "code": entry["code"], "group": entry["group"], "articles": len(articles)})
        all_articles += articles
        print(f"{info['slug']}: {len(articles)} articles, edition: {info['edition']}")

    dup = [k for k, v in Counter(a.doc_id for a in all_articles).items() if v > 1]
    if dup:
        raise SystemExit(f"duplicate doc ids: {dup[:20]}")
    groups = {e["code"]: e["group"] for e in cfg["codes"]}
    rows = [{**asdict(a), "group": groups[a.code]} for a in all_articles]
    write_jsonl(CORPUS / "articles_all.jsonl", rows)
    active = [r for r in rows if r["status"] == "active"]
    write_jsonl(CORPUS / "articles.jsonl", active)
    (CORPUS / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=1), encoding="utf-8")
    status = Counter((r["code"], r["status"]) for r in rows)
    print(dict(status))
    print(f"total {len(rows)}, active {len(active)}")


if __name__ == "__main__":
    main()
