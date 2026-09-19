"""Download codes of law from legalacts.ru as raw HTML.

Two page layouts exist on the site (2026):
- ``full``: the whole code on one page (TK, GK parts 1-3, ZhK, KoAP, ZoZPP);
- ``toc``: a table of contents with links to one page per article (GK part 4, SK).

Raw HTML is cached in ``data/raw/<slug>/``; re-running skips files that exist,
so an interrupted download resumes where it stopped.

Usage: python -m rlr download [--only tk sk] [--limit N]
"""

import argparse
import json
import re
import sys
import time
from datetime import date

import httpx
from bs4 import BeautifulSoup

from rlr.config import codes_config
from rlr.env import DATA

RAW = DATA / "raw"


def slug_of(path: str) -> str:
    return path.strip("/").split("/")[-1]


def fetch(client: httpx.Client, url: str, retries: int = 4) -> str:
    for attempt in range(1, retries + 1):
        try:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError as exc:
            if attempt == retries:
                raise
            print(f"  retry {attempt} for {url}: {exc}", file=sys.stderr)
            time.sleep(3 * attempt)
    raise RuntimeError("unreachable")


def toc_article_links(html: str, code_path: str) -> list[str]:
    """Article links of this code only (pages also link to other codes)."""
    pattern = re.compile(rf"^{re.escape(code_path)}.*statja-[^/]+/$")
    soup = BeautifulSoup(html, "html.parser")
    seen: dict[str, None] = {}
    for a in soup.find_all("a", href=pattern):
        seen.setdefault(a["href"], None)
    return list(seen)


def article_file_name(href: str) -> str:
    # /kodeks/SK-RF/razdel-i/glava-1/statja-1/ -> razdel-i__glava-1__statja-1.html
    parts = href.strip("/").split("/")[2:]
    return "__".join(parts) + ".html"


def download_code(client: httpx.Client, entry: dict, base: str, delay: float, limit: int = 0) -> None:
    slug = slug_of(entry["path"])
    out_dir = RAW / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    main_file = out_dir / "index.html"
    if not main_file.exists():
        main_file.write_text(fetch(client, base + entry["path"]), encoding="utf-8")
        time.sleep(delay)
    meta = {
        "code": entry["code"],
        "name": entry["name"],
        "url": base + entry["path"],
        "layout": entry["layout"],
        "downloaded": date.today().isoformat(),
    }
    meta_file = out_dir / "meta.json"
    if meta_file.exists():  # keep the original download date
        meta["downloaded"] = json.loads(meta_file.read_text(encoding="utf-8"))["downloaded"]

    if entry["layout"] == "toc":
        links = toc_article_links(main_file.read_text(encoding="utf-8"), entry["path"])
        if limit:
            links = links[:limit]
        meta["articles_in_toc"] = len(links)
        art_dir = out_dir / "articles"
        art_dir.mkdir(exist_ok=True)
        todo = [h for h in links if not (art_dir / article_file_name(h)).exists()]
        print(f"{slug}: {len(links)} article links, {len(todo)} to download")
        for i, href in enumerate(todo, 1):
            (art_dir / article_file_name(href)).write_text(fetch(client, base + href), encoding="utf-8")
            if i % 50 == 0:
                print(f"  {slug}: {i}/{len(todo)}")
            time.sleep(delay)
        (out_dir / "links.json").write_text(json.dumps(links, ensure_ascii=False, indent=1), encoding="utf-8")
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{slug}: done")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr download")
    parser.add_argument("--only", nargs="*", help="code ids to download (default: all)")
    parser.add_argument("--limit", type=int, default=0, help="per-article pages limit (debug)")
    args = parser.parse_args(argv)

    cfg = codes_config()
    headers = {"User-Agent": cfg["user_agent"]}
    with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
        for entry in cfg["codes"]:
            if args.only and entry["code"] not in args.only:
                continue
            download_code(client, entry, cfg["source"], cfg["delay"], args.limit)


if __name__ == "__main__":
    main()
