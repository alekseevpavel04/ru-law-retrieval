"""Filters for generated questions.

Each filter returns ``None`` if the question passes or a short reason string.
The order of filters is fixed so that per-filter drop counts are reproducible.
"""

import re
from collections.abc import Iterable

WORD_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)

# "статья 81", "ст. 81", "статьи 22.1", "пункт 3", "ч. 2", "гл. 13"
ARTICLE_REF_RE = re.compile(
    r"(\bстат(ь[яеиюй]|ей|ьям)\s*№?\s*\d)|(\bст\.\s*\d)|(\bч\.\s*\d)|(\bп\.\s*\d)|(\bгл(\.|ав[аеыу])\s*\d)",
    re.IGNORECASE,
)
META_PHRASES_RE = re.compile(
    r"(согласно\s+(данной\s+|этой\s+)?стать)|(в\s+(данной|этой|настоящей)\s+стать)|(по\s+(данной|этой)\s+стать)"
    r"|(данн(ая|ой)\s+стать)|(настоящ(ий|его)\s+кодекс)|(в\s+тексте\s+(статьи|закона))",
    re.IGNORECASE,
)

LENGTH_LIMITS = {  # chars
    "everyday": (20, 400),
    "search": (8, 120),
    "legal": (20, 400),
}
SEARCH_WORDS = (2, 9)


def normalize(text: str) -> str:
    return " ".join(w.lower().replace("ё", "е") for w in WORD_RE.findall(text))


def words(text: str) -> list[str]:
    return normalize(text).split()


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def check_length(question: str, qtype: str) -> str | None:
    lo, hi = LENGTH_LIMITS[qtype]
    if not lo <= len(question.strip()) <= hi:
        return "length"
    if qtype == "search":
        n = len(words(question))
        if not SEARCH_WORDS[0] <= n <= SEARCH_WORDS[1]:
            return "length"
    return None


def check_article_ref(question: str) -> str | None:
    if ARTICLE_REF_RE.search(question) or META_PHRASES_RE.search(question):
        return "article_ref"
    return None


def check_copy(question: str, source_ngrams: set[tuple[str, ...]], n: int = 6) -> str | None:
    """Reject if the question shares > n-1 consecutive words with the source text."""
    if ngrams(words(question), n) & source_ngrams:
        return "copy_ngram"
    return None


def run_filters(question: str, qtype: str, source_text: str, n: int = 6, copy_check: bool = True) -> str | None:
    if not question or not question.strip():
        return "empty"
    for reason in (
        check_length(question, qtype),
        check_article_ref(question),
        check_copy(question, ngrams(words(source_text), n), n) if copy_check else None,
    ):
        if reason:
            return reason
    return None


def exact_duplicates(texts: Iterable[str]) -> list[bool]:
    """True for every text whose normalized form was already seen earlier."""
    seen: set[str] = set()
    flags = []
    for t in texts:
        key = normalize(t)
        flags.append(key in seen)
        seen.add(key)
    return flags
