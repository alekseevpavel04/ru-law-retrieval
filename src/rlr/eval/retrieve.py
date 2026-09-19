"""Retrievers: dense (sentence-transformers), BM25 (bm25s + PyStemmer), RRF hybrid.

Both corpus views are supported:
- ``article``: one document per article (header + text), truncated at ``max_seq_length`` tokens;
- ``chunk``: chunks are scored, the article score is the max over its chunks.

Search is exact (matrix product on GPU), the corpus is small.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rlr.data.parse import CORPUS, read_jsonl
from rlr.env import DATA

EMB_CACHE = DATA / "cache" / "emb"
TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)


@dataclass
class Corpus:
    view: str  # "article" | "chunk"
    unit_ids: list[str]  # doc ids (article view) or chunk ids (chunk view)
    texts: list[str]
    unit_article: np.ndarray  # index of the article for every unit
    article_ids: list[str]


def load_corpus(view: str) -> Corpus:
    docs = read_jsonl(CORPUS / "docs_article.jsonl")
    article_ids = [d["doc_id"] for d in docs]
    index = {d: i for i, d in enumerate(article_ids)}
    if view == "article":
        texts = [f"{d['title']}\n{d['text']}" for d in docs]
        return Corpus(view, article_ids, texts, np.arange(len(docs)), article_ids)
    # "chunk" -> chunks.jsonl; any other view name -> <view>.jsonl (e.g. chunk_tkfmt)
    chunks = read_jsonl(CORPUS / ("chunks.jsonl" if view == "chunk" else f"{view}.jsonl"))
    return Corpus(
        view,
        [c["chunk_id"] for c in chunks],
        [c["text"] for c in chunks],
        np.array([index[c["doc_id"]] for c in chunks]),
        article_ids,
    )


# ---------------------------------------------------------------- aggregation / ranking


def aggregate_max(unit_scores: torch.Tensor, unit_article: np.ndarray, n_articles: int) -> torch.Tensor:
    """(n_queries, n_units) -> (n_queries, n_articles), article score = max over its units."""
    idx = torch.as_tensor(unit_article, device=unit_scores.device).unsqueeze(0).expand_as(unit_scores)
    out = torch.full((unit_scores.shape[0], n_articles), float("-inf"), device=unit_scores.device)
    return out.scatter_reduce(1, idx, unit_scores, reduce="amax", include_self=True)


def topk_articles(article_scores: torch.Tensor, article_ids: list[str], k: int = 100) -> list[list[tuple[str, float]]]:
    k = min(k, article_scores.shape[1])
    vals, idx = article_scores.topk(k, dim=1)
    vals, idx = vals.cpu().numpy(), idx.cpu().numpy()
    return [
        [(article_ids[j], float(v)) for j, v in zip(row_i, row_v, strict=True)]
        for row_i, row_v in zip(idx, vals, strict=True)
    ]


def rrf(
    runs: list[list[list[tuple[str, float]]]],
    k: int = 60,
    depth: int = 100,
    weights: list[float] | None = None,
) -> list[list[tuple[str, float]]]:
    """(Weighted) reciprocal rank fusion of several ranked runs (same query order)."""
    weights = weights or [1.0] * len(runs)
    fused = []
    for per_query in zip(*runs, strict=True):
        scores: dict[str, float] = {}
        for w, ranking in zip(weights, per_query, strict=True):
            for rank, (doc, _) in enumerate(ranking):
                scores[doc] = scores.get(doc, 0.0) + w / (k + rank + 1)
        fused.append(sorted(scores.items(), key=lambda x: -x[1])[:depth])
    return fused


# ---------------------------------------------------------------- dense


class DenseEncoder:
    def __init__(
        self,
        path: str,
        query_prompt: str = "",
        doc_prompt: str = "",
        max_seq_length: int = 512,
        dtype: str = "float16",
        padding_side: str | None = None,
        name: str | None = None,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = name or path
        self.path = path
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype) if self.device == "cuda" else torch.float32
        try:
            self.model = SentenceTransformer(
                path, device=self.device, model_kwargs={"torch_dtype": torch_dtype}, trust_remote_code=False
            )
        except TypeError:
            # sentence-transformers 6 cannot load old configs where Normalize has path "" (e.g. deepvk/USER-base):
            # it reads the model config.json as the Normalize config. Build the same pipeline by hand.
            self.model = build_from_modules(path, self.device, torch_dtype)
        # the same truncation for every model (fine-tuned checkpoints are saved with the training length 208)
        self.model.max_seq_length = max_seq_length
        if padding_side:
            self.model.tokenizer.padding_side = padding_side

    def encode(self, texts: list[str], prompt: str, batch_size: int = 64) -> np.ndarray:
        # sort by length to reduce padding, then restore order
        order = np.argsort([-len(t) for t in texts])
        emb = self.model.encode(
            [texts[i] for i in order],
            prompt=prompt,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        out = np.empty_like(emb)
        out[order] = emb
        return out

    def encode_queries(self, queries: list[str], batch_size: int = 128) -> np.ndarray:
        return self.encode(queries, self.query_prompt, batch_size)

    def encode_corpus(self, corpus: Corpus, cache_key: str | None = None, batch_size: int = 32) -> np.ndarray:
        """Encode corpus units; cached on disk by (model, view, prompt, max_len, texts hash)."""
        h = hashlib.md5()
        for t in corpus.texts:
            h.update(t.encode())
        key = cache_key or self.name.replace("/", "__")
        fname = f"{key}_{corpus.view}_{self.model.max_seq_length}_{h.hexdigest()[:10]}.npy"
        path = EMB_CACHE / fname
        if path.exists():
            return np.load(path)
        emb = self.encode(corpus.texts, self.doc_prompt, batch_size)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        return emb


def build_from_modules(path: str, device: str, torch_dtype):
    """Transformer + Pooling (from ``1_Pooling/config.json``) + Normalize, as in the model's modules.json."""
    import json

    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Normalize, Pooling, Transformer

    pooling_cfg = json.loads(Path(hf_hub_download(path, "1_Pooling/config.json")).read_text(encoding="utf-8"))
    transformer = Transformer(path, model_kwargs={"torch_dtype": torch_dtype})
    pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode=_pooling_mode(pooling_cfg))
    return SentenceTransformer(modules=[transformer, pooling, Normalize()], device=device)


def _pooling_mode(cfg: dict) -> str:
    for key, mode in (
        ("pooling_mode_cls_token", "cls"),
        ("pooling_mode_mean_tokens", "mean"),
        ("pooling_mode_max_tokens", "max"),
        ("pooling_mode_lasttoken", "lasttoken"),
    ):
        if cfg.get(key):
            return mode
    return cfg.get("pooling_mode", "mean")


def dense_search(
    query_emb: np.ndarray, unit_emb: np.ndarray, corpus: Corpus, k: int = 100, device: str | None = None
) -> list[list[tuple[str, float]]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    units = torch.as_tensor(unit_emb, device=device)
    results: list[list[tuple[str, float]]] = []
    for i in range(0, len(query_emb), 256):
        q = torch.as_tensor(query_emb[i : i + 256], device=device)
        scores = q @ units.T
        if corpus.view == "chunk":
            scores = aggregate_max(scores, corpus.unit_article, len(corpus.article_ids))
        results += topk_articles(scores, corpus.article_ids, k)
    return results


def unit_scores_dense(query_emb: np.ndarray, unit_emb: np.ndarray, device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.as_tensor(query_emb, device=device)
    u = torch.as_tensor(unit_emb, device=device)
    return (q @ u.T).cpu().numpy()


# ---------------------------------------------------------------- BM25


def bm25_tokenize(texts: list[str], stemmer) -> list[list[str]]:
    out = []
    for t in texts:
        toks = [w.lower().replace("ё", "е") for w in TOKEN_RE.findall(t)]
        out.append(stemmer.stemWords(toks) if stemmer is not None else toks)
    return out


class BM25Retriever:
    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75, stemmer: str | None = "russian") -> None:
        import bm25s
        import Stemmer

        self.corpus = corpus
        self.stemmer = Stemmer.Stemmer(stemmer) if stemmer else None
        self.model = bm25s.BM25(k1=k1, b=b)
        self.model.index(bm25_tokenize(corpus.texts, self.stemmer), show_progress=False)
        self.name = "BM25"

    def search(self, queries: list[str], k: int = 100) -> list[list[tuple[str, float]]]:
        n_units = len(self.corpus.unit_ids)
        results: list[list[tuple[str, float]]] = []
        tokenized = bm25_tokenize(queries, self.stemmer)
        # full score vectors (units are few) -> exact max aggregation over chunks
        for i in range(0, len(tokenized), 256):
            batch = tokenized[i : i + 256]
            scores = np.stack([self.model.get_scores(q) if q else np.zeros(n_units) for q in batch]).astype(np.float32)
            st = torch.as_tensor(scores)
            if self.corpus.view == "chunk":
                st = aggregate_max(st, self.corpus.unit_article, len(self.corpus.article_ids))
            results += topk_articles(st, self.corpus.article_ids, k)
        return results


def save_run(path: Path, qids: list[str], run: list[list[tuple[str, float]]], depth: int = 100) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid, ranking in zip(qids, run, strict=True):
            f.write(json.dumps({"qid": qid, "ranking": [[d, round(s, 5)] for d, s in ranking[:depth]]}) + "\n")


def load_run(path: Path) -> dict[str, list[tuple[str, float]]]:
    return {r["qid"]: [(d, s) for d, s in r["ranking"]] for r in read_jsonl(path)}
