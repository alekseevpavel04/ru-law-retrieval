"""Dev evaluator for training: chunk protocol, article-level nDCG@10.

Uses the same retrieval and metric code as the final evaluation
(``rlr.eval.retrieve.dense_search`` + ``rlr.eval.metrics``). Keeps the full
history of dev metrics and saves the best model to ``best_dir``.
"""

import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator

from rlr.eval.metrics import mean_metrics, query_metrics
from rlr.eval.retrieve import Corpus, dense_search


class DevRetrievalEvaluator(SentenceEvaluator):
    def __init__(
        self,
        queries: list[dict],
        corpus: Corpus,
        query_prompt: str = "query: ",
        doc_prompt: str = "passage: ",
        best_dir: Path | None = None,
        batch_size: int = 128,
        name: str = "dev",
    ) -> None:
        super().__init__()
        self.queries = queries
        self.corpus = corpus
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt
        self.best_dir = best_dir
        self.batch_size = batch_size
        self.name = name
        self.primary_metric = "ndcg@10"
        self.greater_is_better = True
        self.history: list[dict] = []
        self.best_score = float("-inf")
        self.best_step = -1

    def _encode(self, model, texts: list[str], prompt: str) -> np.ndarray:
        order = np.argsort([-len(t) for t in texts])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            emb = model.encode(
                [texts[i] for i in order],
                prompt=prompt,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        out = np.empty_like(emb, dtype=np.float32)
        out[order] = emb
        return out

    def __call__(self, model, output_path: str | None = None, epoch: float = -1, steps: int = -1) -> dict:
        t0 = time.time()
        was_training = model.training
        model.eval()
        unit_emb = self._encode(model, self.corpus.texts, self.doc_prompt)
        q_emb = self._encode(model, [q["text"] for q in self.queries], self.query_prompt)
        run = dense_search(q_emb, unit_emb, self.corpus)
        per_query = [query_metrics([d for d, _ in r], q["qrels"]) for q, r in zip(self.queries, run, strict=True)]
        metrics = mean_metrics(per_query)
        for slice_name in sorted({q.get("slice", "") for q in self.queries}):
            sub = [m for m, q in zip(per_query, self.queries, strict=True) if q.get("slice", "") == slice_name]
            metrics[f"ndcg@10_{slice_name}"] = mean_metrics(sub)["ndcg@10"]
        if was_training:
            model.train()

        step = max(steps, 0)
        improved = metrics["ndcg@10"] > self.best_score
        if improved:
            self.best_score, self.best_step = metrics["ndcg@10"], step
            if self.best_dir is not None:
                model.save(str(self.best_dir))
        self.history.append(
            {"step": step, "epoch": epoch, **metrics, "is_best": improved, "eval_seconds": round(time.time() - t0, 1)}
        )
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics
