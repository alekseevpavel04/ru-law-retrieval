"""E6: speed and size of embedding models on CPU (Ryzen 5 5600X) and GPU (RTX 3070).

Per model and device:
- throughput: queries/s when encoding the test queries in batches;
- latency: one query end-to-end (encode + exact search over the chunk index + max
  aggregation to articles), p50 / p95 over ``--n-latency`` queries;
- size: model weights on disk, parameters, index size = dim x chunks x 4 bytes (fp32).

Usage: python -m rlr speed --models e5-small=intfloat/multilingual-e5-small ft=models/x/best e5-large=...
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from rlr.config import load_yaml
from rlr.data.parse import read_jsonl
from rlr.env import DATA, RESULTS
from rlr.eval.retrieve import dense_search, load_corpus


def dir_size_mb(path: str) -> float | None:
    p = Path(path)
    if not p.exists():
        from huggingface_hub import snapshot_download

        p = Path(snapshot_download(path, allow_patterns=["*.safetensors", "*.bin", "*.json"]))
    files = list(p.rglob("*.safetensors")) or list(p.rglob("*.bin"))
    return round(sum(f.stat().st_size for f in files) / 2**20, 1)


def bench(name: str, path: str, qp: str, dp: str, device: str, queries: list[str], n_latency: int) -> dict:
    from sentence_transformers import SentenceTransformer

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = SentenceTransformer(path, device=device, model_kwargs={"torch_dtype": dtype})
    model.max_seq_length = 512
    corpus = load_corpus("chunk")
    # index from cache if possible (fp16 GPU encodings are reused for the CPU index too)
    rng = np.random.default_rng(0)
    dim = model.get_sentence_embedding_dimension()
    index = rng.standard_normal((len(corpus.unit_ids), dim)).astype(np.float32)  # timing only, content irrelevant
    index /= np.linalg.norm(index, axis=1, keepdims=True)

    def enc(texts: list[str], bs: int) -> np.ndarray:
        return model.encode(texts, prompt=qp, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)

    enc(queries[:32], 32)  # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    enc(queries, 64)
    if device == "cuda":
        torch.cuda.synchronize()
    throughput = len(queries) / (time.perf_counter() - t0)

    lat = []
    for q in queries[:n_latency]:
        t0 = time.perf_counter()
        e = enc([q], 1)
        dense_search(e, index, corpus, k=10, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    return {
        "model": name,
        "device": device,
        "threads": torch.get_num_threads() if device == "cpu" else None,
        "queries_per_s": round(throughput, 1),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "dim": dim,
        "weights_mb": dir_size_mb(path),
        "index_chunks": len(corpus.unit_ids),
        "index_mb_fp32": round(len(corpus.unit_ids) * dim * 4 / 2**20, 1),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr speed")
    parser.add_argument("--models", nargs="+", required=True, help="name=path (prompts from baselines.yaml or e5)")
    parser.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    parser.add_argument("--n-queries", type=int, default=512)
    parser.add_argument("--n-latency", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--out", default="speed.json")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.cpu_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    prompts = {m["name"]: (m["query_prompt"], m["doc_prompt"]) for m in load_yaml("baselines.yaml")["models"]}
    queries = [q["text"] for q in read_jsonl(DATA / "dataset" / "test.jsonl")][: args.n_queries]
    out_path = RESULTS / args.out
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    for pair in args.models:
        name, path = pair.split("=", 1)
        qp, dp = prompts.get(name, ("query: ", "passage: "))
        for device in args.devices:
            r = bench(name, path, qp, dp, device, queries, args.n_latency)
            results = [x for x in results if not (x["model"] == name and x["device"] == device)] + [r]
            print(json.dumps(r), flush=True)
            out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
