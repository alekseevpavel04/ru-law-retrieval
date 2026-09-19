"""Fine-tune an embedding model with CachedMultipleNegativesRankingLoss (GradCache).

One YAML config = one run. Everything needed to reproduce and plot the run is
saved: config, seed, git commit, time, peak memory, the full train log (loss,
lr, grad norm every ``logging_steps``) and the dev history (every 1/4 epoch).

Outputs:
  models/<run>/best/                 best checkpoint by dev nDCG@10 (only this one is kept)
  results/train/<run>.json           run summary + full history
  results/train/curves/<run>_*.csv   raw curves (train log, dev history)
  results/figures/train/<run>.png    loss / lr / dev curves

Usage: python -m rlr train configs/train/e1_small_llm_hn.yaml [--max-steps 20] [--wandb]
"""

import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from rlr.config import load_yaml
from rlr.data.parse import read_jsonl
from rlr.env import DATA, MODELS, RESULTS, ROOT
from rlr.eval.retrieve import load_corpus
from rlr.train.evaluator import DevRetrievalEvaluator
from rlr.train.sampler import make_sampler_factory

DATASET = DATA / "dataset"
TRAIN_RESULTS = RESULTS / "train"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def load_rows(cfg: dict) -> tuple[list[dict], dict]:
    """Training rows: anchor, positive, (negative), plus article keys for the sampler."""
    data = cfg["data"]
    hn = cfg.get("hard_negatives", False)
    suffix = f"_hn_{cfg['mining_name']}" if hn else ""
    rows: list[dict] = []
    info: dict = {}
    if data.get("llm", True):
        llm = read_jsonl(DATASET / f"train_llm{suffix}.jsonl")
        frac = data.get("fraction", 1.0)
        if frac < 1.0:
            # nested subsets: the 25% subset is a prefix of the 50% subset (fixed data seed)
            order = list(range(len(llm)))
            random.Random(data.get("fraction_seed", 0)).shuffle(order)
            llm = [llm[i] for i in sorted(order[: round(len(llm) * frac)])]
        rows += llm
        info["llm"] = len(llm)
    if data.get("titles", False):
        titles = read_jsonl(DATASET / f"train_titles{suffix}.jsonl")
        rows += titles
        info["titles"] = len(titles)
    if hn:
        # a few queries have no valid negative: use a random chunk of another article
        chunks = read_jsonl(DATA / "corpus" / "chunks.jsonl")
        rng = random.Random(cfg["seed"])
        filled = 0
        for r in rows:
            if r.get("neg_text"):
                continue
            while True:
                c = rng.choice(chunks)
                if c["doc_id"] != r["doc_id"]:
                    break
            r["neg_chunk_id"] = c["chunk_id"]
            r["neg_text"] = c["body"] if r["qtype"] == "title" else c["text"]
            filled += 1
        info["random_negatives_filled"] = filled
    return rows, info


def load_teacher_rows(cfg: dict) -> tuple[list[dict], dict]:
    """v2 rows annotated by the teacher (``rlr teacher``): noise filter, data fraction, format augmentation."""
    rows = read_jsonl(DATASET / f"{cfg['train_file']}.jsonl")
    info: dict = {"rows_total": len(rows)}
    if cfg.get("prompt_versions"):
        rows = [r for r in rows if r["prompt_version"] in cfg["prompt_versions"]]
        info["rows_prompt_versions"] = len(rows)
    k = cfg.get("teacher_filter_rank")
    if k:
        kept = [r for r in rows if r["teacher_gold_rank"] <= k]
        info["dropped_by_teacher_rank"] = len(rows) - len(kept)
        rows = kept
    frac = cfg.get("fraction", 1.0)
    if frac < 1.0:
        order = list(range(len(rows)))
        random.Random(cfg.get("fraction_seed", 0)).shuffle(order)
        rows = [rows[i] for i in sorted(order[: round(len(rows) * frac)])]
    p = cfg.get("format_aug", 0.0)
    rng = random.Random(cfg["seed"])
    n_aug = 0
    for r in rows:
        if p > 0 and rng.random() < p:  # the same question with chunks in the tk-rf-rag format (500/75, no code)
            r["pos_text"], r["neg_text"] = r["pos_b_text"], r["neg_b_text"]
            r["cand_texts"], r["labels"] = r["cand_b_texts"], r["labels_b"]
            n_aug += 1
    info["format_aug_rows"] = n_aug
    info["rows"] = len(rows)
    return rows, info


def to_distill_dataset(rows: list[dict], teacher_temperature: float) -> Dataset:
    """(query, positive, cand_1..cand_n) with teacher logits = cosine / teacher_temperature."""
    n = len(rows[0]["cand_texts"])
    cols: dict[str, list] = {"query": [r["text"] for r in rows], "positive": [r["pos_text"] for r in rows]}
    for i in range(n):
        cols[f"cand_{i}"] = [r["cand_texts"][i] for r in rows]
    cols["label"] = [[v / teacher_temperature for v in r["labels"]] for r in rows]
    return Dataset.from_dict(cols)


def to_dataset(rows: list[dict], hn: bool) -> tuple[Dataset, list[frozenset[str]]]:
    cols: dict[str, list[str]] = {"anchor": [r["text"] for r in rows], "positive": [r["pos_text"] for r in rows]}
    keys = []
    for r in rows:
        k = {r["doc_id"]}
        if hn:
            k.add(r["neg_chunk_id"].split("#")[0])
        keys.append(frozenset(k))
    if hn:
        cols["negative"] = [r["neg_text"] for r in rows]
    return Dataset.from_dict(cols), keys


def run(cfg: dict, max_steps: int = -1, use_wandb: bool = False, out_name: str | None = None) -> dict:
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
        DistillKLDivLoss,
    )
    from sentence_transformers.util import pairwise_dot_score
    from transformers import TrainerCallback

    name = out_name or cfg["name"]
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    loss_name = cfg.get("loss", "mnrl")
    if cfg.get("train_file"):
        rows, data_info = load_teacher_rows(cfg)
        hn = cfg.get("negatives", "teacher") == "teacher"
    else:
        rows, data_info = load_rows(cfg)
        hn = cfg.get("hard_negatives", False)
    if loss_name == "distill_kl":
        train_ds, row_keys = to_distill_dataset(rows, cfg.get("teacher_temperature", 0.05)), None
    else:
        train_ds, row_keys = to_dataset(rows, hn)
    dev = read_jsonl(DATASET / "dev.jsonl")
    corpus = {view: load_corpus(view) for view in cfg.get("dev_views", ["chunk"])}

    model = SentenceTransformer(cfg["base_model"], device="cuda")
    model.max_seq_length = cfg["max_seq_length"]
    if cfg.get("freeze_embeddings", True):
        model[0].auto_model.get_input_embeddings().requires_grad_(False)
    if cfg.get("grad_checkpointing", False):  # only without GradCache (distillation stage): activations do not fit
        model[0].auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out_dir = MODELS / name
    best_dir = out_dir / "best"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    evaluator = DevRetrievalEvaluator(dev, corpus, best_dir=best_dir)

    steps_per_epoch = -(-len(train_ds) // cfg["batch_size"])
    total_steps = max_steps if max_steps > 0 else round(steps_per_epoch * cfg["epochs"])
    eval_steps = max(1, round(steps_per_epoch * cfg.get("eval_every", 0.25)))

    if loss_name == "distill_kl":
        # student logits = scale * cosine (as in MNRL), teacher logits = cosine / teacher_temperature;
        # temperature=1 inside the loss, so gradients are not shrunk by T^2
        scale = cfg.get("scale", 20.0)
        loss = DistillKLDivLoss(model, similarity_fct=lambda a, b: scale * pairwise_dot_score(a, b), temperature=1.0)
    else:
        loss = CachedMultipleNegativesRankingLoss(
            model, mini_batch_size=cfg["mini_batch_size"], scale=cfg.get("scale", 20.0)
        )
    prompts = {
        c: ("query: " if c in ("anchor", "query") else "passage: ") for c in train_ds.column_names if c != "label"
    }
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=cfg["epochs"],
        max_steps=max_steps,
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        warmup_steps=cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type="linear",
        weight_decay=cfg.get("weight_decay", 0.01),
        bf16=True,
        tf32=True,
        batch_sampler=make_sampler_factory(row_keys) if row_keys is not None else "batch_sampler",
        prompts=prompts,
        eval_strategy="steps" if max_steps < 0 or max_steps > eval_steps else "no",
        eval_steps=eval_steps,
        save_strategy="no",
        logging_steps=cfg.get("logging_steps", 2),
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        report_to=["wandb"] if use_wandb else "none",
        run_name=name,
        disable_tqdm=True,
    )

    train_log: list[dict] = []
    t_start = time.time()

    class LogCallback(TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **kwargs):
            if logs and "loss" in logs:
                train_log.append(
                    {
                        "step": state.global_step,
                        "epoch": state.epoch,
                        "loss": logs["loss"],
                        "lr": logs.get("learning_rate"),
                        "grad_norm": logs.get("grad_norm"),
                        "elapsed_s": round(time.time() - t_start, 2),
                    }
                )

    torch.cuda.reset_peak_memory_stats()
    evaluator(model, epoch=0, steps=0)  # starting point of the dev curve (= base model)
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_ds, loss=loss, evaluator=evaluator, callbacks=[LogCallback()]
    )
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    if evaluator.history[-1]["step"] != trainer.state.global_step:
        evaluator(model, epoch=trainer.state.epoch, steps=trainer.state.global_step)
    peak_gb = torch.cuda.max_memory_allocated() / 2**30

    summary = {
        "name": name,
        "config": cfg,
        "seed": seed,
        "git_commit": git_commit(),
        "base_model": cfg["base_model"],
        "data": data_info,
        "train_rows": len(train_ds),
        "params": n_params,
        "trainable_params": n_trainable,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": trainer.state.global_step,
        "eval_steps": eval_steps,
        "train_seconds": round(train_seconds, 1),
        "seconds_per_step": round(train_seconds / max(trainer.state.global_step, 1), 3),
        "peak_mem_gb": round(peak_gb, 2),
        "dev_base": evaluator.history[0],
        "dev_best": max(evaluator.history, key=lambda h: h["ndcg@10"]),
        "dev_history": evaluator.history,
        "train_log": train_log,
        "best_model_dir": str(best_dir.relative_to(ROOT)),
        "planned_total_steps": total_steps,
    }
    if max_steps < 0:
        TRAIN_RESULTS.mkdir(parents=True, exist_ok=True)
        (TRAIN_RESULTS / "curves").mkdir(exist_ok=True)
        (TRAIN_RESULTS / f"{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        pd.DataFrame(train_log).to_csv(TRAIN_RESULTS / "curves" / f"{name}_train.csv", index=False)
        pd.DataFrame(evaluator.history).to_csv(TRAIN_RESULTS / "curves" / f"{name}_dev.csv", index=False)
        from rlr.plots import plot_run

        plot_run(summary, RESULTS / "figures" / "train" / f"{name}.png")
    shutil.rmtree(out_dir / "trainer", ignore_errors=True)
    if max_steps > 0:  # probe: keep nothing on disk
        shutil.rmtree(out_dir, ignore_errors=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlr train")
    parser.add_argument("config")
    parser.add_argument("--max-steps", type=int, default=-1, help="probe run (results are not saved)")
    parser.add_argument("--seed", type=int, help="override the config seed (run name gets a suffix)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-run even if results exist")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    name = cfg["name"]
    if args.seed is not None and args.seed != cfg["seed"]:
        cfg["seed"] = args.seed
        name = f"{name}_s{args.seed}"
    if args.max_steps < 0 and (TRAIN_RESULTS / f"{name}.json").exists() and not args.force:
        print(f"{name}: results exist, skipping (use --force)")
        return
    summary = run(cfg, args.max_steps, args.wandb, out_name=name if args.max_steps < 0 else f"probe_{name}")
    keys = ("name", "train_rows", "total_steps", "train_seconds", "seconds_per_step", "peak_mem_gb", "trainable_params")
    print(json.dumps({k: summary[k] for k in keys}, indent=1))
    print("dev base:", {k: round(v, 4) for k, v in summary["dev_base"].items() if k.startswith("ndcg")})
    print("dev best:", {k: v for k, v in summary["dev_best"].items() if k.startswith("ndcg") or k == "step"})


if __name__ == "__main__":
    main()
