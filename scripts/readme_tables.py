"""Print markdown tables for README from files in results/ (numbers are never typed by hand).

Usage: python scripts/readme_tables.py > results/readme_tables.md
"""

import json
from pathlib import Path

import pandas as pd

R = Path("results")


def fmt(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.3f}"


def main_table(qset: str = "test", protocol: str = "chunk") -> str:
    t = pd.read_csv(R / "main_table.csv")
    t = t[(t["set"] == qset) & (t["protocol"] == protocol)].sort_values("ndcg@10", ascending=False)
    lines = [
        "| Модель | Параметры, M | nDCG@10 [95% ДИ] | Recall@10 | seen | unseen_articles | unseen_codes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in t.iterrows():
        name = f"**{r['model']}**" if r["finetuned"] else r["model"]
        lines.append(
            f"| {name} | {r['params_m']:.0f} | {fmt(r['ndcg@10'])} [{fmt(r['ci_low'])}; {fmt(r['ci_high'])}] "
            f"| {fmt(r['recall@10'])} | {fmt(r.get('ndcg@10_seen'))} | {fmt(r.get('ndcg@10_unseen_articles'))} "
            f"| {fmt(r.get('ndcg@10_unseen_codes'))} |"
        )
    return "\n".join(lines)


def protocol_table() -> str:
    t = pd.read_csv(R / "main_table.csv")
    t = t[t["set"] == "test"].pivot_table(index="model", columns="protocol", values="ndcg@10")
    t = t.sort_values("chunk", ascending=False)
    lines = ["| Модель | article | chunk |", "|---|---:|---:|"]
    lines += [f"| {m} | {fmt(r['article'])} | {fmt(r['chunk'])} |" for m, r in t.iterrows()]
    return "\n".join(lines)


def golden_table() -> str:
    t = pd.read_csv(R / "main_table.csv")
    t = t[(t["set"] == "golden") & (t["protocol"] == "chunk")].sort_values("ndcg@10", ascending=False)
    lines = ["| Модель | nDCG@10 [95% ДИ] | Recall@10 |", "|---|---:|---:|"]
    lines += [
        f"| {r['model']} | {fmt(r['ndcg@10'])} [{fmt(r['ci_low'])}; {fmt(r['ci_high'])}] | {fmt(r['recall@10'])} |"
        for _, r in t.iterrows()
    ]
    return "\n".join(lines)


def bootstrap_table(path: str = "bootstrap.csv") -> str:
    b = pd.read_csv(R / path)
    lines = ["| A | B | набор | срез | n | Δ nDCG@10 | 95% ДИ | p |", "|---|---|---|---|---:|---:|---:|---:|"]
    for _, r in b.iterrows():
        lines.append(
            f"| {r['a']} | {r['b']} | {r['set']}/{r['protocol']} | {r['group']} | {r['n_queries']} | "
            f"{r['delta']:+.3f} | [{r['ci_low']:+.3f}; {r['ci_high']:+.3f}] | {r['p_value']:.3f} |"
        )
    return "\n".join(lines)


def train_table() -> str:
    t = pd.read_csv(R / "train_runs.csv")
    lines = [
        "| Прогон | Модель | Данные | Hard neg | Seed | Пар | Лучшая эпоха | dev nDCG@10 (база → лучший) | Время, мин | Пик памяти, ГБ |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in t.iterrows():
        lines.append(
            f"| {r['run']} | {r['base_model']} | {r['data']} x{r['fraction']:g} | {'да' if r['hard_negatives'] else 'нет'} "
            f"| {r['seed']} | {r['train_rows']} | {r['best_epoch']:.2f} | {r['dev_ndcg@10_base']:.3f} → "
            f"{r['dev_ndcg@10_best']:.3f} | {r['train_min']:.1f} | {r['peak_mem_gb']:.2f} |"
        )
    return "\n".join(lines)


def dataset_table() -> str:
    s = json.loads((R / "corpus_stats.json").read_text(encoding="utf-8"))
    lines = [
        "| Кодекс | Статей | Действующих | Исключено (утр. силу / пустые) | Медиана, симв. | p95, симв. | Медиана, ток. | p95, ток. | > 512 ток. | Фрагментов |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for code, v in s["by_code"].items():
        lines.append(
            f"| {code} | {v['articles_total']} | {v['articles_active']} | {v['excluded_repealed']} / {v['excluded_empty']} "
            f"| {v['chars']['median']:.0f} | {v['chars']['p95']:.0f} | {v['tokens']['median']:.0f} | {v['tokens']['p95']:.0f} "
            f"| {v['over_512_tokens']} | {v['chunks']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    for title, fn in (
        ("Corpus", dataset_table),
        ("Main (test, chunk)", main_table),
        ("Protocols", protocol_table),
        ("Golden", golden_table),
        ("Training runs", train_table),
        ("Bootstrap", bootstrap_table),
    ):
        try:
            print(f"## {title}\n\n{fn()}\n")
        except FileNotFoundError as exc:
            print(f"## {title}\n\n(missing: {exc.filename})\n")
