"""Write results/readme_tables.md from files in results/ (numbers are never typed by hand).

Usage: python scripts/readme_tables.py
"""

import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"


def fmt(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.3f}"


def pval(p: float) -> str:
    """Monte-Carlo p has a floor of 2 / (n + 1); never print it as an exact zero."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


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
            f"{r['delta']:+.3f} | [{r['ci_low']:+.3f}; {r['ci_high']:+.3f}] | {pval(r['p_value'])} |"
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


def recipe_stages_table() -> str:
    """How the recipe was chosen: every stage, every run, dev only."""
    sel = json.loads((R / "recipe_selection.json").read_text(encoding="utf-8"))
    titles = {
        "A": "данные и учитель",
        "B": "аугментация формата фрагментов",
        "C": "дистилляция FRIDA (второй этап)",
        "D": "сиды рецепта A + C",
        "E": "длительность обучения, батч, lr",
        "F": "дистилляция поверх E",
    }
    lines = [
        "| Этап | Прогон | dev nDCG@10 | dev (наш формат) | dev (формат tk-rf-rag) | Решение |",
        "|---|---|---:|---:|---:|---|",
    ]
    rows = []
    for st, v in sel["stages"].items():
        for r in v.get("runs", []) + v.get("seeds", []):
            b = r["best"]
            chosen = "выбран" if r["name"] in (v.get("chosen"), sel["final"]["name"]) else ""
            if st in ("B", "C", "E", "F") and not v.get("kept") and r["name"] != v.get("chosen"):
                chosen = chosen or "не принят"
            rows.append(
                (st, r["name"], b["ndcg@10"], b.get("ndcg@10_view_chunk"), b.get("ndcg@10_view_chunk_tkfmt"), chosen)
            )
    for st, name, m, c1, c2, chosen in rows:
        lines.append(f"| {st}: {titles.get(st, '')} | `{name}` | {m:.4f} | {c1:.4f} | {c2:.4f} | {chosen} |")
    return "\n".join(lines)


def tk_hard_table() -> str:
    t = pd.read_csv(R / "main_table.csv")
    t = t[(t["set"] == "tk_hard") & (t["protocol"] == "chunk")].sort_values("ndcg@10", ascending=False)
    lines = ["| Модель | nDCG@10 [95% ДИ] | Recall@10 | seen | unseen_articles |", "|---|---:|---:|---:|---:|"]
    for _, r in t.iterrows():
        name = f"**{r['model']}**" if r["finetuned"] else r["model"]
        lines.append(
            f"| {name} | {fmt(r['ndcg@10'])} [{fmt(r['ci_low'])}; {fmt(r['ci_high'])}] | {fmt(r['recall@10'])} "
            f"| {fmt(r.get('ndcg@10_seen'))} | {fmt(r.get('ndcg@10_unseen_articles'))} |"
        )
    return "\n".join(lines)


def ablation_table() -> str:
    t = pd.read_csv(R / "ablation.csv")
    lines = ["| Прогон | dev | test | seen | unseen_articles | unseen_codes |", "|---|---:|---:|---:|---:|---:|"]
    lines += [
        f"| {r['run']} | {fmt(r['dev'])} | {fmt(r['test'])} | {fmt(r.get('seen'))} "
        f"| {fmt(r.get('unseen_articles'))} | {fmt(r.get('unseen_codes'))} |"
        for _, r in t.iterrows()
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    out = ["<!-- Generated by scripts/readme_tables.py from results/. Do not edit by hand. -->\n"]
    for title, fn in (
        ("Corpus", dataset_table),
        ("Main (test, chunk)", main_table),
        ("TK-hard (chunk)", tk_hard_table),
        ("Golden (chunk)", golden_table),
        ("Protocols (test): article vs chunk", protocol_table),
        ("Significance (paired bootstrap)", bootstrap_table),
        ("Recipe: every stage, dev only", recipe_stages_table),
        ("Ablations (E1 / E2 / E4)", ablation_table),
        ("Training runs", train_table),
    ):
        out.append(f"## {title}\n\n{fn()}\n")
    path = R / "readme_tables.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {path} ({len(out) - 1} tables)")
