# data/

Everything here is produced by the pipeline and is **not** committed (see `.gitignore`); this file is
the only tracked one. Run the steps in the README's «Воспроизведение» section to rebuild it.

| Path | Written by | What it holds |
|---|---|---|
| `raw/` | `rlr download` | cached HTML pages of the six laws from legalacts.ru |
| `corpus/` | `rlr parse`, `rlr chunk` | `articles.jsonl`, `articles_all.jsonl` (before filtering), `chunks*.jsonl`, `sources.json` |
| `splits/splits.json` | `rlr splits` | article-level splits, fixed once by seed (summary: `results/splits_stats.json`) |
| `gen/` | `rlr generate`, `rlr judge` | raw LLM generations and judge verdicts, one file per pass |
| `dataset/` | `rlr build-dataset`, `rlr teacher` | the final `train_*.jsonl`, `dev/test/golden/tk_hard.jsonl` and the teacher-annotated training pairs |
| `golden/` | `rlr golden`, `rlr annotate` | golden annotation items and the resulting qrels |
| `runs/` | `rlr baselines`, `rlr evaluate` | top-100 rankings per model, set and protocol (input to the RRF hybrids) |
| `cache/` | `rlr evaluate` | corpus embeddings, keyed by checkpoint, prompt, length and corpus hash |
| `external/` | `rlr baselines` | the tk-rf-rag question set, converted, when `TK_RF_RAG_QUESTIONS` is set |
| `export/` | `scripts/export_model.py` | the checkpoint in the classic sentence-transformers layout, as published |

Only `results/` is committed: it holds every number and figure the README cites.
