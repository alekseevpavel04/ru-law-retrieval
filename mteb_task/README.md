# Задача RuLawRetrieval для MTEB

Статус: **PR отправлен** — [embeddings-benchmark/mteb#5496](https://github.com/embeddings-benchmark/mteb/pull/5496), ветка `add-ru-law-retrieval` форка, перебазирована на их `main`.

- `ru_law_retrieval.py`: класс задачи `RuLawRetrieval` для `mteb/tasks/retrieval/rus/`.
- `descriptive_stats.json`: описательная статистика из `task.calculate_descriptive_statistics()`. Дублей нет, минимальная длина документа 136 символов.
- `mteb_results.json`: прогоны штатным `mteb.evaluate`:
  - случайный энкодер — nDCG@10 0.0010 (задача не решается случайностью);
  - `intfloat/multilingual-e5-small` — 0.7564;
  - `alekseevpavel04/multilingual-e5-small-ru-law` — 0.8226.

  Те же числа даёт наш собственный код в протоколе article (0.757 и 0.823): независимая перекрёстная проверка всей оценки.
- `0001-...patch`: коммит для форка MTEB (задача + регистрация в `__init__.py` + статистика). Тесты метаданных MTEB (`test_metadata.py`, `test_get_tasks.py`) на форке проходят.
- `PR_DESCRIPTION.md`: текст PR по чек-листу MTEB.

Формальные требования выполнены: формат, метаданные, описательная статистика, прогон случайной и маленькой модели, нетривиальные и неслучайные скоры. Что могут спросить на ревью:

- вопросы синтетические (у MTEB такие задачи есть, в метаданных это указано через `LM-generated and verified`);
- смешанная лицензия (`multiple`): тексты законов не охраняются авторским правом, вопросы dev/test сгенерированы моделью со своей лицензией, код — MIT.

Как это отправлялось (если понадобится повторить для обновления задачи):

```bash
cd <клон mteb>                     # ветка add-ru-law-retrieval, перебазированная на origin/main
ruff format mteb/tasks/retrieval/rus/ru_law_retrieval.py   # их стиль форматирования
pytest tests/test_tasks/test_metadata.py -o addopts=""     # 1973 passed
gh repo fork embeddings-benchmark/mteb
git push fork add-ru-law-retrieval
gh pr create --repo embeddings-benchmark/mteb --base main --head alekseevpavel04:add-ru-law-retrieval   --title "Add RuLawRetrieval (Russian legal article retrieval)"   --body-file ../ru-law-retrieval/mteb_task/PR_DESCRIPTION.md
```
