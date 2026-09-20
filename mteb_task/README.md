# Задача RuLawRetrieval для MTEB

Статус: **задача готова и проверена локально, PR в MTEB не отправлен** — публикация в чужом репозитории и дальнейшая переписка с ревьюерами остаются за автором.

- `ru_law_retrieval.py`: класс задачи `RuLawRetrieval` для `mteb/tasks/retrieval/rus/`.
- `descriptive_stats.json`: описательная статистика из `task.calculate_descriptive_statistics()`. Дублей нет, минимальная длина документа 136 символов.
- `mteb_results.json`: прогоны штатным `mteb.evaluate`:
  - случайный энкодер — nDCG@10 0.0010 (задача не решается случайностью);
  - `intfloat/multilingual-e5-small` — 0.7564;
  - `alekseevpavel04/multilingual-e5-small-ru-law` — 0.8226.

  Те же числа даёт наш собственный код в протоколе article (0.757 и 0.823): независимая перекрёстная проверка всей оценки.
- `0001-...patch`: коммит для форка MTEB (задача + регистрация в `__init__.py` + статистика). Тесты MTEB `test_metadata.py` и `test_get_tasks.py` проходят (2004 passed).
- `PR_DESCRIPTION.md`: текст PR по чек-листу MTEB.

Формальные требования выполнены: формат, метаданные, описательная статистика, прогон случайной и маленькой модели, нетривиальные и неслучайные скоры. Что могут спросить на ревью:

- вопросы синтетические (у MTEB такие задачи есть, в метаданных это указано через `LM-generated and verified`);
- смешанная лицензия (`multiple`): тексты законов не охраняются авторским правом, вопросы dev/test сгенерированы моделью со своей лицензией, код — MIT.

Как отправить (после форка, одна команда):

```bash
cd <клон mteb>                     # ветка add-ru-law-retrieval уже с коммитом из 0001-*.patch
gh repo fork embeddings-benchmark/mteb --remote --remote-name fork
git push fork add-ru-law-retrieval
gh pr create --repo embeddings-benchmark/mteb --head alekseevpavel04:add-ru-law-retrieval \
  --title "Add RuLawRetrieval (Russian legal article retrieval)" \
  --body-file ../ru-law-retrieval/mteb_task/PR_DESCRIPTION.md
```
