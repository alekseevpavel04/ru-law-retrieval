# Задача для MTEB (этап 8)

Статус: **задача готова и проверена локально, PR в MTEB не отправлен** (публикация в чужом репозитории делается только после «да» автора).

- `ru_law_retrieval.py`: класс задачи `RuLawRetrieval` для `mteb/tasks/retrieval/rus/`.
- `descriptive_stats.json`: описательная статистика из `task.calculate_descriptive_statistics()`. Дублей нет, минимальная длина документа 136 символов.
- `mteb_results.json`: штатный `mteb.evaluate`:
  - случайный энкодер: nDCG@10 0.0010;
  - e5-small: 0.7564;
  - дообученная модель: 0.8052.

  Те же числа, что у нашего кода в протоколе article (0.757 и 0.805): это перекрёстная проверка оценки.
- `0001-...patch`: коммит для форка MTEB (задача + регистрация в `__init__.py` + статистика). Тесты MTEB `test_metadata.py` и `test_get_tasks.py` проходят (2004 passed).
- `PR_DESCRIPTION.md`: текст PR по чек-листу MTEB.

Реально ли? Да, формальные требования выполнены: формат, метаданные, статистика, прогон случайной и маленькой модели, нетривиальные и неслучайные скоры. Риски на ревью:
- синтетические вопросы (у MTEB есть такие задачи, в метаданных это указано через `LM-generated and verified`);
- смешанная лицензия (`multiple`).

Как отправить (одна команда после форка):

```bash
cd D:/VScode_projects/mteb-src        # ветка add-ru-law-retrieval уже с коммитом
gh repo fork embeddings-benchmark/mteb --remote --remote-name fork
git push fork add-ru-law-retrieval
gh pr create --repo embeddings-benchmark/mteb --head alekseevpavel04:add-ru-law-retrieval \
  --title "Add RuLawRetrieval (Russian legal article retrieval)" --body-file ../ru-law-retrieval/mteb_task/PR_DESCRIPTION.md
```
