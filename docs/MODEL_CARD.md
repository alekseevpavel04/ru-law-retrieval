---
language:
- ru
license: mit
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: intfloat/multilingual-e5-small
tags:
- sentence-transformers
- feature-extraction
- retrieval
- legal
- russian
datasets:
- alekseevpavel04/ru-law-retrieval
model-index:
- name: multilingual-e5-small-ru-law
  results:
  - task:
      type: retrieval
    dataset:
      name: RuLawRetrieval (test, chunk protocol)
      type: alekseevpavel04/ru-law-retrieval
      split: test
    metrics:
    - type: ndcg_at_10
      value: 0.8447
    - type: recall_at_10
      value: 0.948
---

# multilingual-e5-small-ru-law

Модель `intfloat/multilingual-e5-small` (118M параметров, 384 измерения), дообученная для поиска статей законов РФ по вопросам людей. Обучение заняло 3.9 минуты на одной RTX 3070 на 9 139 синтетических вопросах к статьям Трудового, Гражданского, Жилищного кодексов и КоАП.

*English summary: `multilingual-e5-small` fine-tuned in 3.9 minutes on 9.1k synthetic Russian legal questions. On the RuLawRetrieval test set nDCG@10 goes from 0.802 to 0.845 (+0.043, 95% CI [+0.028; +0.058]). This is statistically indistinguishable from `multilingual-e5-large` (0.859), with 6.7x lower CPU latency. The largest gain is on colloquial questions (0.569 → 0.687). It does not generalize to unseen codes and slightly forgets general-domain retrieval (RuBQ 0.686 → 0.651).*

Код, данные и все эксперименты: [github.com/alekseevpavel04/ru-law-retrieval](https://github.com/alekseevpavel04/ru-law-retrieval).

## Использование

Префиксы обязательны, как у исходной e5: `query: ` для вопроса и `passage: ` для документа.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("alekseevpavel04/multilingual-e5-small-ru-law")
q = model.encode(["query: меня уволили, пока я был на больничном, это законно?"], normalize_embeddings=True)
d = model.encode([
    "passage: Трудовой кодекс РФ, Статья 81. Расторжение трудового договора по инициативе работодателя\n...",
    "passage: Трудовой кодекс РФ, Статья 115. Продолжительность ежегодного основного оплачиваемого отпуска\n...",
], normalize_embeddings=True)
print(q @ d.T)
```

Лучше всего модель работает, когда в поиске участвуют фрагменты статей по ~600 символов с заголовком статьи в начале каждого фрагмента, а score статьи равен максимуму по её фрагментам. Именно так её обучали и оценивали. Модель сохранена в классическом формате sentence-transformers: sentence-transformers 3.3.1 и 6.1.0 дают на ней одинаковые эмбеддинги (проверено). Максимальная длина входа 512 токенов.

## Результаты

Бенчмарк [RuLawRetrieval](https://huggingface.co/datasets/alekseevpavel04/ru-law-retrieval), test (728 вопросов), протокол chunk. Поиск идёт по всем 3 786 статьям шести законов. В скобках 95% ДИ парного бутстрапа.

| Модель | Параметры | nDCG@10 | Recall@10 | CPU, мс/запрос |
|---|---:|---:|---:|---:|
| e5-small (исходная) | 118M | 0.802 | 0.905 | 21.8 |
| **эта модель** | 118M | **0.845** | 0.948 | 21.7 |
| e5-large | 560M | 0.859 | 0.959 | 145.2 |
| FRIDA (лучшая из 12) | 823M | 0.878 | 0.964 | 272.2 |

| Срез / тип | e5-small | эта модель | Δ [95% ДИ] |
|---|---:|---:|---|
| статьи из обучения (`seen`) | 0.785 | 0.835 | +0.050 [+0.026; +0.075] |
| отложенные статьи тех же кодексов | 0.789 | 0.845 | +0.055 [+0.028; +0.085] |
| новые кодексы (СК, ЗоЗПП) | 0.837 | 0.858 | +0.021 [−0.007; +0.049] |
| бытовые вопросы | 0.569 | 0.687 | +0.118 |
| поисковые запросы | 0.896 | 0.907 | +0.011 |
| вопросы юриста | 0.939 | 0.939 | 0.000 |

Прочие цифры:
- три сида: 0.848 ± 0.004;
- golden-подмножество с множественной релевантностью: 0.869 (у исходной 0.831);
- MTEB RuBQRetrieval (общий домен): 0.651 (у исходной 0.686).

Все числа лежат в `results/` репозитория.

## Обучение

- **Данные.**
  - 9 139 вопросов трёх типов (бытовой, поисковый, юридический), сгенерированных Qwen3-8B по статьям ТК, ГК, ЖК и КоАП.
  - 10% статей отложены без вопросов для проверки обобщения.
  - Фильтры: ссылки на номер статьи, копирование фраз из текста, дубли и почти-дубли.
  - Позитив: фрагмент своей статьи с максимальным score исходной модели.
  - Негатив: 1 hard negative из top-50 другой статьи.
- **Лосс.** `CachedMultipleNegativesRankingLoss`: батч 128, мини-батч 32, scale 20.
- **Сэмплер.** Свой: в батче нет двух строк про одну статью.
- **Гиперпараметры.** lr 3e-5, linear, warmup 10%, 2 эпохи, лучший чекпоинт по dev (шаг 72 из 144, то есть 1 эпоха). bf16, max_seq_length 208 при обучении, таблица эмбеддингов словаря заморожена (обучалось 21.6M параметров из 118M). Seed 42.
- **Железо и время.** RTX 3070 8 ГБ, 3.9 минуты, пик памяти 2.1 ГБ.

## Ограничения

- Обучена на синтетических вопросах. На реальных вопросах качество может отличаться.
- Одна редакция законов (сентябрь 2026). Для других законов прирост не гарантирован: на СК и ЗоЗПП он статистически незначим.
- Слегка хуже исходной модели на общем поиске (RuBQ −0.035). Для смешанного сервиса стоит сравнить обе.
- Модель ищет статьи, а не даёт юридические консультации.

## Лицензия

MIT, как у исходной модели. Обучающие вопросы сгенерированы Qwen3-8B (Apache-2.0).
