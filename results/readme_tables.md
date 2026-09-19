## Corpus

| Кодекс | Статей | Действующих | Исключено (утр. силу / пустые) | Медиана, симв. | p95, симв. | Медиана, ток. | p95, ток. | > 512 ток. | Фрагментов |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tk | 540 | 534 | 6 / 0 | 1092 | 4785 | 220 | 918 | 97 | 1894 |
| gk | 1731 | 1717 | 12 / 2 | 862 | 3184 | 186 | 653 | 144 | 4537 |
| zhk | 243 | 241 | 2 / 0 | 1696 | 10537 | 335 | 2027 | 78 | 1564 |
| koap | 1124 | 1071 | 53 / 0 | 1345 | 6421 | 277 | 1252 | 273 | 5127 |
| sk | 175 | 171 | 4 / 0 | 833 | 2758 | 180 | 604 | 11 | 401 |
| zozpp | 55 | 52 | 3 / 0 | 1704 | 5953 | 342 | 1192 | 19 | 286 |

## Main (test, chunk)

| Модель | Параметры, M | nDCG@10 [95% ДИ] | Recall@10 | seen | unseen_articles | unseen_codes |
|---|---:|---:|---:|---:|---:|---:|
| FRIDA | 823 | 0.878 [0.859; 0.896] | 0.964 | 0.874 | 0.873 | 0.890 |
| e5-large-instruct | 560 | 0.865 [0.846; 0.883] | 0.968 | 0.866 | 0.860 | 0.868 |
| **ft-e5-small-v2** | 118 | 0.859 [0.840; 0.878] | 0.962 | 0.865 | 0.847 | 0.863 |
| e5-large | 560 | 0.859 [0.839; 0.877] | 0.959 | 0.867 | 0.845 | 0.861 |
| bge-m3 | 568 | 0.856 [0.836; 0.875] | 0.957 | 0.853 | 0.845 | 0.872 |
| **ft-e5-base** | 278 | 0.847 [0.826; 0.866] | 0.953 | 0.844 | 0.858 | 0.839 |
| USER-bge-m3 | 359 | 0.846 [0.825; 0.866] | 0.952 | 0.851 | 0.831 | 0.853 |
| **ft-e5-small** | 118 | 0.845 [0.824; 0.864] | 0.948 | 0.835 | 0.845 | 0.858 |
| Qwen3-Emb-0.6B | 596 | 0.843 [0.823; 0.863] | 0.953 | 0.834 | 0.840 | 0.859 |
| ru-en-RoSBERTa | 405 | 0.840 [0.819; 0.859] | 0.956 | 0.844 | 0.837 | 0.837 |
| USER2-base | 149 | 0.825 [0.804; 0.846] | 0.945 | 0.825 | 0.815 | 0.837 |
| e5-base | 278 | 0.822 [0.800; 0.843] | 0.934 | 0.810 | 0.799 | 0.860 |
| e5-small | 118 | 0.802 [0.778; 0.826] | 0.905 | 0.785 | 0.789 | 0.837 |
| USER-base | 124 | 0.684 [0.656; 0.710] | 0.845 | 0.674 | 0.671 | 0.710 |
| BM25 | 0 | 0.670 [0.640; 0.700] | 0.776 | 0.679 | 0.640 | 0.689 |
| rubert-tiny2 | 29 | 0.405 [0.376; 0.434] | 0.588 | 0.395 | 0.343 | 0.481 |

## Protocols

| Модель | article | chunk |
|---|---:|---:|
| FRIDA | 0.861 | 0.878 |
| e5-large-instruct | 0.837 | 0.865 |
| ft-e5-small-v2 | 0.823 | 0.859 |
| e5-large | 0.828 | 0.859 |
| bge-m3 | 0.847 | 0.856 |
| ft-e5-base | 0.805 | 0.847 |
| USER-bge-m3 | 0.837 | 0.846 |
| ft-e5-small | 0.805 | 0.845 |
| Qwen3-Emb-0.6B | 0.829 | 0.843 |
| ru-en-RoSBERTa | 0.813 | 0.840 |
| USER2-base | 0.816 | 0.825 |
| e5-base | 0.776 | 0.822 |
| e5-small | 0.757 | 0.802 |
| USER-base | 0.688 | 0.684 |
| BM25 | 0.673 | 0.670 |
| rubert-tiny2 | 0.384 | 0.405 |

## Golden

| Модель | nDCG@10 [95% ДИ] | Recall@10 |
|---|---:|---:|
| FRIDA | 0.916 [0.886; 0.945] | 0.975 |
| e5-large | 0.907 [0.873; 0.939] | 0.966 |
| e5-large-instruct | 0.904 [0.869; 0.936] | 0.967 |
| bge-m3 | 0.899 [0.866; 0.929] | 0.971 |
| USER-bge-m3 | 0.895 [0.860; 0.928] | 0.964 |
| ft-e5-small-v2 | 0.894 [0.858; 0.927] | 0.968 |
| ru-en-RoSBERTa | 0.882 [0.849; 0.914] | 0.984 |
| ft-e5-base | 0.879 [0.844; 0.913] | 0.974 |
| e5-base | 0.872 [0.833; 0.908] | 0.951 |
| ft-e5-small | 0.869 [0.831; 0.904] | 0.957 |
| USER2-base | 0.869 [0.829; 0.905] | 0.946 |
| Qwen3-Emb-0.6B | 0.869 [0.831; 0.904] | 0.952 |
| e5-small | 0.831 [0.785; 0.874] | 0.916 |
| USER-base | 0.725 [0.673; 0.773] | 0.878 |
| BM25 | 0.712 [0.652; 0.768] | 0.785 |
| rubert-tiny2 | 0.406 [0.351; 0.462] | 0.603 |

## Training runs

| Прогон | Модель | Данные | Hard neg | Seed | Пар | Лучшая эпоха | dev nDCG@10 (база → лучший) | Время, мин | Пик памяти, ГБ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| e1_small_both_hn | multilingual-e5-small | llm+titles x1 | да | 42 | 12298 | 1.98 | 0.816 → 0.850 | 5.6 | 2.11 |
| e1_small_both_inb | multilingual-e5-small | llm+titles x1 | нет | 42 | 12298 | 0.74 | 0.816 → 0.844 | 3.6 | 2.09 |
| e1_small_llm_hn | multilingual-e5-small | llm x1 | да | 42 | 9139 | 1.00 | 0.816 → 0.850 | 3.9 | 2.09 |
| e1_small_llm_hn_s43 | multilingual-e5-small | llm x1 | да | 43 | 9139 | 1.75 | 0.816 → 0.845 | 3.9 | 2.09 |
| e1_small_llm_hn_s44 | multilingual-e5-small | llm x1 | да | 44 | 9139 | 1.00 | 0.816 → 0.845 | 3.9 | 2.09 |
| e1_small_llm_inb | multilingual-e5-small | llm x1 | нет | 42 | 9139 | 2.00 | 0.816 → 0.843 | 3.0 | 2.08 |
| e1_small_titles_hn | multilingual-e5-small | titles x1 | да | 42 | 3159 | 0.24 | 0.816 → 0.826 | 2.3 | 2.08 |
| e1_small_titles_inb | multilingual-e5-small | titles x1 | нет | 42 | 3159 | 0.24 | 0.816 → 0.824 | 2.1 | 2.00 |
| e2_base_both_hn | multilingual-e5-base | llm+titles x1 | да | 42 | 12298 | 0.74 | 0.828 → 0.846 | 10.3 | 3.55 |
| e2_base_llm_hn | multilingual-e5-base | llm x1 | да | 42 | 9139 | 1.00 | 0.828 → 0.850 | 8.2 | 3.54 |
| e2_base_llm_hn_s43 | multilingual-e5-base | llm x1 | да | 43 | 9139 | 2.00 | 0.828 → 0.849 | 8.2 | 3.54 |
| e2_base_llm_hn_s44 | multilingual-e5-base | llm x1 | да | 44 | 9139 | 1.00 | 0.828 → 0.854 | 8.2 | 3.54 |
| e4_small_llm_hn_f25 | multilingual-e5-small | llm x0.25 | да | 42 | 2285 | 0.22 | 0.816 → 0.833 | 2.2 | 2.08 |
| e4_small_llm_hn_f50 | multilingual-e5-small | llm x0.5 | да | 42 | 4570 | 1.50 | 0.816 → 0.838 | 2.7 | 2.08 |

## Bootstrap (v1)

| A | B | набор | срез | n | Δ nDCG@10 | 95% ДИ | p |
|---|---|---|---|---:|---:|---:|---:|
| ft-e5-small | e5-small | test/chunk | all | 728 | +0.043 | [+0.028; +0.058] | 0.000 |
| ft-e5-small | e5-small | test/chunk | slice=seen | 293 | +0.050 | [+0.026; +0.075] | 0.000 |
| ft-e5-small | e5-small | test/chunk | slice=unseen_articles | 218 | +0.055 | [+0.028; +0.085] | 0.000 |
| ft-e5-small | e5-small | test/chunk | slice=unseen_codes | 217 | +0.021 | [-0.007; +0.049] | 0.151 |
| ft-e5-small | FRIDA | test/chunk | all | 728 | -0.034 | [-0.052; -0.015] | 0.000 |
| ft-e5-small | FRIDA | test/chunk | slice=seen | 293 | -0.039 | [-0.068; -0.010] | 0.009 |
| ft-e5-small | FRIDA | test/chunk | slice=unseen_articles | 218 | -0.029 | [-0.065; +0.006] | 0.113 |
| ft-e5-small | FRIDA | test/chunk | slice=unseen_codes | 217 | -0.032 | [-0.065; +0.002] | 0.064 |
| ft-e5-small | e5-large | test/chunk | all | 728 | -0.014 | [-0.031; +0.003] | 0.114 |
| ft-e5-small | e5-large | test/chunk | slice=seen | 293 | -0.032 | [-0.061; -0.003] | 0.029 |
| ft-e5-small | e5-large | test/chunk | slice=unseen_articles | 218 | -0.001 | [-0.033; +0.030] | 0.965 |
| ft-e5-small | e5-large | test/chunk | slice=unseen_codes | 217 | -0.003 | [-0.032; +0.026] | 0.828 |
| ft-e5-base | e5-base | test/chunk | all | 728 | +0.025 | [+0.008; +0.041] | 0.002 |
| ft-e5-base | e5-base | test/chunk | slice=seen | 293 | +0.033 | [+0.008; +0.059] | 0.011 |
| ft-e5-base | e5-base | test/chunk | slice=unseen_articles | 218 | +0.059 | [+0.027; +0.092] | 0.000 |
| ft-e5-base | e5-base | test/chunk | slice=unseen_codes | 217 | -0.021 | [-0.049; +0.007] | 0.137 |
| ft-e5-base | FRIDA | test/chunk | all | 728 | -0.032 | [-0.050; -0.013] | 0.000 |
| ft-e5-base | FRIDA | test/chunk | slice=seen | 293 | -0.030 | [-0.058; -0.003] | 0.029 |
| ft-e5-base | FRIDA | test/chunk | slice=unseen_articles | 218 | -0.015 | [-0.049; +0.019] | 0.373 |
| ft-e5-base | FRIDA | test/chunk | slice=unseen_codes | 217 | -0.051 | [-0.087; -0.015] | 0.006 |
| ft-e5-small | e5-small | test/article | all | 728 | +0.048 | [+0.031; +0.065] | 0.000 |
| ft-e5-small | e5-small | test/article | slice=seen | 293 | +0.053 | [+0.025; +0.083] | 0.000 |
| ft-e5-small | e5-small | test/article | slice=unseen_articles | 218 | +0.062 | [+0.029; +0.095] | 0.000 |
| ft-e5-small | e5-small | test/article | slice=unseen_codes | 217 | +0.026 | [-0.002; +0.054] | 0.065 |
| ft-e5-small | FRIDA | test/article | all | 728 | -0.056 | [-0.076; -0.036] | 0.000 |
| ft-e5-small | FRIDA | test/article | slice=seen | 293 | -0.068 | [-0.099; -0.037] | 0.000 |
| ft-e5-small | FRIDA | test/article | slice=unseen_articles | 218 | -0.049 | [-0.085; -0.014] | 0.006 |
| ft-e5-small | FRIDA | test/article | slice=unseen_codes | 217 | -0.045 | [-0.082; -0.009] | 0.015 |
| ft-e5-small | e5-small | golden/chunk | all | 174 | +0.038 | [+0.009; +0.069] | 0.011 |
| ft-e5-small | e5-small | golden/chunk | slice=seen | 58 | +0.059 | [+0.006; +0.118] | 0.031 |
| ft-e5-small | e5-small | golden/chunk | slice=unseen_articles | 58 | +0.037 | [-0.010; +0.089] | 0.131 |
| ft-e5-small | e5-small | golden/chunk | slice=unseen_codes | 58 | +0.019 | [-0.029; +0.076] | 0.471 |
| ft-e5-small | FRIDA | golden/chunk | all | 174 | -0.048 | [-0.078; -0.017] | 0.002 |
| ft-e5-small | FRIDA | golden/chunk | slice=seen | 58 | -0.059 | [-0.109; -0.013] | 0.013 |
| ft-e5-small | FRIDA | golden/chunk | slice=unseen_articles | 58 | -0.038 | [-0.099; +0.022] | 0.200 |
| ft-e5-small | FRIDA | golden/chunk | slice=unseen_codes | 58 | -0.045 | [-0.094; +0.003] | 0.067 |
| RRF(BM25+ft-e5-small) | ft-e5-small | test/chunk | all | 728 | -0.073 | [-0.092; -0.054] | 0.000 |
| RRF(BM25+ft-e5-small) | ft-e5-small | test/chunk | slice=seen | 293 | -0.050 | [-0.078; -0.024] | 0.000 |
| RRF(BM25+ft-e5-small) | ft-e5-small | test/chunk | slice=unseen_articles | 218 | -0.093 | [-0.132; -0.054] | 0.000 |
| RRF(BM25+ft-e5-small) | ft-e5-small | test/chunk | slice=unseen_codes | 217 | -0.082 | [-0.119; -0.046] | 0.000 |
| RRF(BM25+FRIDA) | FRIDA | test/chunk | all | 728 | -0.087 | [-0.107; -0.067] | 0.000 |
| RRF(BM25+FRIDA) | FRIDA | test/chunk | slice=seen | 293 | -0.066 | [-0.096; -0.038] | 0.000 |
| RRF(BM25+FRIDA) | FRIDA | test/chunk | slice=unseen_articles | 218 | -0.115 | [-0.157; -0.075] | 0.000 |
| RRF(BM25+FRIDA) | FRIDA | test/chunk | slice=unseen_codes | 217 | -0.085 | [-0.121; -0.050] | 0.000 |
| RRF(BM25+ft-e5-small) | FRIDA | test/chunk | all | 728 | -0.106 | [-0.129; -0.084] | 0.000 |
| RRF(BM25+ft-e5-small) | FRIDA | test/chunk | slice=seen | 293 | -0.089 | [-0.123; -0.057] | 0.000 |
| RRF(BM25+ft-e5-small) | FRIDA | test/chunk | slice=unseen_articles | 218 | -0.122 | [-0.170; -0.075] | 0.000 |
| RRF(BM25+ft-e5-small) | FRIDA | test/chunk | slice=unseen_codes | 217 | -0.114 | [-0.155; -0.072] | 0.000 |

## Bootstrap (published v2)

| A | B | набор | срез | n | Δ nDCG@10 | 95% ДИ | p |
|---|---|---|---|---:|---:|---:|---:|
| ft-e5-small-v2 | e5-small | test/chunk | all | 728 | +0.057 | [+0.042; +0.073] | 0.000 |
| ft-e5-small-v2 | e5-small | test/chunk | slice=seen | 293 | +0.080 | [+0.056; +0.106] | 0.000 |
| ft-e5-small-v2 | e5-small | test/chunk | slice=unseen_articles | 218 | +0.058 | [+0.031; +0.086] | 0.000 |
| ft-e5-small-v2 | e5-small | test/chunk | slice=unseen_codes | 217 | +0.026 | [-0.000; +0.052] | 0.052 |
| ft-e5-small-v2 | ft-e5-small | test/chunk | all | 728 | +0.015 | [+0.003; +0.026] | 0.012 |
| ft-e5-small-v2 | ft-e5-small | test/chunk | slice=seen | 293 | +0.031 | [+0.013; +0.049] | 0.000 |
| ft-e5-small-v2 | ft-e5-small | test/chunk | slice=unseen_articles | 218 | +0.003 | [-0.017; +0.023] | 0.772 |
| ft-e5-small-v2 | ft-e5-small | test/chunk | slice=unseen_codes | 217 | +0.005 | [-0.015; +0.025] | 0.623 |
| ft-e5-small-v2 | e5-large | test/chunk | all | 728 | +0.001 | [-0.016; +0.017] | 0.961 |
| ft-e5-small-v2 | e5-large | test/chunk | slice=seen | 293 | -0.001 | [-0.029; +0.026] | 0.915 |
| ft-e5-small-v2 | e5-large | test/chunk | slice=unseen_articles | 218 | +0.002 | [-0.029; +0.033] | 0.885 |
| ft-e5-small-v2 | e5-large | test/chunk | slice=unseen_codes | 217 | +0.002 | [-0.027; +0.030] | 0.891 |
| ft-e5-small-v2 | FRIDA | test/chunk | all | 728 | -0.019 | [-0.036; -0.002] | 0.031 |
| ft-e5-small-v2 | FRIDA | test/chunk | slice=seen | 293 | -0.008 | [-0.035; +0.019] | 0.564 |
| ft-e5-small-v2 | FRIDA | test/chunk | slice=unseen_articles | 218 | -0.026 | [-0.061; +0.009] | 0.139 |
| ft-e5-small-v2 | FRIDA | test/chunk | slice=unseen_codes | 217 | -0.027 | [-0.057; +0.004] | 0.090 |
| ft-e5-small-v2 | e5-small | test/chunk_tkfmt | all | 728 | +0.054 | [+0.040; +0.069] | 0.000 |
| ft-e5-small-v2 | e5-small | test/chunk_tkfmt | slice=seen | 293 | +0.077 | [+0.054; +0.102] | 0.000 |
| ft-e5-small-v2 | e5-small | test/chunk_tkfmt | slice=unseen_articles | 218 | +0.049 | [+0.022; +0.077] | 0.001 |
| ft-e5-small-v2 | e5-small | test/chunk_tkfmt | slice=unseen_codes | 217 | +0.029 | [+0.004; +0.054] | 0.025 |
| ft-e5-small-v2 | e5-small | tk_hard/chunk | all | 716 | +0.109 | [+0.091; +0.128] | 0.000 |
| ft-e5-small-v2 | e5-small | tk_hard/chunk | slice=seen | 644 | +0.114 | [+0.095; +0.134] | 0.000 |
| ft-e5-small-v2 | e5-small | tk_hard/chunk | slice=unseen_articles | 72 | +0.064 | [+0.014; +0.115] | 0.012 |
| ft-e5-small-v2 | ft-e5-small | tk_hard/chunk | all | 716 | +0.025 | [+0.012; +0.039] | 0.000 |
| ft-e5-small-v2 | ft-e5-small | tk_hard/chunk | slice=seen | 644 | +0.025 | [+0.011; +0.039] | 0.000 |
| ft-e5-small-v2 | ft-e5-small | tk_hard/chunk | slice=unseen_articles | 72 | +0.023 | [-0.019; +0.067] | 0.297 |
| ft-e5-small-v2 | e5-large | tk_hard/chunk | all | 716 | -0.000 | [-0.020; +0.018] | 0.957 |
| ft-e5-small-v2 | e5-large | tk_hard/chunk | slice=seen | 644 | -0.005 | [-0.025; +0.014] | 0.627 |
| ft-e5-small-v2 | e5-large | tk_hard/chunk | slice=unseen_articles | 72 | +0.040 | [-0.018; +0.099] | 0.180 |
| ft-e5-small-v2 | FRIDA | tk_hard/chunk | all | 716 | -0.034 | [-0.055; -0.015] | 0.000 |
| ft-e5-small-v2 | FRIDA | tk_hard/chunk | slice=seen | 644 | -0.037 | [-0.059; -0.016] | 0.000 |
| ft-e5-small-v2 | FRIDA | tk_hard/chunk | slice=unseen_articles | 72 | -0.007 | [-0.056; +0.040] | 0.792 |
| ft-e5-small-v2 | e5-small | tk_hard/chunk_tkfmt | all | 716 | +0.104 | [+0.086; +0.123] | 0.000 |
| ft-e5-small-v2 | e5-small | tk_hard/chunk_tkfmt | slice=seen | 644 | +0.108 | [+0.089; +0.128] | 0.000 |
| ft-e5-small-v2 | e5-small | tk_hard/chunk_tkfmt | slice=unseen_articles | 72 | +0.071 | [+0.014; +0.130] | 0.013 |
| ft-e5-small | e5-small | tk_hard/chunk_tkfmt | all | 716 | +0.073 | [+0.054; +0.093] | 0.000 |
| ft-e5-small | e5-small | tk_hard/chunk_tkfmt | slice=seen | 644 | +0.078 | [+0.057; +0.099] | 0.000 |
| ft-e5-small | e5-small | tk_hard/chunk_tkfmt | slice=unseen_articles | 72 | +0.031 | [-0.026; +0.089] | 0.279 |
| ft-e5-small-v2 | e5-small | golden/chunk | all | 174 | +0.063 | [+0.033; +0.094] | 0.000 |
| ft-e5-small-v2 | e5-small | golden/chunk | slice=seen | 58 | +0.104 | [+0.049; +0.164] | 0.000 |
| ft-e5-small-v2 | e5-small | golden/chunk | slice=unseen_articles | 58 | +0.041 | [-0.004; +0.090] | 0.074 |
| ft-e5-small-v2 | e5-small | golden/chunk | slice=unseen_codes | 58 | +0.044 | [-0.002; +0.099] | 0.061 |
| ft-e5-small-v2 | ft-e5-small | golden/chunk | all | 174 | +0.025 | [+0.003; +0.047] | 0.026 |
| ft-e5-small-v2 | ft-e5-small | golden/chunk | slice=seen | 58 | +0.045 | [+0.009; +0.087] | 0.013 |
| ft-e5-small-v2 | ft-e5-small | golden/chunk | slice=unseen_articles | 58 | +0.005 | [-0.038; +0.045] | 0.804 |
| ft-e5-small-v2 | ft-e5-small | golden/chunk | slice=unseen_codes | 58 | +0.025 | [-0.002; +0.055] | 0.070 |
| ft-e5-small-v2 | e5-small | test/article | all | 728 | +0.065 | [+0.049; +0.082] | 0.000 |
| ft-e5-small-v2 | e5-small | test/article | slice=seen | 293 | +0.088 | [+0.062; +0.115] | 0.000 |
| ft-e5-small-v2 | e5-small | test/article | slice=unseen_articles | 218 | +0.075 | [+0.044; +0.106] | 0.000 |
| ft-e5-small-v2 | e5-small | test/article | slice=unseen_codes | 217 | +0.026 | [-0.002; +0.055] | 0.069 |

## TK-hard

| Модель | nDCG@10 [95% ДИ] | Recall@10 | seen | unseen_articles |
|---|---:|---:|---:|---:|
| e5-large-instruct | 0.858 [0.838; 0.877] | 0.950 | 0.857 | 0.863 |
| FRIDA | 0.854 [0.834; 0.874] | 0.960 | 0.854 | 0.857 |
| USER-bge-m3 | 0.840 [0.819; 0.860] | 0.953 | 0.841 | 0.827 |
| bge-m3 | 0.839 [0.819; 0.859] | 0.953 | 0.843 | 0.801 |
| **ft-e5-base** | 0.825 [0.803; 0.845] | 0.954 | 0.826 | 0.808 |
| ru-en-RoSBERTa | 0.821 [0.800; 0.842] | 0.953 | 0.823 | 0.801 |
| e5-large | 0.820 [0.797; 0.842] | 0.930 | 0.821 | 0.810 |
| **ft-e5-small-v2** | 0.820 [0.798; 0.841] | 0.937 | 0.816 | 0.850 |
| **ft-e5-small** | 0.795 [0.772; 0.817] | 0.937 | 0.791 | 0.827 |
| USER2-base | 0.792 [0.768; 0.815] | 0.922 | 0.790 | 0.803 |
| Qwen3-Emb-0.6B | 0.791 [0.767; 0.813] | 0.929 | 0.789 | 0.808 |
| e5-base | 0.760 [0.735; 0.785] | 0.899 | 0.758 | 0.778 |
| e5-small | 0.710 [0.683; 0.737] | 0.865 | 0.702 | 0.786 |
| USER-base | 0.622 [0.594; 0.649] | 0.813 | 0.627 | 0.575 |
| BM25 | 0.480 [0.448; 0.512] | 0.615 | 0.475 | 0.524 |
| rubert-tiny2 | 0.346 [0.318; 0.374] | 0.518 | 0.346 | 0.345 |

## v2 stages

| Этап | Прогон | dev nDCG@10 | dev (наш формат) | dev (формат tk-rf-rag) | Решение |
|---|---|---:|---:|---:|---|
| A: данные и учитель | `v2a_small_teacher_v1data` | 0.8431 | 0.8435 | 0.8426 |  |
| A: данные и учитель | `v2a_small_teacher_v12` | 0.8466 | 0.8503 | 0.8429 |  |
| A: данные и учитель | `v2a_small_teacher_v12_f50` | 0.8483 | 0.8523 | 0.8443 | выбран |
| B: аугментация формата фрагментов | `v2b_small_teacher_v12_f50_aug50` | 0.8463 | 0.8509 | 0.8417 | не принят |
| C: дистилляция FRIDA (2 этап) | `v2c_distill_t05` | 0.8483 | 0.8523 | 0.8443 |  |
| C: дистилляция FRIDA (2 этап) | `v2c_distill_t02` | 0.8506 | 0.8540 | 0.8472 | выбран |
| D: сиды рецепта A+C | `v2c_distill_t02_from_s43` | 0.8560 | 0.8558 | 0.8562 |  |
| D: сиды рецепта A+C | `v2c_distill_t02_from_s44` | 0.8564 | 0.8581 | 0.8548 |  |
| E: длительность обучения, батч, lr | `v2e_ep3` | 0.8523 | 0.8580 | 0.8465 | выбран |
| E: длительность обучения, батч, lr | `v2e_bs256` | 0.8437 | 0.8462 | 0.8413 |  |
| E: длительность обучения, батч, lr | `v2e_lr5e5` | 0.8522 | 0.8560 | 0.8484 |  |
| E: длительность обучения, батч, lr | `v2e_ep3_s43` | 0.8510 | 0.8571 | 0.8448 |  |
| E: длительность обучения, батч, lr | `v2e_ep3_s44` | 0.8497 | 0.8524 | 0.8471 |  |
| F: дистилляция поверх E | `v2f_distill_from_v2e_ep3` | 0.8551 | 0.8575 | 0.8527 | выбран |
| F: дистилляция поверх E | `v2f_distill_from_v2e_ep3_s43` | 0.8564 | 0.8559 | 0.8568 |  |
| F: дистилляция поверх E | `v2f_distill_from_v2e_ep3_s44` | 0.8520 | 0.8540 | 0.8500 |  |

