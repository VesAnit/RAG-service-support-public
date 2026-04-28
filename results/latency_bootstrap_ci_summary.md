# Latency: generation vs other (bootstrap 95% ДИ)

generation_latency_ms = latency_total_ms - latency_embed_ms
other_latency_ms = latency_total_ms - latency_llm_ms

## 1) Generation latency (по model + bench + mode)

| Модель | Бенч | Режим | Generation, ms | 95% ДИ |
| --- | --- | --- | ---: | --- |
 | gemma_31 | золотой | baseline | 34568.27 | [33227.12; 36023.08] | 
 | gemma_31 | золотой | full | 34550.86 | [33017.06; 36202.96] | 
 | gemma_31 | шумный | baseline | 35275.01 | [33987.57; 36684.15] | 
 | gemma_31 | шумный | full | 34751.81 | [32108.36; 38297.96] | 
 | nemotron | золотой | baseline | 27692.29 | [26030.60; 29511.78] | 
 | nemotron | золотой | full | 27618.08 | [26198.25; 29144.53] | 
 | nemotron | шумный | baseline | 31504.28 | [29731.26; 33437.08] | 
 | nemotron | шумный | full | 26356.99 | [24891.38; 27940.03] | 
 | qwen_35 | золотой | baseline | 26665.49 | [26056.20; 27293.24] | 
 | qwen_35 | золотой | full | 26893.17 | [26102.84; 27725.57] | 
 | qwen_35 | шумный | baseline | 27287.61 | [26737.55; 27876.63] | 
 | qwen_35 | шумный | full | 23692.73 | [23063.98; 24356.40] | 

## 2) Other latency (по bench + mode, усреднение по моделям per-question)

| Бенч | Режим | Other, ms | 95% ДИ |
| --- | --- | ---: | --- |
 | золотой | baseline | 1221.94 | [1189.24; 1257.22] | 
 | золотой | full | 1659.27 | [1544.89; 1793.87] | 
 | шумный | baseline | 1477.38 | [1437.69; 1522.07] | 
 | шумный | full | 2118.65 | [1507.74; 3205.78] | 

Источник: `results/latency_bootstrap_ci.json`
