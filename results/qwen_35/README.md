# Результаты бенчей: Qwen 3.5

- **`results_gold_bench/`** — один gold-прогон (`benchmark_rag_results_*.json`, сводки).
- **`results_noise_bench/`** — три noise-прогона:
  - `first_results_noise_bench`
  - `second_results_noise_bench`
  - `third_results_noise_bench`

В `.env` для прогона: `BENCHMARK_RAG_RESULTS_DIR` и `BENCHMARK_RETRIEVAL_*` на нужную из трёх папок; для noise — ещё `BENCHMARK_RAG_JSON` = `benchmarks/data/noisy_bench/synthetic_vague_qa.json`. Excel review (`benchmark_qa_generation_review_*.xlsx`) создаётся в той же папке, что и JSON результатов. См. `benchmarks/benchmark_rag_modes.md`.
