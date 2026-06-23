# Qwen 3.5 benchmark results

- **`results_gold_bench/`** — single gold run (`benchmark_rag_results_*.json`, summaries).
- **`results_noise_bench/`** — three noise runs:
  - `first_results_noise_bench`
  - `second_results_noise_bench`
  - `third_results_noise_bench`

For a run, set `.env`: `BENCHMARK_RAG_RESULTS_DIR` and `BENCHMARK_RETRIEVAL_*` to the target subfolder; for noise also set `BENCHMARK_RAG_JSON` = `benchmarks/data/noisy_bench/synthetic_vague_qa.json`. Excel review files (`benchmark_qa_generation_review_*.xlsx`) are created in the same folder as the result JSON. See `benchmarks/benchmark_rag_modes.md`.
