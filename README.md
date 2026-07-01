# AI Assistant Benchmarks (public)

Repository with RAG assistant benchmark results: raw data in `results/`, metric aggregation scripts with bootstrap 95% CIs, and a notebook for LLM-judge score charts.
Recording with explanations of the architecture and methodology for evaluating RAG quality and LLM-side generation:

**[Video](https://drive.google.com/file/d/11xzxuPyrSyIUKUr6uaqt9ztrllFWJ2k-/view?usp=drive_link)** 
(*The practical cap was about 30-40 000 tokens for context per request)

## Contents

| Directory / file | Purpose |
| --- | --- |
| `results/` | JSON and XLSX by model, mode, and bench (gold / noise) |
| `scripts/` | CLI to rebuild bootstrap artifacts and summaries |
| `notebooks/` | Charts from `judge_score_bootstrap_ci.json` |
| `charts/` | Exported figures |
| `prompts/` | Generation judge prompt |

## Scripts

From the repository root (requires [uv](https://github.com/astral-sh/uv) or an environment with dependencies from `pyproject.toml`):

```bash
uv run python scripts/build_judge_score_artifacts.py
uv run python scripts/build_retrieval_score_artifacts.py
uv run python scripts/build_latency_artifacts.py
```

Input fields and methodology are documented in each script’s module docstring.

