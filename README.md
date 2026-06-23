# AI Assistant Benchmarks (public)

Repository with RAG assistant benchmark results: raw data in `results/`, metric aggregation scripts with bootstrap 95% CIs, and a notebook for LLM-judge score charts.

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

## Assistant demo

<details>
<summary><strong>Video: assistant demo</strong> (architecture, RAG and LLM generation evaluation methodology)</summary>

Recording with explanations of the architecture and methodology for evaluating RAG quality and LLM-side generation:

[Open on Google Drive](https://drive.google.com/file/d/16AmSomzB3d26k32He_tMljM02JFkYznW/view?usp=drive_link)

</details>
