"""Benchmark result aggregation scripts.

build_judge_score_artifacts.py
    Rebuilds judge_score_bootstrap_ci.json in results/ (LLM-judge score means
    + percentile bootstrap 95% CIs by model, mode, and metric).

build_retrieval_score_artifacts.py
    Rebuilds retrieval_bootstrap_ci.json and retrieval_bootstrap_ci_summary.md in results/
    (MRR and Hit@1 + bootstrap 95% CIs; pool all models per question).

build_latency_artifacts.py
    Rebuilds latency_bootstrap_ci.json and latency_bootstrap_ci_summary.md in results/
    (generation vs other latency + bootstrap 95% CIs by model/bench/mode).

The notebook notebooks/judge_scores_bar_charts.ipynb reads judge_score_bootstrap_ci.json.
"""
