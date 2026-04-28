"""Скрипты агрегации метрик по результатам бенчмарка.

build_judge_score_artifacts.py
    Пересобирает judge_score_bootstrap_ci.json в results/ (средние оценок LLM-судьи
    + percentile bootstrap 95% ДИ по моделям, режимам и метрикам).

build_retrieval_score_artifacts.py
    Пересобирает retrieval_bootstrap_ci.json и retrieval_bootstrap_ci_summary.md в results/
    (MRR и Hit@1 + bootstrap 95% ДИ; пул всех моделей на вопрос).

build_latency_artifacts.py
    Пересобирает latency_bootstrap_ci.json и latency_bootstrap_ci_summary.md в results/
    (latency на генерацию и «всё остальное» + bootstrap 95% ДИ по model/bench/mode).

Ноутбук notebooks/judge_scores_bar_charts.ipynb читает judge_score_bootstrap_ci.json.
"""
