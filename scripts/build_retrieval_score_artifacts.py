"""Build retrieval_bootstrap_ci.json and retrieval_bootstrap_ci_summary.md for MRR and Hit@1.

Inputs
------
benchmark_rag_results_<mode>.json under results/, one per (model, mode, [run]).
Each record must include:
  bench_index            — integer question index (shared across models and runs);
  chunk_id               — str or list[str] of gold point_id; missing field = no gold,
                           record excluded;
  retrieval_rr_initial   — float: reciprocal rank of the first matching chunk in initial context;
  retrieval_rank_initial — int | None: 1-based rank.

Methodology
-----------
Retrieval does not depend on the LLM model (same RAG pipeline): for each question
(bench_index) all models yield the same retrieval result. Before bootstrap we pool
observations by bench_index — across all models and (for noise) all runs — and average
RR and Hit@1 per question. Bootstrap unit = one question after averaging.

Gold: 3 models × 1 run = 3 rows per question → mean RR and Hit@1 per question.
Noise: 3 models × 3 runs = 9 rows per question → mean RR and Hit@1 per question.
Bootstrap: percentile 95% CI over averaged questions (n ≈ 453 gold, ≈ 310 noise).

MRR: reciprocal rank of the first matching chunk in initial context — typically RR = 1/r
when the gold rank is r; if gold is not found or JSON has no numeric RR, use 0 (as in
retrieval_rr_initial). After pooling by bench_index, RR values are averaged per question;
reported MRR is the mean over questions of those per-question averages.

Hit@1: 1 if gold rank ≤ 1; else 0. If rank is missing (gold not found) — 0.

Run: uv run python scripts/build_retrieval_score_artifacts.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

import numpy as np

logger = logging.getLogger(__name__)

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

RESULTS_DIR: Final[Path] = REPOSITORY_ROOT / "results"
MODELS: Final[tuple[str, str, str]] = ("gemma_4", "nemotron", "qwen_35")
RAG_MODES: Final[tuple[str, str]] = ("baseline", "full")
NOISE_RUN_DIRS: Final[tuple[str, str, str]] = (
    "first_results_noise_bench",
    "second_results_noise_bench",
    "third_results_noise_bench",
)

BOOTSTRAP_JSON: Final[str] = "retrieval_bootstrap_ci.json"
SUMMARY_MD: Final[str] = "retrieval_bootstrap_ci_summary.md"


class RetrievalBootstrapBuilder:
    """Reads benchmark_rag_results JSON, pools by bench_index, and computes bootstrap CIs.

    Public entry point — build(): writes JSON and MD, returns their paths.
    """

    def __init__(
        self,
        results_dir: Path,
        *,
        num_bootstrap_replicates: int = 10_000,
        alpha: float = 0.05,
        random_seed: int = 42,
    ) -> None:
        self._results: Path = results_dir.resolve()
        self._num_bootstrap: int = num_bootstrap_replicates
        self._alpha: float = alpha
        self._random_seed: int = random_seed

    def build(self) -> tuple[Path, Path]:
        """Rebuild JSON and MD; return their paths."""
        records = self._build_records()
        json_path = self._write_json(records)
        md_path = self._write_md(records)
        return json_path, md_path

    def _build_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for bench in ("gold", "noise"):
            for mode in RAG_MODES:
                result_file_paths = self._gold_paths(mode) if bench == "gold" else self._noise_paths(mode)
                mrr_per_question, hit_at_1_per_question, total_pooled_rows = self._per_question_averages(result_file_paths)

                if not mrr_per_question:
                    logger.warning("No data for %s / %s", bench, mode)

                mrr_stats = self._bootstrap_ci(mrr_per_question)
                hit_at_1_stats = self._bootstrap_ci(hit_at_1_per_question)
                records.append(
                    {
                        "bench": bench,
                        "mode": mode,
                        "n_questions": mrr_stats["n"],
                        "n_pooled_rows": total_pooled_rows,
                        "mrr_mean": mrr_stats["mean"],
                        "mrr_ci_low": mrr_stats["ci_low"],
                        "mrr_ci_high": mrr_stats["ci_high"],
                        "hit_at_1_mean": hit_at_1_stats["mean"],
                        "hit_at_1_ci_low": hit_at_1_stats["ci_low"],
                        "hit_at_1_ci_high": hit_at_1_stats["ci_high"],
                        "n_bootstrap": self._num_bootstrap,
                        "alpha": self._alpha,
                    }
                )
        return records

    def _per_question_averages(
        self, result_file_paths: list[Path]
    ) -> tuple[list[float], list[float], int]:
        """Average RR and Hit@1 by bench_index; return (mrr_vec, hit_vec, n_pooled_rows)."""
        by_bench_index: dict[int, list[tuple[float, float]]] = defaultdict(list)
        total_pooled_rows = 0
        for result_path in result_file_paths:
            for record in self._load_records_with_gold(result_path):
                question_index = self._bench_index(record)
                if question_index is None:
                    continue
                by_bench_index[question_index].append((self._reciprocal_rank(record), self._hit_at_1(record)))
                total_pooled_rows += 1

        mrr_per_question: list[float] = []
        hit_at_1_per_question: list[float] = []
        for question_index in sorted(by_bench_index.keys()):
            observations = by_bench_index[question_index]
            mrr_per_question.append(sum(observation[0] for observation in observations) / len(observations))
            hit_at_1_per_question.append(sum(observation[1] for observation in observations) / len(observations))
        return mrr_per_question, hit_at_1_per_question, total_pooled_rows

    def _gold_paths(self, mode: str) -> list[Path]:
        """Paths to one mode's JSON for all three models (gold bench)."""
        return [
            self._results / model / "results_gold_bench" / f"benchmark_rag_results_{mode}.json"
            for model in MODELS
        ]

    def _noise_paths(self, mode: str) -> list[Path]:
        """Paths to all 9 files (3 models × 3 runs) for noise bench."""
        return [
            self._results / model / "results_noise_bench" / run_dir / f"benchmark_rag_results_{mode}.json"
            for model in MODELS
            for run_dir in NOISE_RUN_DIRS
        ]

    @staticmethod
    def _load_records_with_gold(path: Path) -> list[dict[str, Any]]:
        """Load JSON and return only records with non-empty chunk_id (has gold)."""
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping %s: %s", path, exc)
            return []
        if not isinstance(data, list):
            return []
        return [record for record in data if isinstance(record, dict) and record.get("chunk_id")]

    @staticmethod
    def _bench_index(record: dict[str, Any]) -> int | None:
        """Normalize bench_index to int > 0 or return None."""
        raw = record.get("bench_index")
        if raw is None or isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return raw if raw > 0 else None
        if isinstance(raw, float):
            if not raw.is_integer():
                return None
            integer_value = int(raw)
            return integer_value if integer_value > 0 else None
        try:
            integer_value = int(raw)
            return integer_value if integer_value > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _reciprocal_rank(record: dict[str, Any]) -> float:
        """Reciprocal rank in initial context; 0 if no gold."""
        raw_value = record.get("retrieval_rr_initial")
        return float(raw_value) if isinstance(raw_value, (int, float)) else 0.0

    @staticmethod
    def _hit_at_1(record: dict[str, Any]) -> float:
        """1.0 if gold at rank 1, else 0.0."""
        rank = record.get("retrieval_rank_initial")
        if rank is None:
            return 0.0
        return 1.0 if int(rank) <= 1 else 0.0

    def _bootstrap_ci(self, values: list[float]) -> dict[str, float | int]:
        """Percentile bootstrap: mean + 95% CI. Empty list → nan."""
        n_questions = len(values)
        if n_questions == 0:
            return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}

        values_array = np.asarray(values, dtype=np.float64)
        sample_mean = float(values_array.mean())
        random_generator = np.random.default_rng(self._random_seed)
        bootstrap_sample_means = np.empty(self._num_bootstrap, dtype=np.float64)
        for sample_index in range(self._num_bootstrap):
            bootstrap_sample_means[sample_index] = float(
                random_generator.choice(values_array, size=n_questions, replace=True).mean()
            )

        lower_percentile = 100.0 * self._alpha / 2.0
        upper_percentile = 100.0 * (1.0 - self._alpha / 2.0)
        ci = np.percentile(bootstrap_sample_means, [lower_percentile, upper_percentile])
        return {
            "mean": sample_mean,
            "ci_low": float(ci.flat[0]),
            "ci_high": float(ci.flat[1]),
            "n": n_questions,
        }

    def _write_json(self, records: list[dict[str, Any]]) -> Path:
        path = (self._results / BOOTSTRAP_JSON).resolve()
        _ = path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Retrieval bootstrap CI: %s records → %s", len(records), path)
        return path

    def _write_md(self, records: list[dict[str, Any]]) -> Path:
        """Short MD table for the repository."""

        def fmt(value: Any, digits: int = 4) -> str:
            if value is None or (isinstance(value, float) and (value != value)):  # nan check
                return "—"
            return f"{float(value):.{digits}f}"

        lines = [
            "# Retrieval: MRR and Hit@1 (bootstrap 95% CI)",
            "",
            "Pool all models per question (bench_index). "
            "Gold: 3 models × 1 run. Noise: 3 models × 3 runs. "
            "Per question — mean RR and Hit@1 over the pool; bootstrap over questions. "
            "n — number of unique questions after pooling.",
            "",
            "| Bench | Mode | n questions | MRR | 95% CI MRR | Hit@1 | 95% CI Hit@1 |",
            "| --- | --- | ---: | ---: | --- | ---: | --- |",
        ]
        for record in records:
            lines.append(
                f"| {record['bench']} "
                f"| {record['mode']} "
                f"| {record['n_questions']} "
                f"| {fmt(record['mrr_mean'])} "
                f"| [{fmt(record['mrr_ci_low'])}; {fmt(record['mrr_ci_high'])}] "
                f"| {fmt(record['hit_at_1_mean'])} "
                f"| [{fmt(record['hit_at_1_ci_low'])}; {fmt(record['hit_at_1_ci_high'])}] |"
            )

        lines += ["", f"Source: `results/{BOOTSTRAP_JSON}`"]

        path = (self._results / SUMMARY_MD).resolve()
        _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("MD summary → %s", path)
        return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results_dir = RESULTS_DIR.resolve()
    if not results_dir.is_dir():
        logger.error("Missing results directory: %s", results_dir)
        sys.exit(1)
    _ = RetrievalBootstrapBuilder(results_dir).build()


if __name__ == "__main__":
    main()
