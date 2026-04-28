"""Сборка latency_bootstrap_ci.json и latency_bootstrap_ci_summary.md по benchmark JSON.

Входные данные
--------------
benchmark_rag_results_<mode>.json под results/<model>/results_{gold|noise}_bench/.
Ожидаемые поля на строку:
  bench_index        — идентификатор вопроса;
  latency_total_ms   — total latency;
  latency_embed_ms   — latency на эмбеддинг;
  latency_llm_ms     — latency LLM-генерации.

Метрики
-------
generation_latency_ms = latency_total_ms - latency_embed_ms
other_latency_ms      = latency_total_ms - latency_llm_ms

Методология
-----------
Единица bootstrap-выборки — вопрос (bench_index) после усреднения:
  - внутри одного файла дубликаты bench_index усредняются;
  - generation (таблица 1): по (model, bench, mode), bootstrap по вопросам;
  - other (таблица 2): по (bench, mode), сначала среднее по моделям на вопрос,
    затем bootstrap по вопросам;
  - noise: для каждого model сначала среднее по 3 прогонам на вопрос (пересечение ключей).

ДИ: непараметрический percentile bootstrap, 95% (alpha=0.05), B=10_000.

Запуск: uv run python scripts/build_latency_artifacts.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Final, cast

import numpy as np

logger = logging.getLogger(__name__)

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
RESULTS_DIR: Final[Path] = REPOSITORY_ROOT / "results"

MODELS: Final[tuple[str, str, str]] = ("gemma_31", "nemotron", "qwen_35")
RAG_MODES: Final[tuple[str, str]] = ("baseline", "full")
NOISE_RUN_DIRS: Final[tuple[str, str, str]] = (
    "first_results_noise_bench",
    "second_results_noise_bench",
    "third_results_noise_bench",
)

BOOTSTRAP_JSON: Final[str] = "latency_bootstrap_ci.json"
SUMMARY_MD: Final[str] = "latency_bootstrap_ci_summary.md"


class LatencyBootstrapBuilder:
    """Считает средние latency-метрики и bootstrap ДИ по model/bench/mode."""

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
        records = self._build_records()
        json_path = self._write_json(records)
        md_path = self._write_md(records)
        return json_path, md_path

    def _build_records(self) -> dict[str, list[dict[str, object]]]:
        generation_by_model: list[dict[str, object]] = []
        for model in MODELS:
            available_modes = self._detect_modes(model)
            if not available_modes:
                logger.warning("Нет benchmark_rag_results_*.json в results_gold_bench для %s", model)
                continue
            for mode in available_modes:
                gold_generation_values, _, gold_question_count = self._gold_vectors(model, mode)
                gold_generation_stats = self._bootstrap_ci(gold_generation_values)
                generation_by_model.append(
                    {
                        "model": model,
                        "bench": "gold",
                        "mode": mode,
                        "n_questions": gold_question_count,
                        "n_pooled_rows": gold_question_count,
                        "generation_latency_mean_ms": gold_generation_stats["mean"],
                        "generation_latency_ci_low_ms": gold_generation_stats["ci_low"],
                        "generation_latency_ci_high_ms": gold_generation_stats["ci_high"],
                        "n_bootstrap": self._num_bootstrap,
                        "alpha": self._alpha,
                    }
                )

                noise_generation_values, _, noise_question_count, noise_pooled_row_count = self._noise_vectors(model, mode)
                if noise_question_count == 0:
                    logger.warning("Noise: пустое пересечение bench_index для %s / %s", model, mode)
                noise_generation_stats = self._bootstrap_ci(noise_generation_values)
                generation_by_model.append(
                    {
                        "model": model,
                        "bench": "noise",
                        "mode": mode,
                        "n_questions": noise_question_count,
                        "n_pooled_rows": noise_pooled_row_count,
                        "generation_latency_mean_ms": noise_generation_stats["mean"],
                        "generation_latency_ci_low_ms": noise_generation_stats["ci_low"],
                        "generation_latency_ci_high_ms": noise_generation_stats["ci_high"],
                        "n_bootstrap": self._num_bootstrap,
                        "alpha": self._alpha,
                    }
                )

        other_by_bench_mode: list[dict[str, object]] = []
        for bench in ("gold", "noise"):
            for mode in RAG_MODES:
                other_latency_per_question, other_question_count = self._other_per_question_averaged_across_models(bench, mode)
                other_stats = self._bootstrap_ci(other_latency_per_question)
                other_by_bench_mode.append(
                    {
                        "bench": bench,
                        "mode": mode,
                        "n_questions": other_question_count,
                        "other_latency_mean_ms": other_stats["mean"],
                        "other_latency_ci_low_ms": other_stats["ci_low"],
                        "other_latency_ci_high_ms": other_stats["ci_high"],
                        "n_bootstrap": self._num_bootstrap,
                        "alpha": self._alpha,
                    }
                )

        generation_by_model.sort(key=lambda record: (str(record["model"]), str(record["bench"]), str(record["mode"])))
        other_by_bench_mode.sort(key=lambda record: (str(record["bench"]), str(record["mode"])))
        return {
            "generation_by_model": generation_by_model,
            "other_by_bench_mode": other_by_bench_mode,
        }

    def _gold_vectors(self, model: str, mode: str) -> tuple[list[float], list[float], int]:
        path = self._results / model / "results_gold_bench" / f"benchmark_rag_results_{mode}.json"
        per_question_data = self._load_per_question(path)
        generation_latency_values = [values[0] for _, values in sorted(per_question_data.items())]
        other_latency_values = [values[1] for _, values in sorted(per_question_data.items())]
        return generation_latency_values, other_latency_values, len(per_question_data)

    def _noise_vectors(self, model: str, mode: str) -> tuple[list[float], list[float], int, int]:
        per_run_data = [
            self._load_per_question(
                self._results / model / "results_noise_bench" / run_dir / f"benchmark_rag_results_{mode}.json"
            )
            for run_dir in NOISE_RUN_DIRS
        ]
        if not per_run_data:
            return [], [], 0, 0

        common_bench_indices = set(per_run_data[0].keys())
        for run_data in per_run_data[1:]:
            common_bench_indices &= set(run_data.keys())
        if not common_bench_indices:
            return [], [], 0, sum(len(run_data) for run_data in per_run_data)

        generation_latency_values: list[float] = []
        other_latency_values: list[float] = []
        for question_index in sorted(common_bench_indices):
            generation_latency_values.append(sum(run_data[question_index][0] for run_data in per_run_data) / len(per_run_data))
            other_latency_values.append(sum(run_data[question_index][1] for run_data in per_run_data) / len(per_run_data))
        return generation_latency_values, other_latency_values, len(common_bench_indices), sum(len(run_data) for run_data in per_run_data)

    def _other_per_question_averaged_across_models(self, bench: str, mode: str) -> tuple[list[float], int]:
        per_model_data: list[dict[int, tuple[float, float]]] = []
        for model in MODELS:
            if bench == "gold":
                path = self._results / model / "results_gold_bench" / f"benchmark_rag_results_{mode}.json"
                per_model_data.append(self._load_per_question(path))
            else:
                noise_run_data = [
                    self._load_per_question(
                        self._results / model / "results_noise_bench" / run_dir / f"benchmark_rag_results_{mode}.json"
                    )
                    for run_dir in NOISE_RUN_DIRS
                ]
                common_bench_indices = set(noise_run_data[0].keys())
                for noise_run in noise_run_data[1:]:
                    common_bench_indices &= set(noise_run.keys())
                noise_averaged_by_question: dict[int, tuple[float, float]] = {}
                for question_index in sorted(common_bench_indices):
                    noise_averaged_by_question[question_index] = (
                        sum(noise_run[question_index][0] for noise_run in noise_run_data) / len(noise_run_data),
                        sum(noise_run[question_index][1] for noise_run in noise_run_data) / len(noise_run_data),
                    )
                per_model_data.append(noise_averaged_by_question)

        if not per_model_data:
            return [], 0

        common_bench_indices = set(per_model_data[0].keys())
        for model_data in per_model_data[1:]:
            common_bench_indices &= set(model_data.keys())
        if not common_bench_indices:
            return [], 0

        other_latency_per_question: list[float] = []
        for question_index in sorted(common_bench_indices):
            other_latency_per_question.append(
                sum(model_data[question_index][1] for model_data in per_model_data) / len(per_model_data)
            )
        return other_latency_per_question, len(common_bench_indices)

    def _detect_modes(self, model: str) -> list[str]:
        gold_dir = self._results / model / "results_gold_bench"
        available_modes = {
            path.name[len("benchmark_rag_results_") : -len(".json")]
            for path in gold_dir.glob("benchmark_rag_results_*.json")
        }
        return [mode for mode in RAG_MODES if mode in available_modes]

    def _load_per_question(self, path: Path) -> dict[int, tuple[float, float]]:
        if not path.is_file():
            return {}
        try:
            loaded: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Пропуск %s: %s", path, exc)
            return {}
        if not isinstance(loaded, list):
            return {}
        data = cast(list[object], loaded)

        by_bench_index: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for entry in data:
            if not isinstance(entry, dict):
                continue
            record = cast(dict[str, object], entry)
            question_index = self._bench_index(record)
            latency_pair = self._latencies(record)
            if question_index is None or latency_pair is None:
                continue
            by_bench_index[question_index].append(latency_pair)

        averages_per_question: dict[int, tuple[float, float]] = {}
        for question_index, observations in by_bench_index.items():
            averages_per_question[question_index] = (
                sum(observation[0] for observation in observations) / len(observations),
                sum(observation[1] for observation in observations) / len(observations),
            )
        return averages_per_question

    @staticmethod
    def _bench_index(record: dict[str, object]) -> int | None:
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
        if isinstance(raw, str):
            stripped = raw.strip()
            if not stripped:
                return None
            try:
                integer_value = int(stripped)
                return integer_value if integer_value > 0 else None
            except ValueError:
                return None
        return None

    @staticmethod
    def _latencies(record: dict[str, object]) -> tuple[float, float] | None:
        total = record.get("latency_total_ms")
        embed = record.get("latency_embed_ms")
        llm = record.get("latency_llm_ms")
        if not isinstance(total, (int, float)):
            return None
        if not isinstance(embed, (int, float)):
            return None
        if not isinstance(llm, (int, float)):
            return None
        return float(total - embed), float(total - llm)

    def _bootstrap_ci(self, values: list[float]) -> dict[str, float | int]:
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
        ci_low, ci_high = np.percentile(bootstrap_sample_means, [lower_percentile, upper_percentile]).tolist()
        return {
            "mean": sample_mean,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n": n_questions,
        }

    def _write_json(self, records: dict[str, list[dict[str, object]]]) -> Path:
        path = (self._results / BOOTSTRAP_JSON).resolve()
        _ = path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "Latency bootstrap CI: %s generation + %s other записей → %s",
            len(records["generation_by_model"]),
            len(records["other_by_bench_mode"]),
            path,
        )
        return path

    def _write_md(self, records: dict[str, list[dict[str, object]]]) -> Path:
        def fmt(value: object, digits: int = 2) -> str:
            if value is None or (isinstance(value, float) and (value != value)):
                return "—"
            if isinstance(value, (int, float)):
                return f"{float(value):.{digits}f}"
            return "—"

        bench_ru = {"gold": "золотой", "noise": "шумный"}
        lines = [
            "# Latency: generation vs other (bootstrap 95% ДИ)",
            "",
            "generation_latency_ms = latency_total_ms - latency_embed_ms",
            "other_latency_ms = latency_total_ms - latency_llm_ms",
            "",
            "## 1) Generation latency (по model + bench + mode)",
            "",
            "| Модель | Бенч | Режим | Generation, ms | 95% ДИ |",
            "| --- | --- | --- | ---: | --- |",
        ]
        for record in records["generation_by_model"]:
            model = str(record["model"])
            bench = str(record["bench"])
            mode = str(record["mode"])
            row = " | ".join(
                [
                    "",
                    model,
                    bench_ru.get(bench, bench),
                    mode,
                    fmt(record["generation_latency_mean_ms"]),
                    f"[{fmt(record['generation_latency_ci_low_ms'])}; {fmt(record['generation_latency_ci_high_ms'])}]",
                    "",
                ]
            )
            lines.append(row)
        lines += [
            "",
            "## 2) Other latency (по bench + mode, усреднение по моделям per-question)",
            "",
            "| Бенч | Режим | Other, ms | 95% ДИ |",
            "| --- | --- | ---: | --- |",
        ]
        for record in records["other_by_bench_mode"]:
            bench = str(record["bench"])
            mode = str(record["mode"])
            row = " | ".join(
                [
                    "",
                    bench_ru.get(bench, bench),
                    mode,
                    fmt(record["other_latency_mean_ms"]),
                    f"[{fmt(record['other_latency_ci_low_ms'])}; {fmt(record['other_latency_ci_high_ms'])}]",
                    "",
                ]
            )
            lines.append(row)
        lines += ["", f"Источник: `results/{BOOTSTRAP_JSON}`"]

        path = (self._results / SUMMARY_MD).resolve()
        _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("MD-сводка → %s", path)
        return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results_dir = RESULTS_DIR.resolve()
    if not results_dir.is_dir():
        logger.error("Нет каталога results: %s", results_dir)
        sys.exit(1)
    _ = LatencyBootstrapBuilder(results_dir).build()


if __name__ == "__main__":
    main()
