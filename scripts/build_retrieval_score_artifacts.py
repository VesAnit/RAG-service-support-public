"""Сборка retrieval_bootstrap_ci.json и retrieval_bootstrap_ci_summary.md по MRR и Hit@1.

Входные данные
--------------
benchmark_rag_results_<mode>.json под results/, по одному на (модель, режим, [прогон]).
В каждой записи ожидаются поля:
  bench_index           — целый порядковый номер вопроса (общий для всех моделей и прогонов);
  chunk_id              — str или list[str] с эталонными point_id; отсутствие поля = нет эталона,
                          запись не участвует в расчёте;
  retrieval_rr_initial  — float: reciprocal rank первого подходящего чанка в initial-контексте;
  retrieval_rank_initial — int | None: 1-based ранг.

Методология
-----------
Ретривал не зависит от LLM-модели (один и тот же RAG-конвейер): для каждого вопроса
(bench_index) все модели дают одинаковый ретривал-результат. Поэтому перед bootstrap
мы пулируем наблюдения по bench_index — сквозь все модели и (для noise) все прогоны —
и усредняем RR и индикатор Hit@1 по каждому вопросу. Единица bootstrap-выборки —
один вопрос после усреднения.

Gold: 3 модели × 1 прогон = 3 строки на вопрос → средние RR и Hit@1 на вопрос.
Noise: 3 модели × 3 прогона = 9 строк на вопрос → средние RR и Hit@1 на вопрос.
Bootstrap: percentile 95% ДИ по набору усреднённых вопросов (n ≈ 453 gold, ≈ 310 noise).

Hit@1: 1, если ранг эталона ≤ 1; иначе 0. При отсутствующем ранге (эталон не найден) — 0.

Запуск: uv run python scripts/build_retrieval_score_artifacts.py
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

_RESULTS_DIR: Final[Path] = REPOSITORY_ROOT / "results"
_MODELS: Final[tuple[str, str, str]] = ("gemma_31", "nemotron", "qwen_35")
_RAG_MODES: Final[tuple[str, str]] = ("baseline", "full")
_NOISE_RUN_DIRS: Final[tuple[str, str, str]] = (
    "first_results_noise_bench",
    "second_results_noise_bench",
    "third_results_noise_bench",
)

_BOOTSTRAP_JSON: Final[str] = "retrieval_bootstrap_ci.json"
_SUMMARY_MD: Final[str] = "retrieval_bootstrap_ci_summary.md"


class RetrievalBootstrapBuilder:
    """Читает benchmark_rag_results JSON, пулирует по bench_index и считает bootstrap CI.

    Публичная точка входа — build(): записывает JSON и MD, возвращает их пути.
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
        """Пересобрать JSON и MD; вернуть их пути."""
        records = self._build_records()
        json_path = self._write_json(records)
        md_path = self._write_md(records)
        return json_path, md_path

    # Сборка данных

    def _gold_paths(self, mode: str) -> list[Path]:
        """Пути к JSON одного режима для всех трёх моделей (gold bench)."""
        return [
            self._results / model / "results_gold_bench" / f"benchmark_rag_results_{mode}.json"
            for model in _MODELS
        ]

    def _noise_paths(self, mode: str) -> list[Path]:
        """Пути ко всем 9 файлам (3 модели × 3 прогона) для noise bench."""
        return [
            self._results / model / "results_noise_bench" / run / f"benchmark_rag_results_{mode}.json"
            for model in _MODELS
            for run in _NOISE_RUN_DIRS
        ]

    @staticmethod
    def _load_records_with_gold(path: Path) -> list[dict[str, Any]]:
        """Загрузить JSON и вернуть только записи с непустым chunk_id (есть эталон)."""
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Пропуск %s: %s", path, exc)
            return []
        if not isinstance(data, list):
            return []
        return [r for r in data if isinstance(r, dict) and r.get("chunk_id")]

    @staticmethod
    def _bench_index(rec: dict[str, Any]) -> int | None:
        """Нормализовать bench_index к int > 0 или вернуть None."""
        raw = rec.get("bench_index")
        if raw is None or isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return raw if raw > 0 else None
        if isinstance(raw, float):
            if not raw.is_integer():
                return None
            i = int(raw)
            return i if i > 0 else None
        try:
            i = int(raw)
            return i if i > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _rr(rec: dict[str, Any]) -> float:
        """Reciprocal rank initial-контекста; при отсутствии эталона — 0."""
        v = rec.get("retrieval_rr_initial")
        return float(v) if isinstance(v, (int, float)) else 0.0

    @staticmethod
    def _hit1(rec: dict[str, Any]) -> float:
        """1.0 если эталон на первом месте, иначе 0.0."""
        rank = rec.get("retrieval_rank_initial")
        if rank is None:
            return 0.0
        return 1.0 if int(rank) <= 1 else 0.0

    def _per_question_averages(
        self, paths: list[Path]
    ) -> tuple[list[float], list[float], int]:
        """Усредняет RR и Hit@1 по bench_index; возвращает (mrr_vec, hit_vec, n_pooled_rows)."""
        by_index: dict[int, list[tuple[float, float]]] = defaultdict(list)
        n_total = 0
        for path in paths:
            for rec in self._load_records_with_gold(path):
                key = self._bench_index(rec)
                if key is None:
                    continue
                by_index[key].append((self._rr(rec), self._hit1(rec)))
                n_total += 1

        mrr_vec: list[float] = []
        hit_vec: list[float] = []
        for key in sorted(by_index.keys()):
            pairs = by_index[key]
            mrr_vec.append(sum(p[0] for p in pairs) / len(pairs))
            hit_vec.append(sum(p[1] for p in pairs) / len(pairs))
        return mrr_vec, hit_vec, n_total

    # Bootstrap

    def _bootstrap_ci(self, values: list[float]) -> dict[str, float | int]:
        """Percentile bootstrap: mean + 95% ДИ. При пустом списке — nan."""
        n = len(values)
        if n == 0:
            return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}

        arr = np.asarray(values, dtype=np.float64)
        sample_mean = float(arr.mean())
        rng = np.random.default_rng(self._random_seed)
        boot_means = np.empty(self._num_bootstrap, dtype=np.float64)
        for i in range(self._num_bootstrap):
            boot_means[i] = float(rng.choice(arr, size=n, replace=True).mean())

        lo = 100.0 * self._alpha / 2.0
        hi = 100.0 * (1.0 - self._alpha / 2.0)
        ci = np.percentile(boot_means, [lo, hi])
        return {
            "mean": sample_mean,
            "ci_low": float(ci.flat[0]),
            "ci_high": float(ci.flat[1]),
            "n": n,
        }

    # Сборка записей

    def _build_records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for bench in ("gold", "noise"):
            for mode in _RAG_MODES:
                paths = self._gold_paths(mode) if bench == "gold" else self._noise_paths(mode)
                mrr_vec, hit_vec, n_pooled = self._per_question_averages(paths)

                if not mrr_vec:
                    logger.warning("Нет данных для %s / %s", bench, mode)

                mrr_stats = self._bootstrap_ci(mrr_vec)
                hit_stats = self._bootstrap_ci(hit_vec)
                out.append(
                    {
                        "bench": bench,
                        "mode": mode,
                        "n_questions": mrr_stats["n"],
                        "n_pooled_rows": n_pooled,
                        "mrr_mean": mrr_stats["mean"],
                        "mrr_ci_low": mrr_stats["ci_low"],
                        "mrr_ci_high": mrr_stats["ci_high"],
                        "hit_at_1_mean": hit_stats["mean"],
                        "hit_at_1_ci_low": hit_stats["ci_low"],
                        "hit_at_1_ci_high": hit_stats["ci_high"],
                        "n_bootstrap": self._num_bootstrap,
                        "alpha": self._alpha,
                    }
                )
        return out

    # Запись артефактов

    def _write_json(self, records: list[dict[str, Any]]) -> Path:
        path = (self._results / _BOOTSTRAP_JSON).resolve()
        _ = path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Retrieval bootstrap CI: %s записей → %s", len(records), path)
        return path

    def _write_md(self, records: list[dict[str, Any]]) -> Path:
        """Краткая MD-таблица для коммита в репо."""

        def fmt(v: Any, digits: int = 4) -> str:
            if v is None or (isinstance(v, float) and (v != v)):  # nan check
                return "—"
            return f"{float(v):.{digits}f}"

        bench_ru = {"gold": "золотой", "noise": "шумный"}

        lines = [
            "# Retrieval: MRR и Hit@1 (bootstrap 95% ДИ)",
            "",
            "Пул всех моделей на каждый вопрос (bench_index). "
            "Gold: 3 модели × 1 прогон. Noise: 3 модели × 3 прогона. "
            "На каждый вопрос — среднее RR и Hit@1 по пулу; bootstrap по вопросам. "
            "n — число уникальных вопросов после пулинга.",
            "",
            "| Бенч | Режим | n вопросов | MRR | 95% ДИ MRR | Hit@1 | 95% ДИ Hit@1 |",
            "| --- | --- | ---: | ---: | --- | ---: | --- |",
        ]
        for r in records:
            lines.append(
                f"| {bench_ru.get(r['bench'], r['bench'])} "
                f"| {r['mode']} "
                f"| {r['n_questions']} "
                f"| {fmt(r['mrr_mean'])} "
                f"| [{fmt(r['mrr_ci_low'])}; {fmt(r['mrr_ci_high'])}] "
                f"| {fmt(r['hit_at_1_mean'])} "
                f"| [{fmt(r['hit_at_1_ci_low'])}; {fmt(r['hit_at_1_ci_high'])}] |"
            )

        lines += ["", f"Источник: `results/{_BOOTSTRAP_JSON}`"]

        path = (self._results / _SUMMARY_MD).resolve()
        _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("MD-сводка → %s", path)
        return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results_dir = _RESULTS_DIR.resolve()
    if not results_dir.is_dir():
        logger.error("Нет каталога results: %s", results_dir)
        sys.exit(1)
    _ = RetrievalBootstrapBuilder(results_dir).build()


if __name__ == "__main__":
    main()
