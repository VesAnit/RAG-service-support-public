"""Сборка judge_score_bootstrap_ci.json по review-таблицам LLM-судьи.

Входные данные
--------------
benchmark_qa_generation_review_*.xlsx под results/, лист «Оценка_генерации»:
  столбец 1  — порядковый индекс вопроса в бенче (bench_index);
  столбец 5  — текст вопроса (пустая ячейка = строка пропускается);
  столбцы 25–28 — четыре оценки LLM-судьи: релевантность, корректность,
                   достоверность, полнота; допустимая шкала [1, 5].

Выходной артефакт
-----------------
judge_score_bootstrap_ci.json
    Одна запись на (модель, bench, режим, метрика): точечная оценка mean
    и границы 95% доверительного интервала ci_low / ci_high.


Статистическая методология
---------------------------
Точечная оценка
    Обычное выборочное среднее — несмещённая оценка генерального среднего
    при любом размере выборки.

Метод доверительного интервала: непараметрический percentile bootstrap
    Bootstrap по выборке из n = 453 (gold) или n = 310 (noise) наблюдений, B = 10 000 репликаций.
    95% ДИ задаётся как пара перцентилей этого набора: 2.5-й и 97.5-й (нижняя и верхняя границы).

    В коде q задаётся в процентах:
        ci = np.percentile(bootstrap_sample_means, [lower_percentile, upper_percentile])

    Выбор percentile bootstrap вместо интервала на основе SEM (x̄ ± z·σ/√n),
    предлагаемого в https://www.anthropic.com/research/statistical-approach-to-model-evals,
    обусловлен следующим. Интервал через SEM предполагает симметрию, тогда как
    распределение оценок LLM-судьи имеет левосторонний хвост: модели чаще получают
    высокие оценки, и симметричный интервал недооценивает неопределённость в левом
    хвосте. Percentile bootstrap не накладывает предположений на форму распределения
    и автоматически воспроизводит асимметрию из данных. При n = 450 (gold) и
    n = 310 (noise) аппроксимация достаточно точная, поправка BCa численно незначима.

Конструирование выборки для noise
    Три независимых прогона нельзя конкатенировать: строки одного вопроса из разных
    прогонов — повторные измерения одного объекта, а не независимые наблюдения.
    Конкатенация искусственно увеличивает n в три раза и сужает ДИ.

    Корректная процедура (Anthropic, recommendation #3): для каждого вопроса,
    оцениваемого во всех трёх прогонах (идентификация по bench_index), вычисляется
    среднее по каждой метрике отдельно. Результат — одно наблюдение на вопрос;
    именно по этому набору (n ≈ 310 вопросов) строится bootstrap.

Запуск: uv run python scripts/build_judge_score_artifacts.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import numpy as np
from openpyxl import load_workbook
from openpyxl.cell.cell import Cell, MergedCell

logger = logging.getLogger(__name__)

RowScores = tuple[float, float, float, float]


class JudgeScoreArtifactsBuilder:
    """Читает review Excel и записывает judge_score_bootstrap_ci.json.

    Всё чтение Excel проходит через _iter_generation_rows — единственное место
    открытия/закрытия workbook. Парсинг ячеек изолирован в статических методах
    и переиспользуется для gold и noise без дублирования.
    """

    _SHEET: Final[str] = "Оценка_генерации"
    _COL_BENCH_INDEX: Final[int] = 1   # порядковый номер вопроса — ключ для noise-выравнивания
    _COL_QUESTION: Final[int] = 5      # непустая ячейка = строка относится к конкретному вопросу
    _COL_SCORES: Final[tuple[int, int, int, int]] = (25, 26, 27, 28)  # relevance / correctness / faithfulness / completeness
    _ITER_MAX_COL: Final[int] = 28

    _REVIEW_PREFIX: Final[str] = "benchmark_qa_generation_review_"

    _BOOTSTRAP_JSON: Final[str] = "judge_score_bootstrap_ci.json"

    # Порядок метрик совпадает с порядком столбцов 25–28 и фиксирован во всём пайплайне.
    _METRIC_KEYS: Final[tuple[str, str, str, str]] = (
        "relevance",
        "correctness",
        "faithfulness",
        "completeness",
    )

    _MODELS: Final[tuple[str, str, str]] = ("gemma_31", "nemotron", "qwen_35")
    _RAG_MODES: Final[tuple[str, str]] = ("baseline", "full")

    # Имена подкаталогов трёх noise-прогонов под results/<model>/results_noise_bench/.
    _NOISE_RUN_DIRS: Final[tuple[str, str, str]] = (
        "first_results_noise_bench",
        "second_results_noise_bench",
        "third_results_noise_bench",
    )

    def __init__(
        self,
        project_root: Path,
        *,
        num_bootstrap_replicates: int = 10_000,
        alpha: float = 0.05,
        random_seed: int = 42,
    ) -> None:
        self._root: Path = project_root.resolve()
        self._results: Path = (self._root / "results").resolve()
        self._num_bootstrap: int = num_bootstrap_replicates
        self._alpha: float = alpha
        self._random_seed: int = random_seed

    def build(self) -> Path:
        """Точка входа: пересобрать bootstrap JSON и вернуть путь к файлу."""
        return self._write_bootstrap_json()

    def _write_bootstrap_json(self) -> Path:
        if not self._results.is_dir():
            logger.error("Нет каталога results: %s", self._results)
            sys.exit(1)

        records = self._build_bootstrap_records()
        path = (self._results / self._BOOTSTRAP_JSON).resolve()
        _ = path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Bootstrap CI: %s записей → %s", len(records), path)
        return path

    def _build_bootstrap_records(self) -> list[dict[str, object]]:
        """Внешний цикл: (модель × режим) → gold + noise → четыре метрики каждый.

        Единый random_generator передаётся через все вызовы, чтобы воспроизводимость
        работала при фиксированном random_seed независимо от порядка операций.
        """
        random_generator = np.random.default_rng(self._random_seed)
        records: list[dict[str, object]] = []

        for model in self._MODELS:
            for rag_mode in self._RAG_MODES:
                gold_workbook_path = (
                    self._results / model / "results_gold_bench" / f"{self._REVIEW_PREFIX}{rag_mode}.xlsx"
                )
                self._append_bootstrap_records(records, model, "gold", rag_mode, self._gold_rows(gold_workbook_path), random_generator)

                noise_averaged_rows = self._noise_rows(model, rag_mode)
                if not noise_averaged_rows:
                    logger.warning(
                        "Noise: пустое пересечение по bench_index для %s / %s — в JSON будут nan",
                        model,
                        rag_mode,
                    )
                self._append_bootstrap_records(records, model, "noise", rag_mode, noise_averaged_rows, random_generator)

        records.sort(key=lambda record: (str(record["model"]), str(record["bench"]), str(record["mode"]), str(record["metric"])))
        return records

    def _gold_rows(self, workbook_path: Path) -> list[RowScores]:
        """Все валидные строки из одного gold-файла в порядке следования.

        Для gold bench_index не нужен: один прогон, одно наблюдение = одна строка.
        """
        valid_rows: list[RowScores] = []
        for row in self._iter_generation_rows(workbook_path):
            if not self._has_question(row):
                continue
            parsed_scores = self._parse_scores(row)
            if parsed_scores is not None:
                valid_rows.append(parsed_scores)
        return valid_rows

    def _noise_rows(self, model: str, rag_mode: str) -> list[RowScores]:
        """Построить bootstrap-выборку для noise: одно наблюдение = один вопрос.

        1. Читаем три noise-прогона в словари {bench_index: scores}.
        2. Берём пересечение ключей — только вопросы, оценённые во всех трёх прогонах.
        3. Для каждого вопроса и каждой метрики отдельно усредняем три значения.
        Это убирает между-прогоновую дисперсию до bootstrap, не увеличивая n.
        """
        noise_bench_dir = self._results / model / "results_noise_bench"
        per_run_scores = [
            self._rows_indexed_by_bench(noise_bench_dir / run_dir / f"{self._REVIEW_PREFIX}{rag_mode}.xlsx")
            for run_dir in self._NOISE_RUN_DIRS
        ]

        common_bench_indices = set(per_run_scores[0].keys())
        for run_scores_by_index in per_run_scores[1:]:
            common_bench_indices &= set(run_scores_by_index.keys())

        if not common_bench_indices:
            return []

        averaged_rows: list[RowScores] = []
        for question_index in sorted(common_bench_indices):
            # scores_matrix: матрица (3 прогона × 4 метрики); mean по оси прогонов даёт 4 числа.
            scores_matrix = np.array(
                [per_run_scores[0][question_index], per_run_scores[1][question_index], per_run_scores[2][question_index]],
                dtype=np.float64,
            )
            metric_means = np.mean(scores_matrix, axis=0)
            averaged_rows.append(
                (
                    float(metric_means.flat[0]),
                    float(metric_means.flat[1]),
                    float(metric_means.flat[2]),
                    float(metric_means.flat[3]),
                )
            )
        return averaged_rows

    def _append_bootstrap_records(
        self,
        output_records: list[dict[str, object]],
        model: str,
        bench: str,
        rag_mode: str,
        rows: list[RowScores],
        random_generator: np.random.Generator,
    ) -> None:
        """Добавить четыре записи (по одной на метрику) для одной комбинации (модель, bench, режим).

        Матрица (n × 4) нарезается по столбцам — каждый столбец идёт в _bootstrap_ci отдельно,
        что обеспечивает независимые ДИ для каждой метрики при общем random_generator-состоянии.
        """
        scores_matrix = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 4), dtype=np.float64)

        for metric_index, metric_name in enumerate(self._METRIC_KEYS):
            metric_column = scores_matrix[:, metric_index] if scores_matrix.size else np.zeros(0, dtype=np.float64)
            output_records.append(
                {
                    "model": model,
                    "bench": bench,
                    "mode": rag_mode,
                    "metric": metric_name,
                    **self._bootstrap_ci(metric_column, random_generator),
                    "n_bootstrap": self._num_bootstrap,
                    "alpha": self._alpha,
                }
            )

    def _rows_indexed_by_bench(self, workbook_path: Path) -> dict[int, RowScores]:
        """Прочитать один noise-файл в словарь {bench_index: scores}.

        При дублирующемся индексе сохраняется первая строка.
        """
        scores_by_bench_index: dict[int, RowScores] = {}
        for row in self._iter_generation_rows(workbook_path):
            if not self._has_question(row):
                continue
            question_index = self._bench_index(row[self._COL_BENCH_INDEX - 1].value)
            if question_index is None:
                continue
            parsed_scores = self._parse_scores(row)
            if parsed_scores is None:
                continue
            if question_index not in scores_by_bench_index:
                scores_by_bench_index[question_index] = parsed_scores
        return scores_by_bench_index

    def _iter_generation_rows(self, workbook_path: Path) -> Iterator[tuple[Cell | MergedCell, ...]]:
        """Генератор строк листа «Оценка_генерации» начиная со второй (первая — заголовок).

        Открывает книгу в read_only + data_only, чтобы не держать формулы и не делать
        полную десериализацию. Закрывает книгу в finally даже при исключении.
        """
        if not workbook_path.is_file():
            return
        workbook = load_workbook(workbook_path, read_only=True, data_only=True)
        try:
            if self._SHEET not in workbook.sheetnames:
                return
            yield from workbook[self._SHEET].iter_rows(min_row=2, max_col=self._ITER_MAX_COL)
        finally:
            workbook.close()

    @staticmethod
    def _bench_index(raw: object) -> int | None:
        """Привести значение ячейки столбца 1 к целому индексу > 0.

        openpyxl отдаёт числа как int или float в зависимости от формата ячейки;
        строки возможны при текстовом формате столбца. Всё, что не целое > 0, → None.
        """
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
                float_value = float(stripped)
            except ValueError:
                return None
            if not float_value.is_integer():
                return None
            integer_value = int(float_value)
            return integer_value if integer_value > 0 else None
        return None

    @staticmethod
    def _has_question(row: tuple[Cell | MergedCell, ...]) -> bool:
        """Строка относится к конкретному вопросу, если ячейка столбца 5 непустая."""
        cell_value = row[JudgeScoreArtifactsBuilder._COL_QUESTION - 1].value
        return cell_value is not None and str(cell_value).strip() != ""

    def _parse_scores(self, row: tuple[Cell | MergedCell, ...]) -> RowScores | None:
        """Извлечь четыре оценки из столбцов 25–28; вернуть None при любой неполноте.

        Валидная строка — четыре числа в [1, 5]. Пропуск, bool, нечисловая строка
        или выход за диапазон — вся строка не участвует в bootstrap-выборке.
        """
        score_values: list[float] = []
        for col in self._COL_SCORES:
            raw = row[col - 1].value
            if raw is None or isinstance(raw, bool):
                return None
            if isinstance(raw, str):
                stripped = raw.strip()
                if not stripped:
                    return None
                try:
                    score_value = float(stripped)
                except ValueError:
                    return None
            elif isinstance(raw, (int, float)):
                score_value = float(raw)
            else:
                return None
            if not (1.0 <= score_value <= 5.0):
                return None
            score_values.append(score_value)
        return (score_values[0], score_values[1], score_values[2], score_values[3])

    def _bootstrap_ci(self, values: np.ndarray, random_generator: np.random.Generator) -> dict[str, float | int]:
        """Непараметрический percentile bootstrap для одного вектора наблюдений.

        Возвращает mean, ci_low, ci_high (границы двустороннего ДИ уровня 1 - alpha) и n.
        При пустом векторе — nan по всем полям.
        """
        n_observations = int(values.size)
        if n_observations == 0:
            return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}

        sample_mean = float(values.mean())

        bootstrap_sample_means = np.empty(self._num_bootstrap, dtype=np.float64)
        for sample_index in range(self._num_bootstrap):
            bootstrap_sample_means[sample_index] = float(
                random_generator.choice(values, size=n_observations, replace=True).mean()
            )

        # Двусторонний интервал уровня (1−α): слева и справа отсекаем по α/2 массы
        # эмпирического распределения bootstrap-средних → перцентили α/2 и 1−α/2 (в %).
        lower_percentile = 100.0 * self._alpha / 2.0
        upper_percentile = 100.0 * (1.0 - self._alpha / 2.0)
        ci = np.percentile(bootstrap_sample_means, [lower_percentile, upper_percentile])

        return {
            "mean": sample_mean,
            "ci_low": float(ci.flat[0]),
            "ci_high": float(ci.flat[1]),
            "n": n_observations,
        }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(__file__).resolve().parent.parent
    _ = JudgeScoreArtifactsBuilder(root).build()


if __name__ == "__main__":
    main()
