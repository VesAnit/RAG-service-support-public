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
    Этот файл — единственный источник данных для ноутбука: и высота столбца,
    и полосы ошибок берутся из одного и того же place.

Статистическая методология
---------------------------
Точечная оценка
    Обычное выборочное среднее — несмещённая оценка генерального среднего
    при любом размере выборки.

Метод доверительного интервала: непараметрический percentile bootstrap
    Из n наблюдений генерируется B = 10 000 выборок с возвращением; по каждой
    вычисляется среднее. ДИ = [2.5-й перцентиль, 97.5-й перцентиль] полученного
    эмпирического распределения средних.

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

    # Чтение Excel

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
            i = int(raw)
            return i if i > 0 else None
        if isinstance(raw, str):
            stripped = raw.strip()
            if not stripped:
                return None
            try:
                as_float = float(stripped)
            except ValueError:
                return None
            if not as_float.is_integer():
                return None
            i = int(as_float)
            return i if i > 0 else None
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
        scores: list[float] = []
        for col in self._COL_SCORES:
            raw = row[col - 1].value
            if raw is None or isinstance(raw, bool):
                return None
            if isinstance(raw, str):
                stripped = raw.strip()
                if not stripped:
                    return None
                try:
                    value = float(stripped)
                except ValueError:
                    return None
            elif isinstance(raw, (int, float)):
                value = float(raw)
            else:
                return None
            if not (1.0 <= value <= 5.0):
                return None
            scores.append(value)
        return (scores[0], scores[1], scores[2], scores[3])

    # Сборка выборок для bootstrap

    def _gold_rows(self, workbook_path: Path) -> list[RowScores]:
        """Все валидные строки из одного gold-файла в порядке следования.

        Для gold bench_index не нужен: один прогон, одно наблюдение = одна строка.
        """
        rows: list[RowScores] = []
        for row in self._iter_generation_rows(workbook_path):
            if not self._has_question(row):
                continue
            parsed = self._parse_scores(row)
            if parsed is not None:
                rows.append(parsed)
        return rows

    def _rows_indexed_by_bench(self, workbook_path: Path) -> dict[int, RowScores]:
        """Прочитать один noise-файл в словарь {bench_index: scores}.

        При дублирующемся индексе сохраняется первая строка.
        """
        by_index: dict[int, RowScores] = {}
        for row in self._iter_generation_rows(workbook_path):
            if not self._has_question(row):
                continue
            key = self._bench_index(row[self._COL_BENCH_INDEX - 1].value)
            if key is None:
                continue
            parsed = self._parse_scores(row)
            if parsed is None:
                continue
            if key not in by_index:
                by_index[key] = parsed
        return by_index

    def _noise_rows(self, model: str, rag_mode: str) -> list[RowScores]:
        """Построить bootstrap-выборку для noise: одно наблюдение = один вопрос.

        1. Читаем три noise-прогона в словари {bench_index: scores}.
        2. Берём пересечение ключей — только вопросы, оценённые во всех трёх прогонах.
        3. Для каждого вопроса и каждой метрики отдельно усредняем три значения.
        Это убирает между-прогоновую дисперсию до bootstrap, не увеличивая n.
        """
        noise_root = self._results / model / "results_noise_bench"
        per_run = [
            self._rows_indexed_by_bench(noise_root / run / f"{self._REVIEW_PREFIX}{rag_mode}.xlsx")
            for run in self._NOISE_RUN_DIRS
        ]

        common_keys = set(per_run[0].keys())
        for run_map in per_run[1:]:
            common_keys &= set(run_map.keys())

        if not common_keys:
            return []

        averaged: list[RowScores] = []
        for key in sorted(common_keys):
            # stack: матрица (3 прогона × 4 метрики); mean по оси прогонов даёт 4 числа.
            stack = np.array([per_run[0][key], per_run[1][key], per_run[2][key]], dtype=np.float64)
            col_means = np.mean(stack, axis=0)
            averaged.append(
                (
                    float(col_means.flat[0]),
                    float(col_means.flat[1]),
                    float(col_means.flat[2]),
                    float(col_means.flat[3]),
                )
            )
        return averaged

    # Bootstrap

    def _bootstrap_ci(self, values: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
        """Непараметрический percentile bootstrap для одного вектора наблюдений.

        Возвращает mean, ci_low, ci_high (границы двустороннего ДИ уровня 1 - alpha) и n.
        При пустом векторе — nan по всем полям.
        """
        n = int(values.size)
        if n == 0:
            return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}

        sample_mean = float(values.mean())

        boot_means = np.empty(self._num_bootstrap, dtype=np.float64)
        for i in range(self._num_bootstrap):
            boot_means[i] = float(rng.choice(values, size=n, replace=True).mean())

        lo = 100.0 * self._alpha / 2.0
        hi = 100.0 * (1.0 - self._alpha / 2.0)
        ci = np.percentile(boot_means, [lo, hi])

        return {
            "mean": sample_mean,
            "ci_low": float(ci.flat[0]),
            "ci_high": float(ci.flat[1]),
            "n": n,
        }

    def _bootstrap_records_for(
        self,
        sink: list[dict[str, object]],
        model: str,
        bench: str,
        rag_mode: str,
        rows: list[RowScores],
        rng: np.random.Generator,
    ) -> None:
        """Добавить четыре записи (по одной на метрику) для одной комбинации (модель, bench, режим).

        Матрица (n × 4) нарезается по столбцам — каждый столбец идёт в _bootstrap_ci отдельно,
        что обеспечивает независимые ДИ для каждой метрики при общем rng-состоянии.
        """
        matrix = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 4), dtype=np.float64)

        for j, metric_name in enumerate(self._METRIC_KEYS):
            column = matrix[:, j] if matrix.size else np.zeros(0, dtype=np.float64)
            sink.append(
                {
                    "model": model,
                    "bench": bench,
                    "mode": rag_mode,
                    "metric": metric_name,
                    **self._bootstrap_ci(column, rng),
                    "n_bootstrap": self._num_bootstrap,
                    "alpha": self._alpha,
                }
            )

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

        Единый rng передаётся через все вызовы, чтобы воспроизводимость работала
        при фиксированном random_seed независимо от порядка операций.
        """
        rng = np.random.default_rng(self._random_seed)
        out: list[dict[str, object]] = []

        for model in self._MODELS:
            for rag_mode in self._RAG_MODES:
                gold_path = (
                    self._results / model / "results_gold_bench" / f"{self._REVIEW_PREFIX}{rag_mode}.xlsx"
                )
                self._bootstrap_records_for(out, model, "gold", rag_mode, self._gold_rows(gold_path), rng)

                noise_rows = self._noise_rows(model, rag_mode)
                if not noise_rows:
                    logger.warning(
                        "Noise: пустое пересечение по bench_index для %s / %s — в JSON будут nan",
                        model,
                        rag_mode,
                    )
                self._bootstrap_records_for(out, model, "noise", rag_mode, noise_rows, rng)

        out.sort(key=lambda r: (str(r["model"]), str(r["bench"]), str(r["mode"]), str(r["metric"])))
        return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(__file__).resolve().parent.parent
    _ = JudgeScoreArtifactsBuilder(root).build()


if __name__ == "__main__":
    main()
