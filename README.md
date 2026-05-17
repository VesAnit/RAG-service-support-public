# AI Assistant Benchmarks (public)

Репозиторий с результатами бенчмарка RAG-ассистента: сырые данные в `results/`, скрипты агрегации метрик с bootstrap 95% ДИ и ноутбук для визуализации оценок LLM-судьи.

## Содержимое

| Каталог / файл | Назначение |
| --- | --- |
| `results/` | JSON и XLSX по моделям, режимам и бенчам (gold / noise) |
| `scripts/` | CLI для пересборки артефактов bootstrap и сводок |
| `notebooks/` | Ноутбук для отрисовки графиков `judge_score_bootstrap_ci.json` |
| `charts/` | Графики |
| `prompts/` | Промпт для судьи генерации |

## Скрипты

Из корня репозитория (нужен [uv](https://github.com/astral-sh/uv) или окружение с зависимостями из pyproject.toml):

uv run python scripts/build_judge_score_artifacts.py
uv run python scripts/build_retrieval_score_artifacts.py
uv run python scripts/build_latency_artifacts.py


Подробности по полям входных файлов и методологии — в docstring-ах соответствующих скриптов.

## Демонстрация ассистента

<details>
<summary><strong>Видео: демо работы ассистента</strong> (архитектура, методика оценки RAG и генерации LLM)</summary>

Запись с пояснениями по архитектуре и методике оценки качества работы RAG и генерации на стороне LLM:

[Открыть на Google Drive](https://drive.google.com/file/d/16AmSomzB3d26k32He_tMljM02JFkYznW/view?usp=drive_link)

</details>
