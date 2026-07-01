# Аудит версий Colab и локальных файлов

Дата подготовки: 2026-07-01.

Источник публичной сборки: `/Users/boris/Documents/diploma/CHB-0r1s-diploma`, commit `5510fe0`.

## Source of truth

Для этой публичной папки source of truth считается локальный repo commit `5510fe0`. Открытые Google Colab runtime недоступны для автоматического чтения из этой среды, поэтому неподтверждённые ручные правки в Colab нужно отдельно экспортировать в `.ipynb` и сравнить.

## Проверка экспортов из Colab

После первичной сборки были отдельно скачаны и сравнены с `diploma-public/notebooks/` следующие Colab exports:

- `/Users/boris/Downloads/selection_entropy.ipynb`
- `/Users/boris/Downloads/selection_loss_rho.ipynb`
- `/Users/boris/Downloads/selection_quality_classifier.ipynb`
- `/Users/boris/Downloads/eval_ru_mt_bench_fastchat_official.ipynb`

Результат: по нормализованному тексту ячеек все четыре файла совпадают с публичными копиями и исходным repo. Отличия были только в JSON-представлении `source` и Colab metadata, то есть дополнительных ручных code/markdown правок в этих экспортированных Colab-файлах нет.

## Подтверждённые сигналы расхождений

- `/Users/boris/Documents/diploma/baseline_masked_packed.ipynb` является stale-дубликатом вне рабочего repo. Он меньше по размеру, отличается по hash, использует старую Drive mount-ячейку `drive.mount("/content/drive")` и не содержит финальный no-leak `COMMON_VAL_HOLDOUT_SIZE`. Его не публиковать.
- Актуальная версия `baseline_masked_packed.ipynb` находится в `/Users/boris/Documents/diploma/CHB-0r1s-diploma/baseline_masked_packed.ipynb` и скопирована в `notebooks/`.
- `eval_ru_mt_bench_fastchat_official.ipynb` имел несколько Colab-driven фиксов после runtime-ошибок: `No module named fastchat`, переход на явную shell install cell, robust defaults для partial rerun, сброс duplicate `max_new_tokens` / `max_length` warning. Актуальной считается repo-версия commit `5510fe0`, скопированная в `notebooks/`.
- `selection_entropy.ipynb` и `selection_loss_rho.ipynb` проходили ручной debug/runtime цикл в Colab. Финальные no-leak, Drive mount, RHO split и `90k x 1` правки находятся в repo и скопированы в `notebooks/`.

## Потенциальные Colab-only риски

Не найдено подтверждённых случаев, где финальная актуальная версия существует только в Google Colab и отсутствует в repo. Тем не менее риск есть для тетрадок, которые активно правились во время прогонов:

- `selection_entropy.ipynb`
- `selection_loss_rho.ipynb`
- `selection_quality_classifier.ipynb`
- `eval_ru_mt_bench_fastchat_official.ipynb`

Для этих четырёх файлов риск закрыт экспортами из `/Users/boris/Downloads`: дополнительные source-level различия не найдены.

Если в Colab у этих файлов виден статус unsaved / `Save in GitHub to keep changes`, нужно перед публикацией экспортировать текущую Colab-копию и сравнить её с `diploma-public/notebooks/<name>.ipynb`.

## Одноразовые диагностические ячейки

В ходе отладки в Colab использовались одноразовые диагностические ячейки, например:

- проверка `os.path.exists(...)` для Drive paths;
- ручной `drive.mount(..., force_remount=True)`, который падал на занятый mountpoint;
- проверка `outputs/` и baseline metrics paths;
- smoke-runs install/import для FastChat.

Эти ячейки не являются частью финального публичного протокола. В публичной папке оставлены только repo-версии тетрадок.

## Как проверить вручную

1. В Colab открыть нужную тетрадку.
2. `File -> Download -> Download .ipynb`.
3. Сравнить экспорт с соответствующим файлом из `diploma-public/notebooks/`.
4. Если различия только в outputs, execution counts или Colab metadata, repo-версия остаётся актуальной.
5. Если различия в code/markdown cells, нужно отдельно решить, переносить ли их в публичную папку.
