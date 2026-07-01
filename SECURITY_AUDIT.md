# Security audit

Дата подготовки: 2026-07-01.

## Результат

В публичной папке не найдено hardcoded credentials по типичным паттернам:

- OpenAI keys: `sk-...`
- Anthropic keys: `sk-ant-...`
- Hugging Face tokens: `hf_...`
- GitHub tokens: `ghp_...`, `github_pat_...`
- Google API keys: `AIza...`
- PEM private keys

`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` и похожие строки в тетрадках используются как имена переменных окружения, а не как значения секретов.

## Правила для публичного репозитория

- Не коммитить `.env`, Colab Secrets exports, API keys, Hugging Face tokens.
- Не коммитить `outputs/`, checkpoints, LoRA adapters, merged models, `final_metrics.json`, `batch_id.txt`, `annotations.json`, scorer caches и `scores*.npy`.
- Для запуска judge/evaluation задавать ключи через environment variables или Colab Secrets.
- Перед публикацией нового commit повторять secret scan по всей папке.

## Проверка, выполненная при сборке

Проверялись файлы внутри `diploma-public` после копирования и очистки notebooks. Дополнительно ранее проверялись исходные notebooks в рабочем repo; реальных значений секретов найдено не было.
