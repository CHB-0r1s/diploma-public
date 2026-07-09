# Публичные тетрадки дипломного эксперимента

Эта папка содержит финальные версии тетрадок для дипломной работы по сравнению стратегий отбора данных для SFT русскоязычной LLM. Тетрадки очищены от outputs, execution counts и временной metadata, чтобы их можно было безопасно перенести в публичный GitHub-репозиторий.

Источник: локальный репозиторий `/Users/boris/Documents/diploma/CHB-0r1s-diploma`, commit `5510fe0`.

## Общий протокол

- Базовая модель target-обучения: `Qwen/Qwen2.5-1.5B`.
- Обучение: QLoRA, assistant-only masked loss, packing, `SUBSAMPLE_SIZE = 90_000`, `NUM_EPOCHS = 1`.
- Precision: только bf16; если bf16 недоступен, тетрадки должны останавливаться без fallback на fp16.
- Chat template: явно фиксируется Qwen2.5 ChatML через `get_chat_template(..., "qwen-2.5")`; не используется неявный stock `apply_chat_template`.
- Валидация: общий no-leak common val резервируется до построения train/selection pools: `ds_full.shuffle(seed=SEED)[0:4500]`.
- Selection pool: no-leak pool начинается после common val; common-val примеры недоступны для скоринга и отбора.
- Артефакты обучения сохраняются в Google Drive, обычно в `MyDrive/diploma/outputs/...`.

## Тетрадки

Запускать в таком порядке:

1. [`baseline_masked_packed.ipynb`](notebooks/baseline_masked_packed.ipynb)  
   Random baseline: masked assistant-only loss, packing, 90k unique examples, 1 epoch, common no-leak val.

2. [`selection_entropy.ipynb`](notebooks/selection_entropy.ipynb)  
   Entropy selection: top-90k из no-leak 200k pool по средней entropy base-модели на assistant-токенах.

3. [`selection_loss_rho.ipynb`](notebooks/selection_loss_rho.ipynb)  
   RHO / reducible loss: proxy IL-модель обучается на `D_ho`, затем top-90k выбираются из disjoint `D_pool` по `L_base - L_IL`.

4. [`selection_ifd.ipynb`](notebooks/selection_ifd.ipynb)  
   IFD selection: top-90k по `IFD = PPL(y|x) / PPL(y)` на assistant-токенах.  
   Скоринг и отбор вынесены в компаньон-модуль [`ifd_select.py`](notebooks/ifd_select.py) (`IFDSelector`): подаёшь готовые `(model, tok)` — получаешь IFD-скоры и индексы top-K, с резюмируемым кешем скоров. Ядро model-agnostic, дефолтные chat-маркеры под Qwen-2.5.

5. [`selection_quality_classifier.ipynb`](notebooks/selection_quality_classifier.ipynb)  
   Quality classifier: Claude-labeled 3k subset -> Qwen2.5-0.5B quality scorer -> top-90k quality selection.

6. [`eval_ru_mt_bench_fastchat_official.ipynb`](notebooks/eval_ru_mt_bench_fastchat_official.ipynb)  
   Оценка обученных моделей на `t-tech/ru-mt-bench` через официальный FastChat judge loop. Кастомная часть только генерирует ответы из локальных Unsloth LoRA adapters в FastChat-compatible `model_answer/*.jsonl`.

## Требования к запуску

- Google Colab или совместимая среда с CUDA GPU, поддерживающим bf16: A100, L4, H100 или другой `sm_80+`.
- Google Drive для сохранения чекпоинтов и финальных adapters.
- Доступ к Hugging Face datasets/models.
- Для `selection_quality_classifier.ipynb`: `ANTHROPIC_API_KEY` в переменной окружения или Colab Secrets.
- Для `eval_ru_mt_bench_fastchat_official.ipynb`: `OPENAI_API_KEY` для GPT judge или `ANTHROPIC_API_KEY` для Claude judge, если judge model переключён на Claude.

Ключи API не должны записываться в тетрадки. Используйте переменные окружения или Colab Secrets.

## Что не включено

В публичный строгий набор намеренно не входят:

- `baseline_full.ipynb` и `baseline_masked.ipynb`: ранние baseline-версии на instruct-модели и старом режиме `30k x 3`.
- `selection_template.ipynb`: шаблон/промежуточная заготовка.
- `eval_ru_mt_bench.ipynb`: lightweight custom LLM-as-a-judge контур; он не является официальным FastChat judge loop.
- `step*.md`, `implementation_plan.md`: рабочие планы и промежуточные заметки.
- любые `outputs/`, adapters, checkpoints, метрики, annotations и batch ids.

Дополнительные проверки см. в [`COLAB_VERSION_AUDIT.md`](COLAB_VERSION_AUDIT.md) и [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md).
