"""Train the random baseline with Hydra, TRL/Unsloth, and W&B logging.

Example:
    python scripts/train.py
    python scripts/train.py debug=true max_steps=5 wandb.mode=offline
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import hydra
    from hydra.utils import to_absolute_path
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing experiment dependencies. Install them with: "
        'python -m pip install -e ".[experiments]"'
    ) from exc

from diploma_sft.config import to_plain_dict  # noqa: E402
from diploma_sft.data import load_random_baseline_split, save_split_artifacts  # noqa: E402
from diploma_sft.runtime import environment_snapshot, require_bf16_cuda  # noqa: E402
from diploma_sft.wandb_utils import init_wandb, log_files_as_artifact  # noqa: E402


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.selection.method != "random":
        raise NotImplementedError("scripts/train.py currently supports selection.method=random only")

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.resolved.json", resolved_cfg)
    _write_json(output_dir / "environment.json", environment_snapshot())

    run = init_wandb(cfg, config=to_plain_dict(cfg))

    precision = require_bf16_cuda()

    # Unsloth must patch Transformers and TRL before either package is imported.
    from unsloth import FastLanguageModel  # noqa: I001
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer_preview = get_chat_template(
        AutoTokenizer.from_pretrained(cfg.model.name),
        chat_template=cfg.model.chat_template,
    )
    ds_split, data_artifacts = load_random_baseline_split(cfg, tokenizer_preview)
    split_paths = save_split_artifacts(data_artifacts["indices"], str(output_dir))
    _write_json(output_dir / "dataset_metadata.json", data_artifacts["metadata"])
    log_files_as_artifact(
        run,
        name=f"{cfg.experiment_name}-splits",
        artifact_type="dataset-splits",
        paths=split_paths.values(),
        metadata=data_artifacts["metadata"],
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_len,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.model.chat_template)
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        target_modules=list(cfg.lora.target_modules),
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    eval_subset = ds_split["test"].select(range(min(cfg.dataset.eval_samples, len(ds_split["test"]))))
    report_to = ["wandb"] if cfg.wandb.enabled else []
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.train.num_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        optim=cfg.train.optim,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        # A float below 1 is resolved against the post-packing training steps.
        warmup_steps=cfg.train.warmup_ratio,
        bf16=precision["bf16"],
        fp16=precision["fp16"],
        max_length=cfg.model.max_seq_len,
        eos_token=cfg.model.eos_token,
        logging_steps=cfg.train.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        run_name=cfg.experiment_name,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        seed=cfg.seed,
        data_seed=cfg.seed,
        max_steps=cfg.max_steps,
    )
    training_args.packing = cfg.train.packing
    training_args.eval_packing = cfg.train.eval_packing

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=ds_split["train"],
        eval_dataset=eval_subset,
    )
    if cfg.train.train_on_responses_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    _write_json(output_dir / "train_result.json", metrics)
    log_files_as_artifact(
        run,
        name=f"{cfg.experiment_name}-adapter",
        artifact_type="lora-adapter",
        paths=[str(path) for path in adapter_dir.glob("*") if path.is_file()],
        metadata={"base_model": cfg.model.name, "selection_method": cfg.selection.method},
    )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    main()
