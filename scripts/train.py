"""Train one fixed selection artifact with Hydra, TRL/Unsloth, and W&B.

Example:
    python scripts/train.py selection_artifact=selections/random_90000/selection_manifest.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

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

from diploma_sft.artifacts import (  # noqa: E402
    load_selection_artifact,
    prepare_training_run_manifest,
)
from diploma_sft.config import to_plain_dict  # noqa: E402
from diploma_sft.data import load_selected_training_dataset  # noqa: E402
from diploma_sft.runtime import (  # noqa: E402
    environment_snapshot,
    latest_checkpoint,
    require_bf16_cuda,
)
from diploma_sft.wandb_utils import init_wandb, log_files_as_artifact  # noqa: E402


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def _resolve_resume_checkpoint(cfg: DictConfig, output_dir: Path) -> Optional[Path]:
    value = cfg.resume_from_checkpoint
    if value is None or str(value).lower() in {"", "false", "none"}:
        return None
    if str(value).lower() == "auto":
        return latest_checkpoint(output_dir)
    path = Path(to_absolute_path(str(value)))
    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.selection_artifact:
        raise ValueError(
            "selection_artifact is required. Run scripts/select_data.py first, then pass its manifest."
        )

    selection_manifest_path = Path(to_absolute_path(str(cfg.selection_artifact)))
    selection_manifest, selected_indices = load_selection_artifact(selection_manifest_path)
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = _resolve_resume_checkpoint(cfg, output_dir)
    training_protocol = {
        "seed": int(cfg.seed),
        "max_steps": int(cfg.max_steps),
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "lora": OmegaConf.to_container(cfg.lora, resolve=True),
        "train": OmegaConf.to_container(cfg.train, resolve=True),
    }
    run_manifest = prepare_training_run_manifest(
        output_dir=output_dir,
        selection_indices_sha256=selection_manifest["indices"]["sha256"],
        training_protocol=training_protocol,
        resume_checkpoint=resume_checkpoint,
    )
    _write_json(output_dir / "config.resolved.json", OmegaConf.to_container(cfg, resolve=True))
    _write_json(output_dir / "environment.json", environment_snapshot(REPO_ROOT))

    selection_copy = output_dir / "selection_manifest.json"
    indices_copy = output_dir / "selected_indices.npy"
    shutil.copy2(selection_manifest_path, selection_copy)
    shutil.copy2(selection_manifest_path.parent / selection_manifest["indices"]["file"], indices_copy)

    run = init_wandb(
        cfg,
        config=to_plain_dict(cfg),
        job_type="train",
        extra_tags=[selection_manifest["selection"]["method"]],
    )
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
    train_dataset, data_metadata = load_selected_training_dataset(
        selection_manifest,
        selected_indices,
        tokenizer_preview,
    )
    _write_json(output_dir / "dataset_metadata.json", data_metadata)

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
        eval_strategy="no",
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=False,
        report_to=["wandb"] if cfg.wandb.enabled else [],
        run_name=cfg.experiment_name,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        seed=cfg.seed,
        data_seed=cfg.seed,
        max_steps=cfg.max_steps,
    )
    training_args.packing = cfg.train.packing

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
    )
    if cfg.train.train_on_responses_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    metrics = train_result.metrics
    metrics.update(
        {
            "selected_rows": len(train_dataset),
            "selection_method": selection_manifest["selection"]["method"],
            "selection_indices_sha256": selection_manifest["indices"]["sha256"],
            "training_protocol_sha256": run_manifest["training_protocol_sha256"],
        }
    )
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    _write_json(adapter_dir / "training_run_manifest.json", run_manifest)
    _write_json(output_dir / "train_result.json", metrics)
    log_files_as_artifact(
        run,
        name=f"{cfg.experiment_name}-adapter",
        artifact_type="lora-adapter",
        paths=[str(path) for path in adapter_dir.glob("*") if path.is_file()],
        metadata={
            "base_model": cfg.model.name,
            "selection_method": selection_manifest["selection"]["method"],
            "selection_indices_sha256": selection_manifest["indices"]["sha256"],
            "training_protocol_sha256": run_manifest["training_protocol_sha256"],
            "selected_rows": len(train_dataset),
        },
    )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    main()
