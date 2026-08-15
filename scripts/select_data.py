"""Create a versioned dataset-selection artifact.

Examples:
    python scripts/select_data.py selection=random
    python scripts/select_data.py selection=ifd
    python scripts/select_data.py selection=entropy
    python scripts/select_data.py selection=rho
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

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
    prepare_scoring_cache_manifest,
    sha256_file,
    write_selection_artifact,
)
from diploma_sft.config import to_plain_dict  # noqa: E402
from diploma_sft.data import (  # noqa: E402
    dataset_fingerprint,
    random_pool_indices,
    resolve_dataset_revision,
    validate_dataset_layout,
)
from diploma_sft.runtime import (  # noqa: E402
    environment_snapshot,
    git_commit,
    latest_checkpoint,
    require_bf16_cuda,
)
from diploma_sft.wandb_utils import init_wandb, log_files_as_artifact  # noqa: E402


class DatasetColumnView(Sequence):
    """Lazy row-wise view that avoids materializing 200k conversations in RAM."""

    def __init__(self, dataset: Any, column: str):
        self.dataset = dataset
        self.column = column

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.dataset[i][self.column] for i in range(*index.indices(len(self)))]
        return self.dataset[int(index)][self.column]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def _score_file_metadata(
    output_dir: Path,
    filenames: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    metadata = {}
    for filename in filenames:
        path = output_dir / filename
        metadata[filename] = {
            "file": filename,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return metadata


def _resolved_model_identity(model: Any) -> Dict[str, str]:
    name = getattr(model.config, "_name_or_path", None)
    commit = getattr(model.config, "_commit_hash", None)
    if name and not commit:
        from huggingface_hub import HfApi

        commit = HfApi().model_info(repo_id=name).sha
    if not name or not commit:
        raise RuntimeError("Could not resolve the scorer model to an immutable Hugging Face commit")
    return {"name_or_path": name, "commit_hash": commit}


def _ifd_selection(
    cfg: DictConfig,
    dataset: Any,
    dataset_identity: Dict[str, Any],
    output_dir: Path,
    environment: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any], List[Path], Dict[str, float]]:
    require_bf16_cuda()

    # Unsloth must patch Transformers before model/tokenizer loading.
    from unsloth import FastLanguageModel  # noqa: I001
    from unsloth.chat_templates import get_chat_template

    from ifd_select import IFDSelector

    layout = {
        "shuffle_seed": int(cfg.seed),
        "common_holdout_start": 0,
        "common_holdout_size": int(cfg.dataset.common_val_holdout_size),
        "pool_start": int(cfg.dataset.common_val_holdout_size),
        "pool_size": int(cfg.selection.pool_size),
    }
    shuffled = dataset.shuffle(seed=layout["shuffle_seed"])
    pool = shuffled.select(range(layout["pool_start"], layout["pool_start"] + layout["pool_size"]))
    scorer_source = REPO_ROOT / "notebooks" / "ifd_select.py"
    scoring_protocol = {
        "implementation": "ifd-ppl-ratio-v1",
        "implementation_sha256": sha256_file(scorer_source),
        "dataset_resolved_revision": dataset_identity["resolved_revision"],
        "dataset_fingerprint": dataset_identity["fingerprint"],
        "pool_fingerprint": getattr(pool, "_fingerprint", None),
        "layout": layout,
        "conversation_column": cfg.dataset.conversation_column,
        "scorer_model": cfg.selection.scorer_model,
        "chat_template": cfg.model.chat_template,
        "max_seq_len": int(cfg.model.max_seq_len),
        "load_in_4bit": bool(cfg.model.load_in_4bit),
        "batch_size": int(cfg.selection.batch_size),
        "ifd_threshold": float(cfg.selection.ifd_threshold),
        "packages": {
            name: environment["packages"].get(name)
            for name in ("torch", "transformers", "unsloth", "numpy")
        },
    }
    cache_filenames = ("scores_cond.npy", "scores_uncond.npy", "scores_ifd.npy")

    scorer, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.selection.scorer_model,
        max_seq_length=cfg.model.max_seq_len,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    scoring_protocol["resolved_scorer_model"] = _resolved_model_identity(scorer)
    cache_manifest = prepare_scoring_cache_manifest(
        output_dir,
        scoring_protocol=scoring_protocol,
        cache_filenames=cache_filenames,
    )
    FastLanguageModel.for_inference(scorer)
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.model.chat_template)
    tokenizer.padding_side = "right"

    selector = IFDSelector(
        scorer,
        tokenizer,
        max_seq_len=cfg.model.max_seq_len,
        batch_size=cfg.selection.batch_size,
        ifd_threshold=cfg.selection.ifd_threshold,
        cache_dir=str(output_dir),
        checkpoint_every=cfg.selection.checkpoint_every,
    )
    conversations = DatasetColumnView(pool, cfg.dataset.conversation_column)
    scores = selector.score(conversations)
    selected_indices = selector.select(scores, k=cfg.selection.subsample_size)

    valid = scores[~np.isnan(scores)]
    selected_scores = scores[selected_indices]
    stats = {
        "valid_scores": int(len(valid)),
        "nan_scores": int(np.isnan(scores).sum()),
        "selectable_below_threshold": int(
            ((~np.isnan(scores)) & (scores < cfg.selection.ifd_threshold)).sum()
        ),
        "ifd_min": float(valid.min()),
        "ifd_mean": float(valid.mean()),
        "ifd_median": float(np.median(valid)),
        "ifd_max": float(valid.max()),
        "selected_ifd_min": float(selected_scores.min()),
        "selected_ifd_mean": float(selected_scores.mean()),
        "selected_ifd_max": float(selected_scores.max()),
    }
    selection_details = {
        "scorer_model": cfg.selection.scorer_model,
        "batch_size": int(cfg.selection.batch_size),
        "ifd_threshold": float(cfg.selection.ifd_threshold),
        "checkpoint_every": int(cfg.selection.checkpoint_every),
        "scoring_protocol_sha256": cache_manifest["scoring_protocol_sha256"],
        "score_files": _score_file_metadata(output_dir, cache_filenames),
        "score_statistics": stats,
    }
    extra_paths = [output_dir / "scoring_cache_manifest.json"] + [
        output_dir / name for name in cache_filenames
    ]

    del selector, scorer, tokenizer
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return selected_indices, selection_details, extra_paths, stats


def _entropy_selection(
    cfg: DictConfig,
    dataset: Any,
    dataset_identity: Dict[str, Any],
    output_dir: Path,
    environment: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any], List[Path], Dict[str, float]]:
    require_bf16_cuda()

    # Unsloth must patch Transformers before model/tokenizer loading.
    from unsloth import FastLanguageModel  # noqa: I001
    from unsloth.chat_templates import get_chat_template

    from diploma_sft.entropy_selection import EntropySelector

    layout = {
        "shuffle_seed": int(cfg.seed),
        "common_holdout_start": 0,
        "common_holdout_size": int(cfg.dataset.common_val_holdout_size),
        "pool_start": int(cfg.dataset.common_val_holdout_size),
        "pool_size": int(cfg.selection.pool_size),
    }
    shuffled = dataset.shuffle(seed=layout["shuffle_seed"])
    pool = shuffled.select(range(layout["pool_start"], layout["pool_start"] + layout["pool_size"]))
    scorer_source = REPO_ROOT / "diploma_sft" / "entropy_selection.py"
    scoring_protocol = {
        "implementation": "mean-assistant-token-entropy-v1",
        "implementation_sha256": sha256_file(scorer_source),
        "dataset_resolved_revision": dataset_identity["resolved_revision"],
        "dataset_fingerprint": dataset_identity["fingerprint"],
        "pool_fingerprint": getattr(pool, "_fingerprint", None),
        "layout": layout,
        "conversation_column": cfg.dataset.conversation_column,
        "scorer_model": cfg.selection.scorer_model,
        "chat_template": cfg.model.chat_template,
        "max_seq_len": int(cfg.model.max_seq_len),
        "load_in_4bit": bool(cfg.model.load_in_4bit),
        "batch_size": int(cfg.selection.batch_size),
        "aggregation": "mean over all assistant response tokens",
        "packages": {
            name: environment["packages"].get(name)
            for name in ("torch", "transformers", "unsloth", "numpy")
        },
    }
    cache_filenames = ("scores_entropy.npy",)

    scorer, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.selection.scorer_model,
        max_seq_length=cfg.model.max_seq_len,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    scoring_protocol["resolved_scorer_model"] = _resolved_model_identity(scorer)
    cache_manifest = prepare_scoring_cache_manifest(
        output_dir,
        scoring_protocol=scoring_protocol,
        cache_filenames=cache_filenames,
    )
    FastLanguageModel.for_inference(scorer)
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.model.chat_template)
    tokenizer.padding_side = "right"

    selector = EntropySelector(
        scorer,
        tokenizer,
        max_seq_len=cfg.model.max_seq_len,
        batch_size=cfg.selection.batch_size,
        cache_dir=str(output_dir),
        checkpoint_every=cfg.selection.checkpoint_every,
    )
    conversations = DatasetColumnView(pool, cfg.dataset.conversation_column)
    scores = selector.score(conversations)
    selected_indices = selector.select(scores, k=cfg.selection.subsample_size)

    valid = scores[np.isfinite(scores)]
    selected_scores = scores[selected_indices]
    stats = {
        "valid_scores": int(len(valid)),
        "nan_scores": int(np.isnan(scores).sum()),
        "entropy_min": float(valid.min()),
        "entropy_mean": float(valid.mean()),
        "entropy_median": float(np.median(valid)),
        "entropy_max": float(valid.max()),
        "selected_entropy_min": float(selected_scores.min()),
        "selected_entropy_mean": float(selected_scores.mean()),
        "selected_entropy_max": float(selected_scores.max()),
    }
    selection_details = {
        "scorer_model": cfg.selection.scorer_model,
        "batch_size": int(cfg.selection.batch_size),
        "checkpoint_every": int(cfg.selection.checkpoint_every),
        "aggregation": "mean over all assistant response tokens",
        "scoring_protocol_sha256": cache_manifest["scoring_protocol_sha256"],
        "score_files": _score_file_metadata(output_dir, cache_filenames),
        "score_statistics": stats,
    }
    extra_paths = [
        output_dir / "scoring_cache_manifest.json",
        output_dir / "scores_entropy.npy",
    ]

    del selector, scorer, tokenizer
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return selected_indices, selection_details, extra_paths, stats


def _rho_selection(
    cfg: DictConfig,
    dataset: Any,
    dataset_identity: Dict[str, Any],
    output_dir: Path,
    environment: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any], List[Path], Dict[str, float]]:
    precision = require_bf16_cuda()

    # Unsloth must patch Transformers and TRL before either package is imported.
    from unsloth import FastLanguageModel  # noqa: I001
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    from diploma_sft.rho_selection import AssistantLossScorer, select_highest_rho

    holdout_size = int(cfg.selection.rho_holdout_size)
    pool_start = int(cfg.dataset.common_val_holdout_size)
    pool_size = int(cfg.selection.pool_size)
    holdout_start = pool_start + pool_size
    required_size = holdout_start + holdout_size
    if required_size > len(dataset):
        raise ValueError(
            f"RHO requires {required_size} shuffled rows for common holdout, "
            f"selection pool, and D_ho; dataset has {len(dataset)}"
        )

    shuffled = dataset.shuffle(seed=int(cfg.seed))
    pool = shuffled.select(range(pool_start, pool_start + pool_size))
    holdout = shuffled.select(range(holdout_start, holdout_start + holdout_size))
    holdout_split = holdout.train_test_split(
        test_size=float(cfg.selection.rho_holdout_val_fraction),
        seed=int(cfg.seed),
    )
    layout = {
        "shuffle_seed": int(cfg.seed),
        "common_holdout_start": 0,
        "common_holdout_size": int(cfg.dataset.common_val_holdout_size),
        "pool_start": pool_start,
        "pool_size": pool_size,
        "rho_holdout_start": holdout_start,
        "rho_holdout_size": holdout_size,
        "rho_holdout_train_size": len(holdout_split["train"]),
        "rho_holdout_eval_size": len(holdout_split["test"]),
    }
    if pool_size < int(cfg.selection.subsample_size):
        raise ValueError("RHO candidate pool is smaller than the requested selection")

    il_model, il_tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.selection.scorer_model,
        max_seq_length=cfg.model.max_seq_len,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    resolved_model = _resolved_model_identity(il_model)
    il_tokenizer = get_chat_template(il_tokenizer, chat_template=cfg.model.chat_template)
    il_tokenizer.padding_side = "right"

    scorer_source = REPO_ROOT / "diploma_sft" / "rho_selection.py"
    scoring_protocol = {
        "implementation": "rho-reducible-assistant-loss-v2",
        "implementation_sha256": sha256_file(scorer_source),
        "runner_sha256": sha256_file(Path(__file__)),
        "dataset_resolved_revision": dataset_identity["resolved_revision"],
        "dataset_fingerprint": dataset_identity["fingerprint"],
        "pool_fingerprint": getattr(pool, "_fingerprint", None),
        "rho_holdout_fingerprint": getattr(holdout, "_fingerprint", None),
        "layout": layout,
        "conversation_column": cfg.dataset.conversation_column,
        "scorer_model": cfg.selection.scorer_model,
        "resolved_scorer_model": resolved_model,
        "chat_template": cfg.model.chat_template,
        "max_seq_len": int(cfg.model.max_seq_len),
        "load_in_4bit": bool(cfg.model.load_in_4bit),
        "batch_size": int(cfg.selection.batch_size),
        "score": "mean assistant-token cross-entropy",
        "rho": "base_loss - irreducible_loss",
        "il_training": {
            "max_steps": int(cfg.selection.rho_il_max_steps),
            "eval_steps": int(cfg.selection.rho_il_eval_steps),
            "save_steps": int(cfg.selection.rho_il_save_steps),
            "warmup_steps": int(cfg.selection.rho_il_warmup_steps),
            "holdout_val_fraction": float(cfg.selection.rho_holdout_val_fraction),
            "per_device_train_batch_size": int(cfg.train.per_device_train_batch_size),
            "gradient_accumulation_steps": int(cfg.train.gradient_accumulation_steps),
            "learning_rate": float(cfg.train.learning_rate),
            "optim": cfg.train.optim,
            "lr_scheduler_type": cfg.train.lr_scheduler_type,
            "lora_r": int(cfg.selection.rho_il_lora_r),
            "lora_alpha": int(cfg.selection.rho_il_lora_alpha),
            "lora_dropout": float(cfg.selection.rho_il_lora_dropout),
            "lora_bias": cfg.lora.bias,
            "target_modules": list(cfg.lora.target_modules),
            "seed": int(cfg.selection.rho_il_seed),
            "packing": True,
            "train_on_responses_only": True,
            "best_checkpoint_metric": "eval_loss",
        },
        "packages": {
            name: environment["packages"].get(name)
            for name in ("torch", "transformers", "trl", "unsloth", "numpy")
        },
    }
    cache_filenames = ("scores_base.npy", "scores_il.npy", "scores_rho.npy")
    cache_manifest = prepare_scoring_cache_manifest(
        output_dir,
        scoring_protocol=scoring_protocol,
        cache_filenames=cache_filenames,
    )
    protocol_sha = cache_manifest["scoring_protocol_sha256"]

    il_training_dir = output_dir / "il_training"
    il_adapter_dir = output_dir / "il_adapter"
    il_manifest_path = il_adapter_dir / "il_training_manifest.json"
    il_log_path = output_dir / "il_training_log.json"
    existing_il_adapter = (il_adapter_dir / "adapter_config.json").is_file()
    if existing_il_adapter:
        if not il_manifest_path.is_file():
            raise RuntimeError("Existing IL adapter has no protocol manifest")
        with il_manifest_path.open(encoding="utf-8") as stream:
            il_manifest = json.load(stream)
        if il_manifest.get("scoring_protocol_sha256") != protocol_sha:
            raise RuntimeError("Existing IL adapter belongs to a different RHO protocol")
        del il_model, il_tokenizer
    else:
        def render(examples):
            return {
                "text": [
                    il_tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    for conversation in examples[cfg.dataset.conversation_column]
                ]
            }

        holdout_train = holdout_split["train"].map(
            render,
            batched=True,
            remove_columns=holdout_split["train"].column_names,
            desc="render RHO D_ho train",
        )
        holdout_eval = holdout_split["test"].map(
            render,
            batched=True,
            remove_columns=holdout_split["test"].column_names,
            desc="render RHO D_ho eval",
        )
        il_model = FastLanguageModel.get_peft_model(
            il_model,
            r=int(cfg.selection.rho_il_lora_r),
            target_modules=list(cfg.lora.target_modules),
            lora_alpha=int(cfg.selection.rho_il_lora_alpha),
            lora_dropout=float(cfg.selection.rho_il_lora_dropout),
            bias=cfg.lora.bias,
            use_gradient_checkpointing="unsloth",
            random_state=int(cfg.selection.rho_il_seed),
        )
        il_args = SFTConfig(
            output_dir=str(il_training_dir),
            max_steps=int(cfg.selection.rho_il_max_steps),
            per_device_train_batch_size=int(cfg.train.per_device_train_batch_size),
            gradient_accumulation_steps=int(cfg.train.gradient_accumulation_steps),
            optim=cfg.train.optim,
            learning_rate=float(cfg.train.learning_rate),
            lr_scheduler_type=cfg.train.lr_scheduler_type,
            warmup_steps=int(cfg.selection.rho_il_warmup_steps),
            bf16=precision["bf16"],
            fp16=precision["fp16"],
            max_length=int(cfg.model.max_seq_len),
            eos_token=cfg.model.eos_token,
            logging_steps=int(cfg.train.logging_steps),
            eval_strategy="steps",
            eval_steps=int(cfg.selection.rho_il_eval_steps),
            save_strategy="steps",
            save_steps=int(cfg.selection.rho_il_save_steps),
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            seed=int(cfg.selection.rho_il_seed),
            data_seed=int(cfg.seed),
        )
        il_args.packing = True
        il_args.eval_packing = False
        il_trainer = SFTTrainer(
            model=il_model,
            processing_class=il_tokenizer,
            args=il_args,
            train_dataset=holdout_train,
            eval_dataset=holdout_eval,
        )
        il_trainer = train_on_responses_only(
            il_trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        resume_checkpoint = latest_checkpoint(il_training_dir)
        if resume_checkpoint:
            print(f"Resuming RHO IL training from {resume_checkpoint}")
        il_trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
        )
        il_adapter_dir.mkdir(parents=True, exist_ok=True)
        il_trainer.model.save_pretrained(str(il_adapter_dir))
        il_tokenizer.save_pretrained(str(il_adapter_dir))
        il_manifest = {
            "scoring_protocol_sha256": protocol_sha,
            "best_model_checkpoint": il_trainer.state.best_model_checkpoint,
            "best_metric": il_trainer.state.best_metric,
            "global_step": int(il_trainer.state.global_step),
        }
        _write_json(il_manifest_path, il_manifest)
        _write_json(il_log_path, il_trainer.state.log_history)
        del il_trainer, il_model, il_tokenizer, holdout_train, holdout_eval

    gc.collect()
    torch = __import__("torch")
    torch.cuda.empty_cache()

    conversations = DatasetColumnView(pool, cfg.dataset.conversation_column)
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.selection.scorer_model,
        max_seq_length=cfg.model.max_seq_len,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    FastLanguageModel.for_inference(base_model)
    base_tokenizer = get_chat_template(base_tokenizer, chat_template=cfg.model.chat_template)
    base_tokenizer.padding_side = "right"
    base_scorer = AssistantLossScorer(
        base_model,
        base_tokenizer,
        cache_filename="scores_base.npy",
        max_seq_len=int(cfg.model.max_seq_len),
        batch_size=int(cfg.selection.batch_size),
        cache_dir=str(output_dir),
        checkpoint_every=int(cfg.selection.checkpoint_every),
    )
    base_losses = base_scorer.score(conversations, desc="RHO base loss")
    del base_scorer, base_model, base_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    il_scorer_model, il_scorer_tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(il_adapter_dir),
        max_seq_length=cfg.model.max_seq_len,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    FastLanguageModel.for_inference(il_scorer_model)
    il_scorer_tokenizer = get_chat_template(
        il_scorer_tokenizer,
        chat_template=cfg.model.chat_template,
    )
    il_scorer_tokenizer.padding_side = "right"
    il_scorer = AssistantLossScorer(
        il_scorer_model,
        il_scorer_tokenizer,
        cache_filename="scores_il.npy",
        max_seq_len=int(cfg.model.max_seq_len),
        batch_size=int(cfg.selection.batch_size),
        cache_dir=str(output_dir),
        checkpoint_every=int(cfg.selection.checkpoint_every),
    )
    il_losses = il_scorer.score(conversations, desc="RHO irreducible loss")
    selected_indices, rho_scores = select_highest_rho(
        base_losses,
        il_losses,
        k=int(cfg.selection.subsample_size),
    )
    np.save(output_dir / "scores_rho.npy", rho_scores, allow_pickle=False)

    valid_mask = np.isfinite(rho_scores)
    valid_rho = rho_scores[valid_mask]
    selected_rho = rho_scores[selected_indices]
    stats = {
        "valid_scores": int(valid_mask.sum()),
        "nan_scores": int(np.isnan(rho_scores).sum()),
        "negative_scores": int((valid_rho < 0).sum()),
        "rho_min": float(valid_rho.min()),
        "rho_mean": float(valid_rho.mean()),
        "rho_median": float(np.median(valid_rho)),
        "rho_max": float(valid_rho.max()),
        "selected_rho_min": float(selected_rho.min()),
        "selected_rho_mean": float(selected_rho.mean()),
        "selected_rho_max": float(selected_rho.max()),
        "il_best_eval_loss": float(il_manifest["best_metric"]),
        "il_global_step": int(il_manifest["global_step"]),
    }
    selection_details = {
        "scorer_model": cfg.selection.scorer_model,
        "resolved_scorer_model": resolved_model,
        "batch_size": int(cfg.selection.batch_size),
        "checkpoint_every": int(cfg.selection.checkpoint_every),
        "rho_layout": layout,
        "rho_formula": "mean_assistant_ce(base) - mean_assistant_ce(IL)",
        "il_training": scoring_protocol["il_training"],
        "il_adapter": str(il_adapter_dir),
        "il_best_model_checkpoint": il_manifest["best_model_checkpoint"],
        "scoring_protocol_sha256": protocol_sha,
        "score_files": _score_file_metadata(output_dir, cache_filenames),
        "score_statistics": stats,
    }
    extra_paths = [
        output_dir / "scoring_cache_manifest.json",
        output_dir / "scores_base.npy",
        output_dir / "scores_il.npy",
        output_dir / "scores_rho.npy",
        il_manifest_path,
    ]
    if il_log_path.is_file():
        extra_paths.append(il_log_path)
    extra_paths.extend(path for path in il_adapter_dir.glob("*") if path.is_file())

    del il_scorer, il_scorer_model, il_scorer_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return selected_indices, selection_details, extra_paths, stats


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.selection.method not in {"random", "ifd", "entropy", "rho"}:
        raise NotImplementedError(
            f"selection.method={cfg.selection.method!r} is not implemented; "
            "use random, ifd, entropy, or rho"
        )

    from datasets import load_dataset

    output_dir = Path(to_absolute_path(str(cfg.selection_output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    environment = environment_snapshot(REPO_ROOT)
    _write_json(output_dir / "config.resolved.json", resolved_cfg)
    _write_json(output_dir / "environment.json", environment)

    run = init_wandb(
        cfg,
        config=to_plain_dict(cfg),
        job_type="selection",
        run_name=f"selection-{cfg.selection.method}-{cfg.selection.subsample_size}",
        extra_tags=[cfg.selection.method],
    )

    resolved_revision = resolve_dataset_revision(cfg.dataset.name, cfg.dataset.revision)
    dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.split,
        revision=resolved_revision,
    )
    validate_dataset_layout(
        dataset_size=len(dataset),
        common_holdout_size=cfg.dataset.common_val_holdout_size,
        pool_size=cfg.selection.pool_size,
    )
    identity = {
        "name": cfg.dataset.name,
        "split": cfg.dataset.split,
        "requested_revision": cfg.dataset.revision,
        "resolved_revision": resolved_revision,
        "conversation_column": cfg.dataset.conversation_column,
        **dataset_fingerprint(dataset),
    }

    extra_paths: List[Path] = []
    selection_details: Dict[str, Any] = {}
    selection_metrics: Dict[str, float] = {}
    if cfg.selection.method == "random":
        selected_indices = random_pool_indices(
            pool_size=cfg.selection.pool_size,
            selected_count=cfg.selection.subsample_size,
            seed=cfg.selection.seed,
        )
    elif cfg.selection.method == "ifd":
        selected_indices, selection_details, extra_paths, selection_metrics = _ifd_selection(
            cfg,
            dataset,
            identity,
            output_dir,
            environment,
        )
    elif cfg.selection.method == "entropy":
        selected_indices, selection_details, extra_paths, selection_metrics = _entropy_selection(
            cfg,
            dataset,
            identity,
            output_dir,
            environment,
        )
    else:
        selected_indices, selection_details, extra_paths, selection_metrics = _rho_selection(
            cfg,
            dataset,
            identity,
            output_dir,
            environment,
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(REPO_ROOT),
        "dataset": identity,
        "layout": {
            "shuffle_seed": int(cfg.seed),
            "common_holdout_start": 0,
            "common_holdout_size": int(cfg.dataset.common_val_holdout_size),
            "pool_start": int(cfg.dataset.common_val_holdout_size),
            "pool_size": int(cfg.selection.pool_size),
        },
        "selection": {
            "method": cfg.selection.method,
            "seed": int(cfg.selection.seed),
            "selected_count": int(cfg.selection.subsample_size),
            **selection_details,
        },
    }
    artifact_paths = write_selection_artifact(output_dir, selected_indices, manifest)
    if run is not None and selection_metrics:
        run.log({f"selection/{key}": value for key, value in selection_metrics.items()})
    log_files_as_artifact(
        run,
        name=f"selection-{cfg.selection.method}-{cfg.selection.subsample_size}",
        artifact_type="dataset-selection",
        paths=[str(path) for path in list(artifact_paths.values()) + extra_paths],
        metadata={
            "method": cfg.selection.method,
            "selected_count": int(cfg.selection.subsample_size),
            "dataset_revision": resolved_revision,
        },
    )
    if run is not None:
        run.finish()

    print(f"Selection artifact: {artifact_paths['manifest']}")


if __name__ == "__main__":
    main()
