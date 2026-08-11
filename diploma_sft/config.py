"""Typed experiment config helpers used by Hydra entrypoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetConfig:
    name: str = "d0rj/ru-instruct"
    split: str = "train"
    revision: Optional[str] = None
    conversation_column: str = "conversations"
    val_size: float = 0.05
    common_val_holdout_size: int = 4_500
    eval_samples: int = 500
    final_eval_samples: int = 200


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-1.5B"
    chat_template: str = "qwen-2.5"
    max_seq_len: int = 2048
    load_in_4bit: bool = True


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TrainConfig:
    num_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 20
    eval_steps: int = 200
    save_steps: int = 400
    save_total_limit: int = 2
    packing: bool = True
    eval_packing: bool = False
    train_on_responses_only: bool = True


@dataclass
class SelectionConfig:
    method: str = "random"
    subsample_size: int = 90_000
    pool_limit: Optional[int] = None


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "diploma-sft"
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["baseline", "sft"])
    mode: Optional[str] = None
    log_model: bool = False


@dataclass
class ExperimentConfig:
    experiment_name: str = "baseline_random_qwen15b_90k"
    seed: int = 42
    output_dir: str = "outputs/${experiment_name}"
    debug: bool = False
    max_steps: int = -1
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def to_plain_dict(cfg: Any) -> Dict[str, Any]:
    """Convert a dataclass or OmegaConf object into a plain JSON-serializable dict."""
    if hasattr(cfg, "__dataclass_fields__"):
        return asdict(cfg)
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except Exception:
        return dict(cfg)

