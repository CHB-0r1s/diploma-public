"""Pinned dataset layout and rendering helpers for the experiment pipeline."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np


def dataset_fingerprint(dataset: Any) -> Dict[str, Any]:
    """Return the Hugging Face dataset identity recorded in every artifact."""
    info = getattr(dataset, "info", None)
    return {
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "builder_name": getattr(info, "builder_name", None),
        "config_name": getattr(info, "config_name", None),
        "version": str(getattr(info, "version", "")) if info else None,
        "dataset_size": getattr(info, "dataset_size", None),
        "num_rows": len(dataset),
    }


def resolve_dataset_revision(name: str, revision: Optional[str]) -> str:
    """Resolve a branch/tag to an immutable Hugging Face dataset commit SHA."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id=name, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Could not resolve an immutable revision for dataset {name}")
    return info.sha


def validate_dataset_layout(dataset_size: int, common_holdout_size: int, pool_size: int) -> None:
    if common_holdout_size <= 0:
        raise ValueError("common_holdout_size must be positive")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    required = common_holdout_size + pool_size
    if required > dataset_size:
        raise ValueError(f"common holdout + selection pool = {required} exceeds dataset_size={dataset_size}")


def random_pool_indices(pool_size: int, selected_count: int, seed: int) -> np.ndarray:
    """Choose a deterministic random subset in selection-pool coordinates."""
    if selected_count <= 0 or selected_count > pool_size:
        raise ValueError(f"selected_count={selected_count} must be in [1, pool_size={pool_size}]")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool_size, size=selected_count, replace=False)).astype(np.int64)


def render_chat_dataset(dataset: Any, tokenizer: Any, conversation_column: str, desc: str) -> Any:
    """Render a conversational dataset to fixed ChatML text."""

    def render(examples: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text": [
                tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                for conv in examples[conversation_column]
            ]
        }

    return dataset.map(
        render,
        batched=True,
        remove_columns=dataset.column_names,
        desc=desc,
    )


def load_dataset_from_manifest(manifest: Dict[str, Any]) -> Any:
    from datasets import load_dataset

    identity = manifest["dataset"]
    dataset = load_dataset(
        identity["name"],
        split=identity["split"],
        revision=identity["resolved_revision"],
    )
    actual = dataset_fingerprint(dataset)
    expected = identity["fingerprint"]
    if actual["fingerprint"] != expected:
        raise RuntimeError(
            "Dataset fingerprint mismatch: "
            f"artifact={expected}, loaded={actual['fingerprint']}"
        )
    if actual["num_rows"] != identity["num_rows"]:
        raise RuntimeError("Dataset row count does not match the selection artifact")
    return dataset


def load_selected_training_dataset(
    manifest: Dict[str, Any],
    selected_indices: np.ndarray,
    tokenizer: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Rebuild and render exactly the rows declared by a selection artifact."""
    dataset = load_dataset_from_manifest(manifest)
    layout = manifest["layout"]
    shuffled = dataset.shuffle(seed=int(layout["shuffle_seed"]))
    pool_start = int(layout["pool_start"])
    pool_size = int(layout["pool_size"])
    pool = shuffled.select(range(pool_start, pool_start + pool_size))
    selected = pool.select(selected_indices.tolist())
    rendered = render_chat_dataset(
        selected,
        tokenizer=tokenizer,
        conversation_column=manifest["dataset"]["conversation_column"],
        desc="render selected train dataset",
    )
    metadata = {
        "dataset": dataset_fingerprint(dataset),
        "shuffled_fingerprint": getattr(shuffled, "_fingerprint", None),
        "pool_fingerprint": getattr(pool, "_fingerprint", None),
        "selected_fingerprint": getattr(selected, "_fingerprint", None),
        "rendered_fingerprint": getattr(rendered, "_fingerprint", None),
        "train_rows": len(rendered),
        "common_holdout_rows": int(layout["common_holdout_size"]),
        "pool_rows": len(pool),
    }
    if len(rendered) != int(manifest["selection"]["selected_count"]):
        raise RuntimeError("Rendered train rows do not match selected_count")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return rendered, metadata


def load_common_evaluation_dataset(
    manifest: Dict[str, Any],
    tokenizer: Any,
    sample_limit: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the shared no-leak holdout declared by the selection artifact."""
    dataset = load_dataset_from_manifest(manifest)
    layout = manifest["layout"]
    shuffled = dataset.shuffle(seed=int(layout["shuffle_seed"]))
    holdout_size = int(layout["common_holdout_size"])
    common = shuffled.select(range(holdout_size))
    if sample_limit is not None:
        common = common.select(range(min(sample_limit, len(common))))
    rendered = render_chat_dataset(
        common,
        tokenizer=tokenizer,
        conversation_column=manifest["dataset"]["conversation_column"],
        desc="render common evaluation holdout",
    )
    metadata = {
        "common_eval_rows": len(rendered),
        "common_eval_fingerprint": getattr(common, "_fingerprint", None),
        "rendered_fingerprint": getattr(rendered, "_fingerprint", None),
    }
    return rendered, metadata
