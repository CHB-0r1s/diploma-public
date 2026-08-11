"""Dataset preparation and no-leak split helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import numpy as np


def dataset_fingerprint(dataset: Any) -> Dict[str, Any]:
    """Return stable-ish dataset metadata exposed by Hugging Face Datasets."""
    info = getattr(dataset, "info", None)
    return {
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "builder_name": getattr(info, "builder_name", None),
        "config_name": getattr(info, "config_name", None),
        "version": str(getattr(info, "version", "")) if info else None,
        "dataset_size": getattr(info, "dataset_size", None),
        "num_rows": len(dataset),
    }


def random_baseline_indices(
    dataset_size: int,
    common_val_holdout_size: int,
    subsample_size: int,
) -> Dict[str, np.ndarray]:
    """Indices in the shuffled dataset for common-val and random baseline train pool."""
    train_start = common_val_holdout_size
    train_end = train_start + subsample_size
    if train_end > dataset_size:
        raise ValueError(
            "common_val_holdout_size + subsample_size "
            f"= {train_end} exceeds dataset_size={dataset_size}"
        )
    return {
        "common_val": np.arange(0, common_val_holdout_size, dtype=np.int64),
        "selected": np.arange(train_start, train_end, dtype=np.int64),
    }


def save_split_artifacts(indices: Dict[str, np.ndarray], output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for name, values in indices.items():
        path = os.path.join(output_dir, f"{name}_indices.npy")
        np.save(path, values)
        paths[name] = path
    manifest_path = os.path.join(output_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {name: {"path": path, "count": int(len(indices[name]))} for name, path in paths.items()},
            f,
            indent=2,
            ensure_ascii=False,
        )
    paths["manifest"] = manifest_path
    return paths


def render_chat_dataset(dataset: Any, tokenizer: Any, conversation_column: str) -> Any:
    """Render a conversational dataset to a single text column with the tokenizer template."""

    def render(examples: Dict[str, Any]) -> Dict[str, Any]:
        texts = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in examples[conversation_column]
        ]
        return {"text": texts}

    return dataset.map(
        render,
        batched=True,
        remove_columns=dataset.column_names,
        desc="render chat template",
    )


def load_random_baseline_split(cfg: Any, tokenizer: Any, verbose: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """Load d0rj/ru-instruct, reserve common val, render random baseline train split."""
    from datasets import load_dataset

    ds_full = load_dataset(cfg.dataset.name, split=cfg.dataset.split, revision=cfg.dataset.revision)
    indices = random_baseline_indices(
        dataset_size=len(ds_full),
        common_val_holdout_size=cfg.dataset.common_val_holdout_size,
        subsample_size=cfg.selection.subsample_size,
    )

    ds_shuffled = ds_full.shuffle(seed=cfg.seed)
    ds_selected = ds_shuffled.select(indices["selected"].tolist())
    ds_rendered = render_chat_dataset(
        ds_selected,
        tokenizer=tokenizer,
        conversation_column=cfg.dataset.conversation_column,
    )
    ds_split = ds_rendered.train_test_split(test_size=cfg.dataset.val_size, seed=cfg.seed)

    metadata = {
        "dataset": dataset_fingerprint(ds_full),
        "shuffled_fingerprint": getattr(ds_shuffled, "_fingerprint", None),
        "selected_fingerprint": getattr(ds_selected, "_fingerprint", None),
        "rendered_fingerprint": getattr(ds_rendered, "_fingerprint", None),
        "train_rows": len(ds_split["train"]),
        "val_rows": len(ds_split["test"]),
        "common_val_rows": int(len(indices["common_val"])),
        "selected_rows": int(len(indices["selected"])),
    }
    if verbose:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return ds_split, {"indices": indices, "metadata": metadata}

