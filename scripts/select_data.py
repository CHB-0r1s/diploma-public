"""Create a versioned dataset-selection artifact.

Examples:
    python scripts/select_data.py selection=random
    python scripts/select_data.py selection=ifd
    python scripts/select_data.py selection=entropy
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
    resolved_model_name = getattr(scorer.config, "_name_or_path", None)
    resolved_model_commit = getattr(scorer.config, "_commit_hash", None)
    if resolved_model_name and not resolved_model_commit:
        from huggingface_hub import HfApi

        resolved_model_commit = HfApi().model_info(repo_id=resolved_model_name).sha
    if not resolved_model_name or not resolved_model_commit:
        raise RuntimeError("Could not resolve the scorer model to an immutable Hugging Face commit")
    scoring_protocol["resolved_scorer_model"] = {
        "name_or_path": resolved_model_name,
        "commit_hash": resolved_model_commit,
    }
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
    resolved_model_name = getattr(scorer.config, "_name_or_path", None)
    resolved_model_commit = getattr(scorer.config, "_commit_hash", None)
    if resolved_model_name and not resolved_model_commit:
        from huggingface_hub import HfApi

        resolved_model_commit = HfApi().model_info(repo_id=resolved_model_name).sha
    if not resolved_model_name or not resolved_model_commit:
        raise RuntimeError("Could not resolve the scorer model to an immutable Hugging Face commit")
    scoring_protocol["resolved_scorer_model"] = {
        "name_or_path": resolved_model_name,
        "commit_hash": resolved_model_commit,
    }
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.selection.method not in {"random", "ifd", "entropy"}:
        raise NotImplementedError(
            f"selection.method={cfg.selection.method!r} is not implemented; "
            "use random, ifd, or entropy"
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
    else:
        selected_indices, selection_details, extra_paths, selection_metrics = _entropy_selection(
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
