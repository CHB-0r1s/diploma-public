"""Create a versioned dataset-selection artifact.

Example:
    python scripts/select_data.py selection=random
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
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

from diploma_sft.artifacts import write_selection_artifact  # noqa: E402
from diploma_sft.config import to_plain_dict  # noqa: E402
from diploma_sft.data import (  # noqa: E402
    dataset_fingerprint,
    random_pool_indices,
    resolve_dataset_revision,
    validate_dataset_layout,
)
from diploma_sft.runtime import environment_snapshot, git_commit  # noqa: E402
from diploma_sft.wandb_utils import init_wandb, log_files_as_artifact  # noqa: E402


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.selection.method != "random":
        raise NotImplementedError(
            f"selection.method={cfg.selection.method!r} is not implemented yet; use random"
        )

    from datasets import load_dataset

    output_dir = Path(to_absolute_path(str(cfg.selection_output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    _write_json(output_dir / "config.resolved.json", resolved_cfg)
    _write_json(output_dir / "environment.json", environment_snapshot(REPO_ROOT))

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
    selected_indices = random_pool_indices(
        pool_size=cfg.selection.pool_size,
        selected_count=cfg.selection.subsample_size,
        seed=cfg.selection.seed,
    )
    identity = dataset_fingerprint(dataset)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(REPO_ROOT),
        "dataset": {
            "name": cfg.dataset.name,
            "split": cfg.dataset.split,
            "requested_revision": cfg.dataset.revision,
            "resolved_revision": resolved_revision,
            "conversation_column": cfg.dataset.conversation_column,
            **identity,
        },
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
        },
    }
    artifact_paths = write_selection_artifact(output_dir, selected_indices, manifest)

    run = init_wandb(
        cfg,
        config=to_plain_dict(cfg),
        job_type="selection",
        run_name=f"selection-{cfg.selection.method}-{cfg.selection.subsample_size}",
        extra_tags=[cfg.selection.method],
    )
    log_files_as_artifact(
        run,
        name=f"selection-{cfg.selection.method}-{cfg.selection.subsample_size}",
        artifact_type="dataset-selection",
        paths=[str(path) for path in artifact_paths.values()],
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
