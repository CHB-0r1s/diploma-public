"""Small W&B helpers for experiment artifacts."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional


def init_wandb(
    cfg: Any,
    config: Dict[str, Any],
    job_type: Optional[str] = None,
    run_name: Optional[str] = None,
    extra_tags: Optional[Iterable[str]] = None,
):
    if not cfg.wandb.enabled:
        return None
    import wandb

    kwargs = {
        "project": cfg.wandb.project,
        "name": run_name or cfg.experiment_name,
        "config": config,
        "tags": list(cfg.wandb.tags) + list(extra_tags or []),
    }
    if cfg.wandb.entity:
        kwargs["entity"] = cfg.wandb.entity
    if cfg.wandb.group:
        kwargs["group"] = cfg.wandb.group
    if cfg.wandb.mode:
        kwargs["mode"] = cfg.wandb.mode
    if job_type:
        kwargs["job_type"] = job_type
    return wandb.init(**kwargs)


def log_files_as_artifact(
    run: Any,
    name: str,
    artifact_type: str,
    paths: Iterable[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
    for path in paths:
        if os.path.exists(path):
            artifact.add_file(path)
    run.log_artifact(artifact)
