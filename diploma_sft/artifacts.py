"""Versioned selection artifact contract shared by selection, training, and evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

SELECTION_ARTIFACT_SCHEMA_VERSION = 1


def sha256_json(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_training_run_manifest(
    output_dir: Path,
    selection_indices_sha256: str,
    training_protocol: Dict[str, Any],
    resume_checkpoint: Optional[Path],
) -> Dict[str, Any]:
    """Create or verify the immutable inputs required for checkpoint resume."""
    run_manifest = {
        "selection_indices_sha256": selection_indices_sha256,
        "training_protocol": training_protocol,
        "training_protocol_sha256": sha256_json(training_protocol),
    }
    path = output_dir / "training_run_manifest.json"
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != run_manifest:
            raise RuntimeError(
                "Output directory belongs to a different selection artifact or training protocol"
            )
    elif resume_checkpoint is not None:
        raise RuntimeError("Refusing to resume a checkpoint without training_run_manifest.json")
    elif (output_dir / "adapter").exists():
        raise RuntimeError("Output directory already contains an adapter but no resumable checkpoint")
    with path.open("w", encoding="utf-8") as stream:
        json.dump(run_manifest, stream, indent=2, ensure_ascii=False)
    return run_manifest


def validate_adapter_lineage(adapter_dir: Path, selection_indices_sha256: str) -> Dict[str, Any]:
    """Verify that an adapter was trained from the supplied selection artifact."""
    path = adapter_dir / "training_run_manifest.json"
    if not path.is_file():
        raise RuntimeError("Adapter is missing training_run_manifest.json")
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("selection_indices_sha256") != selection_indices_sha256:
        raise RuntimeError("Adapter and selection artifact have different selection checksums")
    return manifest


def write_selection_artifact(
    output_dir: Path,
    selected_indices: np.ndarray,
    manifest: Dict[str, Any],
) -> Dict[str, Path]:
    """Write canonical pool-relative indices and their versioned manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = np.asarray(selected_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("selected_indices must be one-dimensional")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("selected_indices must be unique")

    indices_path = output_dir / "selected_indices.npy"
    np.save(indices_path, indices, allow_pickle=False)

    payload = dict(manifest)
    payload.update(
        {
            "artifact_type": "dataset-selection",
            "schema_version": SELECTION_ARTIFACT_SCHEMA_VERSION,
            "indices": {
                "file": indices_path.name,
                "coordinate_system": "selection_pool",
                "count": int(len(indices)),
                "sha256": sha256_file(indices_path),
            },
        }
    )
    manifest_path = output_dir / "selection_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)

    return {"manifest": manifest_path, "indices": indices_path}


def load_selection_artifact(manifest_path: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    """Load and validate a selection artifact before it reaches the trainer."""
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    if manifest.get("artifact_type") != "dataset-selection":
        raise ValueError("Not a dataset-selection artifact")
    if manifest.get("schema_version") != SELECTION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported selection artifact schema: {manifest.get('schema_version')}")

    for section in ("dataset", "layout", "selection", "indices"):
        if section not in manifest:
            raise ValueError(f"Selection manifest is missing '{section}'")

    indices_meta = manifest["indices"]
    if indices_meta.get("coordinate_system") != "selection_pool":
        raise ValueError("Selection indices must be relative to selection_pool")
    indices_path = manifest_path.parent / indices_meta["file"]
    if sha256_file(indices_path) != indices_meta["sha256"]:
        raise ValueError("Selection indices checksum mismatch")

    indices = np.load(indices_path, allow_pickle=False)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Selection indices must be a one-dimensional integer array")
    if len(indices) != int(indices_meta["count"]):
        raise ValueError("Selection index count does not match the manifest")
    if len(indices) != int(manifest["selection"]["selected_count"]):
        raise ValueError("Selected count does not match the manifest")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Selection indices are not unique")

    layout = manifest["layout"]
    common_start = int(layout["common_holdout_start"])
    common_size = int(layout["common_holdout_size"])
    pool_start = int(layout["pool_start"])
    pool_size = int(layout["pool_size"])
    if common_start != 0:
        raise ValueError("Common holdout must start at shuffled index 0")
    if pool_start < common_start + common_size:
        raise ValueError("Selection pool overlaps the common holdout")
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= pool_size):
        raise ValueError("Selection indices fall outside the declared pool")
    return manifest, indices.astype(np.int64, copy=False)
