"""Runtime checks for Colab/GPU SFT runs."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional


def require_bf16_cuda() -> Dict[str, object]:
    """Require CUDA with Ampere+ compute capability and return precision flags."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This experiment requires a bf16-capable GPU.")

    major, minor = torch.cuda.get_device_capability(0)
    if major < 8:
        raise RuntimeError(
            f"bf16 is unavailable on GPU sm_{major}{minor}. "
            "Use A100/L4/H100 or another sm_80+ GPU."
        )
    return {"bf16": True, "fp16": False, "compute_capability": f"sm_{major}{minor}"}


def package_versions(packages: Iterable[str]) -> Dict[str, str]:
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def environment_snapshot(repo_root: Optional[Path] = None) -> Dict[str, object]:
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "compute_capability": f"sm_{major}{minor}",
            }
    except (ImportError, AttributeError):
        pass

    snapshot = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(
            [
                "torch",
                "transformers",
                "trl",
                "peft",
                "datasets",
                "huggingface-hub",
                "numpy",
                "unsloth",
                "bitsandbytes",
                "triton",
                "wandb",
                "hydra-core",
            ]
        ),
        "gpu": gpu,
    }
    if repo_root is not None:
        snapshot["git_commit"] = git_commit(repo_root)
    return snapshot


def git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Return the numerically latest Trainer checkpoint, if one exists."""
    checkpoints = []
    if output_dir.exists():
        for path in output_dir.glob("checkpoint-*"):
            try:
                step = int(path.name.removeprefix("checkpoint-"))
            except ValueError:
                continue
            if path.is_dir():
                checkpoints.append((step, path))
    return max(checkpoints, default=(None, None))[1]
