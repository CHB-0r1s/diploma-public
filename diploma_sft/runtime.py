"""Runtime checks for Colab/GPU SFT runs."""

from __future__ import annotations

import importlib.metadata
import platform
from typing import Dict, Iterable


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


def environment_snapshot() -> Dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(
            ["torch", "transformers", "trl", "peft", "datasets", "unsloth", "wandb", "hydra-core"]
        ),
    }

