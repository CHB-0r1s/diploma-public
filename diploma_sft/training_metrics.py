"""Training metrics that reuse logits from the optimization forward pass."""

from __future__ import annotations

from typing import Any, Optional


def masked_token_entropy(
    logits: Any,
    supervised_mask: Any,
    chunk_size: int = 32,
) -> Optional[float]:
    """Return mean categorical entropy in nats at positions selected by a mask."""
    import torch

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if logits.ndim != 3 or supervised_mask.ndim != 2:
        raise ValueError("Expected logits [batch, seq, vocab] and mask [batch, seq]")
    if logits.shape[:2] != supervised_mask.shape:
        raise ValueError("Logits and mask must have matching batch and sequence dimensions")

    supervised = torch.nonzero(supervised_mask.reshape(-1), as_tuple=False).flatten()
    if supervised.numel() == 0:
        return None

    flat_logits = logits.reshape(-1, logits.shape[-1])
    entropy_sum = torch.zeros((), dtype=torch.float32, device=logits.device)
    with torch.no_grad():
        for start in range(0, supervised.numel(), chunk_size):
            indices = supervised[start : start + chunk_size]
            token_logits = flat_logits[indices].float()
            log_normalizer = torch.logsumexp(token_logits, dim=-1)
            expected_logit = (torch.softmax(token_logits, dim=-1) * token_logits).sum(
                dim=-1
            )
            entropy_sum += (log_normalizer - expected_logit).sum()
    return float((entropy_sum / supervised.numel()).item())


def assistant_token_entropy(
    logits: Any,
    labels: Any,
    chunk_size: int = 32,
    ignore_index: int = -100,
) -> Optional[float]:
    """Return mean next-token entropy in nats on supervised assistant positions."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [batch, seq, vocab] and labels [batch, seq]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("Logits and labels must have matching batch and sequence dimensions")

    shifted_logits = logits[..., :-1, :]
    shifted_labels = labels[..., 1:]
    return masked_token_entropy(
        shifted_logits,
        shifted_labels != ignore_index,
        chunk_size=chunk_size,
    )
