"""Pure helpers for paired comparison of benchmark predictions."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Tuple

import numpy as np


def exact_mcnemar_pvalue(baseline_only: int, candidate_only: int) -> float:
    """Two-sided exact McNemar p-value for discordant correctness pairs."""
    if baseline_only < 0 or candidate_only < 0:
        raise ValueError("Discordant counts must be non-negative")
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, candidate_only)
    log_probabilities = [
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2.0)
        for index in range(tail + 1)
    ]
    largest = max(log_probabilities)
    one_sided = math.exp(largest) * sum(
        math.exp(value - largest) for value in log_probabilities
    )
    return min(1.0, 2.0 * one_sided)


def paired_bootstrap_interval(
    baseline_correct: Iterable[bool],
    candidate_correct: Iterable[bool],
    resamples: int = 10_000,
    seed: int = 42,
    batch_size: int = 200,
) -> Tuple[float, float]:
    """Percentile bootstrap interval for candidate minus baseline accuracy."""
    baseline = np.asarray(list(baseline_correct), dtype=np.int8)
    candidate = np.asarray(list(candidate_correct), dtype=np.int8)
    if baseline.shape != candidate.shape or baseline.ndim != 1 or not len(baseline):
        raise ValueError("Correctness vectors must be non-empty and equally sized")
    if resamples <= 0 or batch_size <= 0:
        raise ValueError("resamples and batch_size must be positive")

    differences = candidate - baseline
    generator = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        sampled = generator.integers(0, len(differences), size=(stop - start, len(differences)))
        estimates[start:stop] = differences[sampled].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_pairs(
    rows: Iterable[Dict[str, Any]],
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    """Summarize aligned predictions with paired significance statistics."""
    values = list(rows)
    if not values:
        raise ValueError("No aligned predictions")
    baseline = [bool(row["baseline_correct"]) for row in values]
    candidate = [bool(row["candidate_correct"]) for row in values]
    both_correct = sum(left and right for left, right in zip(baseline, candidate))
    baseline_only = sum(left and not right for left, right in zip(baseline, candidate))
    candidate_only = sum(not left and right for left, right in zip(baseline, candidate))
    both_wrong = len(values) - both_correct - baseline_only - candidate_only
    baseline_accuracy = sum(baseline) / len(values)
    candidate_accuracy = sum(candidate) / len(values)
    lower, upper = paired_bootstrap_interval(
        baseline,
        candidate,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return {
        "examples": len(values),
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_delta": candidate_accuracy - baseline_accuracy,
        "accuracy_delta_ci95": [lower, upper],
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "both_wrong": both_wrong,
        "discordant": baseline_only + candidate_only,
        "mcnemar_exact_pvalue": exact_mcnemar_pvalue(baseline_only, candidate_only),
        "prediction_agreement": sum(
            row["baseline_prediction"] == row["candidate_prediction"] for row in values
        )
        / len(values),
    }
