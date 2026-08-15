import pytest

from diploma_sft.paired_analysis import (
    exact_mcnemar_pvalue,
    paired_bootstrap_interval,
    summarize_pairs,
)


def _row(baseline, candidate, baseline_prediction="A", candidate_prediction="A"):
    return {
        "baseline_correct": baseline,
        "candidate_correct": candidate,
        "baseline_prediction": baseline_prediction,
        "candidate_prediction": candidate_prediction,
    }


def test_exact_mcnemar_handles_symmetric_and_one_sided_discordance():
    assert exact_mcnemar_pvalue(2, 2) == pytest.approx(1.0)
    assert exact_mcnemar_pvalue(5, 0) == pytest.approx(0.0625)
    assert exact_mcnemar_pvalue(0, 0) == pytest.approx(1.0)


def test_paired_summary_uses_direction_candidate_minus_baseline():
    rows = [
        _row(True, True),
        _row(True, False, "A", "B"),
        _row(True, False, "A", "C"),
        _row(False, True, "D", "A"),
        _row(False, False, "B", "C"),
    ]

    result = summarize_pairs(rows, bootstrap_resamples=200, bootstrap_seed=7)

    assert result["baseline_accuracy"] == pytest.approx(0.6)
    assert result["candidate_accuracy"] == pytest.approx(0.4)
    assert result["accuracy_delta"] == pytest.approx(-0.2)
    assert result["baseline_only_correct"] == 2
    assert result["candidate_only_correct"] == 1
    assert result["both_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["prediction_agreement"] == pytest.approx(0.2)


def test_paired_bootstrap_is_deterministic():
    baseline = [True, True, False, False]
    candidate = [True, False, True, False]

    first = paired_bootstrap_interval(baseline, candidate, resamples=100, seed=11)
    second = paired_bootstrap_interval(baseline, candidate, resamples=100, seed=11)

    assert first == second
