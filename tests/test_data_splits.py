import numpy as np
import pytest

from diploma_sft.config import compute_warmup_steps
from diploma_sft.data import random_baseline_indices


def test_random_baseline_indices_reserve_common_val_before_train():
    out = random_baseline_indices(
        dataset_size=100,
        common_val_holdout_size=10,
        subsample_size=20,
    )

    np.testing.assert_array_equal(out["common_val"], np.arange(0, 10))
    np.testing.assert_array_equal(out["selected"], np.arange(10, 30))
    assert set(out["common_val"]).isdisjoint(set(out["selected"]))


def test_random_baseline_indices_raises_when_dataset_too_small():
    with pytest.raises(ValueError, match="exceeds dataset_size"):
        random_baseline_indices(
            dataset_size=29,
            common_val_holdout_size=10,
            subsample_size=20,
        )


def test_compute_warmup_steps_uses_max_steps_for_smoke_run():
    assert compute_warmup_steps(128, 4, 2, 1, 5, 0.03) == 1


def test_compute_warmup_steps_uses_full_training_schedule():
    assert compute_warmup_steps(90_000, 4, 2, 1, -1, 0.03) == 338
