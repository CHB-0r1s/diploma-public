import numpy as np
import pytest

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
