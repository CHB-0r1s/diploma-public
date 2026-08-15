import types

import numpy as np
import pytest

from diploma_sft.entropy_selection import EntropySelector, select_highest_entropy


class FakeTokenizer:
    padding_side = "right"

    def __call__(self, text, add_special_tokens=True):
        markers = {
            "<|im_start|>assistant\n": [1, 2],
            "<|im_start|>user\n": [3],
        }
        return types.SimpleNamespace(input_ids=markers.get(text, [ord(char) for char in text]))


def test_select_highest_entropy_is_deterministic_and_dataset_sorted():
    scores = np.array([0.2, 0.9, np.nan, 0.7, 0.9])

    selected = select_highest_entropy(scores, k=3)

    assert selected.tolist() == [1, 3, 4]


def test_select_highest_entropy_rejects_too_few_valid_scores():
    with pytest.raises(ValueError, match="Only 1 valid"):
        select_highest_entropy(np.array([np.nan, 0.5]), k=2)


def test_entropy_selector_requires_right_padding():
    tokenizer = FakeTokenizer()
    tokenizer.padding_side = "left"

    with pytest.raises(ValueError, match="right-padding"):
        EntropySelector(model=None, tokenizer=tokenizer)


def test_entropy_cache_round_trip_and_size_validation(tmp_path):
    selector = EntropySelector(model=None, tokenizer=FakeTokenizer(), cache_dir=str(tmp_path))
    scores = np.array([0.2, np.nan, 0.7], dtype=np.float32)
    selector._save(scores)

    np.testing.assert_array_equal(selector._load_partial(3), scores)
    with pytest.raises(RuntimeError, match="does not match"):
        selector._load_partial(4)
