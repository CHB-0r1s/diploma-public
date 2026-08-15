import types

import numpy as np
import pytest

from diploma_sft.rho_selection import AssistantLossScorer, select_highest_rho


class FakeTokenizer:
    padding_side = "right"

    def __call__(self, text, add_special_tokens=True):
        markers = {
            "<|im_start|>assistant\n": [1, 2],
            "<|im_start|>user\n": [3],
        }
        return types.SimpleNamespace(input_ids=markers.get(text, [ord(char) for char in text]))


def test_select_highest_rho_is_deterministic_and_dataset_sorted():
    base = np.array([1.0, 1.2, np.nan, 0.9, 1.1])
    irreducible = np.array([0.8, 0.4, 0.1, 0.5, 0.3])

    selected, rho = select_highest_rho(base, irreducible, k=3)

    assert selected.tolist() == [1, 3, 4]
    np.testing.assert_allclose(rho[[1, 3, 4]], [0.8, 0.4, 0.8])


def test_select_highest_rho_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        select_highest_rho(np.array([1.0]), np.array([1.0, 2.0]), k=1)


def test_select_highest_rho_rejects_too_few_valid_scores():
    with pytest.raises(ValueError, match="Only 1 valid"):
        select_highest_rho(np.array([np.nan, 1.0]), np.array([0.0, 0.5]), k=2)


def test_assistant_loss_scorer_requires_right_padding():
    tokenizer = FakeTokenizer()
    tokenizer.padding_side = "left"

    with pytest.raises(ValueError, match="right-padding"):
        AssistantLossScorer(
            model=None,
            tokenizer=tokenizer,
            cache_filename="scores.npy",
        )


def test_rho_cache_round_trip_and_size_validation(tmp_path):
    scorer = AssistantLossScorer(
        model=None,
        tokenizer=FakeTokenizer(),
        cache_filename="scores.npy",
        cache_dir=str(tmp_path),
    )
    scores = np.array([0.2, np.nan, 0.7], dtype=np.float32)
    scorer._save(scores)

    np.testing.assert_array_equal(scorer._load_partial(3), scores)
    with pytest.raises(RuntimeError, match="does not match"):
        scorer._load_partial(4)
