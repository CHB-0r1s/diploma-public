import math

import pytest

from diploma_sft.training_metrics import assistant_token_entropy


def test_assistant_token_entropy_uses_only_supervised_shifted_positions():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "tensor"):
        pytest.skip("real torch is not installed")
    logits = torch.zeros((1, 4, 2), dtype=torch.float32)
    labels = torch.tensor([[-100, 0, -100, 1]])

    entropy = assistant_token_entropy(logits, labels, chunk_size=1)

    assert entropy == pytest.approx(math.log(2))


def test_assistant_token_entropy_returns_none_without_supervised_tokens():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "tensor"):
        pytest.skip("real torch is not installed")
    logits = torch.zeros((1, 3, 2), dtype=torch.float32)
    labels = torch.full((1, 3), -100)

    assert assistant_token_entropy(logits, labels) is None
