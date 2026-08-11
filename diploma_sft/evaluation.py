"""Assistant-only evaluation matching the masked SFT objective."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence


def assistant_token_mask(
    input_ids: Sequence[int],
    user_marker_ids: Sequence[int],
    assistant_marker_ids: Sequence[int],
) -> List[bool]:
    """Mark assistant spans, including their turn-ending token, in ChatML token IDs."""
    mask = [False] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if list(input_ids[i : i + len(assistant_marker_ids)]) == list(assistant_marker_ids):
            start = i + len(assistant_marker_ids)
            end = len(input_ids)
            for j in range(start, len(input_ids) - len(user_marker_ids) + 1):
                if list(input_ids[j : j + len(user_marker_ids)]) == list(user_marker_ids):
                    end = j
                    break
            for position in range(start, end):
                mask[position] = True
            i = end
        else:
            i += 1
    return mask


def compute_assistant_only_perplexity(
    model: Any,
    tokenizer: Any,
    examples: Iterable[Dict[str, Any]],
    max_length: int,
) -> Dict[str, Any]:
    """Compute token-weighted loss and perplexity on assistant spans only."""
    import torch
    from tqdm.auto import tqdm

    user_marker_ids = tokenizer(
        "<|im_start|>user\n", add_special_tokens=False
    )["input_ids"]
    assistant_marker_ids = tokenizer(
        "<|im_start|>assistant\n", add_special_tokens=False
    )["input_ids"]
    total_loss = 0.0
    total_tokens = 0
    evaluated_examples = 0
    skipped_examples = 0

    for example in tqdm(examples, desc="assistant-only common eval"):
        encoded = tokenizer(
            example["text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        input_ids = encoded["input_ids"][0]
        mask = assistant_token_mask(
            input_ids.tolist(),
            user_marker_ids=user_marker_ids,
            assistant_marker_ids=assistant_marker_ids,
        )
        assistant_tokens = sum(mask)
        if assistant_tokens == 0:
            skipped_examples += 1
            continue

        labels = input_ids.clone()
        labels[:] = -100
        labels[torch.tensor(mask, device=labels.device, dtype=torch.bool)] = input_ids[
            torch.tensor(mask, device=input_ids.device, dtype=torch.bool)
        ]
        with torch.no_grad():
            output = model(**encoded, labels=labels.unsqueeze(0))
        total_loss += float(output.loss.item()) * assistant_tokens
        total_tokens += assistant_tokens
        evaluated_examples += 1

    if total_tokens == 0:
        raise RuntimeError("No assistant tokens found in the common evaluation holdout")
    mean_loss = total_loss / total_tokens
    return {
        "assistant_only_loss": mean_loss,
        "assistant_only_perplexity": math.exp(mean_loss),
        "assistant_tokens": total_tokens,
        "evaluated_examples": evaluated_examples,
        "skipped_examples": skipped_examples,
    }
