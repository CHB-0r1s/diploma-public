"""Resumable assistant-only loss scoring for RHO-Loss data selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch

from diploma_sft.evaluation import assistant_token_mask

Conversation = list


def select_highest_rho(
    base_losses: np.ndarray,
    irreducible_losses: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dataset-sorted top-k indices and reducible-loss scores."""
    base = np.asarray(base_losses)
    irreducible = np.asarray(irreducible_losses)
    if base.shape != irreducible.shape:
        raise ValueError("Base and irreducible losses must have matching shapes")
    rho = base - irreducible
    valid_indices = np.flatnonzero(np.isfinite(rho))
    if len(valid_indices) < k:
        raise ValueError(f"Only {len(valid_indices)} valid RHO scores remain for k={k}")
    ranked = np.argsort(-rho[valid_indices], kind="stable")[:k]
    return np.sort(valid_indices[ranked]), rho.astype(np.float32, copy=False)


@dataclass
class AssistantLossScorer:
    """Score mean cross-entropy over assistant response tokens."""

    model: Any
    tokenizer: Any
    cache_filename: str
    max_seq_len: int = 2048
    batch_size: int = 8
    assistant_marker: str = "<|im_start|>assistant\n"
    user_marker: str = "<|im_start|>user\n"
    cache_dir: Optional[str] = None
    checkpoint_every: int = 10_000

    def __post_init__(self) -> None:
        if self.tokenizer.padding_side != "right":
            raise ValueError("Assistant loss scoring requires right-padding")
        self._assistant_ids = self.tokenizer(
            self.assistant_marker,
            add_special_tokens=False,
        ).input_ids
        self._user_ids = self.tokenizer(
            self.user_marker,
            add_special_tokens=False,
        ).input_ids
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def score(self, conversations: Sequence[Conversation], desc: str) -> np.ndarray:
        scores = self._load_partial(len(conversations))
        pending = np.flatnonzero(np.isnan(scores)).tolist()
        if not pending:
            return scores
        pending.sort(key=lambda index: self._char_len(conversations[index]))

        from tqdm.auto import tqdm

        progress = tqdm(total=len(pending), desc=f"{desc} (batch={self.batch_size})")
        since_save = 0
        for start in range(0, len(pending), self.batch_size):
            indices = pending[start : start + self.batch_size]
            batch = [conversations[index] for index in indices]
            try:
                batch_scores = self._loss_batch(batch)
            except Exception as exc:
                print(
                    f"\n[!] {desc} batch idx={indices[0]}..{indices[-1]}: "
                    f"{type(exc).__name__}: {exc}"
                )
                batch_scores = [float("nan")] * len(indices)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            for index, value in zip(indices, batch_scores):
                scores[index] = value
            progress.update(len(indices))
            since_save += len(indices)
            if since_save >= self.checkpoint_every:
                self._save(scores)
                since_save = 0
        progress.close()
        self._save(scores)
        missing = int(np.isnan(scores).sum())
        print(
            f"{desc} scoring complete. NaN {missing}/{len(scores)} "
            f"({100 * missing / max(len(scores), 1):.2f}%)"
        )
        return scores

    @torch.no_grad()
    def _loss_batch(self, conversations: Sequence[Conversation]) -> list[float]:
        texts = [
            self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            for conversation in conversations
        ]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        encoded = {name: value.to(self.model.device) for name, value in encoded.items()}
        logits = self.model(**encoded, use_cache=False, return_dict=True).logits
        if not hasattr(logits, "shape"):
            raise RuntimeError("Assistant loss scorer did not return tensor logits")

        results = []
        for batch_index in range(len(conversations)):
            valid_length = int(encoded["attention_mask"][batch_index].sum().item())
            token_ids = encoded["input_ids"][batch_index, :valid_length].tolist()
            target_mask = assistant_token_mask(
                token_ids,
                user_marker_ids=self._user_ids,
                assistant_marker_ids=self._assistant_ids,
            )
            target_positions = [
                index for index, selected in enumerate(target_mask) if selected and index
            ]
            if not target_positions:
                results.append(float("nan"))
                continue
            logit_positions = torch.tensor(
                [index - 1 for index in target_positions],
                device=logits.device,
            )
            targets = torch.tensor(
                [token_ids[index] for index in target_positions],
                device=logits.device,
            )
            token_logits = logits[batch_index, logit_positions].float()
            losses = torch.nn.functional.cross_entropy(
                token_logits,
                targets,
                reduction="none",
            )
            results.append(float(losses.mean().item()))
        return results

    def _cache_path(self) -> Optional[str]:
        return os.path.join(self.cache_dir, self.cache_filename) if self.cache_dir else None

    def _save(self, scores: np.ndarray) -> None:
        path = self._cache_path()
        if not path:
            return
        temporary = f"{path}.tmp"
        with open(temporary, "wb") as stream:
            np.save(stream, scores, allow_pickle=False)
        os.replace(temporary, path)

    def _load_partial(self, size: int) -> np.ndarray:
        path = self._cache_path()
        if path and os.path.exists(path):
            scores = np.load(path, allow_pickle=False)
            if len(scores) != size:
                raise RuntimeError(f"RHO cache size {len(scores)} does not match {size}")
            complete = int(np.isfinite(scores).sum())
            print(f"Found {complete}/{size} cached assistant losses in {path}")
            return scores.astype(np.float32)
        return np.full(size, np.nan, dtype=np.float32)

    @staticmethod
    def _char_len(conversation: Conversation) -> int:
        return sum(len(message.get("content", "")) for message in conversation)
