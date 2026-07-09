"""ifd_select — микро-либа для отбора данных по IFD (Instruction-Following Difficulty).

IFD(x, y) = PPL(y | x) / PPL(y)
    PPL(y | x) — перплексия assistant-ответа y при видимой инструкции x
    PPL(y)     — перплексия того же ответа y без контекста (пустой user + ответ)

Отбираются top-K примеров среди тех, у кого IFD < threshold (шум с IFD >= 1
отбрасывается автоматически — см. Li et al. 2023, Cherry_LLM / Superfiltering).

Ядро model-agnostic: принимает готовые (model, tok) — любой causal-LM + токенизатор,
у которого УЖЕ настроен chat template (напр. через unsloth.get_chat_template(..., "qwen-2.5")).
Дефолтные маркеры — под Qwen-2.5, но конфигурируемы.

Пример
------
    from ifd_select import IFDSelector

    sel = IFDSelector(model, tok, cache_dir=OUTPUT_DIR)          # резюмируемый кеш
    convs = [pool_ds[i]["conversations"] for i in range(len(pool_ds))]
    idx = sel.fit_select(convs, k=90_000)                        # np.array индексов в convs

    # или по шагам:
    ifd = sel.score(convs)                                       # np.array IFD (с NaN где не вышло)
    idx = sel.select(ifd, k=90_000)

Один conversation = список сообщений [{"role": ..., "content": ...}, ...]
с финальным assistant-ответом.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm не обязателен
    def tqdm(x=None, **_):
        return x if x is not None else _NullBar()

    class _NullBar:
        def update(self, *a, **k):
            pass

        def close(self):
            pass


Conversation = list  # list[dict[str, str]]


@dataclass
class IFDSelector:
    """Скоринг IFD и отбор top-K. См. модульный docstring."""

    model: object
    tok: object
    max_seq_len: int = 2048
    batch_size: int = 8
    ifd_threshold: float = 1.0
    assistant_marker: str = "<|im_start|>assistant\n"
    user_marker: str = "<|im_start|>user\n"
    cache_dir: str | None = None
    checkpoint_every: int = 10_000

    def __post_init__(self):
        # Батчи с padding + поиск assistant-сегментов корректны только при right-padding.
        assert self.tok.padding_side == "right", (
            f"tok.padding_side={self.tok.padding_side!r} — нужен 'right'. "
            "Установи tok.padding_side = 'right' перед скорингом."
        )
        self._asst_ids = self.tok(self.assistant_marker, add_special_tokens=False).input_ids
        self._user_ids = self.tok(self.user_marker, add_special_tokens=False).input_ids
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ scoring

    def score(self, conversations: list[Conversation]) -> np.ndarray:
        """IFD-скор на каждый пример. Резюмируется из cache_dir, если задан.

        Возвращает np.array длины len(conversations); NaN там, где ответ не найден
        или батч упал.
        """
        cond = self._score_mode(conversations, "conditional", "scores_cond.npy")
        uncond = self._score_mode(conversations, "unconditional", "scores_uncond.npy")
        with np.errstate(divide="ignore", invalid="ignore"):
            ifd = cond / uncond
        self._save("scores_ifd.npy", ifd)
        return ifd

    def _score_mode(self, conversations, mode, cache_name) -> np.ndarray:
        n = len(conversations)
        scores = self._load_partial(cache_name, n)
        todo = np.where(np.isnan(scores))[0].tolist()
        if not todo:
            return scores

        # length-bucketing: похожие длины в один батч → меньше паддинга.
        char_lens = [self._char_len(conversations[i]) for i in todo]
        todo = [i for _, i in sorted(zip(char_lens, todo))]

        path = self._path(cache_name)
        pbar = tqdm(total=len(todo), desc=f"IFD {mode} (batch={self.batch_size})")
        since_save = 0
        for start in range(0, len(todo), self.batch_size):
            idx = todo[start : start + self.batch_size]
            batch = [conversations[i] for i in idx]
            try:
                ppls = self._ppl_batch(batch, mode)
            except Exception as e:  # OOM и т.п. — не роняем весь прогон
                print(f"\n[!] Батч idx={idx[0]}..{idx[-1]}: {type(e).__name__}: {e}")
                ppls = [float("nan")] * len(idx)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            for i, p in zip(idx, ppls):
                scores[i] = p
            if hasattr(pbar, "update"):
                pbar.update(len(idx))
            since_save += len(idx)
            if path and since_save >= self.checkpoint_every:
                np.save(path, scores)
                since_save = 0
        if hasattr(pbar, "close"):
            pbar.close()
        if path:
            np.save(path, scores)
        nan_n = int(np.isnan(scores).sum())
        print(f"IFD {mode}: готово. NaN {nan_n}/{n} ({100 * nan_n / max(n, 1):.2f}%)")
        return scores

    @torch.no_grad()
    def _ppl_batch(self, conversations: list[Conversation], mode: str) -> list[float]:
        """PPL = exp(mean CE) на assistant-токенах для батча.

        mode='conditional'   — полный диалог как есть.
        mode='unconditional' — пустой user + только финальный assistant.
        """
        if mode == "conditional":
            msgs = conversations
        elif mode == "unconditional":
            msgs = [self._strip_context(c) for c in conversations]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        texts = [
            self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in msgs
        ]
        enc = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        logits = self.model(**enc).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)

        results = []
        for b in range(len(conversations)):
            ids = enc["input_ids"][b].tolist()
            valid_len = int(enc["attention_mask"][b].sum().item())
            positions = self._assistant_positions(ids, valid_len)
            if not positions:
                results.append(float("nan"))
                continue
            pos_list = sorted(positions)
            targets = torch.tensor([ids[p + 1] for p in pos_list], device=log_probs.device)
            pos_t = torch.tensor(pos_list, device=log_probs.device)
            nll = -log_probs[b, pos_t, targets]
            results.append(float(np.exp(nll.mean().item())))
        return results

    def _assistant_positions(self, ids: list[int], valid_len: int) -> set[int]:
        """Позиции (индексы логитов) для предсказания assistant-токенов.

        Для assistant-сегмента [start, end) целевые токены — ids[start..end),
        а логиты, их предсказывающие, стоят на позициях k-1.
        """
        asst, user = self._asst_ids, self._user_ids
        positions: set[int] = set()
        i = 0
        while i < valid_len:
            if ids[i : i + len(asst)] == asst:
                start = i + len(asst)
                end = valid_len
                for j in range(start, valid_len - len(user) + 1):
                    if ids[j : j + len(user)] == user:
                        end = j
                        break
                for k in range(start, end):
                    if 0 < k <= valid_len - 1:
                        positions.add(k - 1)
                i = end
            else:
                i += 1
        return positions

    @staticmethod
    def _strip_context(conversation: Conversation) -> Conversation:
        """Пустой user + только финальный assistant — для PPL(y) без контекста."""
        assistants = [m for m in conversation if m.get("role") == "assistant"]
        if not assistants:
            return [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]
        return [{"role": "user", "content": ""}, assistants[-1]]

    @staticmethod
    def _char_len(conversation: Conversation) -> int:
        return sum(len(m.get("content", "")) for m in conversation)

    # ---------------------------------------------------------------- selection

    def select(self, scores: np.ndarray, k: int) -> np.ndarray:
        """Top-K индексов среди примеров с IFD < threshold.

        Возвращает отсортированные по возрастанию индексы (стабильно к порядку в датасете).
        """
        scores = np.asarray(scores)
        valid = ~np.isnan(scores)
        selectable = valid & (scores < self.ifd_threshold)
        n_selectable = int(selectable.sum())
        if n_selectable < k:
            raise ValueError(
                f"После фильтра IFD < {self.ifd_threshold} осталось {n_selectable} < k={k}. "
                "Подними ifd_threshold или уменьши k."
            )
        cand = np.where(selectable)[0]
        top_local = np.argsort(-scores[cand])[:k]  # top-K по убыванию IFD
        return np.sort(cand[top_local])

    def fit_select(self, conversations: list[Conversation], k: int) -> np.ndarray:
        """score() + select() одним вызовом."""
        return self.select(self.score(conversations), k)

    # ------------------------------------------------------------------- cache

    def _path(self, name: str) -> str | None:
        return os.path.join(self.cache_dir, name) if self.cache_dir else None

    def _save(self, name: str, arr: np.ndarray) -> None:
        path = self._path(name)
        if path:
            np.save(path, arr)

    def _load_partial(self, name: str, n: int) -> np.ndarray:
        path = self._path(name)
        if path and os.path.exists(path):
            scores = np.load(path)
            if len(scores) != n:
                raise RuntimeError(
                    f"Размер кеша {path} ({len(scores)}) != {n}. Удали файл для пересчёта."
                )
            done = int((~np.isnan(scores)).sum())
            print(f"Найдено {done}/{n} посчитанных скоров в {path}")
            return scores.astype(np.float32)
        return np.full(n, np.nan, dtype=np.float32)
