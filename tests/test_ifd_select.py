"""Юнит-тесты чистой логики IFDSelector (без модели и без настоящего torch)."""

import types

import numpy as np
import pytest

from ifd_select import IFDSelector


class FakeTokenizer:
    """Минимальный токенизатор для __post_init__ и парсинга сегментов.

    Маркеры мапятся в фиксированные id, всё остальное — коды символов.
    """

    padding_side = "right"

    _MARKERS = {
        "<|im_start|>assistant\n": [1, 2],
        "<|im_start|>user\n": [3],
    }

    def __call__(self, text, add_special_tokens=True):
        ids = self._MARKERS.get(text, [ord(c) for c in text])
        return types.SimpleNamespace(input_ids=ids)


def make_selector(**kwargs) -> IFDSelector:
    return IFDSelector(model=None, tok=FakeTokenizer(), **kwargs)


# --------------------------------------------------------------------------- init

def test_post_init_requires_right_padding():
    tok = FakeTokenizer()
    tok.padding_side = "left"
    with pytest.raises(AssertionError):
        IFDSelector(model=None, tok=tok)


def test_post_init_tokenizes_markers():
    sel = make_selector()
    assert sel._asst_ids == [1, 2]
    assert sel._user_ids == [3]


# ------------------------------------------------------------------------- select

def test_select_topk_by_descending_ifd_returns_sorted_indices():
    sel = make_selector(ifd_threshold=1.0)
    scores = np.array([0.9, 0.5, 1.2, np.nan, 0.7])
    #   idx0=0.9, idx1=0.5, idx2=1.2(>=thr excl), idx3=nan(excl), idx4=0.7
    #   top-2 по убыванию IFD → {0, 4}, возвращается отсортированным
    out = sel.select(scores, k=2)
    assert out.tolist() == [0, 4]


def test_select_excludes_scores_at_or_above_threshold():
    sel = make_selector(ifd_threshold=0.8)
    scores = np.array([0.9, 0.5, 0.7])  # только idx1, idx2 проходят фильтр < 0.8
    out = sel.select(scores, k=2)
    assert out.tolist() == [1, 2]


def test_select_raises_when_too_few_selectable():
    sel = make_selector(ifd_threshold=1.0)
    scores = np.array([0.5, np.nan, 1.5])  # selectable = только idx0
    with pytest.raises(ValueError, match="осталось 1 < k=2"):
        sel.select(scores, k=2)


def test_select_all_nan_raises():
    sel = make_selector()
    scores = np.full(5, np.nan)
    with pytest.raises(ValueError):
        sel.select(scores, k=1)


# --------------------------------------------------------------- assistant parsing

def test_assistant_positions_finds_all_segments():
    sel = make_selector()
    #                0  1  2  3  4  5  6  7  8  9 10
    ids = [3, 5, 1, 2, 7, 8, 3, 9, 1, 2, 4]
    #  seg1: asst[1,2]@2 → targets ids[4],ids[5] → logit-позиции 3,4, стоп на user[3]@6
    #  seg2: asst[1,2]@8 → target ids[10] → logit-позиция 9 (до конца, user не найден)
    positions = sel._assistant_positions(ids, valid_len=len(ids))
    assert positions == {3, 4, 9}


def test_assistant_positions_no_marker_is_empty():
    sel = make_selector()
    assert sel._assistant_positions([5, 6, 7], valid_len=3) == set()


# --------------------------------------------------------------------- strip / lens

def test_strip_context_keeps_only_last_assistant():
    conv = [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "первый"},
        {"role": "user", "content": "ещё"},
        {"role": "assistant", "content": "последний"},
    ]
    out = IFDSelector._strip_context(conv)
    assert out == [{"role": "user", "content": ""}, {"role": "assistant", "content": "последний"}]


def test_strip_context_without_assistant_returns_empty_pair():
    out = IFDSelector._strip_context([{"role": "user", "content": "x"}])
    assert out == [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]


def test_char_len_sums_content():
    conv = [{"role": "user", "content": "abc"}, {"role": "assistant", "content": "de"}]
    assert IFDSelector._char_len(conv) == 5


# ---------------------------------------------------------------------- cache paths

def test_load_partial_no_file_returns_all_nan(tmp_path):
    sel = make_selector(cache_dir=str(tmp_path))
    arr = sel._load_partial("scores_cond.npy", n=4)
    assert arr.shape == (4,)
    assert np.isnan(arr).all()


def test_load_partial_reads_existing_file(tmp_path):
    sel = make_selector(cache_dir=str(tmp_path))
    saved = np.array([0.1, np.nan, 0.3, 0.4], dtype=np.float32)
    np.save(tmp_path / "scores_cond.npy", saved)
    loaded = sel._load_partial("scores_cond.npy", n=4)
    np.testing.assert_array_equal(loaded, saved)


def test_load_partial_size_mismatch_raises(tmp_path):
    sel = make_selector(cache_dir=str(tmp_path))
    np.save(tmp_path / "scores_cond.npy", np.zeros(3, dtype=np.float32))
    with pytest.raises(RuntimeError, match="!= 4"):
        sel._load_partial("scores_cond.npy", n=4)


def test_no_cache_dir_returns_all_nan_without_touching_disk():
    sel = make_selector()  # cache_dir=None
    arr = sel._load_partial("scores_cond.npy", n=3)
    assert np.isnan(arr).all()
