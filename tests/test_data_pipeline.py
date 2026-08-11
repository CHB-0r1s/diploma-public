from pathlib import Path

import numpy as np
import pytest

from diploma_sft.artifacts import (
    load_selection_artifact,
    prepare_scoring_cache_manifest,
    prepare_training_run_manifest,
    validate_adapter_lineage,
    write_selection_artifact,
)
from diploma_sft.data import (
    random_pool_indices,
    train_audit_pool_indices,
    validate_dataset_layout,
)
from diploma_sft.evaluation import assistant_token_mask
from diploma_sft.runtime import latest_checkpoint


def _manifest(pool_size=100, selected_count=20):
    return {
        "dataset": {"name": "example/dataset"},
        "layout": {
            "common_holdout_start": 0,
            "common_holdout_size": 10,
            "pool_start": 10,
            "pool_size": pool_size,
        },
        "selection": {"method": "random", "selected_count": selected_count},
    }


def test_dataset_layout_reserves_holdout_before_pool():
    validate_dataset_layout(dataset_size=110, common_holdout_size=10, pool_size=100)
    with pytest.raises(ValueError, match="exceeds dataset_size"):
        validate_dataset_layout(dataset_size=109, common_holdout_size=10, pool_size=100)


def test_random_pool_indices_are_deterministic_unique_and_in_bounds():
    first = random_pool_indices(pool_size=100, selected_count=20, seed=42)
    second = random_pool_indices(pool_size=100, selected_count=20, seed=42)

    np.testing.assert_array_equal(first, second)
    assert len(first) == len(np.unique(first)) == 20
    assert first.min() >= 0
    assert first.max() < 100


def test_random_pool_indices_reject_invalid_selection_size():
    with pytest.raises(ValueError, match="must be in"):
        random_pool_indices(pool_size=10, selected_count=11, seed=42)


def test_train_audit_indices_are_deterministic_subset():
    selected = random_pool_indices(pool_size=100, selected_count=20, seed=42)

    first = train_audit_pool_indices(selected, sample_limit=8, seed=7)
    second = train_audit_pool_indices(selected, sample_limit=8, seed=7)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 8
    assert set(first).issubset(set(selected))


def test_selection_artifact_round_trip(tmp_path: Path):
    indices = random_pool_indices(pool_size=100, selected_count=20, seed=42)
    paths = write_selection_artifact(tmp_path, indices, _manifest())

    manifest, loaded = load_selection_artifact(paths["manifest"])

    np.testing.assert_array_equal(loaded, indices)
    assert manifest["indices"]["coordinate_system"] == "selection_pool"
    assert manifest["indices"]["count"] == 20


def test_selection_artifact_detects_checksum_mismatch(tmp_path: Path):
    indices = random_pool_indices(pool_size=100, selected_count=20, seed=42)
    paths = write_selection_artifact(tmp_path, indices, _manifest())
    with paths["indices"].open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_selection_artifact(paths["manifest"])


def test_selection_artifact_rejects_pool_overlap(tmp_path: Path):
    indices = random_pool_indices(pool_size=100, selected_count=20, seed=42)
    manifest = _manifest()
    manifest["layout"]["pool_start"] = 9
    paths = write_selection_artifact(tmp_path, indices, manifest)

    with pytest.raises(ValueError, match="overlaps"):
        load_selection_artifact(paths["manifest"])


def test_assistant_token_mask_handles_multiple_turns():
    mask = assistant_token_mask(
        input_ids=[10, 1, 20, 2, 3, 10, 4, 20, 5],
        user_marker_ids=[10],
        assistant_marker_ids=[20],
    )

    assert mask == [False, False, False, True, True, False, False, False, True]


def test_latest_checkpoint_uses_numeric_step(tmp_path: Path):
    (tmp_path / "checkpoint-9").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-invalid").mkdir()

    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-100"


def test_resume_rejects_changed_training_protocol(tmp_path: Path):
    prepare_training_run_manifest(tmp_path, "selection-sha", {"seed": 42}, None)

    with pytest.raises(RuntimeError, match="different selection artifact or training protocol"):
        prepare_training_run_manifest(
            tmp_path,
            "selection-sha",
            {"seed": 43},
            tmp_path / "checkpoint-100",
        )


def test_adapter_lineage_rejects_different_selection(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    prepare_training_run_manifest(adapter_dir, "first-selection", {"seed": 42}, None)

    with pytest.raises(RuntimeError, match="different selection checksums"):
        validate_adapter_lineage(adapter_dir, "second-selection")


def test_score_cache_rejects_changed_protocol(tmp_path: Path):
    prepare_scoring_cache_manifest(tmp_path, {"model": "first"}, ("scores.npy",))

    with pytest.raises(RuntimeError, match="different scoring protocol"):
        prepare_scoring_cache_manifest(tmp_path, {"model": "second"}, ("scores.npy",))
