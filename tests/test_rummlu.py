from pathlib import Path

import pytest

from diploma_sft.artifacts import prepare_benchmark_run_manifest
from diploma_sft.rummlu import (
    aggregate_subject_results,
    build_five_shot_prompt,
    format_question,
)


def _example(index: int, output: str = "A"):
    return {
        "instruction": "Тема {subject}\n{text}\nA {option_a}\nB {option_b}\nC {option_c}\nD {option_d}\nОтвет:",
        "inputs": {
            "subject": "тест",
            "text": f"Вопрос {index}",
            "option_a": "один",
            "option_b": "два",
            "option_c": "три",
            "option_d": "четыре",
        },
        "outputs": output,
    }


def test_five_shot_prompt_uses_instruction_once_and_target_last():
    demonstrations = [_example(index, output="ABCD"[index % 4]) for index in range(5)]

    prompt = build_five_shot_prompt(_example(99), demonstrations)

    assert prompt.count("Тема тест") == 1
    assert "Ответ: A\n\nВопрос 1" in prompt
    assert prompt.endswith(format_question(_example(99)))
    assert not prompt.endswith(" A")


def test_aggregate_subject_results_reports_micro_and_macro_accuracy():
    metrics = aggregate_subject_results(
        [
            {"subject": "large", "correct": 8, "examples": 10, "accuracy": 0.8},
            {"subject": "small", "correct": 0, "examples": 2, "accuracy": 0.0},
        ]
    )

    assert metrics["accuracy"] == pytest.approx(8 / 12)
    assert metrics["macro_subject_accuracy"] == pytest.approx(0.4)
    assert metrics["subjects"] == 2


def test_benchmark_manifest_rejects_changed_protocol(tmp_path: Path):
    prepare_benchmark_run_manifest(tmp_path, {"adapter": "first"})

    with pytest.raises(RuntimeError, match="different protocol"):
        prepare_benchmark_run_manifest(tmp_path, {"adapter": "second"})
