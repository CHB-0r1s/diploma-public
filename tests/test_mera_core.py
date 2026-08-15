import pytest

from diploma_sft.mera_core import (
    TASK_SPECS,
    aggregate_task_results,
    build_prompt,
    format_science_question,
    macro_f1,
    summarize_task,
)


def _science_example(index: int, output: str = "A"):
    return {
        "instruction": "Вопрос: {question}\nA {option_a}\nB {option_b}\nC {option_c}\nD {option_d}\nОтвет:",
        "inputs": {
            "question": f"Научный вопрос {index}",
            "option_a": "один",
            "option_b": "два",
            "option_c": "три",
            "option_d": "четыре",
        },
        "outputs": output,
    }


def test_zero_shot_prompt_renders_dataset_instruction():
    example = {
        "instruction": "Ситуация: {premise}\nОтвет:",
        "inputs": {"premise": "тест"},
        "outputs": "1",
    }

    assert build_prompt("parus", example, []) == "Ситуация: тест\nОтвет:"


def test_science_five_shot_prompt_excludes_target_answer():
    demonstrations = [
        _science_example(index, output="ABCD"[index % 4]) for index in range(5)
    ]
    target = _science_example(99)

    prompt = build_prompt("ruopenbookqa", target, demonstrations)

    assert prompt.count("Вопрос:") == 1
    assert "Ответ: A\n\nНаучный вопрос 1" in prompt
    assert prompt.endswith(format_science_question(target))
    assert not prompt.endswith(" A")


def test_macro_f1_averages_all_declared_labels():
    score = macro_f1(
        gold=["A", "A", "B", "B"],
        predicted=["A", "B", "B", "B"],
        labels=["A", "B"],
    )

    assert score == pytest.approx((2 / 3 + 0.8) / 2)


def test_task_and_suite_summaries_report_expected_metrics():
    predictions = [
        {"gold": "1", "prediction": "1", "correct": True, "prompt_truncated": False},
        {"gold": "2", "prediction": "1", "correct": False, "prompt_truncated": True},
        {"gold": "3", "prediction": "3", "correct": True, "prompt_truncated": False},
    ]
    rcb = summarize_task("rcb", predictions, "rcb.json")
    parus = {
        "task": "parus",
        "accuracy": 0.5,
        "correct": 1,
        "examples": 2,
    }

    assert rcb["accuracy"] == pytest.approx(2 / 3)
    assert "macro_f1" in rcb
    aggregate = aggregate_task_results([rcb, parus])
    assert aggregate["accuracy"] == pytest.approx(3 / 5)
    assert aggregate["macro_task_accuracy"] == pytest.approx(((2 / 3) + 0.5) / 2)


def test_core_task_specs_use_only_public_labeled_splits():
    assert TASK_SPECS["parus"].evaluation_split == "validation"
    assert TASK_SPECS["rcb"].evaluation_split == "validation"
    assert TASK_SPECS["rwsd"].evaluation_split == "validation"
    assert TASK_SPECS["ruopenbookqa"].evaluation_split == "train"
    assert TASK_SPECS["ruworldtree"].evaluation_split == "train"
