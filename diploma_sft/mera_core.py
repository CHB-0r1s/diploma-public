"""Public labeled MERA Core task definitions and metric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


@dataclass(frozen=True)
class MeraTaskSpec:
    name: str
    evaluation_split: str
    candidates: Tuple[str, ...]
    num_fewshot: int
    reports_macro_f1: bool = False


TASK_SPECS = {
    "parus": MeraTaskSpec("parus", "validation", ("1", "2"), 0),
    "rcb": MeraTaskSpec("rcb", "validation", ("1", "2", "3"), 0, True),
    "rwsd": MeraTaskSpec("rwsd", "validation", ("Да", "Нет"), 0),
    "ruopenbookqa": MeraTaskSpec(
        "ruopenbookqa", "train", ("A", "B", "C", "D"), 5, True
    ),
    "ruworldtree": MeraTaskSpec(
        "ruworldtree", "train", ("A", "B", "C", "D"), 5, True
    ),
}


def format_instruction(example: Dict[str, Any]) -> str:
    return example["instruction"].format(**example["inputs"]).strip()


def format_science_question(example: Dict[str, Any]) -> str:
    inputs = example["inputs"]
    return (
        f"{inputs['question']}\n"
        f"A) {inputs['option_a']}\n"
        f"B) {inputs['option_b']}\n"
        f"C) {inputs['option_c']}\n"
        f"D) {inputs['option_d']}\n"
        "Ответ:"
    ).strip()


def build_prompt(
    task: str,
    example: Dict[str, Any],
    demonstrations: Iterable[Dict[str, Any]],
) -> str:
    """Build zero-shot or official-style five-shot public MERA prompts."""
    shots = list(demonstrations)
    if not shots:
        return format_instruction(example)
    if task not in {"ruopenbookqa", "ruworldtree"}:
        raise ValueError(f"Few-shot prompt formatting is not defined for {task}")

    rendered = []
    for index, shot in enumerate(shots):
        prompt = format_instruction(shot) if index == 0 else format_science_question(shot)
        rendered.append(f"{prompt} {shot['outputs']}")
    rendered.append(format_science_question(example))
    return "\n\n".join(rendered)


def macro_f1(
    gold: Iterable[str],
    predicted: Iterable[str],
    labels: Iterable[str],
) -> float:
    gold_values = list(gold)
    predicted_values = list(predicted)
    if len(gold_values) != len(predicted_values):
        raise ValueError("gold and predicted must have equal length")
    scores = []
    for label in labels:
        true_positive = sum(
            expected == label and actual == label
            for expected, actual in zip(gold_values, predicted_values)
        )
        false_positive = sum(
            expected != label and actual == label
            for expected, actual in zip(gold_values, predicted_values)
        )
        false_negative = sum(
            expected == label and actual != label
            for expected, actual in zip(gold_values, predicted_values)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def summarize_task(
    task: str,
    predictions: Iterable[Dict[str, Any]],
    result_file: str,
) -> Dict[str, Any]:
    rows = list(predictions)
    if not rows:
        raise ValueError(f"No predictions for MERA task {task}")
    spec = TASK_SPECS[task]
    correct = sum(bool(row["correct"]) for row in rows)
    summary: Dict[str, Any] = {
        "task": task,
        "accuracy": correct / len(rows),
        "correct": correct,
        "examples": len(rows),
        "truncated_prompts": sum(bool(row["prompt_truncated"]) for row in rows),
        "result_file": result_file,
    }
    if spec.reports_macro_f1:
        summary["macro_f1"] = macro_f1(
            (row["gold"] for row in rows),
            (row["prediction"] for row in rows),
            spec.candidates,
        )
    return summary


def aggregate_task_results(task_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(task_results)
    total = sum(int(row["examples"]) for row in rows)
    correct = sum(int(row["correct"]) for row in rows)
    if not rows or total == 0:
        raise ValueError("No completed MERA Core examples to aggregate")
    per_task = {row["task"]: float(row["accuracy"]) for row in rows}
    macro_f1_values = {
        row["task"]: float(row["macro_f1"])
        for row in rows
        if row.get("macro_f1") is not None
    }
    return {
        "accuracy": correct / total,
        "macro_task_accuracy": sum(per_task.values()) / len(per_task),
        "correct": correct,
        "examples": total,
        "tasks": len(per_task),
        "per_task_accuracy": per_task,
        "per_task_macro_f1": macro_f1_values,
    }
