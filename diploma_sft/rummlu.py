"""Pure helpers for the public five-shot ruMMLU evaluation protocol."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

CHOICES = ("A", "B", "C", "D")


def format_instruction(example: Dict[str, Any]) -> str:
    """Render the instruction supplied by the benchmark dataset."""
    return example["instruction"].format(**example["inputs"]).strip()


def format_question(example: Dict[str, Any]) -> str:
    """Render the instruction-free form used after the first demonstration."""
    inputs = example["inputs"]
    return (
        f"{inputs['text']}\n"
        f"A) {inputs['option_a']}\n"
        f"B) {inputs['option_b']}\n"
        f"C) {inputs['option_c']}\n"
        f"D) {inputs['option_d']}\n"
        "Ответ:"
    ).strip()


def build_five_shot_prompt(
    example: Dict[str, Any],
    demonstrations: Iterable[Dict[str, Any]],
) -> str:
    """Match the official MERA ruMMLU few-shot context construction."""
    shots = list(demonstrations)
    if not shots:
        return format_instruction(example)

    rendered: List[str] = []
    for index, shot in enumerate(shots):
        prompt = format_instruction(shot) if index == 0 else format_question(shot)
        rendered.append(f"{prompt} {shot['outputs']}")
    rendered.append(format_question(example))
    return "\n\n".join(rendered)


def aggregate_subject_results(subject_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute micro and macro accuracy from completed subject summaries."""
    rows = list(subject_results)
    total = sum(int(row["examples"]) for row in rows)
    correct = sum(int(row["correct"]) for row in rows)
    if not rows or total == 0:
        raise ValueError("No completed ruMMLU examples to aggregate")
    per_subject = {row["subject"]: float(row["accuracy"]) for row in rows}
    return {
        "accuracy": correct / total,
        "macro_subject_accuracy": sum(per_subject.values()) / len(per_subject),
        "correct": correct,
        "examples": total,
        "subjects": len(per_subject),
        "per_subject_accuracy": per_subject,
    }
