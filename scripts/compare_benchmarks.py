"""Paired comparison of two saved ruMMLU or MERA Core benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diploma_sft.paired_analysis import summarize_pairs  # noqa: E402


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _detect_benchmark(directory: Path) -> Tuple[str, str, str]:
    if (directory / "subjects").is_dir():
        return "rummlu", "subjects", "rummlu_metrics.json"
    if (directory / "tasks").is_dir():
        return "mera_core", "tasks", "mera_core_metrics.json"
    raise ValueError(f"Cannot detect benchmark layout in {directory}")


def _load_groups(directory: Path, subdirectory: str) -> Dict[str, Dict[str, Any]]:
    paths = sorted((directory / subdirectory).glob("*.json"))
    if not paths:
        raise ValueError(f"No prediction files found in {directory / subdirectory}")
    return {path.stem: _read_json(path) for path in paths}


def _align_predictions(
    baseline_groups: Dict[str, Dict[str, Any]],
    candidate_groups: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if baseline_groups.keys() != candidate_groups.keys():
        missing_candidate = sorted(baseline_groups.keys() - candidate_groups.keys())
        missing_baseline = sorted(candidate_groups.keys() - baseline_groups.keys())
        raise ValueError(
            "Benchmark groups differ: "
            f"missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
        )

    aligned = []
    for group in sorted(baseline_groups):
        baseline_rows = {
            int(row["index"]): row for row in baseline_groups[group]["predictions"]
        }
        candidate_rows = {
            int(row["index"]): row for row in candidate_groups[group]["predictions"]
        }
        if baseline_rows.keys() != candidate_rows.keys():
            raise ValueError(f"Prediction indices differ for {group}")
        for index in sorted(baseline_rows):
            baseline = baseline_rows[index]
            candidate = candidate_rows[index]
            if baseline["gold"] != candidate["gold"]:
                raise ValueError(f"Gold label differs for {group} index {index}")
            aligned.append(
                {
                    "group": group,
                    "index": index,
                    "gold": baseline["gold"],
                    "baseline_prediction": baseline["prediction"],
                    "candidate_prediction": candidate["prediction"],
                    "baseline_correct": bool(baseline["correct"]),
                    "candidate_correct": bool(candidate["correct"]),
                    "baseline_scores": baseline.get(
                        "choice_log_likelihoods",
                        baseline.get("candidate_log_likelihoods"),
                    ),
                    "candidate_scores": candidate.get(
                        "choice_log_likelihoods",
                        candidate.get("candidate_log_likelihoods"),
                    ),
                }
            )
    return aligned


def _lineage(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        name: metrics.get(name)
        for name in (
            "adapter_path",
            "selection_indices_sha256",
            "training_protocol_sha256",
            "benchmark_protocol_sha256",
            "dataset_name",
            "dataset_resolved_revision",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="random")
    parser.add_argument("--candidate-name", default="ifd")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_kind, baseline_subdir, baseline_metrics_name = _detect_benchmark(
        args.baseline_dir
    )
    candidate_kind, candidate_subdir, candidate_metrics_name = _detect_benchmark(
        args.candidate_dir
    )
    if baseline_kind != candidate_kind:
        raise ValueError(f"Benchmark types differ: {baseline_kind} vs {candidate_kind}")

    baseline_metrics = _read_json(args.baseline_dir / baseline_metrics_name)
    candidate_metrics = _read_json(args.candidate_dir / candidate_metrics_name)
    for field in ("dataset_name", "dataset_resolved_revision"):
        if baseline_metrics.get(field) != candidate_metrics.get(field):
            raise ValueError(f"Benchmark metadata differs for {field}")

    rows = _align_predictions(
        _load_groups(args.baseline_dir, baseline_subdir),
        _load_groups(args.candidate_dir, candidate_subdir),
    )
    group_results = {}
    for group in sorted({row["group"] for row in rows}):
        group_rows = [row for row in rows if row["group"] == group]
        group_results[group] = summarize_pairs(
            group_rows,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.seed,
        )
    report = {
        "benchmark": baseline_kind,
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "baseline_lineage": _lineage(baseline_metrics),
        "candidate_lineage": _lineage(candidate_metrics),
        "overall": summarize_pairs(
            rows,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.seed,
        ),
        "per_group": group_results,
        "bootstrap": {"resamples": args.bootstrap_resamples, "seed": args.seed},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "paired_analysis.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    disagreements_path = args.output_dir / "disagreements.jsonl"
    with disagreements_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            if row["baseline_correct"] != row["candidate_correct"]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report: {report_path}")
    print(f"Saved discordant examples: {disagreements_path}")


if __name__ == "__main__":
    main()
