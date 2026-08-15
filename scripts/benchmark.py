"""Run a reproducible external benchmark against a trained LoRA adapter.

Example:
    python scripts/benchmark.py benchmark=rummlu adapter_path=outputs/run/adapter
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import hydra
    from hydra.utils import to_absolute_path
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing experiment dependencies. Install them with: "
        'python -m pip install -e ".[experiments]"'
    ) from exc

from diploma_sft.artifacts import (  # noqa: E402
    prepare_benchmark_run_manifest,
    sha256_file,
)
from diploma_sft.config import to_plain_dict  # noqa: E402
from diploma_sft.data import resolve_dataset_revision  # noqa: E402
from diploma_sft.mera_core import (  # noqa: E402
    TASK_SPECS,
    aggregate_task_results,
    summarize_task,
)
from diploma_sft.mera_core import build_prompt as build_mera_prompt  # noqa: E402
from diploma_sft.rummlu import (  # noqa: E402
    CHOICES,
    aggregate_subject_results,
    build_five_shot_prompt,
)
from diploma_sft.runtime import environment_snapshot, require_bf16_cuda  # noqa: E402
from diploma_sft.wandb_utils import init_wandb, log_files_as_artifact  # noqa: E402


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _load_adapter_manifest(adapter_path: Path) -> Dict[str, Any]:
    path = adapter_path / "training_run_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"Adapter is missing training lineage: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _score_prompts(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    candidates: Sequence[str],
    max_length: int,
) -> Tuple[List[List[float]], List[bool]]:
    """Score candidate continuations and return log-likelihoods plus truncation flags."""
    import torch
    import torch.nn.functional as F

    candidate_ids = [
        tokenizer(choice, add_special_tokens=False)["input_ids"]
        for choice in candidates
    ]
    prompt_sequences: List[List[int]] = []
    truncated_flags: List[bool] = []
    max_candidate_length = max(len(ids) for ids in candidate_ids)
    for prompt in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        available = max_length - max_candidate_length
        was_truncated = len(prompt_ids) > available
        if was_truncated:
            prompt_ids = prompt_ids[-available:]
        prompt_sequences.append(prompt_ids)
        truncated_flags.append(was_truncated)

    # Qwen encodes each answer label as one token. In that common
    # case one prompt forward supplies all four continuation probabilities.
    if all(len(ids) == 1 for ids in candidate_ids):
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        width = max(len(sequence) for sequence in prompt_sequences)
        input_ids = torch.full(
            (len(prompt_sequences), width),
            int(pad_id),
            dtype=torch.long,
            device=model.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(prompt_sequences):
            size = len(sequence)
            input_ids[row, :size] = torch.tensor(sequence, device=model.device)
            attention_mask[row, :size] = 1
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        choice_token_ids = torch.tensor(
            [ids[0] for ids in candidate_ids],
            device=logits.device,
        )
        scores = []
        for row, sequence in enumerate(prompt_sequences):
            next_token_log_probs = F.log_softmax(
                logits[row, len(sequence) - 1].float(),
                dim=-1,
            )
            scores.append(next_token_log_probs[choice_token_ids].tolist())
        return scores, truncated_flags

    sequences: List[List[int]] = []
    prompt_lengths: List[int] = []
    candidate_lengths: List[int] = []
    for prompt_ids in prompt_sequences:
        for ids in candidate_ids:
            sequences.append(prompt_ids + ids)
            prompt_lengths.append(len(prompt_ids))
            candidate_lengths.append(len(ids))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), width),
        int(pad_id),
        dtype=torch.long,
        device=model.device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        size = len(sequence)
        input_ids[row, :size] = torch.tensor(sequence, device=model.device)
        attention_mask[row, :size] = 1

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    flat_scores: List[float] = []
    for row, (prompt_length, candidate_length) in enumerate(
        zip(prompt_lengths, candidate_lengths)
    ):
        positions = torch.arange(
            prompt_length - 1,
            prompt_length + candidate_length - 1,
            device=logits.device,
        )
        targets = input_ids[row, prompt_length : prompt_length + candidate_length]
        token_log_probs = F.log_softmax(logits[row, positions].float(), dim=-1)
        flat_scores.append(
            float(token_log_probs.gather(1, targets.unsqueeze(1)).sum().item())
        )
    width = len(candidates)
    return [flat_scores[i : i + width] for i in range(0, len(flat_scores), width)], truncated_flags


def _evaluate_subject(
    subject: str,
    dev: Any,
    test: Any,
    model: Any,
    tokenizer: Any,
    cfg: DictConfig,
    output_path: Path,
    protocol_sha256: str,
) -> Dict[str, Any]:
    expected = len(test)
    payload: Dict[str, Any] = {
        "subject": subject,
        "protocol_sha256": protocol_sha256,
        "expected_examples": expected,
        "predictions": [],
    }
    if output_path.exists():
        with output_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError(f"Cached subject {subject} belongs to a different protocol")
        if int(payload.get("expected_examples", -1)) != expected:
            raise RuntimeError(f"Cached subject {subject} has a different test size")

    predictions = payload["predictions"]
    if len(predictions) > expected:
        raise RuntimeError(f"Cached ruMMLU subject {subject} has too many predictions")
    cached_indices = [int(row["index"]) for row in predictions]
    expected_indices = list(range(len(predictions)))
    if cached_indices != expected_indices:
        raise RuntimeError(f"Cached ruMMLU subject {subject} has non-sequential indices")
    start = len(predictions)
    if len(dev) < int(cfg.benchmark.num_fewshot):
        raise RuntimeError(
            f"Subject {subject} has only {len(dev)} dev examples for "
            f"{cfg.benchmark.num_fewshot}-shot evaluation"
        )
    demonstrations = [dev[index] for index in range(int(cfg.benchmark.num_fewshot))]
    checkpoint_every = int(cfg.benchmark.checkpoint_every)
    last_checkpoint = start
    from tqdm.auto import tqdm

    for batch_start in tqdm(
        range(start, expected, int(cfg.benchmark.batch_size)),
        desc=f"ruMMLU {subject}",
    ):
        batch_end = min(batch_start + int(cfg.benchmark.batch_size), expected)
        examples = [test[index] for index in range(batch_start, batch_end)]
        prompts = [build_five_shot_prompt(example, demonstrations) for example in examples]
        scores, truncated_flags = _score_prompts(
            model,
            tokenizer,
            prompts,
            candidates=CHOICES,
            max_length=int(cfg.benchmark.max_seq_len),
        )
        for offset, (example, choice_scores, was_truncated) in enumerate(
            zip(examples, scores, truncated_flags)
        ):
            prediction = CHOICES[max(range(4), key=lambda index: choice_scores[index])]
            predictions.append(
                {
                    "index": batch_start + offset,
                    "gold": example["outputs"],
                    "prediction": prediction,
                    "correct": prediction == example["outputs"],
                    "choice_log_likelihoods": dict(zip(CHOICES, choice_scores)),
                    "prompt_truncated": was_truncated,
                }
            )
        if len(predictions) - last_checkpoint >= checkpoint_every:
            _atomic_write_json(output_path, payload)
            last_checkpoint = len(predictions)

    _atomic_write_json(output_path, payload)
    correct = sum(bool(row["correct"]) for row in predictions)
    return {
        "subject": subject,
        "accuracy": correct / expected,
        "correct": correct,
        "examples": expected,
        "truncated_prompts": sum(bool(row["prompt_truncated"]) for row in predictions),
        "result_file": output_path.name,
    }


def _evaluate_mera_task(
    task: str,
    dataset: Any,
    model: Any,
    tokenizer: Any,
    cfg: DictConfig,
    output_path: Path,
    protocol_sha256: str,
) -> Dict[str, Any]:
    spec = TASK_SPECS[task]
    demonstrations = [dataset[index] for index in range(spec.num_fewshot)]
    evaluation = dataset.select(range(spec.num_fewshot, len(dataset)))
    if cfg.benchmark.max_samples_per_task is not None:
        evaluation = evaluation.select(
            range(min(int(cfg.benchmark.max_samples_per_task), len(evaluation)))
        )
    expected = len(evaluation)
    if expected == 0:
        raise RuntimeError(f"No labeled evaluation examples remain for MERA task {task}")

    payload: Dict[str, Any] = {
        "task": task,
        "protocol_sha256": protocol_sha256,
        "expected_examples": expected,
        "predictions": [],
    }
    if output_path.exists():
        with output_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError(f"Cached MERA task {task} belongs to a different protocol")
        if int(payload.get("expected_examples", -1)) != expected:
            raise RuntimeError(f"Cached MERA task {task} has a different evaluation size")

    predictions = payload["predictions"]
    if len(predictions) > expected:
        raise RuntimeError(f"Cached MERA task {task} has too many predictions")
    cached_indices = [int(row["index"]) for row in predictions]
    expected_indices = list(range(spec.num_fewshot, spec.num_fewshot + len(predictions)))
    if cached_indices != expected_indices:
        raise RuntimeError(f"Cached MERA task {task} has non-sequential indices")
    start = len(predictions)
    checkpoint_every = int(cfg.benchmark.checkpoint_every)
    last_checkpoint = start
    from tqdm.auto import tqdm

    for batch_start in tqdm(
        range(start, expected, int(cfg.benchmark.batch_size)),
        desc=f"MERA Core {task}",
    ):
        batch_end = min(batch_start + int(cfg.benchmark.batch_size), expected)
        examples = [evaluation[index] for index in range(batch_start, batch_end)]
        for example in examples:
            if not example["outputs"]:
                raise RuntimeError(f"MERA task {task} evaluation split contains hidden labels")
        prompts = [
            build_mera_prompt(task, example, demonstrations) for example in examples
        ]
        scores, truncated_flags = _score_prompts(
            model,
            tokenizer,
            prompts,
            candidates=spec.candidates,
            max_length=int(cfg.benchmark.max_seq_len),
        )
        for offset, (example, candidate_scores, was_truncated) in enumerate(
            zip(examples, scores, truncated_flags)
        ):
            prediction = spec.candidates[
                max(range(len(spec.candidates)), key=lambda index: candidate_scores[index])
            ]
            predictions.append(
                {
                    "index": batch_start + offset + spec.num_fewshot,
                    "gold": example["outputs"],
                    "prediction": prediction,
                    "correct": prediction == example["outputs"],
                    "candidate_log_likelihoods": dict(
                        zip(spec.candidates, candidate_scores)
                    ),
                    "prompt_truncated": was_truncated,
                }
            )
        if len(predictions) - last_checkpoint >= checkpoint_every:
            _atomic_write_json(output_path, payload)
            last_checkpoint = len(predictions)

    _atomic_write_json(output_path, payload)
    return summarize_task(task, predictions, output_path.name)


def _run_mera_core(
    cfg: DictConfig,
    adapter_path: Path,
    adapter_manifest: Dict[str, Any],
    adapter_config: Dict[str, Any],
    adapter_weights_path: Path,
    output_dir: Path,
) -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    tasks = list(cfg.benchmark.tasks)
    unknown = sorted(set(tasks) - set(TASK_SPECS))
    if unknown:
        raise ValueError(f"Unknown MERA Core tasks: {unknown}")
    resolved_revision = resolve_dataset_revision(
        cfg.benchmark.dataset_name,
        cfg.benchmark.revision,
    )
    base_model_name = adapter_config["base_model_name_or_path"]
    base_model_revision = HfApi().model_info(repo_id=base_model_name).sha
    environment = environment_snapshot(REPO_ROOT)
    task_datasets = {}
    task_protocols = {}
    for task in tasks:
        spec = TASK_SPECS[task]
        dataset = load_dataset(
            cfg.benchmark.dataset_name,
            task,
            split=spec.evaluation_split,
            revision=resolved_revision,
        )
        labeled_rows = sum(bool(row["outputs"]) for row in dataset)
        if labeled_rows != len(dataset):
            raise RuntimeError(
                f"MERA task {task} split {spec.evaluation_split} has "
                f"{len(dataset) - labeled_rows} hidden labels"
            )
        task_datasets[task] = dataset
        task_protocols[task] = {
            "evaluation_split": TASK_SPECS[task].evaluation_split,
            "candidates": list(TASK_SPECS[task].candidates),
            "num_fewshot": TASK_SPECS[task].num_fewshot,
            "exclude_demonstrations_from_evaluation": True,
            "reports_macro_f1": TASK_SPECS[task].reports_macro_f1,
            "source_rows": len(dataset),
            "evaluation_rows_before_limit": len(dataset) - spec.num_fewshot,
            "fingerprint": getattr(dataset, "_fingerprint", None),
        }
    protocol = {
        "implementation": "public-labeled-mera-core-loglikelihood-v1",
        "implementation_sha256": {
            "runner": sha256_file(Path(__file__)),
            "helpers": sha256_file(REPO_ROOT / "diploma_sft" / "mera_core.py"),
        },
        "scope": "public labeled splits; not the closed MERA leaderboard test",
        "adapter_path": str(adapter_path),
        "adapter_config_sha256": sha256_file(adapter_path / "adapter_config.json"),
        "adapter_weights_sha256": sha256_file(adapter_weights_path),
        "base_model": {
            "name": base_model_name,
            "resolved_revision": base_model_revision,
        },
        "selection_indices_sha256": adapter_manifest["selection_indices_sha256"],
        "training_protocol_sha256": adapter_manifest["training_protocol_sha256"],
        "dataset": {
            "name": cfg.benchmark.dataset_name,
            "resolved_revision": resolved_revision,
        },
        "tasks": task_protocols,
        "max_samples_per_task": cfg.benchmark.max_samples_per_task,
        "max_seq_len": int(cfg.benchmark.max_seq_len),
        "batch_size": int(cfg.benchmark.batch_size),
        "chat_template": cfg.model.chat_template,
        "runtime": {
            "gpu": environment["gpu"],
            "packages": {
                name: environment["packages"].get(name)
                for name in (
                    "torch",
                    "transformers",
                    "datasets",
                    "huggingface-hub",
                    "unsloth",
                    "bitsandbytes",
                )
            },
        },
    }
    run_manifest = prepare_benchmark_run_manifest(output_dir, protocol)
    _atomic_write_json(output_dir / "config.resolved.json", OmegaConf.to_container(cfg, resolve=True))
    _atomic_write_json(output_dir / "environment.json", environment)

    run = init_wandb(
        cfg,
        config=to_plain_dict(cfg),
        job_type="benchmark",
        run_name=f"{cfg.experiment_name}-mera-core",
        extra_tags=["mera-core"],
    )
    require_bf16_cuda()
    from unsloth import FastLanguageModel  # noqa: I001
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=cfg.benchmark.max_seq_len,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    loaded_base_commit = getattr(model.config, "_commit_hash", None)
    if loaded_base_commit and loaded_base_commit != base_model_revision:
        raise RuntimeError(
            "Loaded base model revision differs from the benchmark protocol: "
            f"loaded={loaded_base_commit}, expected={base_model_revision}"
        )
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.model.chat_template)
    tokenizer.padding_side = "right"
    FastLanguageModel.for_inference(model)

    started = time.perf_counter()
    task_results = []
    task_paths = []
    for task in tasks:
        dataset = task_datasets[task]
        task_path = output_dir / "tasks" / f"{task}.json"
        result = _evaluate_mera_task(
            task,
            dataset,
            model,
            tokenizer,
            cfg,
            task_path,
            run_manifest["benchmark_protocol_sha256"],
        )
        task_results.append(result)
        task_paths.append(task_path)
        if run is not None:
            values = {f"mera_core/task/{task}/accuracy": result["accuracy"]}
            if "macro_f1" in result:
                values[f"mera_core/task/{task}/macro_f1"] = result["macro_f1"]
            run.log(values)

    metrics = aggregate_task_results(task_results)
    metrics.update(
        {
            "runtime_seconds": time.perf_counter() - started,
            "scope": "public labeled splits; not the closed MERA leaderboard test",
            "dataset_name": cfg.benchmark.dataset_name,
            "dataset_resolved_revision": resolved_revision,
            "adapter_path": str(adapter_path),
            "selection_indices_sha256": adapter_manifest["selection_indices_sha256"],
            "training_protocol_sha256": adapter_manifest["training_protocol_sha256"],
            "benchmark_protocol_sha256": run_manifest["benchmark_protocol_sha256"],
            "truncated_prompts": sum(row["truncated_prompts"] for row in task_results),
            "task_results": task_results,
        }
    )
    metrics_path = output_dir / "mera_core_metrics.json"
    _atomic_write_json(metrics_path, metrics)
    if run is not None:
        run.log(
            {
                "mera_core/accuracy": metrics["accuracy"],
                "mera_core/macro_task_accuracy": metrics["macro_task_accuracy"],
                "mera_core/examples": metrics["examples"],
                "mera_core/truncated_prompts": metrics["truncated_prompts"],
            }
        )
    log_files_as_artifact(
        run,
        name=f"{cfg.experiment_name}-mera-core",
        artifact_type="benchmark-results",
        paths=[
            str(metrics_path),
            str(output_dir / "benchmark_run_manifest.json"),
            *map(str, task_paths),
        ],
        metadata={
            "accuracy": metrics["accuracy"],
            "dataset_revision": resolved_revision,
            "benchmark_protocol_sha256": run_manifest["benchmark_protocol_sha256"],
        },
    )
    if run is not None:
        run.finish()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.benchmark.name not in {"rummlu", "mera_core"}:
        raise NotImplementedError("Use benchmark=rummlu or benchmark=mera_core")
    if not cfg.adapter_path:
        raise ValueError("adapter_path is required")

    adapter_path = Path(to_absolute_path(str(cfg.adapter_path)))
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
    adapter_manifest = _load_adapter_manifest(adapter_path)
    with (adapter_path / "adapter_config.json").open(encoding="utf-8") as stream:
        adapter_config = json.load(stream)
    adapter_weights_path = adapter_path / "adapter_model.safetensors"
    if not adapter_weights_path.is_file():
        raise RuntimeError(f"Adapter weights are missing: {adapter_weights_path}")
    output_dir = Path(to_absolute_path(str(cfg.benchmark_output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.benchmark.name == "mera_core":
        _run_mera_core(
            cfg,
            adapter_path,
            adapter_manifest,
            adapter_config,
            adapter_weights_path,
            output_dir,
        )
        return

    from datasets import get_dataset_config_names, load_dataset

    resolved_revision = resolve_dataset_revision(
        cfg.benchmark.dataset_name,
        cfg.benchmark.revision,
    )
    available_subjects = sorted(
        get_dataset_config_names(cfg.benchmark.dataset_name, revision=resolved_revision)
    )
    subjects = (
        list(cfg.benchmark.subjects)
        if cfg.benchmark.subjects is not None
        else available_subjects
    )
    unknown = sorted(set(subjects) - set(available_subjects))
    if unknown:
        raise ValueError(f"Unknown ruMMLU subjects: {unknown}")

    from huggingface_hub import HfApi

    base_model_name = adapter_config["base_model_name_or_path"]
    base_model_revision = HfApi().model_info(repo_id=base_model_name).sha
    environment = environment_snapshot(REPO_ROOT)

    protocol = {
        "implementation": "public-rummlu-five-shot-loglikelihood-v1",
        "implementation_sha256": {
            "runner": sha256_file(Path(__file__)),
            "helpers": sha256_file(REPO_ROOT / "diploma_sft" / "rummlu.py"),
        },
        "adapter_path": str(adapter_path),
        "adapter_config_sha256": sha256_file(adapter_path / "adapter_config.json"),
        "adapter_weights_sha256": sha256_file(adapter_weights_path),
        "base_model": {
            "name": base_model_name,
            "resolved_revision": base_model_revision,
        },
        "selection_indices_sha256": adapter_manifest["selection_indices_sha256"],
        "training_protocol_sha256": adapter_manifest["training_protocol_sha256"],
        "dataset": {
            "name": cfg.benchmark.dataset_name,
            "resolved_revision": resolved_revision,
        },
        "subjects": subjects,
        "num_fewshot": int(cfg.benchmark.num_fewshot),
        "max_samples_per_subject": cfg.benchmark.max_samples_per_subject,
        "max_seq_len": int(cfg.benchmark.max_seq_len),
        "batch_size": int(cfg.benchmark.batch_size),
        "choice_continuations": list(CHOICES),
        "chat_template": cfg.model.chat_template,
        "runtime": {
            "gpu": environment["gpu"],
            "packages": {
                name: environment["packages"].get(name)
                for name in (
                    "torch",
                    "transformers",
                    "datasets",
                    "huggingface-hub",
                    "unsloth",
                    "bitsandbytes",
                )
            },
        },
    }
    run_manifest = prepare_benchmark_run_manifest(output_dir, protocol)
    _atomic_write_json(output_dir / "config.resolved.json", OmegaConf.to_container(cfg, resolve=True))
    _atomic_write_json(output_dir / "environment.json", environment)

    run = init_wandb(
        cfg,
        config=to_plain_dict(cfg),
        job_type="benchmark",
        run_name=f"{cfg.experiment_name}-rummlu",
        extra_tags=["rummlu"],
    )
    require_bf16_cuda()
    from unsloth import FastLanguageModel  # noqa: I001
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=cfg.benchmark.max_seq_len,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    loaded_base_commit = getattr(model.config, "_commit_hash", None)
    if loaded_base_commit and loaded_base_commit != base_model_revision:
        raise RuntimeError(
            "Loaded base model revision differs from the benchmark protocol: "
            f"loaded={loaded_base_commit}, expected={base_model_revision}"
        )
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.model.chat_template)
    tokenizer.padding_side = "right"
    FastLanguageModel.for_inference(model)

    started = time.perf_counter()
    subject_results = []
    subject_paths = []
    for subject in subjects:
        dev = load_dataset(
            cfg.benchmark.dataset_name,
            subject,
            split="dev",
            revision=resolved_revision,
        )
        test = load_dataset(
            cfg.benchmark.dataset_name,
            subject,
            split="test",
            revision=resolved_revision,
        )
        if cfg.benchmark.max_samples_per_subject is not None:
            test = test.select(
                range(min(int(cfg.benchmark.max_samples_per_subject), len(test)))
            )
        subject_path = output_dir / "subjects" / f"{subject}.json"
        result = _evaluate_subject(
            subject,
            dev,
            test,
            model,
            tokenizer,
            cfg,
            subject_path,
            run_manifest["benchmark_protocol_sha256"],
        )
        subject_results.append(result)
        subject_paths.append(subject_path)
        if run is not None:
            run.log({f"rummlu/subject/{subject}": result["accuracy"]})

    metrics = aggregate_subject_results(subject_results)
    metrics.update(
        {
            "runtime_seconds": time.perf_counter() - started,
            "dataset_name": cfg.benchmark.dataset_name,
            "dataset_resolved_revision": resolved_revision,
            "num_fewshot": int(cfg.benchmark.num_fewshot),
            "adapter_path": str(adapter_path),
            "selection_indices_sha256": adapter_manifest["selection_indices_sha256"],
            "training_protocol_sha256": adapter_manifest["training_protocol_sha256"],
            "benchmark_protocol_sha256": run_manifest["benchmark_protocol_sha256"],
            "truncated_prompts": sum(row["truncated_prompts"] for row in subject_results),
            "subject_results": subject_results,
        }
    )
    metrics_path = output_dir / "rummlu_metrics.json"
    _atomic_write_json(metrics_path, metrics)
    if run is not None:
        run.log(
            {
                "rummlu/accuracy": metrics["accuracy"],
                "rummlu/macro_subject_accuracy": metrics["macro_subject_accuracy"],
                "rummlu/examples": metrics["examples"],
                "rummlu/truncated_prompts": metrics["truncated_prompts"],
            }
        )
    log_files_as_artifact(
        run,
        name=f"{cfg.experiment_name}-rummlu",
        artifact_type="benchmark-results",
        paths=[str(metrics_path), str(output_dir / "benchmark_run_manifest.json"), *map(str, subject_paths)],
        metadata={
            "accuracy": metrics["accuracy"],
            "dataset_revision": resolved_revision,
            "benchmark_protocol_sha256": run_manifest["benchmark_protocol_sha256"],
        },
    )
    if run is not None:
        run.finish()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
